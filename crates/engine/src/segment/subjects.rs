//! Opt-in Subject picker prototype. Suggestions remain transient until confirmed.
//!
//! Discovery runs in two stages, neither on the UI thread: BLIP describes the photo on a thread of its
//! own, then the SAM 3 worker finds each phrase in the photo, reusing the photo it has encoded (and
//! leaving it encoded for Object and Describe). People are offered one by one, sharpest first (the
//! one the camera focused on, in a sports photo), and the sharpest is chosen already.
use crate::Session;
use lightcraft_catalog::PhotoId;
use lightcraft_develop::{Mask, MaskComponent, MaskOp, MaskShape, SegMask};
use lightcraft_geom::Point;

#[derive(Clone, serde::Serialize)]
pub struct Candidate {
    pub label: String,
    pub selected: bool,
    pub seg: SegMask,
    #[serde(skip)]
    pub rgba: Vec<u8>,
    pub size: [usize; 2],
    /// A person: how sharp they are (higher is sharper; candidates are ordered by it). `None` for things.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub focus: Option<f32>,
    /// A person: a point inside them, stored as the Object click that refines this one person.
    #[serde(skip)]
    pub click: Option<Point>,
}

/// What the worker found for one phrase: a person (one of several) or everything a phrase names.
#[cfg(feature = "sam")]
pub(crate) struct Found {
    pub label: String,
    pub logits: Vec<f32>,
    pub person: bool,
}

/// The rendered model input (8-bit RGB) the suggestions are about.
#[cfg(feature = "sam")]
pub(crate) struct Image {
    rgb: Vec<u8>,
    w: usize,
    h: usize,
}

#[cfg(feature = "sam")]
enum Stage {
    /// BLIP is describing the photo (its own thread).
    Caption(std::sync::mpsc::Receiver<Result<(Vec<String>, Image), String>>),
    /// The SAM 3 worker is finding the phrases.
    Segment(std::sync::mpsc::Receiver<Result<Vec<Found>, String>>, Image),
}

#[derive(Default)]
pub struct Picker {
    pub photo: Option<PhotoId>,
    pub open: bool,
    pub busy: bool,
    pub background: bool,
    pub error: Option<String>,
    pub candidates: Vec<Candidate>,
    key: u64,
    #[cfg(feature = "sam")]
    stage: Option<Stage>,
}

/// Most people offered one by one.
#[cfg(feature = "sam")]
const MAX_PEOPLE: usize = 8;

impl Session {
    pub fn subject_poll(&mut self) {
        #[cfg(feature = "sam")]
        {
            use std::sync::mpsc::TryRecvError;
            let Some(stage) = self.segmenter.subjects.stage.take() else { return };
            let valid = |s: &Session| s.segmenter.subjects.photo.and_then(|id| s.segment_key(id).map(|x| x.0)) == Some(s.segmenter.subjects.key);
            let fail = |s: &mut Session, e: String| {
                let picker = &mut s.segmenter.subjects;
                picker.busy = false;
                picker.error = Some(e);
            };
            match stage {
                Stage::Caption(rx) => match rx.try_recv() {
                    Err(TryRecvError::Empty) => self.segmenter.subjects.stage = Some(Stage::Caption(rx)),
                    Err(TryRecvError::Disconnected) => fail(self, "The subject worker stopped. Try again.".into()),
                    Ok(Err(e)) => fail(self, e),
                    Ok(Ok(_)) if !valid(self) => fail(self, "The photo changed. Close this picker and try again.".into()),
                    Ok(Ok((labels, _))) if labels.is_empty() => fail(self, "No suggestions could be selected. Try Describe or Object.".into()),
                    Ok(Ok((labels, image))) => {
                        let (tx, rx) = std::sync::mpsc::channel();
                        // (the job's own outcome carries nothing: nobody listens to it)
                        let (outcome, _) = std::sync::mpsc::channel();
                        let id = self.segmenter.subjects.photo;
                        let job = id.ok_or_else(|| "Photo unavailable".to_string()).and_then(|id| {
                            self.segment_job(id, super::worker::Kind::Subjects { labels, reply: tx }, super::Tag::Wait, outcome, None)
                        });
                        match job.and_then(|job| self.segmenter.worker.submit(job)) {
                            Ok(()) => self.segmenter.subjects.stage = Some(Stage::Segment(rx, image)),
                            Err(e) => fail(self, e),
                        }
                    }
                },
                Stage::Segment(rx, image) => match rx.try_recv() {
                    Err(TryRecvError::Empty) => self.segmenter.subjects.stage = Some(Stage::Segment(rx, image)),
                    Err(TryRecvError::Disconnected) => fail(self, "The SAM 3 worker stopped. Try again.".into()),
                    Ok(Err(e)) => fail(self, e),
                    Ok(Ok(_)) if !valid(self) => fail(self, "The photo changed. Close this picker and try again.".into()),
                    Ok(Ok(found)) => {
                        let candidates = candidates(&image, found);
                        if candidates.is_empty() {
                            fail(self, "No suggestions could be selected. Try Describe or Object.".into());
                        } else {
                            let picker = &mut self.segmenter.subjects;
                            picker.busy = false;
                            picker.candidates = candidates;
                        }
                    }
                },
            }
        }
    }

    pub fn subject_start(&mut self, background: bool) -> Result<(), String> {
        if self.segmenter.subjects.busy {
            return Err("Subject discovery is already running.".into());
        }
        let id = self.active().ok_or("Select a photo first.")?;
        let picker = &self.segmenter.subjects;
        if picker.photo == Some(id)
            && !picker.candidates.is_empty()
            && picker.error.is_none()
            && self.segment_key(id).map(|x| x.0) == Some(picker.key)
        {
            self.segmenter.subjects.open = true;
            self.segmenter.subjects.background = background;
            return Ok(());
        }

        #[cfg(feature = "sam")]
        {
            let blip = std::env::var_os("LIGHTCRAFT_BLIP_DIR").map(std::path::PathBuf::from).ok_or(
                "This prototype needs LIGHTCRAFT_BLIP_DIR pointing to BLIP-base (pytorch_model.bin and vocab.txt). Use Describe or Object meanwhile.",
            )?;
            if !blip.join("pytorch_model.bin").is_file() || !blip.join("vocab.txt").is_file() {
                return Err("BLIP-base files are missing. Use Describe or Object meanwhile.".into());
            }
            if !self.segmenter.installed() {
                return Err("This local prototype needs SAM 3 installed.".into());
            }
            let (key, settings) = self.segment_key(id).ok_or("Photo unavailable")?;
            // the model input, rendered as the SAM 3 worker renders it (same size, look and key)
            let job = self.preview_job(id, super::INPUT_EDGE, super::INPUT_EDGE, false, &settings).ok_or("Photo unavailable")?;
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("subject-caption".into())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| caption(job, &blip)))
                        .unwrap_or_else(|_| Err("The subject models failed unexpectedly. Please try again.".into()));
                    let _ = tx.send(result);
                })
                .map_err(|e| e.to_string())?;
            self.segmenter.subjects =
                Picker { photo: Some(id), open: true, busy: true, background, key, stage: Some(Stage::Caption(rx)), ..Default::default() };
            Ok(())
        }
        #[cfg(not(feature = "sam"))]
        {
            let _ = (id, background);
            Err("Subject discovery is unavailable in this build.".into())
        }
    }

    pub fn subject_create(&mut self) -> Result<(), String> {
        let picker = &self.segmenter.subjects;
        if !picker.open {
            return Err("The subject picker is closed.".into());
        }
        let id = picker.photo.ok_or("No subject suggestions")?;
        if self.active() != Some(id) || self.segment_key(id).map(|x| x.0) != Some(picker.key) {
            return Err("The photo changed. Find subjects again.".into());
        }
        if picker.busy {
            return Err("Wait for the suggestions to finish.".into());
        }
        let chosen: Vec<_> = picker.candidates.iter().filter(|c| c.selected).collect();
        if chosen.is_empty() {
            return Err("Choose at least one subject.".into());
        }
        let mut d = (*self.develop_of(id).ok_or("Photo unavailable")?).clone();
        let mid = d.next_mask_id();
        let components = chosen
            .iter()
            .map(|c| MaskComponent {
                name: Some(c.label.clone()),
                op: MaskOp::Add,
                invert: false,
                // one person is an Object selection (+ / − clicks and the detail pass refine that person
                // alone); a phrase stays a Describe selection
                shape: match c.click {
                    Some(at) => MaskShape::Object { hint: vec![at], exclude: Vec::new(), seg: Some(c.seg.clone()), detail: Vec::new(), edge: 0.0 },
                    None => MaskShape::Prompt { text: c.label.clone(), seg: Some(c.seg.clone()), detail: Vec::new(), edge: 0.0 },
                },
            })
            .collect();
        d.masks.push(Mask {
            id: mid,
            name: if picker.background { "Background" } else { "Subject" }.into(),
            invert: picker.background,
            components,
            ..Default::default()
        });
        self.set_develop(id, d, "Create Subject Selection").map_err(|e| e.to_string())?;
        self.active_mask = Some(mid);
        self.segmenter.subjects.open = false;
        Ok(())
    }
}

/// Render the model input and describe it with BLIP: the phrases to look for.
#[cfg(feature = "sam")]
fn caption(job: crate::media::RenderJob, blip: &std::path::Path) -> Result<(Vec<String>, Image), String> {
    let img = job.run().rendered.map_err(|e| format!("Could not render this photo: {e}"))?.image;
    let rgb: Vec<u8> = img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let caption = lightcraft_segment::caption::describe(blip, &rgb, img.width, img.height).map_err(|e| e.to_string())?;
    log::info!("subject caption: {caption}");
    Ok((lightcraft_segment::caption::candidates(&caption), Image { rgb, w: img.width, h: img.height }))
}

/// On the SAM 3 worker, with the photo encoded: every person on their own (when a phrase names people),
/// everything else a phrase names as one selection.
#[cfg(feature = "sam")]
pub(crate) fn find(model: &mut lightcraft_segment::Sam3, enc: &mut lightcraft_segment::Encoded, labels: &[String]) -> Result<Vec<Found>, String> {
    let mut out = Vec::new();
    let mut people = false;
    for label in labels {
        if label == "person" {
            if !std::mem::replace(&mut people, true) {
                let found = model.segment_instances(enc, "person", 0.5).map_err(|e| e.to_string())?;
                out.extend(found.into_iter().take(MAX_PEOPLE).map(|p| Found { label: "person".into(), logits: p.logits, person: true }));
            }
        } else if let Some(p) = model.segment_text(enc, label, 0.5).map_err(|e| e.to_string())? {
            out.push(Found { label: label.clone(), logits: p.logits, person: false });
        }
    }
    Ok(out)
}

/// The picker's cards: people sharpest first (numbered; the sharpest chosen), then the things, without
/// near-duplicates.
#[cfg(feature = "sam")]
fn candidates(img: &Image, found: Vec<Found>) -> Vec<Candidate> {
    let side = lightcraft_segment::MASK_SIDE;
    let sharp = laplacian(img);
    let mut people: Vec<(f32, Found)> = Vec::new();
    let mut things: Vec<Found> = Vec::new();
    for f in found {
        if f.person {
            people.push((focus(img, &sharp, &f.logits, side), f));
        } else {
            things.push(f);
        }
    }
    people.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out: Vec<Candidate> = Vec::new();
    let n = people.len();
    for (i, (score, f)) in people.into_iter().enumerate() {
        let seg = SegMask::from_logits(side, &f.logits);
        if out.iter().any(|c| same_selection(&c.seg, &seg)) {
            continue;
        }
        let label = if n == 1 { "person".to_string() } else { format!("person {}", i + 1) };
        let click = inside_point(&f.logits, side);
        let (rgba, size) = thumbnail(img, &f.logits, side);
        out.push(Candidate { label, selected: out.is_empty(), seg, rgba, size, focus: Some(score), click });
    }
    for f in things {
        let seg = SegMask::from_logits(side, &f.logits);
        if out.iter().any(|c| same_selection(&c.seg, &seg)) {
            continue;
        }
        let (rgba, size) = thumbnail(img, &f.logits, side);
        out.push(Candidate { label: f.label, selected: false, seg, rgba, size, focus: None, click: None });
    }
    out
}

/// |Laplacian| of the image's luma (0 on the border): fine detail, which is where the photo is in focus.
#[cfg(feature = "sam")]
fn laplacian(img: &Image) -> Vec<f32> {
    let (w, h) = (img.w, img.h);
    let luma: Vec<f32> =
        img.rgb.as_chunks::<3>().0.iter().map(|p| 0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])).collect();
    let mut out = vec![0f32; w * h];
    if w < 3 || h < 3 || luma.len() != w * h {
        return out;
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            out[i] = (4.0 * luma[i] - luma[i - 1] - luma[i + 1] - luma[i - w] - luma[i + w]).abs();
        }
    }
    out
}

/// How sharp the person `logits` selects is: the 90th percentile of the detail well inside them (their
/// outline is left out: a blurred person against a contrasting background has sharp-looking edges).
#[cfg(feature = "sam")]
fn focus(img: &Image, sharp: &[f32], logits: &[f32], side: usize) -> f32 {
    let (w, h) = (img.w, img.h);
    let inside = |x: usize, y: usize| logits.get((y * side / h.max(1)) * side + x * side / w.max(1)).is_some_and(|l| *l > 2.0);
    let mut values: Vec<f32> = Vec::new();
    for y in 2..h.saturating_sub(2) {
        for x in 2..w.saturating_sub(2) {
            if inside(x, y) && inside(x - 2, y) && inside(x + 2, y) && inside(x, y - 2) && inside(x, y + 2) {
                values.push(sharp[y * w + x]);
            }
        }
    }
    if values.len() < 16 {
        return 0.0;
    }
    let k = values.len() * 9 / 10;
    *values.select_nth_unstable_by(k, f32::total_cmp).1
}

/// The most confident point of a selection, in normalized photo coordinates.
#[cfg(feature = "sam")]
fn inside_point(logits: &[f32], side: usize) -> Option<Point> {
    let (i, l) = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1))?;
    (*l > 0.0).then(|| Point::new(((i % side) as f64 + 0.5) / side as f64, ((i / side) as f64 + 0.5) / side as f64))
}

/// A card's picture: the photo at most 160 px, the selection tinted red.
#[cfg(feature = "sam")]
fn thumbnail(img: &Image, logits: &[f32], side: usize) -> (Vec<u8>, [usize; 2]) {
    let longest = img.w.max(img.h).max(1);
    let width = (160 * img.w / longest).clamp(1, 160);
    let height = (160 * img.h / longest).clamp(1, 160);
    let mut rgba = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let at = ((y * img.h / height) * img.w + x * img.w / width) * 3;
            let px = img.rgb.get(at..at + 3).unwrap_or(&[0, 0, 0]);
            let logit = logits.get((y * side / height) * side + x * side / width).copied().unwrap_or(-16.0);
            let alpha = 0.45 / (1.0 + (-logit).exp());
            for (v, tint) in px.iter().zip([255.0, 35.0, 70.0]) {
                rgba.push((f32::from(*v) * (1.0 - alpha) + tint * alpha).round() as u8);
            }
            rgba.push(255);
        }
    }
    (rgba, [width, height])
}

#[cfg_attr(not(feature = "sam"), allow(dead_code))]
fn same_selection(a: &SegMask, b: &SegMask) -> bool {
    if a.side != b.side {
        return false;
    }
    let (Some(a), Some(b)) = (a.logits(), b.logits()) else { return false };
    let (mut intersection, mut union) = (0usize, 0usize);
    for (a, b) in a.iter().zip(b.iter()) {
        if *a > 0.0 || *b > 0.0 {
            union += 1;
        }
        if *a > 0.0 && *b > 0.0 {
            intersection += 1;
        }
    }
    union > 0 && intersection as f64 / union as f64 > 0.9
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn ready() -> Session {
        let mut s = Session::with_demo();
        s.execute("library.select", &json!({"ids": [1]})).unwrap();
        let id = s.active().unwrap();
        let key = s.segment_key(id).unwrap().0;
        let seg = SegMask::from_logits(2, &[12.0, -12.0, 12.0, -12.0]);
        s.segmenter.subjects = Picker {
            photo: Some(id),
            key,
            open: true,
            candidates: ["person", "helmet"]
                .iter()
                .map(|label| Candidate {
                    label: (*label).into(),
                    selected: false,
                    seg: seg.clone(),
                    rgba: vec![],
                    size: [0, 0],
                    focus: None,
                    click: None,
                })
                .collect(),
            ..Default::default()
        };
        s
    }
    #[test]
    fn subject_picker_combines_only_chosen_candidates_and_undoes_once() {
        let mut s = ready();
        assert!(s.execute("mask.subject.create", &json!({})).is_err());
        for i in [0, 1] {
            s.execute("mask.subject.choose", &json!({"index": i, "selected": true})).unwrap();
        }
        s.execute("mask.subject.create", &json!({})).unwrap();
        let d = s.develop_of(s.active().unwrap()).unwrap();
        assert_eq!(d.masks.len(), 1);
        assert_eq!(d.masks[0].components.len(), 2);
        s.execute("edit.undo", &json!({})).unwrap();
        assert!(s.develop_of(s.active().unwrap()).unwrap().masks.is_empty());
    }
    #[test]
    fn subject_picker_background_and_cancel_do_not_rerun_models() {
        let mut s = ready();
        s.subject_start(true).unwrap();
        assert!(!s.segmenter.subjects.busy);
        assert!(s.segmenter.subjects.background);
        let seg = s.segmenter.subjects.candidates[0].seg.clone();
        s.execute("mask.subject.choose", &json!({"index": 0, "selected": true})).unwrap();
        s.execute("mask.subject.create", &json!({})).unwrap();
        let d = s.develop_of(s.active().unwrap()).unwrap();
        assert!(d.masks[0].invert);
        assert!(matches!(&d.masks[0].components[0].shape, MaskShape::Prompt { seg: Some(stored), .. } if *stored == seg));
        let mut s = ready();
        s.execute("mask.subject.dismiss", &json!({})).unwrap();
        assert!(s.execute("mask.subject.create", &json!({})).is_err());
        assert!(s.develop_of(s.active().unwrap()).unwrap().masks.is_empty());
    }
    #[test]
    fn subject_picker_rejects_changed_photos_and_bad_indexes() {
        let mut s = ready();
        assert!(s.execute("mask.subject.choose", &json!({"index": 99, "selected": true})).is_err());
        s.segmenter.subjects.candidates[0].selected = true;
        s.execute("library.select", &json!({"ids": [2]})).unwrap();
        assert!(s.execute("mask.subject.create", &json!({})).is_err());
        s.execute("library.select", &json!({"ids": [1]})).unwrap();
        s.segmenter.subjects.key ^= 1;
        assert!(s.execute("mask.subject.create", &json!({})).is_err());
    }

    /// One person is an Object selection with a click inside them (refining it refines that person); a
    /// phrase stays a Describe selection.
    #[test]
    fn a_chosen_person_is_an_object_selection_of_that_person() {
        let mut s = ready();
        s.segmenter.subjects.candidates[0].click = Some(Point::new(0.25, 0.5));
        for i in [0, 1] {
            s.execute("mask.subject.choose", &json!({"index": i, "selected": true})).unwrap();
        }
        s.execute("mask.subject.create", &json!({})).unwrap();
        let d = s.develop_of(s.active().unwrap()).unwrap();
        assert!(matches!(&d.masks[0].components[0].shape, MaskShape::Object { hint, seg: Some(_), .. } if *hint == vec![Point::new(0.25, 0.5)]));
        assert!(matches!(&d.masks[0].components[1].shape, MaskShape::Prompt { text, .. } if text == "helmet"));
    }

    /// People come one by one, the sharpest first and chosen; things follow, not chosen.
    #[cfg(feature = "sam")]
    #[test]
    fn people_are_ordered_by_sharpness_and_the_sharpest_is_chosen() {
        let side = lightcraft_segment::MASK_SIDE;
        let (w, h) = (96usize, 64usize);
        // left half: fine detail (in focus); right half: smooth (out of focus)
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let v = if x < w / 2 { if (x + y) % 2 == 0 { 230 } else { 20 } } else { 100 + (x as u8) / 4 };
                rgb.extend([v, v, v]);
            }
        }
        let img = Image { rgb, w, h };
        let half = |left: bool| -> Vec<f32> { (0..side * side).map(|i| if ((i % side) < side / 2) == left { 8.0 } else { -8.0 }).collect() };
        let found = vec![
            Found { label: "person".into(), logits: half(false), person: true },
            Found {
                label: "court".into(),
                logits: (0..side * side).map(|i| if i / side > side * 3 / 4 { 8.0 } else { -8.0 }).collect(),
                person: false,
            },
            Found { label: "person".into(), logits: half(true), person: true },
        ];
        let c = candidates(&img, found);
        assert_eq!(c.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(), ["person 1", "person 2", "court"]);
        assert!(c[0].focus.unwrap() > c[1].focus.unwrap(), "{:?} vs {:?}", c[0].focus, c[1].focus);
        assert!(c[0].click.is_some_and(|p| p.x < 0.5), "the sharp person is the left one");
        assert_eq!(c.iter().map(|c| c.selected).collect::<Vec<_>>(), [true, false, false]);
        assert!(c[2].focus.is_none() && c[2].click.is_none());
    }
}
