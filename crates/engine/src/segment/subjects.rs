//! Opt-in Subject picker prototype. Suggestions remain transient until confirmed.
use crate::Session;
use lightcraft_catalog::PhotoId;
use lightcraft_develop::{Mask, MaskComponent, MaskOp, MaskShape, SegMask};

#[derive(Clone, serde::Serialize)]
pub struct Candidate {
    pub label: String,
    pub selected: bool,
    pub seg: SegMask,
    #[serde(skip)]
    pub rgba: Vec<u8>,
    pub size: [usize; 2],
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
    receiver: Option<std::sync::mpsc::Receiver<Result<Vec<Candidate>, String>>>,
}

impl Session {
    pub fn subject_poll(&mut self) {
        #[cfg(feature = "sam")]
        {
            use std::sync::mpsc::TryRecvError;
            let picker = &mut self.segmenter.subjects;
            let Some(rx) = &picker.receiver else { return };
            let result = match rx.try_recv() {
                Ok(r) => r,
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => Err("The subject worker stopped. Try again.".into()),
            };
            picker.receiver = None;
            picker.busy = false;
            let valid = picker.photo.and_then(|id| self.segment_key(id).map(|x| x.0)) == Some(self.segmenter.subjects.key);
            let picker = &mut self.segmenter.subjects;
            match result {
                _ if !valid => picker.error = Some("The photo changed. Close this picker and try again.".into()),
                Ok(c) if c.is_empty() => picker.error = Some("No suggestions could be selected. Try Describe or Object.".into()),
                Ok(c) => picker.candidates = c,
                Err(e) => picker.error = Some(e),
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
            let sam = self.segmenter.dir.clone().ok_or("SAM 3 directory missing")?;
            let (key, settings) = self.segment_key(id).ok_or("Photo unavailable")?;
            let job = self.preview_job(id, super::INPUT_EDGE, super::INPUT_EDGE, false, &settings).ok_or("Photo unavailable")?;
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("subject-discovery".into())
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| discover(job, &blip, &sam)))
                        .unwrap_or_else(|_| Err("The subject models failed unexpectedly. Please try again.".into()));
                    let _ = tx.send(result);
                })
                .map_err(|e| e.to_string())?;
            self.segmenter.subjects = Picker { photo: Some(id), open: true, busy: true, background, key, receiver: Some(rx), ..Default::default() };
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
                shape: MaskShape::Prompt { text: c.label.clone(), seg: Some(c.seg.clone()), detail: Vec::new(), edge: 0.0 },
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

#[cfg(feature = "sam")]
fn discover(job: crate::media::RenderJob, blip: &std::path::Path, sam: &std::path::Path) -> Result<Vec<Candidate>, String> {
    let img = job.run().rendered.map_err(|e| format!("Could not render this photo: {e}"))?.image;
    let rgb: Vec<u8> = img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
    // BLIP is dropped before SAM loads, keeping peak model memory bounded.
    let caption = lightcraft_segment::caption::describe(blip, &rgb, img.width, img.height).map_err(|e| e.to_string())?;
    let labels = lightcraft_segment::caption::candidates(&caption);
    if labels.is_empty() {
        return Ok(Vec::new());
    }
    let mut model = lightcraft_segment::Sam3::load(sam).map_err(|e| e.to_string())?;
    let mut encoded = model.encode(&rgb, img.width, img.height).map_err(|e| e.to_string())?;
    let mut out: Vec<Candidate> = Vec::new();
    for label in labels {
        let Some(prediction) = model.segment_text(&mut encoded, &label, 0.5).map_err(|e| e.to_string())? else { continue };
        let seg = SegMask::from_logits(lightcraft_segment::MASK_SIDE, &prediction.logits);
        if out.iter().any(|c| same_selection(&c.seg, &seg)) {
            continue;
        }
        let longest = img.width.max(img.height).max(1);
        let width = (160 * img.width / longest).clamp(1, 160);
        let height = (160 * img.height / longest).clamp(1, 160);
        let mut rgba = Vec::with_capacity(width * height * 4);
        let side = lightcraft_segment::MASK_SIDE;
        for y in 0..height {
            for x in 0..width {
                let px = img.data.get((y * img.height / height) * img.width + x * img.width / width).copied().unwrap_or([0, 0, 0, 255]);
                let logit = prediction.logits.get((y * side / height) * side + x * side / width).copied().unwrap_or(-16.0);
                let alpha = 0.45 / (1.0 + (-logit).exp());
                for (v, tint) in px.iter().take(3).zip([255.0, 35.0, 70.0]) {
                    rgba.push((f32::from(*v) * (1.0 - alpha) + tint * alpha).round() as u8);
                }
                rgba.push(255);
            }
        }
        out.push(Candidate { label, selected: false, seg, rgba, size: [width, height] });
    }
    Ok(out)
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
                .map(|label| Candidate { label: (*label).into(), selected: false, seg: seg.clone(), rgba: vec![], size: [0, 0] })
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
}
