//! Assisted culling: score photos for focus and clipping, group similar shots taken close
//! together (bursts) and mark the sharpest of each group; optionally reject blurry ones and pick
//! each group's best, and stack each burst with its best photo on top. Focus is measured on the
//! subject where one is known (face and pet regions, else the camera's focus point), else on the
//! frame. Scores are kept on the photos (rules / smart albums: `sharpness`, `bestOfGroup`). Bursts
//! can be arranged by hand (`photo.moveToBurst`, Window → Bursts); culling run again keeps them.

use lightcraft_catalog::{Analysis, Flag, FocusOn, Op, PhotoId};
use lightcraft_geom::Rect;
use lightcraft_pipeline::SourceInfo;
use lightcraft_raster::Rgb32f;
use serde_json::{Value, json};

use super::{CommandSpec, always, cmd};
use crate::media::{DecodedSource, SourceLevel, SourceRef};
use crate::{Result, Session};

/// Seconds between shots of the same burst at most (default of `burstGap`).
const BURST_GAP: f64 = 10.0;
/// Signature similarity for "the same scene" (default of `similarity`).
const SAME: f64 = 0.93;
/// The longest `burstGap` accepted: a day.
const MAX_GAP: f64 = 86_400.0;

fn secs(iso: &str) -> Option<i64> {
    let d = lightcraft_meta::DateTime::parse_iso(iso)?;
    Some(d.unix_seconds())
}

/// How far a subject box is grown on every side before its focus is measured (a share of its
/// width and height): a face region is the face itself, and a little around it keeps hair, ears
/// and the edge of the chin in.
const SUBJECT_PAD: f64 = 0.15;
/// The shortest side (pixels, on the thumbnail) a subject box needs to be scored there; a smaller
/// one is measured on the larger preview source instead.
const SUBJECT_MIN_PX: usize = 48;
/// At most this many subject boxes are measured on a photo (the largest ones).
const MAX_BOXES: usize = 8;

/// Where the photo's subject is, as boxes on the upright (EXIF-oriented) frame — the frame the
/// sources are decoded in and regions and the focus point are stored in, so the numbers are used
/// as they are: face and pet regions (each grown by [`SUBJECT_PAD`]), else the camera's focus
/// point; none when the photo has neither.
fn subject_boxes(p: &lightcraft_catalog::Photo) -> (Vec<Rect>, FocusOn) {
    use lightcraft_meta::RegionKind;
    let grow = |r: &Rect| -> Option<Rect> {
        if ![r.x0, r.y0, r.x1, r.y1].iter().all(|v| v.is_finite()) {
            return None;
        }
        let (x0, x1) = (r.x0.min(r.x1).clamp(0.0, 1.0), r.x0.max(r.x1).clamp(0.0, 1.0));
        let (y0, y1) = (r.y0.min(r.y1).clamp(0.0, 1.0), r.y0.max(r.y1).clamp(0.0, 1.0));
        let (w, h) = (x1 - x0, y1 - y0);
        (w > 0.0 && h > 0.0).then(|| Rect {
            x0: (x0 - w * SUBJECT_PAD).max(0.0),
            y0: (y0 - h * SUBJECT_PAD).max(0.0),
            x1: (x1 + w * SUBJECT_PAD).min(1.0),
            y1: (y1 + h * SUBJECT_PAD).min(1.0),
        })
    };
    let area = |r: &Rect| (r.x1 - r.x0) * (r.y1 - r.y0);
    let mut faces: Vec<Rect> =
        p.meta.regions.iter().filter(|r| matches!(r.kind, RegionKind::Face | RegionKind::Pet)).filter_map(|r| grow(&r.rect)).collect();
    if !faces.is_empty() {
        faces.sort_by(|a, b| area(b).total_cmp(&area(a)));
        faces.truncate(MAX_BOXES);
        return (faces, FocusOn::Face);
    }
    match p.meta.focus.as_ref().and_then(|f| grow(&f.rect)) {
        Some(r) => (vec![r], FocusOn::FocusPoint),
        None => (Vec::new(), FocusOn::Frame),
    }
}

/// Focus on the photo's subject in `img`: the boxes' scores averaged by area, so the prominent
/// faces of a group count most; `None` when no box could be scored.
fn subject_on(img: &Rgb32f, boxes: &[Rect]) -> Option<f32> {
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for r in boxes {
        if let Some(v) = lightcraft_pipeline::cull::sharpness_in(img, r).filter(|v| v.is_finite()) {
            let w = ((r.x1 - r.x0) * (r.y1 - r.y0)).max(f64::MIN_POSITIVE);
            sum += f64::from(v) * w;
            weight += w;
        }
    }
    (weight > 0.0).then(|| (sum / weight) as f32)
}

/// Whether the subject is measured on the preview instead of the thumbnail (`thumb`): some box is
/// too small on the thumbnail and the photo (`full_edge` pixels long) has more pixels than it. All
/// boxes of a photo are measured on one source; if none can be scored on the preview, they are
/// measured on the thumbnail. A score's scale depends on the source (a preview shows finer
/// detail), so scores from different sources are not compared (see `analyze`). The preview is
/// loaded for such a photo only, and not kept: a culling run would push the loupe's out of memory
/// and leave the last few measured in its place.
fn wants_preview(thumb: &Rgb32f, boxes: &[Rect], full_edge: usize) -> bool {
    boxes.iter().any(|r| lightcraft_pipeline::cull::subject_px(thumb, r) < SUBJECT_MIN_PX) && full_edge > thumb.width.max(thumb.height)
}

/// Photos measured side by side on rayon's pool, a batch at a time, results in the order of
/// `jobs` (each `(photo, weight, job)`), off the session. A batch is at most a photo per pool
/// thread, and no more than the memory gate's limit counts for decoding them (each job's `weight`,
/// [`crate::files::gate_weight`]): the decodes count there without waiting (interactive work), so
/// the batch is what keeps them within it. `go(done)` is asked before each batch: `false` stops,
/// and the photos left are not measured.
fn measure_all<J: Send, R: Send>(
    jobs: Vec<(PhotoId, usize, J)>,
    measure: impl Fn(PhotoId, J) -> R + Sync,
    go: &mut dyn FnMut(usize) -> bool,
) -> Vec<R> {
    use rayon::prelude::*;
    let (limit, threads) = (crate::memory::work_gate().usage().1, rayon::current_num_threads().max(1));
    let mut out = Vec::with_capacity(jobs.len());
    let mut rest = jobs.into_iter().peekable();
    while rest.peek().is_some() {
        if !go(out.len()) {
            break;
        }
        let (mut batch, mut held) = (Vec::new(), 0usize);
        while batch.len() < threads
            && let Some(next) = rest.peek()
        {
            if !batch.is_empty() && held.saturating_add(next.1) > limit {
                break;
            }
            held = held.saturating_add(next.1);
            batch.extend(rest.next());
        }
        // No deadlock at the memory gate: a decode waits there only as background work
        // (`memory::in_background`, a thread-local the app's render worker threads set, never
        // rayon's), so these decodes, and the parallel loops nested in them, count without waiting.
        out.extend(batch.into_par_iter().map(|(id, _, j)| measure(id, j)).collect::<Vec<_>>());
    }
    out
}

/// What one photo measured to.
struct Row {
    id: PhotoId,
    time: Option<i64>,
    /// The score culling goes by (the subject's when there is one).
    sharp: f32,
    /// Focus of the whole frame.
    frame: f32,
    subject: Option<f32>,
    /// The subject was measured on the preview (else on the thumbnail).
    on_preview: bool,
    focus_on: FocusOn,
    clip: f32,
    sig: [f32; 64],
}

/// What measuring one photo needs, read from the session by [`cull_prepare`].
struct Job {
    thumb: std::result::Result<SourceRef, String>,
    /// Only for a photo with a subject (a small one is measured on it), and its weight at the gate.
    preview: Option<(SourceRef, usize)>,
    time: Option<i64>,
    boxes: Vec<Rect>,
    kind: FocusOn,
    full_edge: usize,
}

/// What `photo.analyze` was asked to do with the scores.
#[derive(Clone, Copy)]
struct Params {
    reject_below: Option<f32>,
    pick_best: bool,
    gap: f64,
    same: f32,
    stack_groups: bool,
    /// Regroup photos whose burst was arranged by hand too.
    reset: bool,
}

/// Assisted Culling in three steps, so the app measures off its UI thread with a progress dialog
/// and Cancel: the photos and what measuring them needs, read from the session ([`cull_prepare`]),
/// measured anywhere ([`CullPlan::measure`]) and applied to the session ([`cull_finish`]).
/// `photo.analyze` runs all three at once.
pub struct CullPlan {
    params: Params,
    jobs: Vec<(PhotoId, usize, Job)>,
    library: u64,
}

/// At most this many measured thumbnails are kept in memory afterwards (as [`Session::source_now`]
/// keeps them): the last ones measured, so a large run doesn't hold every thumbnail until it ends.
const KEEP_THUMBS: usize = 256;

/// What [`CullPlan::measure`] found, applied by [`cull_finish`].
pub struct CullMeasured {
    params: Params,
    rows: Vec<Row>,
    failed: Vec<Value>,
    kept: std::collections::VecDeque<(PhotoId, DecodedSource)>,
    learned: Vec<(PhotoId, SourceInfo)>,
    library: u64,
    /// Measuring was stopped: nothing is applied.
    pub cancelled: bool,
}

/// A true / false parameter of `photo.analyze` (false when left out).
fn flag_param(p: &Value, key: &str) -> Result<bool> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(v) => v.as_bool().ok_or_else(|| super::bad("photo.analyze", format!("`{key}` must be true or false"))),
    }
}

/// The photos `photo.analyze {p}` culls (the selected ones, else all in view) and what measuring
/// them needs; refused before anything is measured when the parameters are wrong.
pub fn cull_prepare(s: &mut Session, p: &Value) -> Result<CullPlan> {
    let mut ids: Vec<PhotoId> = if p.get("ids").is_some() || s.selection.ids.len() > 1 { s.targets(p)? } else { s.visible_cloned() };
    // a photo named twice is measured (and stacked) once
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| seen.insert(*id));
    let (gap, same) = burst_params(p)?;
    let params = Params {
        reject_below: p.get("rejectBelow").and_then(Value::as_f64).map(|v| v as f32),
        pick_best: p.get("pickBest").and_then(Value::as_bool).unwrap_or(false),
        gap,
        same,
        stack_groups: flag_param(p, "stackGroups")?,
        reset: flag_param(p, "resetBursts")?,
    };
    let jobs = ids
        .into_iter()
        .map(|id| {
            let thumb = s.source_load(id, SourceLevel::Thumb);
            let (time, (boxes, kind), full_edge, size) = match s.catalog.photo(id) {
                Some(ph) => (
                    ph.captured.as_deref().and_then(secs),
                    subject_boxes(ph),
                    ph.width.max(ph.height) as usize,
                    usize::try_from(ph.file_size).unwrap_or(usize::MAX),
                ),
                None => (None, (Vec::new(), FocusOn::Frame), 0, 0),
            };
            let weight = |level: SourceLevel| crate::files::gate_weight(size, level.max_edge());
            let preview =
                if boxes.is_empty() { None } else { s.source_load(id, SourceLevel::Preview).ok().map(|r| (r, weight(SourceLevel::Preview))) };
            (id, weight(SourceLevel::Thumb), Job { thumb, preview, time, boxes, kind, full_edge })
        })
        .collect();
    Ok(CullPlan { params, jobs, library: s.library_generation() })
}

impl CullPlan {
    /// How many photos are measured.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Decode and measure the photos, on all cores, off the session: thumbnail-level sources (fast,
    /// and enough for focus at 512 px), then a subject too small there on the preview (see
    /// `wants_preview`). `progress(done, total)` is asked between batches (the previews add to
    /// `total` once the thumbnails are done); `false` stops, and nothing is applied.
    pub fn measure(self, progress: &mut dyn FnMut(usize, usize) -> bool) -> CullMeasured {
        let CullPlan { params, jobs, library } = self;
        let total = jobs.len();
        let mut cancelled = false;
        let measured = measure_all(
            jobs,
            |id, j: Job| {
                use lightcraft_pipeline::cull;
                let src = match j.thumb.and_then(|r| r.load_source()) {
                    Ok(src) => src,
                    Err(e) => return (Err(json!([id.0, e])), None),
                };
                let (img, frame) = (&src.image, cull::sharpness(&src.image));
                let row = Row {
                    id,
                    time: j.time,
                    sharp: frame,
                    frame,
                    subject: subject_on(img, &j.boxes),
                    on_preview: false,
                    focus_on: j.kind,
                    clip: cull::clipped(img),
                    sig: cull::signature(img),
                };
                let preview = j.preview.filter(|_| wants_preview(img, &j.boxes, j.full_edge)).map(|(r, w)| (r, w, j.boxes));
                (Ok((row, preview)), Some(src))
            },
            &mut |done| {
                let go = progress(done, total);
                cancelled |= !go;
                go
            },
        );
        let (mut rows, mut failed, mut small) = (Vec::new(), Vec::new(), Vec::new());
        let (mut kept, mut learned) = (std::collections::VecDeque::new(), Vec::new());
        for (m, src) in measured {
            match m {
                Ok((r, preview)) => {
                    if let Some(src) = src {
                        kept.push_back((r.id, src));
                        if kept.len() > KEEP_THUMBS
                            && let Some((id, old)) = kept.pop_front()
                            && let Some(info) = old.info
                        {
                            learned.push((id, info));
                        }
                    }
                    if let Some((r2, w, boxes)) = preview {
                        small.push((rows.len(), (r.id, w, (r2, boxes))));
                    }
                    rows.push(r);
                }
                Err(e) => failed.push(e),
            }
        }
        if !cancelled && !small.is_empty() {
            let n = small.len();
            let (at, jobs): (Vec<usize>, Vec<_>) = small.into_iter().unzip();
            let scores = measure_all(
                jobs,
                |_, (r, bx): (SourceRef, Vec<Rect>)| match r.load_source() {
                    Ok(pv) => (subject_on(&pv.image, &bx), pv.info),
                    Err(_) => (None, None),
                },
                &mut |done| {
                    let go = progress(total + done, total + n);
                    cancelled |= !go;
                    go
                },
            );
            for (i, (v, info)) in at.into_iter().zip(scores) {
                let Some(r) = rows.get_mut(i) else { continue };
                if let Some(info) = info {
                    learned.push((r.id, info));
                }
                if let Some(v) = v {
                    (r.subject, r.on_preview) = (Some(v), true);
                }
            }
        }
        for r in &mut rows {
            r.sharp = r.subject.unwrap_or(r.frame);
            if r.subject.is_none() {
                r.focus_on = FocusOn::Frame;
            }
        }
        CullMeasured { params, rows, failed, kept, learned, library, cancelled }
    }
}

/// Apply what was measured: the thumbnails kept and what their decodes read, then (unless measuring
/// was cancelled) the scores, bursts, flags and stacks as one undo step. Photos removed meanwhile
/// are left out; refused when another library was opened meanwhile.
pub fn cull_finish(s: &mut Session, m: CullMeasured) -> Result<Value> {
    if m.library != s.library_generation() {
        return Err(super::bad("photo.analyze", "another library was opened while the photos were measured: nothing was changed"));
    }
    for (id, src) in m.kept {
        if s.catalog.photo(id).is_some() {
            s.keep_source(id, SourceLevel::Thumb, src);
        }
    }
    for (id, info) in &m.learned {
        if s.catalog.photo(*id).is_some() {
            s.learn_source(*id, Some(info));
        }
    }
    if m.cancelled {
        return Ok(json!({"cancelled": true, "measured": m.rows.len()}));
    }
    let rows = m.rows.into_iter().filter(|r| s.catalog.photo(r.id).is_some_and(|p| !p.deleted)).collect();
    apply(s, &m.params, rows, m.failed)
}

fn analyze(s: &mut Session, p: &Value) -> Result<Value> {
    let plan = cull_prepare(s, p)?;
    let measured = plan.measure(&mut |_, _| true);
    cull_finish(s, measured)
}

/// Group the measured photos into bursts, mark each burst's best and record it all (one undo step).
fn apply(s: &mut Session, params: &Params, mut rows: Vec<Row>, failed: Vec<Value>) -> Result<Value> {
    let Params { reject_below, pick_best, gap, same, stack_groups, reset } = *params;
    // bursts: consecutive (by capture time) look-alike shots
    rows.sort_by_key(|r| (r.time.unwrap_or(i64::MAX), r.id));
    // A photo whose burst was arranged by hand keeps it (burst, place, pick): only its scores
    // are measured again. The others are grouped among themselves, in bursts numbered after
    // every burst that stays (earlier runs on other photos, the ones arranged by hand).
    let by_hand = |id: PhotoId| -> Option<Analysis> { s.catalog.photo(id).and_then(|p| p.analysis).filter(|a| !reset && a.by_hand()) };
    let kept: Vec<Option<Analysis>> = rows.iter().map(|r| by_hand(r.id)).collect();
    let auto: Vec<usize> = (0..rows.len()).filter(|i| kept[*i].is_none()).collect();
    let regrouped: std::collections::HashSet<PhotoId> = auto.iter().map(|i| rows[*i].id).collect();
    let mut next = s.catalog.photos().filter(|p| !regrouped.contains(&p.id)).filter_map(|p| p.analysis.and_then(|a| a.group)).max().unwrap_or(0) + 1;
    let mut group_of: Vec<Option<u32>> = vec![None; rows.len()];
    for w in auto.windows(2) {
        let (i, j) = (w[0], w[1]);
        let (a, b) = (&rows[i], &rows[j]);
        let close = matches!((a.time, b.time), (Some(x), Some(y)) if (y.saturating_sub(x)).unsigned_abs() as f64 <= gap);
        if close && lightcraft_pipeline::cull::similarity(&a.sig, &b.sig) >= same {
            let g = *group_of[i].get_or_insert_with(|| {
                next += 1;
                next - 1
            });
            group_of[j] = Some(g);
        }
    }
    // a group's best is its sharpest on the subject when every shot has one measured on the same
    // source (the same thing at the same scale is compared); when only some do (a face the
    // detector missed, a turned head) or a subject straddles the thumbnail / preview limit within
    // the burst, the whole frames are compared instead, so unlike scores are never set against
    // each other
    let mut on_subject: std::collections::HashMap<u32, (bool, Option<bool>)> = Default::default();
    for (i, g) in group_of.iter().enumerate() {
        if let (Some(g), Some(r)) = (g, rows.get(i)) {
            let (all, level) = on_subject.entry(*g).or_insert((true, None));
            *all &= r.subject.is_some() && level.is_none_or(|l| l == r.on_preview);
            *level = Some(r.on_preview);
        }
    }
    let mut best: std::collections::HashMap<u32, (usize, f32)> = Default::default();
    for (i, g) in group_of.iter().enumerate() {
        if let Some(g) = g {
            let score = if on_subject.get(g).is_some_and(|o| o.0) { rows[i].sharp } else { rows[i].frame };
            let e = best.entry(*g).or_insert((i, -1.0));
            if score > e.1 {
                *e = (i, score);
            }
        }
    }
    let mut ops = Vec::new();
    let (mut rejected, mut picked) = (0, 0);
    let mut out = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let scores = |a: Analysis| Analysis { sharpness: r.sharp, clipped: r.clip, subject: r.subject, focus_on: r.focus_on, frame: r.subject.map(|_| r.frame), ..a };
        let analysis = match kept[i] {
            Some(k) => scores(k),
            None => {
                let g = group_of[i];
                scores(Analysis { group: g, best: g.is_some_and(|g| best.get(&g).is_some_and(|b| b.0 == i)), rank: None, ..Default::default() })
            }
        };
        let (g, is_best) = (analysis.group, analysis.best);
        ops.push(Op::SetAnalysis { id: r.id, analysis: Some(analysis) });
        let flag = s.catalog.photo(r.id).map(|p| p.flag).unwrap_or_default();
        if reject_below.is_some_and(|t| r.sharp < t) && flag != Flag::Reject {
            ops.push(Op::SetFlag { id: r.id, flag: Flag::Reject });
            rejected += 1;
        } else if pick_best && is_best && flag == Flag::None {
            ops.push(Op::SetFlag { id: r.id, flag: Flag::Pick });
            picked += 1;
        }
        let one = |v: f32| (v * 10.0).round() / 10.0;
        out.push(json!({
            "id": r.id.0,
            "sharpness": one(r.sharp),
            "subject": r.subject.map(one),
            "focusOn": r.focus_on.id(),
            "measuredOn": r.subject.map(|_| if r.on_preview { "preview" } else { "thumbnail" }),
            "frame": one(r.frame),
            "clipped": r.clip,
            "group": g,
            "best": is_best,
            "byHand": kept[i].is_some(),
        }));
    }
    let mut stacks = 0;
    if stack_groups {
        // Photos already in a stack stay where they are (like `stack.auto`): a burst is stacked
        // from its other photos when two or more remain. A burst culling grouped has the sharpest
        // of those on top; a burst arranged by hand keeps its order, its pick (place 0) on top.
        let index = s.catalog.stack_index();
        let mut members: std::collections::BTreeMap<u32, Vec<(PhotoId, f32)>> = Default::default();
        let mut arranged: std::collections::BTreeMap<u32, Vec<(u32, PhotoId)>> = Default::default();
        for (i, r) in rows.iter().enumerate() {
            if index.contains_key(&r.id) || !s.catalog.photo(r.id).is_some_and(|ph| !ph.deleted) {
                continue;
            }
            match kept[i] {
                Some(k) => {
                    if let Some(g) = k.group {
                        arranged.entry(g).or_default().push((k.rank.unwrap_or(u32::MAX), r.id));
                    }
                }
                None => {
                    if let Some(g) = group_of[i] {
                        // the group's best goes on top (ranked as `best` was, subject or frame)
                        let top = best.get(&g).is_some_and(|b| b.0 == i);
                        members.entry(g).or_default().push((r.id, if top { f32::INFINITY } else { r.sharp }));
                    }
                }
            }
        }
        let mut orders: Vec<Vec<PhotoId>> = Vec::new();
        for group in members.values() {
            let Some(top) = group.iter().fold(None::<(PhotoId, f32)>, |b, m| if b.is_none_or(|b| m.1 > b.1) { Some(*m) } else { b }) else {
                continue;
            };
            let mut photos = vec![top.0];
            photos.extend(group.iter().map(|m| m.0).filter(|p| *p != top.0));
            orders.push(photos);
        }
        for group in arranged.values_mut() {
            group.sort();
            orders.push(group.iter().map(|m| m.1).collect());
        }
        for photos in orders {
            if photos.len() >= 2
                && let Some(op) = s.catalog.group_ops(photos[0], &photos, true)
            {
                ops.push(op);
                stacks += 1;
            }
        }
    }
    if !ops.is_empty() {
        s.commit("Assisted Culling", Op::Batch { ops })?;
    }
    if stacks > 0 {
        keep_active_shown(s);
    }
    let kept = kept.iter().filter(|k| k.is_some()).count();
    Ok(json!({"photos": out, "groups": best.len(), "kept": kept, "stacks": stacks, "rejected": rejected, "picked": picked, "failed": failed}))
}

/// A photo now hidden inside a collapsed stack must not stay the active one: its stack's top
/// takes over, and the rest of the selection stays selected.
fn keep_active_shown(s: &mut Session) {
    let vis = s.visible_cloned();
    if let Some(a) = s.selection.active
        && !vis.contains(&a)
        && let Some(st) = s.catalog.stack_of(a)
    {
        let top = vis.iter().copied().find(|p| st.photos.contains(p)).unwrap_or(st.top());
        s.selection.ids.retain(|id| vis.contains(id));
        if !s.selection.ids.contains(&top) {
            s.selection.ids.push(top);
        }
        s.selection.active = Some(top);
    }
}

/// Where `photo.moveToBurst` puts the photos.
#[derive(Clone, Copy, PartialEq)]
enum Dest {
    Burst(u32),
    New,
    Nowhere,
}

/// Arrange bursts by hand: `ids` go into a burst at `index` (0 = the pick), into a new burst, or
/// out of every burst. Every burst touched is then arranged by hand ([`Analysis::rank`]), so
/// culling run again keeps it. What marks a burst's pick follows its first photo: `best` (the
/// Best of Similar Shots rule), the Pick flag when culling's pick had it, and the top of the
/// burst's stack when the burst is stacked. One undo step.
fn move_to_burst(s: &mut Session, p: &Value) -> Result<Value> {
    use std::collections::{BTreeMap, HashMap};
    const C: &str = "photo.moveToBurst";
    let mut ids = s.targets(p)?;
    let mut seen = std::collections::HashSet::new();
    ids.retain(|id| seen.insert(*id));
    if ids.is_empty() {
        return Err(super::bad(C, "no photos"));
    }
    for id in &ids {
        match s.catalog.photo(*id) {
            Some(ph) if ph.deleted => return Err(super::bad(C, format!("“{}” is deleted", ph.file_name))),
            Some(ph) if ph.analysis.is_none() => {
                return Err(super::bad(C, format!("“{}” has no culling scores yet: run Assisted Culling on it first", ph.file_name)));
            }
            Some(_) => {}
            None => return Err(super::bad(C, format!("no photo {}", id.0))),
        }
    }
    let group = |s: &Session, id: PhotoId| s.catalog.photo(id).and_then(|p| p.analysis).and_then(|a| a.group);
    let index = s.catalog.burst_index();
    let into = match p.get("burst") {
        None => Dest::Burst(group(s, ids[0]).ok_or_else(|| super::bad(C, "the photo isn't in a burst: say which `burst` it goes to"))?),
        Some(Value::Null) => Dest::Nowhere,
        Some(Value::String(n)) if n == "new" => Dest::New,
        Some(v) => match v.as_u64().and_then(|n| u32::try_from(n).ok()) {
            Some(n) if index.contains_key(&n) => Dest::Burst(n),
            Some(n) => return Err(super::bad(C, format!("there is no burst {n}"))),
            None => return Err(super::bad(C, "`burst` must be a burst number, \"new\" or null")),
        },
    };
    let at = match p.get("index") {
        None | Some(Value::Null) => usize::MAX,
        Some(v) => v.as_u64().map(|n| n as usize).ok_or_else(|| super::bad(C, "`index` must be a place in the burst: 0 (the pick), 1, …"))?,
    };
    // the bursts touched, before and after
    let target = match into {
        Dest::Burst(g) => Some(g),
        Dest::New => Some(s.catalog.next_burst_id()),
        Dest::Nowhere => None,
    };
    let mut old: BTreeMap<u32, Vec<PhotoId>> = BTreeMap::new();
    for g in ids.iter().filter_map(|id| group(s, *id)).chain(target) {
        old.entry(g).or_insert_with(|| index.get(&g).cloned().unwrap_or_default());
    }
    let mut new = old.clone();
    for members in new.values_mut() {
        members.retain(|m| !ids.contains(m));
    }
    if let Some(g) = target {
        let members = new.entry(g).or_default();
        let at = at.min(members.len());
        members.splice(at..at, ids.iter().copied());
    }
    // each photo's burst, place and pick: a burst a move leaves with one photo is no burst (the
    // burst moved into stays one, so a new burst may start with one photo for more to join)
    let remains = |g: u32, members: &Vec<PhotoId>| members.len() >= 2 || (target == Some(g) && !members.is_empty());
    let mut placed: HashMap<PhotoId, (Option<u32>, u32, bool)> = ids.iter().map(|id| (*id, (None, 0, false))).collect();
    for (g, members) in &new {
        let stays = remains(*g, members);
        for (i, m) in members.iter().enumerate() {
            placed.insert(*m, if stays { (Some(*g), i as u32, i == 0) } else { (None, 0, false) });
        }
    }
    let mut ops = Vec::new();
    let mut by_id: Vec<_> = placed.iter().collect();
    by_id.sort_by_key(|(id, _)| **id);
    for (id, (g, rank, best)) in by_id {
        let Some(was) = s.catalog.photo(*id).and_then(|p| p.analysis) else { continue };
        let now = Analysis { group: *g, rank: Some(*rank), best: *best, ..was };
        if now != was {
            ops.push(Op::SetAnalysis { id: *id, analysis: Some(now) });
        }
    }
    // what the target burst holds now
    let members: Option<Vec<u64>> = target.and_then(|g| new.get(&g).filter(|m| remains(g, m))).map(|m| m.iter().map(|p| p.0).collect());
    let report = |changed: bool| json!({"burst": members.as_ref().and(target), "pick": members.as_ref().and_then(|m| m.first()), "photos": members, "changed": changed});
    if ops.is_empty() {
        return Ok(report(false));
    }
    // the Pick culling gave a burst's pick goes to its new pick
    let is_pick = |id: PhotoId| placed.get(&id).is_some_and(|p| p.2);
    let flag = |s: &Session, id: PhotoId| s.catalog.photo(id).map(|p| p.flag).unwrap_or_default();
    let mut flags: BTreeMap<PhotoId, Flag> = BTreeMap::new();
    for (g, before) in &old {
        let Some(old_pick) = before.iter().copied().find(|m| s.catalog.photo(*m).and_then(|p| p.analysis).is_some_and(|a| a.best)) else { continue };
        let new_pick = new.get(g).filter(|m| remains(*g, m)).and_then(|m| m.first()).copied();
        if flag(s, old_pick) != Flag::Pick || new_pick == Some(old_pick) {
            continue;
        }
        if !is_pick(old_pick) {
            flags.insert(old_pick, Flag::None);
        }
        // the new pick was chosen by hand: it takes the Pick even over a Reject
        if let Some(n) = new_pick {
            flags.insert(n, Flag::Pick);
        }
    }
    ops.extend(flags.into_iter().map(|(id, flag)| Op::SetFlag { id, flag }));
    // a stacked burst (a stack of exactly its photos, as culling stacks them) follows it: its stack
    // shrinks, grows or goes, in the burst's order. Shrinking first: a photo is in one stack at a time.
    let (mut shrink, mut grow) = (Vec::new(), Vec::new());
    let mut grows_target = false;
    let mut followed = std::collections::HashSet::new();
    for (g, before) in old.iter().filter(|(_, m)| m.len() >= 2) {
        let Some(st) = s.catalog.stack_of(before[0]).filter(|st| st.photos.len() == before.len() && before.iter().all(|m| st.photos.contains(m)))
        else {
            continue;
        };
        followed.insert(st.id);
        let after = new.get(g).filter(|m| m.len() >= 2);
        match after {
            None => shrink.push(Op::RemoveStack { id: st.id }),
            Some(after) if after == &st.photos => {}
            Some(after) => {
                let op = Op::SetStack { id: st.id, photos: after.clone(), collapsed: st.collapsed };
                if after.iter().all(|m| st.photos.contains(m)) {
                    shrink.push(op)
                } else {
                    grows_target = true;
                    grow.push(op)
                }
            }
        }
    }
    if grows_target {
        // photos joining a stacked burst leave any other stack they are in
        let strays: Vec<PhotoId> = ids.iter().copied().filter(|id| s.catalog.stack_of(*id).is_some_and(|st| !followed.contains(&st.id))).collect();
        shrink.extend(s.catalog.remove_from_stacks_ops(&strays));
    }
    let restacked = !shrink.is_empty() || !grow.is_empty();
    ops.extend(shrink);
    ops.extend(grow);
    let same = matches!(into, Dest::Burst(g) if ids.iter().all(|id| group(s, *id) == Some(g)));
    let label = match into {
        Dest::Burst(_) if same && at == 0 => "Choose Burst Pick",
        Dest::Burst(_) if same => "Rearrange Burst",
        Dest::Burst(_) => "Move to Burst",
        Dest::New => "New Burst",
        Dest::Nowhere => "Remove from Burst",
    };
    s.commit(label, Op::Batch { ops })?;
    if restacked {
        keep_active_shown(s);
    }
    Ok(report(true))
}

/// `burstGap` (seconds, 0..=a day) and `similarity` (0..=1) of `photo.analyze`, refused before
/// anything is measured when they are not numbers in range.
fn burst_params(p: &Value) -> Result<(f64, f32)> {
    const C: &str = "photo.analyze";
    let num = |key: &str, d: f64, max: f64| -> Result<f64> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() && (0.0..=max).contains(&x) => Ok(x),
                _ => Err(super::bad(C, format!("`{key}` must be a number from 0 to {max}"))),
            },
        }
    };
    Ok((num("burstGap", BURST_GAP, MAX_GAP)?, num("similarity", SAME, 1.0)? as f32))
}

/// What decoding photo `id`'s thumbnail-level source counts for at the memory gate.
fn thumb_weight(s: &Session, id: PhotoId) -> usize {
    s.catalog.photo(id).map_or(0, |p| crate::files::gate_weight(usize::try_from(p.file_size).unwrap_or(usize::MAX), SourceLevel::Thumb.max_edge()))
}

/// Find Similar: photos that look like `id` (signatures from thumbnail-level sources, cached by
/// content), most similar first; the view is filtered to them.
fn find_similar(s: &mut Session, p: &Value) -> Result<Value> {
    const C: &str = "library.findSimilar";
    let id = super::photo_id(p, C)?.or(s.active()).ok_or_else(|| super::bad(C, "no photo"))?;
    let min = p.get("similarity").and_then(Value::as_f64).unwrap_or(0.8) as f32;
    let all: Vec<PhotoId> = s.catalog.photos().filter(|q| q.in_library()).map(|q| q.id).collect();
    // signatures not cached yet, measured side by side: once for each content, from the next photo
    // with that content when one can't be read (an offline copy)
    let mut todo: Vec<(PhotoId, String)> = std::iter::once(id)
        .chain(all.iter().copied().filter(|q| *q != id))
        .filter_map(|q| Some((q, crate::media::content_key(s.catalog.photo(q)?))))
        .filter(|(_, key)| !s.signatures.contains_key(key))
        .collect();
    while !todo.is_empty() {
        let mut seen = std::collections::HashSet::new();
        let (now, later): (Vec<_>, Vec<_>) = todo.into_iter().partition(|(_, key)| seen.insert(key.clone()));
        let jobs: Vec<_> = now.iter().map(|(q, _)| (*q, thumb_weight(s, *q), s.source_load(*q, SourceLevel::Thumb))).collect();
        // (the thumbnails aren't kept: the whole library may be measured)
        let found = measure_all(
            jobs,
            |_, r: std::result::Result<SourceRef, String>| match r.and_then(|r| r.load_source()) {
                Ok(src) => (Some(lightcraft_pipeline::cull::signature(&src.image)), src.info),
                Err(_) => (None, None),
            },
            &mut |_| true,
        );
        for ((q, key), (v, info)) in now.into_iter().zip(found) {
            s.learn_source(q, info.as_ref());
            if let Some(v) = v {
                s.signatures.insert(key, v);
            }
        }
        todo = later.into_iter().filter(|(_, key)| !s.signatures.contains_key(key)).collect();
    }
    let sig = |s: &Session, id: PhotoId| -> Option<[f32; 64]> {
        let key = s.catalog.photo(id).map(|p| crate::media::content_key(p))?;
        s.signatures.get(&key).copied()
    };
    let me = sig(s, id).ok_or_else(|| super::bad(C, "can't read the photo"))?;
    let mut hits: Vec<(PhotoId, f32)> = Vec::new();
    for q in all {
        if let Some(v) = sig(s, q) {
            let sim = lightcraft_pipeline::cull::similarity(&me, &v);
            if sim >= min {
                hits.push((q, sim));
            }
        }
    }
    hits.sort_by(|a, b| b.1.total_cmp(&a.1));
    if p.get("filter").and_then(Value::as_bool).unwrap_or(true) {
        s.filter.only = hits.iter().map(|h| h.0).collect();
    }
    Ok(json!({"photos": hits.iter().map(|(i, v)| json!({"id": i.0, "similarity": (v * 1000.0).round() / 1000.0})).collect::<Vec<_>>()}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "library.findSimilar",
            "Find Similar Photos",
            ["Photo"],
            None,
            "{id?, similarity?: 0..1 (0.8), filter?: bool (true)} — photos that look like the active one (composition and tones), most similar first; filters the view to them (Clear Filters to go back) → {photos: [{id, similarity}]}",
            always,
            find_similar
        ),
        cmd!(
            "photo.analyze",
            "Assisted Culling",
            [],
            None,
            "{ids?, rejectBelow?: sharpness 0..100, pickBest?: bool, stackGroups?: bool (false; stacks each burst, best on top), burstGap?: seconds 0..86400 (10), similarity?: 0..1 (0.93), resetBursts?: bool (false)} — score the selected photos (else all in view) for focus (0..100) and clipping, group look-alike shots (signature similarity at least `similarity`) taken at most `burstGap` seconds apart and mark the sharpest of each; optionally reject blurry photos and pick each group's best; photos whose burst was arranged by hand (`photo.moveToBurst`) keep their burst, place and pick and are only scored again, unless `resetBursts`; one undo step. Focus is measured on the subject where one is known — the photo's face and pet regions (run `faces.detect` first), else the camera's focus point — and on the whole frame (its sharpest areas) otherwise; `sharpness` is the score used, so `rejectBelow`, the best of a group and the `sharpness` / `bestOfGroup` rules go by the subject (a group whose shots don't all have a subject score measured on the same source — thumbnail, or the preview for a small subject — is compared on the frames) → {photos: [{id, sharpness, subject (null when not measured), focusOn: \"face\"|\"focusPoint\"|\"frame\", measuredOn: \"thumbnail\"|\"preview\"|null, frame, clipped, group, best, byHand}], groups, kept, stacks, rejected, picked, failed}",
            always,
            analyze
        ),
        cmd!(
            "photo.moveToBurst",
            "Move to Burst",
            [],
            None,
            "{ids?, burst?: number | \"new\" | null (default: the first photo's burst), index?: place (0 = the pick; default last)} — arrange Assisted Culling's bursts by hand (Window → Bursts): the photos go into that burst at `index`, into a new burst, or (null) out of every burst; reordering within a burst is the same call. A burst left with one photo is no burst. The pick follows the first place: Best of Similar Shots, the Pick flag culling gave the old pick, and the top of the burst's stack. Every burst touched is kept as arranged when culling runs again (`photo.analyze` `resetBursts` forgets it). Photos must have been culled; one undo step → {burst, photos, pick, changed}",
            always,
            move_to_burst
        ),
    ]
}
