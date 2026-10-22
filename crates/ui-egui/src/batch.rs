//! Long work over many photos, run without holding up the window, with a progress dialog and
//! Cancel: Assisted Culling (measured on a worker thread, applied as one undo step when it ends),
//! Photo ▸ Denoise Selected Photos (the denoise queue makes their pictures) and Photo ▸ Find
//! Subjects in Selected Photos (BLIP and SAM 3, one photo after another). The dialog shows how far
//! the work is and stops it; Hide leaves it running, with its row (and ✕) in the activity stack.
//! One batch runs at a time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::sync::{Arc, Mutex};

use egui::{Align2, RichText, Stroke, vec2};
use lightcraft_engine::activity::{Cancel, TaskGuard};
use lightcraft_engine::cmd::cull::{CullMeasured, cull_finish, cull_prepare};
use serde_json::{Value, json};

use crate::LightcraftApp;
use crate::i18n::{tr, tr_format};
use crate::theme::Tokens;
use crate::widgets::register;

/// The dialog's width (points).
const WIDTH: f32 = 360.0;
/// How often the denoise queue is asked how far the batch's photos are (seconds).
const DENOISE_LOOK: f64 = 0.25;

pub struct Batch {
    /// What the dialog and the activity row are called (English; shown translated).
    pub title: &'static str,
    job: Job,
    /// Its row in the activity stack: ✕ there (and `activity.cancel`) sets `cancel`.
    guard: TaskGuard,
    cancel: Arc<AtomicBool>,
    /// Cancel has been passed on to the work (the denoise queue, the subject batch).
    stopping: bool,
    hidden: bool,
    /// Photos done and in all, as last seen.
    done: usize,
    total: usize,
    detail: String,
}

enum Job {
    /// Measuring on a worker thread: its progress and, at the end, what it measured.
    Cull { progress: Arc<Mutex<(usize, usize)>>, rx: Receiver<CullMeasured> },
    /// The denoise queue makes these photos' pictures; `next` is when it is asked again.
    Denoise { ids: Vec<u64>, next: f64, ready: usize, failed: usize, waiting: usize },
    /// The session's subject batch (`mask.subject.batch`).
    Subjects,
}

impl Batch {
    fn new(app: &mut LightcraftApp, title: &'static str, kind: &'static str, total: usize, job: Job) -> Batch {
        let cancel = Arc::new(AtomicBool::new(false));
        let guard = app.session.activity.start(kind, title, Cancel::Flag(cancel.clone()));
        guard.progress(0, total as u64);
        Batch { title, job, guard, cancel, stopping: false, hidden: false, done: 0, total, detail: String::new() }
    }

    /// The denoise queue should run at full pace while the dialog watches it.
    pub fn wants_denoise(&self) -> bool {
        matches!(self.job, Job::Denoise { .. }) && !self.hidden
    }

    /// `ui.inspect` → `batch`.
    pub fn status(&self) -> Value {
        json!({"title": self.title, "done": self.done, "total": self.total, "detail": self.detail, "hidden": self.hidden, "stopping": self.stopping})
    }
}

fn busy(app: &LightcraftApp) -> Result<(), String> {
    match &app.batch {
        Some(b) => Err(tr_format!("{} is still running: wait for it, or cancel it", tr(b.title))),
        None => Ok(()),
    }
}

/// Assisted Culling's OK: the photos are measured on a worker thread (the session isn't touched
/// there), and what was measured is applied when it ends, unless it was cancelled.
pub fn start_cull(app: &mut LightcraftApp, params: Value) -> Result<Value, String> {
    busy(app)?;
    let plan = cull_prepare(&mut app.session, &params).map_err(|e| e.to_string())?;
    let total = plan.len();
    if total == 0 {
        return app.run("photo.analyze", params);
    }
    let progress = Arc::new(Mutex::new((0, total)));
    let (tx, rx) = channel();
    let batch = Batch::new(app, "Assisted Culling", "cull", total, Job::Cull { progress: progress.clone(), rx });
    let (row, cancel, repaint) = (batch.guard.handle(), batch.cancel.clone(), app.tasks.repaint.clone());
    let work = move || {
        let measured = plan.measure(&mut |done, total| {
            if let Ok(mut p) = progress.lock() {
                *p = (done, total);
            }
            row.progress(done as u64, total as u64);
            !cancel.load(Ordering::Relaxed)
        });
        let _ = tx.send(measured);
        if let Some(c) = repaint {
            c.request_repaint();
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::Builder::new().name("assisted-culling".into()).spawn(work).map_err(|e| format!("Assisted Culling: could not start: {e}"))?;
    #[cfg(target_arch = "wasm32")]
    work();
    app.batch = Some(batch);
    Ok(json!({"background": true, "total": total}))
}

/// Photo ▸ Denoise Selected Photos: AI Denoise on for the selected raw photos (one undo step), then
/// their pictures are made by the denoise queue while the dialog follows them.
pub fn start_denoise(app: &mut LightcraftApp) -> Result<Value, String> {
    busy(app)?;
    let r = app.run("denoise.batch", json!({}))?;
    let ids: Vec<u64> = r["photos"].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect()).unwrap_or_default();
    if ids.is_empty() {
        return Err(tr("None of the selected photos is a raw photo AI Denoise applies to").to_string());
    }
    let total = ids.len();
    let batch = Batch::new(app, "Denoising photos", "denoise", total, Job::Denoise { ids, next: 0.0, ready: 0, failed: 0, waiting: total });
    app.batch = Some(batch);
    Ok(r)
}

/// Photo ▸ Find Subjects in Selected Photos: each selected photo gets a Subject mask, one after
/// another (`mask.subject.batch`).
pub fn start_subjects(app: &mut LightcraftApp) -> Result<Value, String> {
    busy(app)?;
    let r = app.run("mask.subject.batch", json!({}))?;
    let total = r["queued"].as_u64().unwrap_or(0) as usize;
    let batch = Batch::new(app, "Finding subjects", "subjects", total, Job::Subjects);
    app.batch = Some(batch);
    Ok(r)
}

/// Stop the batch: the culling worker stops at its next batch of photos (nothing is applied), the
/// denoise queue forgets the photos still waiting, the subject batch leaves the rest without masks.
fn stop(app: &mut LightcraftApp) {
    let Some(b) = app.batch.as_mut() else { return };
    b.cancel.store(true, Ordering::Relaxed);
    if b.stopping {
        return;
    }
    b.stopping = true;
    match &b.job {
        Job::Cull { .. } => {}
        Job::Denoise { ids, .. } => {
            for id in ids.clone() {
                let _ = app.session.execute("denoise.cancel", &json!({"id": id}));
            }
        }
        Job::Subjects => {
            let _ = app.session.execute("mask.subject.batchCancel", &json!({}));
        }
    }
}

/// Follow the batch (every logic tick, painted or not): take in its progress, finish it when it is done.
pub fn tick(app: &mut LightcraftApp, ctx: &egui::Context) {
    let Some(b) = app.batch.as_ref() else { return };
    if b.cancel.load(Ordering::Relaxed) && !b.stopping {
        // ✕ in the activity stack
        stop(app);
    }
    ctx.request_repaint_after(std::time::Duration::from_millis(100));
    if let Some(message) = poll(app, ctx) {
        app.batch = None;
        app.toast_for(ctx, message, 6.0);
        ctx.request_repaint();
    }
}

/// Draw the progress dialog (every frame).
pub fn show(app: &mut LightcraftApp, ctx: &egui::Context) {
    let Some(b) = app.batch.as_ref() else { return };
    if b.hidden {
        return;
    }
    let (title, done, total, detail, stopping) = (tr(b.title).to_string(), b.done, b.total, b.detail.clone(), b.stopping);
    let t = Tokens::get(ctx);
    let (mut cancel, mut hide) = (false, false);
    egui::Window::new(title.clone())
        .id(egui::Id::new("batch-progress"))
        .collapsible(false)
        .resizable(false)
        .title_bar(false)
        .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
        .frame(egui::Frame::window(&ctx.global_style()).inner_margin(egui::Margin::symmetric(16, 14)))
        .show(ctx, |ui| {
            ui.set_width(WIDTH);
            ui.spacing_mut().item_spacing.y = 6.0;
            ui.label(RichText::new(&title).font(t.font(14.0)).strong().color(t.text));
            let count = if total > 0 { tr_format!("{done} of {total}", done = done.min(total), total = total) } else { String::new() };
            let line = if stopping { tr("Stopping…").to_string() } else { count };
            bar(ui, &t, done, total, stopping);
            ui.label(RichText::new(line).font(t.font(12.0)).color(t.text_dim));
            if !detail.is_empty() && !stopping {
                ui.add(egui::Label::new(RichText::new(&detail).font(t.font(11.5)).color(t.text_dim)).truncate());
            }
            ui.add_space(6.0);
            // a row of its own: `with_layout` would take all the height the window offers
            ui.allocate_ui_with_layout(vec2(WIDTH, 30.0), egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let c = crate::button::Button::new(tr("Cancel")).secondary().show(ui);
                register(ui.ctx(), "batch:cancel", c.rect);
                cancel = c.clicked() && !stopping;
                let h = crate::button::Button::new(tr("Hide")).ghost().show(ui).on_hover_text(tr("Keep it running; it stays in the activity list"));
                register(ui.ctx(), "batch:hide", h.rect);
                hide = h.clicked();
            });
        });
    if cancel {
        stop(app);
    }
    if hide && let Some(b) = app.batch.as_mut() {
        b.hidden = true;
    }
}

/// The progress bar.
fn bar(ui: &mut egui::Ui, t: &Tokens, done: usize, total: usize, stopping: bool) {
    let (rect, _) = ui.allocate_exact_size(vec2(WIDTH, 6.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, t.field);
    p.rect_stroke(rect, 3.0, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    let frac = if total > 0 { (done as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 };
    if frac > 0.0 {
        let fill = egui::Rect::from_min_size(rect.min, vec2(rect.width() * frac, rect.height()));
        p.rect_filled(fill, 3.0, if stopping { t.text_dim } else { t.accent });
    }
}

/// Take in the batch's progress; the message to show when it has ended.
fn poll(app: &mut LightcraftApp, ctx: &egui::Context) -> Option<String> {
    let now = ctx.input(|i| i.time);
    let b = app.batch.as_mut()?;
    match &mut b.job {
        Job::Cull { progress, rx } => {
            if let Ok(p) = progress.lock() {
                (b.done, b.total) = *p;
            }
            b.detail = tr("Measuring focus and finding bursts").to_string();
            let measured = match rx.try_recv() {
                Ok(m) => m,
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => return Some(tr("Assisted Culling stopped unexpectedly: nothing was changed").to_string()),
            };
            let cancelled = measured.cancelled;
            Some(match cull_finish(&mut app.session, measured) {
                Ok(_) if cancelled => tr("Assisted Culling cancelled: nothing was changed").to_string(),
                Ok(r) => tr_format!(
                    "Assisted Culling: {groups} bursts, {picked} picked, {rejected} rejected",
                    groups = r["groups"].as_u64().unwrap_or(0),
                    picked = r["picked"].as_u64().unwrap_or(0),
                    rejected = r["rejected"].as_u64().unwrap_or(0)
                ),
                Err(e) => e.to_string(),
            })
        }
        Job::Denoise { ids, next, ready, failed, waiting } => {
            if now >= *next {
                *next = now + DENOISE_LOOK;
                if let Ok(v) = app.session.execute("denoise.batchStatus", &json!({"ids": ids})) {
                    let n = |k: &str| v[k].as_u64().unwrap_or(0) as usize;
                    (*ready, *failed, *waiting) = (n("ready"), n("failed"), n("waiting"));
                    b.detail = match v["running"].as_object() {
                        Some(r) => {
                            let (d, t) = (r["done"].as_u64().unwrap_or(0), r["total"].as_u64().unwrap_or(0).max(1));
                            tr_format!("Photo {n} of {total}: {pct} %", n = *ready + *failed + 1, total = ids.len(), pct = d * 100 / t)
                        }
                        None => String::new(),
                    };
                }
            }
            b.done = *ready + *failed;
            if *waiting > 0 && !b.stopping {
                b.guard.progress(b.done as u64, b.total as u64);
                return None;
            }
            Some(if b.stopping {
                tr_format!("Denoise cancelled: {ready} of {total} photos done", ready = *ready, total = ids.len())
            } else if *failed > 0 {
                tr_format!("{ready} photos denoised, {failed} failed", ready = *ready, failed = *failed)
            } else {
                tr_format!("{ready} photos denoised", ready = *ready)
            })
        }
        Job::Subjects => {
            let s = &app.session.segmenter.subject_batch;
            (b.done, b.total) = (s.done, s.total);
            b.guard.progress(b.done as u64, b.total as u64);
            b.detail = s
                .current
                .and_then(|id| app.session.catalog.photo(id))
                .map(|p| tr_format!("Describing and selecting: {name}", name = p.file_name.as_str()))
                .unwrap_or_default();
            if s.running() {
                return None;
            }
            let (made, skipped, failed) = (s.made, s.skipped, s.failed.len());
            Some(if s.cancelled {
                tr_format!("Finding subjects cancelled: {made} Subject masks made", made = made)
            } else {
                tr_format!("{made} Subject masks made, {skipped} skipped, {failed} failed", made = made, skipped = skipped, failed = failed)
            })
        }
    }
}
