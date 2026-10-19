//! People masks: one person, picked by their face among SAM 3's "person" instances, and the
//! parts of them asked for (Hair, Clothes, Skin) from further detections, cut to that person so
//! a neighbour's hair stays out. Pure functions over logit grids (no model), tested without it.
//!
//! Logits combine like fuzzy sets: a union is the maximum, an intersection the minimum, "a
//! without b" is min(a, −b). That is the boolean operation wherever the model is sure, and the
//! edges stay as soft as the model's.
//!
//! **Skin is an approximation:** the person without their hair and clothes (SAM 3 has no skin
//! concept), so glasses, jewellery, a bag or anything else that is neither stays in it.

/// The parts a People mask can select now, as the Masking panel names them.
pub const PARTS: [&str; 4] = ["Entire Person", "Hair", "Clothes", "Skin"];

/// Score above which a "person" instance can be the one at the face: lower than Describe's,
/// because the face already says someone is there.
pub const PERSON_THRESHOLD: f32 = 0.3;
/// Score above which hair and clothes instances count (Describe's).
pub const PART_THRESHOLD: f32 = 0.5;
/// What the model is asked for, per part.
pub const HAIR_PROMPT: &str = "hair";
pub const CLOTHES_PROMPT: &str = "clothes";

/// A logit that selects nothing.
const NONE: f32 = -16.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Person,
    Hair,
    Clothes,
    Skin,
}

impl Part {
    /// The part a name means (case and spaces don't matter: "entire person", "EntirePerson",
    /// "person"; "clothing" is Clothes).
    pub fn parse(name: &str) -> Option<Part> {
        let n: String = name.chars().filter(|c| !c.is_whitespace() && *c != '_' && *c != '-').collect::<String>().to_lowercase();
        Some(match n.as_str() {
            "entireperson" | "person" | "wholeperson" => Part::Person,
            "hair" => Part::Hair,
            "clothes" | "clothing" => Part::Clothes,
            "skin" => Part::Skin,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Part::Person => PARTS[0],
            Part::Hair => PARTS[1],
            Part::Clothes => PARTS[2],
            Part::Skin => PARTS[3],
        }
    }
}

/// The parts `names` ask for, once each (none: the entire person); an unknown name is an error
/// that lists the parts there are.
pub fn parts_of(names: &[String]) -> Result<Vec<Part>, String> {
    let mut out = Vec::new();
    for n in names {
        let p = Part::parse(n).ok_or_else(|| format!("unknown part `{n}` of a person (one of: {})", PARTS.join(", ")))?;
        if !out.contains(&p) {
            out.push(p);
        }
    }
    if out.is_empty() {
        out.push(Part::Person);
    }
    Ok(out)
}

/// Whether `parts` need the hair and the clothes detections.
pub fn needs(parts: &[Part]) -> (bool, bool) {
    let any = |want: &[Part]| parts.iter().any(|p| want.contains(p));
    (any(&[Part::Hair, Part::Skin]), any(&[Part::Clothes, Part::Skin]))
}

/// A face box from a client or a document: finite, inside the photo (clamped) and not empty.
pub fn face_box(b: [f64; 4]) -> Option<[f64; 4]> {
    if !b.iter().all(|v| v.is_finite()) {
        return None;
    }
    let [x0, y0, x1, y1] = b.map(|v| v.clamp(0.0, 1.0));
    (x1 > x0 && y1 > y0).then_some([x0, y0, x1, y1])
}

/// One instance the detector found: `side × side` mask logits over the model's input, its score
/// and its box (normalized).
#[derive(Clone, Debug)]
pub struct Candidate {
    pub logits: Vec<f32>,
    pub score: f32,
    pub bbox: [f32; 4],
}

/// The logit of the grid cell at normalized point (`x`, `y`) (nothing outside the grid).
pub fn logit_at(logits: &[f32], side: usize, x: f64, y: f64) -> f32 {
    if !(x.is_finite() && y.is_finite()) || side == 0 {
        return NONE;
    }
    let cell = |v: f64| ((v * side as f64).floor().max(0.0) as usize).min(side - 1);
    logits.get(cell(y) * side + cell(x)).copied().unwrap_or(NONE)
}

/// Which of `cands` is the person whose face is `face`: the best-scoring one whose mask covers
/// the face's centre; failing that, the one whose box holds the centre and most of the face
/// (then the better score). `None`: nobody is there.
pub fn pick_person(cands: &[Candidate], side: usize, face: [f64; 4]) -> Option<usize> {
    let (cx, cy) = ((face[0] + face[2]) / 2.0, (face[1] + face[3]) / 2.0);
    let best = |it: &mut dyn Iterator<Item = (usize, f64, f32)>| it.max_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2))).map(|(i, _, _)| i);
    let mut covering = cands.iter().enumerate().filter(|(_, c)| logit_at(&c.logits, side, cx, cy) > 0.0).map(|(i, c)| (i, 0.0, c.score));
    if let Some(i) = best(&mut covering) {
        return Some(i);
    }
    let face_area = ((face[2] - face[0]) * (face[3] - face[1])).max(1e-12);
    let mut boxed = cands.iter().enumerate().filter_map(|(i, c)| {
        let b = c.bbox.map(f64::from);
        if !(b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3]) {
            return None;
        }
        let w = (face[2].min(b[2]) - face[0].max(b[0])).max(0.0);
        let h = (face[3].min(b[3]) - face[1].max(b[1])).max(0.0);
        Some((i, w * h / face_area, c.score))
    });
    best(&mut boxed)
}

/// The selection of `parts` of a person (all grids `side × side` logits): the union of
/// - Entire Person: `person`;
/// - Hair, Clothes: the `hair` / `clothes` found (anyone's) within `person`;
/// - Skin: `person` without hair and clothes.
///
/// A detection that found nothing (`None`) selects nothing and takes nothing away.
pub fn compose(person: &[f32], hair: Option<&[f32]>, clothes: Option<&[f32]>, parts: &[Part]) -> Vec<f32> {
    let at = |g: Option<&[f32]>, i: usize| g.and_then(|g| g.get(i)).copied().unwrap_or(NONE);
    person
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let (h, c) = (at(hair, i).min(p), at(clothes, i).min(p));
            parts
                .iter()
                .map(|part| match part {
                    Part::Person => p,
                    Part::Hair => h,
                    Part::Clothes => c,
                    Part::Skin => p.min(-at(hair, i)).min(-at(clothes, i)),
                })
                .fold(NONE, f32::max)
        })
        .collect()
}

/// Whether a selection selects anything at all.
pub fn selects_anything(logits: &[f32]) -> bool {
    logits.iter().any(|l| *l > 0.0)
}

/// `face` (normalized to the photo) in the frame of `region` (a part of the photo, as a detail
/// pass sees it); `None` when its centre is outside it.
pub fn face_in(face: [f64; 4], region: [f64; 4]) -> Option<[f64; 4]> {
    let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
    if !(rw > 0.0 && rh > 0.0) {
        return None;
    }
    let (cx, cy) = ((face[0] + face[2]) / 2.0, (face[1] + face[3]) / 2.0);
    if !(region[0] <= cx && cx <= region[2] && region[1] <= cy && cy <= region[3]) {
        return None;
    }
    Some([(face[0] - region[0]) / rw, (face[1] - region[1]) / rh, (face[2] - region[0]) / rw, (face[3] - region[1]) / rh])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIDE: usize = 40;

    /// Logits that select the cells of `[x0, y0, x1, y1]` (normalized) on a `SIDE²` grid.
    fn rect(r: [f64; 4]) -> Vec<f32> {
        (0..SIDE * SIDE)
            .map(|i| {
                let (x, y) = (((i % SIDE) as f64 + 0.5) / SIDE as f64, ((i / SIDE) as f64 + 0.5) / SIDE as f64);
                if r[0] <= x && x < r[2] && r[1] <= y && y < r[3] { 8.0 } else { -8.0 }
            })
            .collect()
    }

    fn union(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(a, b)| a.max(*b)).collect()
    }

    fn cand(r: [f64; 4], score: f32) -> Candidate {
        Candidate { logits: rect(r), score, bbox: r.map(|v| v as f32) }
    }

    fn selected(l: &[f32], x: f64, y: f64) -> bool {
        logit_at(l, SIDE, x, y) > 0.0
    }

    // Two people side by side: Ann on the left (x 0.05..0.45), Ben on the right (0.55..0.95),
    // faces at the top of each; Ann's long hair falls over onto Ben's shoulder.
    const ANN: [f64; 4] = [0.05, 0.1, 0.45, 1.0];
    const BEN: [f64; 4] = [0.55, 0.1, 0.95, 1.0];
    const ANN_FACE: [f64; 4] = [0.2, 0.15, 0.3, 0.3];
    const BEN_FACE: [f64; 4] = [0.7, 0.15, 0.8, 0.3];

    fn people() -> Vec<Candidate> {
        vec![cand(BEN, 0.9), cand(ANN, 0.8)]
    }

    #[test]
    fn the_person_is_the_one_whose_mask_covers_the_face() {
        let c = people();
        assert_eq!(pick_person(&c, SIDE, ANN_FACE), Some(1));
        assert_eq!(pick_person(&c, SIDE, BEN_FACE), Some(0));
        // a face between them belongs to nobody
        assert_eq!(pick_person(&c, SIDE, [0.47, 0.2, 0.53, 0.3]), None);
        assert_eq!(pick_person(&[], SIDE, ANN_FACE), None);
        // two instances over the same face (a duplicate query): the better score
        let mut c = people();
        c.push(cand([0.1, 0.1, 0.4, 0.9], 0.95));
        assert_eq!(pick_person(&c, SIDE, ANN_FACE), Some(2));
    }

    #[test]
    fn without_a_mask_at_the_face_the_box_that_holds_most_of_it_wins() {
        // masks that miss the face (a hole there), boxes that hold it; the second holds more of it
        let mut a = cand([0.0, 0.0, 0.5, 1.0], 0.9);
        a.logits = rect([0.0, 0.5, 0.5, 1.0]);
        a.bbox = [0.0, 0.0, 0.25, 1.0];
        let mut b = cand([0.0, 0.0, 0.5, 1.0], 0.4);
        b.logits = rect([0.0, 0.5, 0.5, 1.0]);
        b.bbox = [0.1, 0.0, 0.5, 1.0];
        assert_eq!(pick_person(&[a.clone(), b.clone()], SIDE, ANN_FACE), Some(1));
        // a box that doesn't hold the face's centre doesn't count
        a.bbox = [0.0, 0.0, 0.22, 1.0];
        b.bbox = [0.27, 0.0, 0.5, 1.0];
        assert_eq!(pick_person(&[a, b], SIDE, ANN_FACE), None);
    }

    #[test]
    fn parts_are_cut_to_the_person() {
        let c = people();
        let ann = &c[pick_person(&c, SIDE, ANN_FACE).unwrap()].logits;
        // Ann's hair runs over Ben's shoulder; Ben has short hair
        let hair = union(&rect([0.1, 0.1, 0.65, 0.4]), &rect([0.68, 0.1, 0.82, 0.16]));
        let clothes = union(&rect([0.05, 0.5, 0.45, 1.0]), &rect([0.55, 0.5, 0.95, 1.0]));
        let hair_only = compose(ann, Some(&hair), Some(&clothes), &[Part::Hair]);
        assert!(selected(&hair_only, 0.15, 0.3), "Ann's hair");
        assert!(!selected(&hair_only, 0.6, 0.3), "not where it lies on Ben");
        assert!(!selected(&hair_only, 0.75, 0.12), "not Ben's hair");
        assert!(!selected(&hair_only, 0.25, 0.45), "not her skin");
        let clothes_only = compose(ann, Some(&hair), Some(&clothes), &[Part::Clothes]);
        assert!(selected(&clothes_only, 0.25, 0.8) && !selected(&clothes_only, 0.75, 0.8));
        // Skin: Ann without her hair and clothes (her face below the hairline, her neck)
        let skin = compose(ann, Some(&hair), Some(&clothes), &[Part::Skin]);
        assert!(selected(&skin, 0.25, 0.45) && selected(&skin, 0.07, 0.2));
        assert!(!selected(&skin, 0.25, 0.3) && !selected(&skin, 0.25, 0.8) && !selected(&skin, 0.75, 0.45));
        // the parts add up: hair + skin + clothes is the whole of Ann, and nothing of Ben
        let all = compose(ann, Some(&hair), Some(&clothes), &[Part::Hair, Part::Skin, Part::Clothes]);
        let whole = compose(ann, Some(&hair), Some(&clothes), &[Part::Person]);
        for (x, y) in [(0.25, 0.3), (0.25, 0.45), (0.25, 0.8), (0.75, 0.45), (0.6, 0.3)] {
            assert_eq!(selected(&all, x, y), selected(&whole, x, y), "({x}, {y})");
            assert_eq!(selected(&whole, x, y), selected(ann, x, y));
        }
    }

    #[test]
    fn a_part_nobody_has_selects_nothing_and_takes_nothing_away() {
        let ann = rect(ANN);
        let hair = compose(&ann, None, None, &[Part::Hair]);
        assert!(!selects_anything(&hair));
        assert_eq!(compose(&ann, None, None, &[Part::Skin]), ann);
        assert!(selects_anything(&compose(&ann, None, None, &[Part::Hair, Part::Person])));
    }

    #[test]
    fn part_names_and_what_they_need() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parts_of(&names(&["entire person", "HAIR", "Hair", "clothing"])).unwrap(), [Part::Person, Part::Hair, Part::Clothes]);
        assert_eq!(parts_of(&[]).unwrap(), [Part::Person]);
        let e = parts_of(&names(&["Lips"])).unwrap_err();
        assert!(e.contains("Lips") && e.contains("Skin"), "{e}");
        for n in PARTS {
            assert_eq!(Part::parse(n).map(Part::name), Some(n));
        }
        assert_eq!(needs(&[Part::Person]), (false, false));
        assert_eq!(needs(&[Part::Hair]), (true, false));
        assert_eq!(needs(&[Part::Clothes, Part::Person]), (false, true));
        assert_eq!(needs(&[Part::Skin]), (true, true));
    }

    #[test]
    fn hostile_face_boxes_and_points() {
        assert_eq!(face_box([0.2, 0.2, 0.4, 0.5]), Some([0.2, 0.2, 0.4, 0.5]));
        assert_eq!(face_box([-1.0, 0.2, 0.4, 2.0]), Some([0.0, 0.2, 0.4, 1.0]));
        assert_eq!(face_box([0.4, 0.2, 0.2, 0.5]), None, "inside out");
        assert_eq!(face_box([f64::NAN, 0.2, 0.4, 0.5]), None);
        assert_eq!(face_box([2.0, 0.2, 3.0, 0.5]), None, "outside the photo");
        let l = rect(ANN);
        assert_eq!(logit_at(&l, SIDE, f64::INFINITY, 0.5), NONE);
        assert_eq!(logit_at(&l, 0, 0.5, 0.5), NONE);
        assert_eq!(logit_at(&l[..10], SIDE, 0.9, 0.9), NONE, "a short grid");
        assert!(logit_at(&rect([0.0, 0.0, 0.5, 1.0]), SIDE, -5.0, 0.5) > 0.0, "clamped to the edge cell");
    }

    #[test]
    fn a_face_in_a_detail_region() {
        let f = face_in(ANN_FACE, [0.0, 0.0, 0.5, 0.5]).unwrap();
        assert!((f[0] - 0.4).abs() < 1e-12 && (f[3] - 0.6).abs() < 1e-12, "{f:?}");
        assert!(face_in(ANN_FACE, [0.5, 0.0, 1.0, 1.0]).is_none());
        assert!(face_in(ANN_FACE, [0.5, 0.0, 0.5, 1.0]).is_none());
    }
}
