//! Throwaway bench for lightcraft#2367: main's inpainter vs the PR's, on procedural textures only.
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

fn scene(name: &str, w: usize, h: usize) -> Vec<f32> {
    let mut r = Rng(42);
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let n = r.f() - 0.5;
            let px = match name {
                // smooth gradient with strong film-like grain
                "grain" => [0.35 + 0.3 * fx + 0.14 * n, 0.45 + 0.15 * fy + 0.14 * n, 0.55 - 0.2 * fx + 0.14 * n],
                // wavy wood-like stripes with grain
                "wood" => {
                    let s = 0.5 + 0.5 * ((x as f32) * 0.18 + 4.0 * (fy * 6.0).sin()).sin();
                    [0.45 + 0.3 * s + 0.08 * n, 0.3 + 0.2 * s + 0.08 * n, 0.15 + 0.1 * s + 0.08 * n]
                }
                // a slanted horizon: smooth sky over a noisy field
                "horizon" => {
                    if fy < 0.5 + 0.15 * (fx - 0.5) {
                        [0.45 + 0.2 * fy, 0.6 + 0.2 * fy, 0.85 + 0.01 * n]
                    } else {
                        [0.25 + 0.12 * n, 0.45 + 0.2 * n, 0.15 + 0.08 * n]
                    }
                }
                // offset bricks with mortar
                _ => {
                    let (bh, bw) = (h as f32 / 12.0, w as f32 / 6.0);
                    let row = (y as f32 / bh) as usize;
                    let xo = x as f32 + if row % 2 == 1 { bw / 2.0 } else { 0.0 };
                    let mortar = (y as f32 % bh) < 2.5 || (xo % bw) < 2.5;
                    if mortar { [0.75 + 0.05 * n, 0.73 + 0.05 * n, 0.68 + 0.05 * n] } else { [0.55 + 0.1 * n, 0.25 + 0.06 * n, 0.18 + 0.05 * n] }
                }
            };
            v.extend(px.map(|c| c.clamp(0.0, 1.0)));
        }
    }
    v
}

fn disc(w: usize, h: usize, r: f32) -> Vec<bool> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    (0..w * h).map(|i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) < r).collect()
}

fn run(which: &str, w: usize, h: usize, img: &[f32], hole: &[bool]) -> Vec<f32> {
    match which {
        "old" => inpaint_old::complete(w, h, 3, img, hole, &inpaint_old::CompleteParams::default()).unwrap(),
        _ => inpaint_new::complete(w, h, 3, img, hole, &inpaint_new::CompleteParams::default()).unwrap(),
    }
}

fn ppm(path: &str, w: usize, h: usize, px: &[[u8; 3]]) {
    let mut b = format!("P6\n{w} {h}\n255\n").into_bytes();
    px.iter().for_each(|p| b.extend(p));
    std::fs::write(path, b).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        // bench <old|new> <box side> <scene>
        "bench" => {
            let n: usize = a[3].parse().unwrap();
            let img = scene(&a[4], n, n);
            let hole = disc(n, n, n as f32 / 6.0);
            let t = Instant::now();
            let out = run(&a[2], n, n, &img, &hole);
            println!("{} {} box {n} hole r {:.0}: {:.0} ms ({})", a[2], a[4], n as f32 / 6.0, t.elapsed().as_secs_f64() * 1e3, out.len());
        }
        // stats: grain (mean |right neighbour diff|) in the hole vs a ring around it, per scene
        "stats" => {
            let n = 192;
            for name in ["grain", "wood", "horizon", "bricks"] {
                let img = scene(name, n, n);
                let hole = disc(n, n, n as f32 / 6.0);
                let ring = |i: usize| { let d = ((i % n) as f32 - n as f32 / 2.0).hypot((i / n) as f32 - n as f32 / 2.0); d >= n as f32 / 6.0 + 2.0 && d < n as f32 / 3.0 };
                let grain = |v: &[f32], inside: bool| {
                    let (mut s, mut k) = (0.0f32, 0);
                    for i in 0..n * n - 1 {
                        let ok = |j: usize| if inside { hole[j] } else { ring(j) };
                        if (i + 1) % n != 0 && ok(i) && ok(i + 1) { for c in 0..3 { s += (v[(i + 1) * 3 + c] - v[i * 3 + c]).abs(); } k += 3; }
                    }
                    s / k as f32
                };
                let (o, p) = (run("old", n, n, &img, &hole), run("new", n, n, &img, &hole));
                println!("{name}: around {:.4}  main fill {:.4} ({:.0}%)  PR fill {:.4} ({:.0}%)", grain(&img, false), grain(&o, true), 100.0 * grain(&o, true) / grain(&img, false), grain(&p, true), 100.0 * grain(&p, true) / grain(&img, false));
            }
        }
        // images <dir>: input | main | PR, side by side, 2x nearest
        _ => {
            let n = 192;
            for name in ["grain", "wood", "horizon", "bricks"] {
                let img = scene(name, n, n);
                let hole = disc(n, n, n as f32 / 6.0);
                let (o, p) = (run("old", n, n, &img, &hole), run("new", n, n, &img, &hole));
                let (gap, s) = (8, 2);
                let tw = 3 * n * s + 2 * gap;
                let mut px = vec![[255u8; 3]; tw * n * s];
                for (k, src) in [&img, &o, &p].into_iter().enumerate() {
                    for y in 0..n * s {
                        for x in 0..n * s {
                            let i = (y / s) * n + x / s;
                            let c = if k == 0 && hole[i] { [255, 0, 255] } else { [0, 1, 2].map(|c| (src[i * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8) };
                            px[y * tw + k * (n * s + gap) + x] = c;
                        }
                    }
                }
                ppm(&format!("{}/{name}.ppm", a[2]), tw, n * s, &px);
            }
        }
    }
}
