//! Experimental local BLIP captioning for subject suggestions. No weights are bundled.
use crate::{Error, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use std::path::Path;

/// A bounded caption from the opt-in BLIP-base files (BSD-3-Clause).
pub fn describe(dir: &Path, rgb: &[u8], w: usize, h: usize) -> Result<String> {
    let error = |e: String| Error::Model(format!("BLIP: {e}"));
    if w == 0 || h == 0 || w > 1008 || h > 1008 || w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
        return Err(error("invalid image dimensions".into()));
    }
    let weights = dir.join("pytorch_model.bin");
    let bytes = std::fs::metadata(&weights).map_err(|e| error(e.to_string()))?.len();
    if bytes > 1_100_000_000 {
        return Err(error("model is too large for BLIP-base".into()));
    }
    let vocab_path = dir.join("vocab.txt");
    if std::fs::metadata(&vocab_path).map_err(|e| error(e.to_string()))?.len() > 1_000_000 {
        return Err(error("vocabulary is too large".into()));
    }
    let vocabulary = std::fs::read_to_string(vocab_path).map_err(|e| error(e.to_string()))?;
    let vocab: Vec<&str> = vocabulary.lines().collect();
    let device = crate::best_device();
    let mut config = blip::Config::image_captioning_large();
    config.vision_config.hidden_size = 768;
    config.vision_config.intermediate_size = 3072;
    config.vision_config.num_hidden_layers = 12;
    config.vision_config.num_attention_heads = 12;
    config.text_config.encoder_hidden_size = 768;
    let vb = VarBuilder::from_pth(weights, DType::F32, &device)?;
    let mut model = blip::BlipForConditionalGeneration::new(&config, vb)?;
    let image = image::RgbImage::from_raw(w as u32, h as u32, rgb.to_vec()).ok_or_else(|| error("invalid RGB image".into()))?;
    let image = image::imageops::resize(&image, 384, 384, image::imageops::FilterType::CatmullRom);
    let xs = Tensor::from_vec(image.into_raw(), (384, 384, 3), &device)?.permute((2, 0, 1))?.to_dtype(DType::F32)?;
    let mean = Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &device)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], &device)?.reshape((3, 1, 1))?;
    let xs = (xs / 255.)?.broadcast_sub(&mean)?.broadcast_div(&std)?.unsqueeze(0)?;
    let encoded = xs.apply(model.vision_model())?;
    let mut token = 30522u32;
    let mut caption = String::new();
    for _ in 0..48 {
        let input = Tensor::new(&[[token]], &device)?;
        let logits = model.text_decoder().forward(&input, &encoded)?.squeeze(0)?;
        let last = logits.dim(0)?.checked_sub(1).ok_or_else(|| error("empty decoder output".into()))?;
        token = logits.get(last)?.argmax(0)?.to_scalar::<u32>()?;
        if token == 102 {
            break;
        }
        let word = vocab.get(token as usize).ok_or_else(|| error("invalid vocabulary index".into()))?;
        if let Some(suffix) = word.strip_prefix("##") {
            caption.push_str(suffix);
        } else {
            if !caption.is_empty() {
                caption.push(' ');
            }
            caption.push_str(word);
        }
    }
    Ok(caption)
}

/// Lightweight phrase extraction for the prototype, not an exhaustive object detector.
/// Normalize demographic words to "person"; only SAM-verified phrases reach the picker.
pub fn candidates(caption: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut phrase = Vec::new();
    let flush = |phrase: &mut Vec<String>, out: &mut Vec<String>| {
        if phrase.is_empty() {
            return;
        }
        let mut label = phrase.join(" ");
        if phrase.iter().any(|s| matches!(s.as_str(), "man" | "woman" | "boy" | "girl" | "person" | "people")) {
            label = "person".into();
        }
        if !out.contains(&label) && out.len() < 6 {
            out.push(label);
        }
        phrase.clear();
    };
    for word in caption.split_whitespace().take(64) {
        let word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if matches!(
            word.as_str(),
            "a" | "an" | "the" | "in" | "on" | "with" | "of" | "and" | "holding" | "sitting" | "standing" | "next" | "to" | "at" | "is" | "are"
        ) {
            flush(&mut phrase, &mut out);
        } else if !word.is_empty()
            && !matches!(word.as_str(), "young" | "old" | "orange" | "red" | "green" | "blue" | "white" | "black" | "wooden" | "small" | "large")
        {
            phrase.push(word);
        }
    }
    flush(&mut phrase, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caption_candidates_are_bounded_and_neutral() {
        assert_eq!(candidates("a young man in an orange space suit holding a helmet"), ["person", "space suit", "helmet"]);
        assert_eq!(candidates("a cup of coffee on a wooden table"), ["cup", "coffee", "table"]);
        assert_eq!(candidates("a cat and a cat"), ["cat"]);
        assert!(candidates("a the in").is_empty());
        assert_eq!(candidates("a b a c a d a e a f a g a h").len(), 6);
    }
}
