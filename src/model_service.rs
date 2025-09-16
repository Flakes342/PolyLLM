use anyhow::{ Result, Context };
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use std::fs::File;

pub struct ModelService {
    model: ModelWeights,
    device: Device,
}

impl ModelService {
    pub fn new(model_path: &str, device: Device) -> Result<Self> {
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("- Model file not found: {}", model_path));
        }

        println!("- Loading GGUF model from: {}", model_path);
        
        let mut file = File::open(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to open model file: {}", e))?;
            
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("Failed to read GGUF content: {}", e))?;
            
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model from GGUF: {}", e))?;

        println!("- Model loaded successfully!");

        Ok(Self { model, device })
    }

    pub fn forward(&mut self, input_tensor: &Tensor, position: usize) -> Result<Tensor> {
        self.model
            .forward(input_tensor, position)
            .context("Model forward pass failed")
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}