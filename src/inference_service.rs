use anyhow::{Result, Context};
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use crate::model_service::ModelService;
use crate::tokenizer_service::TokenizerService;
use std::io::{self, Write};

pub struct InferenceService {
    logits_processor: LogitsProcessor,
}

impl InferenceService {
    pub fn new() -> Self {
        Self {
            logits_processor: LogitsProcessor::new(
                42,          // seed
                Some(0.7),   // temperature
                Some(0.9),   // top_p
            ),
        }
    }

    pub fn update_sampling_params(&mut self, temperature: f64, top_p: f64, seed: u64) {
        self.logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));
    }

    pub fn generate_streaming(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        model_service: &mut ModelService,
        tokenizer_service: &TokenizerService,
        silent: bool,
    ) -> Result<String> {
        // Update sampling parameters
        self.update_sampling_params(temperature, 0.9, 42);

        // Tokenize input
        let input_ids = tokenizer_service.encode(prompt)?;
        let mut all_tokens = input_ids.clone();

        if !silent {
            print!("Assistant: ");
            io::stdout().flush().unwrap();
        }

        // Initial forward pass
        let mut pos = 0;
        let input_tensor = Tensor::new(&input_ids[..], model_service.device())?
            .unsqueeze(0)
            .context("Failed to create input tensor")?;

        let mut logits = model_service.forward(&input_tensor, pos)?
            .squeeze(0)?;
        pos += input_ids.len();

        // Get logits for last token
        if logits.dims().len() == 3 {
            logits = logits.get(logits.dim(1)? - 1)?.squeeze(0)?;
        }

        // Sample first token
        let scaled_logits = (logits / temperature)?;
        let mut next_token = self.logits_processor.sample(&scaled_logits)?;

        let mut stream_buffer = Vec::new();
        let mut generated_count = 0;

        // Generation loop
        for _ in 0..max_tokens {
            all_tokens.push(next_token);
            generated_count += 1;

            // Check for EOS
            if next_token == tokenizer_service.get_eos_token_id() {
                if !silent {
                    println!("\n[EOS after {} tokens]", generated_count);
                }
                break;
            }

            // Stream decoding
            stream_buffer.push(next_token);
            if let Ok(decoded) = tokenizer_service.decode(&stream_buffer, false) {
                if !decoded.is_empty() && !silent {
                    print!("{}", decoded);
                    io::stdout().flush().unwrap();
                    stream_buffer.clear();
                }
            }

            // Forward pass for next token
            let next_input = Tensor::new(&[next_token], model_service.device())?
                .unsqueeze(0)?;
            
            let mut next_logits = model_service.forward(&next_input, pos)?;

            // Handle dimensions
            next_logits = match next_logits.dims().len() {
                3 => next_logits.squeeze(0)?.squeeze(0)?,
                2 => next_logits.squeeze(0)?,
                _ => next_logits,
            };

            pos += 1;

            // Sample next token
            let scaled = (next_logits / temperature)?;
            next_token = self.logits_processor.sample(&scaled)?;
        }

        if !silent && generated_count == max_tokens {
            println!("\n[Max tokens reached: {}]", max_tokens);
        }

        // Final decode
        let response_tokens = &all_tokens[input_ids.len()..];
        let response = tokenizer_service.decode(response_tokens, false)?;
        
        Ok(response)
    }
}