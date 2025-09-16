use anyhow::{Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::io::Write;
use std::fs::File;

pub struct LLMInferenceEngine {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    logits_processor: LogitsProcessor,
}

impl LLMInferenceEngine {
    /// Initialize the inference engine with GGUF model and tokenizer paths
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        device: Device,
    ) -> Result<Self> {
        println!("Loading tokenizer from: {}", tokenizer_path);
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        println!("Loading GGUF model from: {}", model_path);
        
        // Load GGUF model with proper content parsing
        let mut file = File::open(model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        println!("Model loaded successfully!");

        // Initialize logits processor for sampling
        let logits_processor = LogitsProcessor::new(
            42, // seed
            Some(0.8), // temperature
            Some(0.9), // top_p
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            logits_processor,
        })
    }

    /// Generate text from a given prompt, streaming tokens to stdout
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f64>,
    ) -> Result<String> {
        // Tokenize input
        let tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        
        let input_ids = tokens.get_ids();
        let mut all_tokens = input_ids.to_vec();

        println!("\nPrompt: {}", prompt);
        println!("Starting generation...");
        print!("Response: ");
        // std::io::stdout().flush().unwrap();

        // Track position
        let mut pos = 0;

        // Run forward on prompt
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input_tensor, pos)?.squeeze(0)?;
        pos += input_ids.len();

        // Get logits for last token
        if logits.dims().len() == 3 {
            logits = logits.get(logits.dim(1)? - 1)?.squeeze(0)?;
        }

        // Sample first new token
        let mut next_token = {
            let scaled = if let Some(temp) = temperature {
                (logits / temp)?
            } else {
                logits
            };
            self.logits_processor.sample(&scaled)?
        };

        // --- Streaming loop ---
        let mut stream_buffer: Vec<u32> = Vec::new();

        for _ in 0..max_tokens {
            all_tokens.push(next_token);

            // EOS check
            if next_token == self.get_eos_token_id() {
                break;
            }

            // Push to buffer and try decoding
            stream_buffer.push(next_token);

            if let Ok(decoded) = self.tokenizer.decode(&stream_buffer, false) {
                // Only flush if we got a valid UTF-8 chunk
                if !decoded.is_empty() {
                    print!("{}", decoded);
                    std::io::stdout().flush().unwrap();
                    stream_buffer.clear(); // reset buffer once flushed
                }
            }

            // Forward pass with just the new token
            let input_tensor = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let mut logits = self.model.forward(&input_tensor, pos)?;

            // Handle dims
            if logits.dims().len() == 3 {
                logits = logits.squeeze(0)?.squeeze(0)?;
            } else if logits.dims().len() == 2 {
                logits = logits.squeeze(0)?;
            }

            pos += 1;

            // Sample next token
            next_token = {
                let scaled = if let Some(temp) = temperature {
                    (logits / temp)?
                } else {
                    logits
                };
                self.logits_processor.sample(&scaled)?
            };
        }

        println!();

        // Decode final response
        let response_tokens = &all_tokens[input_ids.len()..];
        let response = self.tokenizer
            .decode(response_tokens, false)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(response)
    }

    /// Get the end-of-sequence token ID
    fn get_eos_token_id(&self) -> u32 {
        // Try common EOS tokens
        self.tokenizer.token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| self.tokenizer.token_to_id("<|end|>"))
            .or_else(|| self.tokenizer.token_to_id("<eos>"))
            .unwrap_or(2) // Default EOS token ID
    }

    /// Simple chat interface with system prompt support
    pub fn chat(&mut self) -> Result<()> {
        use std::io::{self, BufRead};
        
        println!("\n=== LLM Chat Interface ===");
        println!("Commands:");
        println!("  'quit' or 'exit' - Exit the chat");
        println!("  'clear' - Clear screen");
        println!("  'help' - Show this help");
        println!("================================\n");
        
        let stdin = io::stdin();
        
        loop {
            print!("User: ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            stdin.lock().read_line(&mut input)?;
            let input = input.trim();
            
            match input.to_lowercase().as_str() {
                "quit" | "exit" => {
                    println!("Goodbye! 👋");
                    break;
                },
                "clear" => {
                    // Clear screen
                    if cfg!(windows) {
                        std::process::Command::new("cls").status().ok();
                    } else {
                        print!("\x1B[2J\x1B[1;1H");
                    }
                    continue;
                },
                "help" => {
                    println!("Available commands:");
                    println!("  quit/exit - Exit the program");
                    println!("  clear - Clear the screen");
                    println!("  help - Show this help message");
                    continue;
                },
                "" => continue,
                _ => {}
            }
            
            match self.generate(input, 10, Some(0.4)) {
                Ok(_) => println!(),
                Err(e) => eprintln!("Generation error: {}", e),
            }
        }
        
        Ok(())
    }

    /// Generate with custom parameters
    pub fn generate_with_params(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<String> {
        // Update logits processor with new parameters
        self.logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));
        self.generate(prompt, max_tokens, Some(temperature))
    }
}

// Helper function to determine the best available device
pub fn get_device() -> Result<Device> {
    if candle_core::utils::cuda_is_available() {
        println!("Using CUDA GPU acceleration");
        Ok(Device::new_cuda(0)?)
    } else if candle_core::utils::metal_is_available() {
        println!("Using Metal GPU acceleration");
        Ok(Device::new_metal(0)?)
    } else {
        println!("Using CPU (consider GPU acceleration for better performance)");
        Ok(Device::Cpu)
    }
}

fn print_system_info() {
    println!("=== System Information ===");
    println!("OS: {}", std::env::consts::OS);
    println!("Architecture: {}", std::env::consts::ARCH);
    
    // Check for GPU availability
    if candle_core::utils::cuda_is_available() {
        println!("CUDA: Available");
    } else {
        println!("CUDA: Not available");
    }
    
    if candle_core::utils::metal_is_available() {
        println!("Metal: Available");
    } else {
        println!("Metal: Not available");
    }
    println!("==========================\n");
}

fn main() -> Result<()> {
    println!("🦀 Minimal LLM Inference Engine 🦀");
    println!("Using Candle with GGUF support\n");
    
    print_system_info();
    
    let device = get_device()?;
    
    // Model file paths
    let model_path = "C:/Users/Ayush/Documents/PolyLLM/PolyLLM/models/Llama-3.2-1B-Instruct-Q4_0.gguf";
    let tokenizer_path = "C:/Users/Ayush/Documents/PolyLLM/PolyLLM/models/tokenizerLlama.json";
    
    // Check if files exist
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model file not found: {}", model_path);
        return Ok(());
    }
    
    if !std::path::Path::new(tokenizer_path).exists() {
        eprintln!("Tokenizer file not found: {}", tokenizer_path);
        eprintln!("   Download tokenizer.json from the same model repository");
        return Ok(());
    }
    
    println!("Loading model files...");
    println!("   Model: {}", model_path);
    println!("   Tokenizer: {}", tokenizer_path);
    
    match LLMInferenceEngine::new(model_path, tokenizer_path, device) {
        Ok(mut engine) => {
            println!("\nModel loaded successfully!");
            
            // // Example single generation
            // println!("\n--- Testing Generation ---");
            // match engine.generate("Hello", 10, Some(0.8)) {
            //     Ok(_) => {},
            //     Err(e) => eprintln!("Test generation failed: {}", e),
            // }
            
            // Start interactive chat
            engine.chat()?;
        },
        Err(e) => {
            eprintln!("Failed to initialize engine: {}", e);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_selection() {
        let device = get_device();
        assert!(device.is_ok());
    }
}