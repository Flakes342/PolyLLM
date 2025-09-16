use anyhow::Result;
mod device_manager; //device manager module for handling device selection
mod model_service; // model service module for loading and running the model
mod tokenizer_service; // tokenizer service module for text-token conversion
mod chat_service; // chat service module for managing chat sessions
mod inference_service; // inference service module for generating responses

use device_manager::DeviceManager;
use model_service::ModelService;
use tokenizer_service::TokenizerService;
use chat_service::ChatService;
use inference_service::InferenceService;

fn main() -> Result<()> {
    println!("🦀 Microserviceable LLM Inference System 🦀");
    println!("Using modular architecture with Candle + GGUF\n");
    
    // Initialize device manager
    let device_manager = DeviceManager::new()?;
    device_manager.print_system_info();
    let device = device_manager.get_best_device()?;
    
    // Configuration
    let model_path = "C:/Users/Ayush/Documents/PolyLLM/PolyLLM/models/Llama-3.2-1B-Instruct-Q4_0.gguf";
    let tokenizer_path = "C:/Users/Ayush/Documents/PolyLLM/PolyLLM/models/tokenizerLlama.json";

    // Sampling parameters
    let temperature = 0.7;
    let top_p = 0.9;
    
    // Initialize services
    let tokenizer_service = TokenizerService::new(tokenizer_path)?;
    let model_service = ModelService::new(model_path, device)?;
    let inference_service = InferenceService::new();
    let mut chat_service = ChatService::new(
        tokenizer_service,
        model_service,
        inference_service,
    ); // need mutable as chat state changes
    
    println!("**All services initialized successfully!**\n");
    
    // Start interactive chat
    chat_service.start_interactive_chat()?;
    
    Ok(())
}