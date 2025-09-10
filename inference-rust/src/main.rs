use llm::{load_dynamic, Model, ModelParameters, OutputToken};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_path = PathBuf::from("models/mistral-7b.gguf");

    // Load the Mistral model
    let model = load_dynamic(
        Some(&model_path),
        llm::ModelArchitecture::Mistral,
        Default::default(),
        llm::load_progress_callback_stdout,
    )?;

    // Wrap the model in an Arc<Mutex> for shared ownership
    let model = Arc::new(Mutex::new(model));

    let prompt = "Explain quantum entanglement in simple terms.";

    // Perform inference
    let response = model.lock().await.generate(prompt).await?;

    // Print the response
    println!("Response: {}", response);

    Ok(())
}
