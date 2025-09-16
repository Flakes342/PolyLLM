use anyhow::{Result};
use tokenizers::Tokenizer;

pub struct TokenizerService {
    tokenizer: Tokenizer,
}

impl TokenizerService {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        if !std::path::Path::new(tokenizer_path).exists() {
            return Err(anyhow::anyhow!("Tokenizer file not found: {}", tokenizer_path));
        }

        println!("- Loading tokenizer from: {}", tokenizer_path);
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("- Failed to load tokenizer: {}", e))?;

        println!("- Tokenizer loaded successfully!");
        
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self.tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(tokens.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    pub fn get_eos_token_id(&self) -> u32 {
        self.tokenizer.token_to_id("</s>")
            .or_else(|| self.tokenizer.token_to_id("<|endoftext|>")) //chain optional operations if the first returns None we try the next
            .or_else(|| self.tokenizer.token_to_id("<|end|>"))
            .or_else(|| self.tokenizer.token_to_id("<eos>"))
            .or_else(|| self.tokenizer.token_to_id("[EOS]"))
            .unwrap_or(2)
    } //somehow the model isn't recognizing eos tokens properly as of now

    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }
}