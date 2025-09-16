use anyhow::Result;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};
use crate::tokenizer_service::TokenizerService;
use crate::model_service::ModelService;
use crate::inference_service::InferenceService;

#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    // pub timestamp: std::time::SystemTime,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
            // timestamp: std::time::SystemTime::now(),
        }
    }
}

pub enum CommandResult {
    Exit,
    Continue, 
    Process,
}

pub struct ChatSession {
    pub messages: VecDeque<ChatMessage>,
    pub system_prompt: String,
    pub max_history: usize,
}

impl ChatSession {
    pub fn new(system_prompt: Option<&str>, max_history: usize) -> Self {
        let default_system = "You are a helpful AI assistant. Be concise, accurate, and friendly.";
        
        Self {
            messages: VecDeque::new(),
            system_prompt: system_prompt.unwrap_or(default_system).to_string(),
            max_history,
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push_back(ChatMessage::new(role, content));
        
        while self.messages.len() > self.max_history {
            self.messages.pop_front();
        }
    }

    pub fn build_prompt(&self, user_input: &str) -> String {
        let mut prompt = format!("<|system|>\n{}\n\n", self.system_prompt);
        
        for msg in &self.messages {
            match msg.role.as_str() {
                "user" => prompt.push_str(&format!("<|user|>\n{}\n\n", msg.content)),
                "assistant" => prompt.push_str(&format!("<|assistant|>\n{}\n\n", msg.content)),
                _ => {}
            }
        }
        
        prompt.push_str(&format!("<|user|>\n{}\n\n<|assistant|>\n", user_input));
        prompt
    }

    pub fn clear_history(&mut self) {
        self.messages.clear();
    }
}

pub struct ChatService {
    tokenizer_service: TokenizerService,
    model_service: ModelService,
    inference_service: InferenceService,
    chat_session: ChatSession,
}

impl ChatService {
    pub fn new(
        tokenizer_service: TokenizerService,
        model_service: ModelService,
        inference_service: InferenceService,
    ) -> Self {
        let chat_session = ChatSession::new(
            Some("You are a helpful AI assistant. Provide clear, accurate, and conversational responses."),
            20
        );

        Self {
            tokenizer_service,
            model_service,
            inference_service,
            chat_session,
        }
    }

    pub fn start_interactive_chat(&mut self) -> Result<()> {
        self.print_chat_header();
        
        let stdin = io::stdin();
        
        loop {
            self.print_user_prompt();
            
            let mut input = String::new();
            match stdin.lock().read_line(&mut input) {
                Ok(_) => {},
                Err(e) => {
                    eprintln!(" Input error: {}", e);
                    continue;
                }
            }
            
            let input = input.trim();
            
            if input.is_empty() {
                continue;
            }
            
            match self.handle_command(input) {
                CommandResult::Exit => break,
                CommandResult::Continue => continue,
                CommandResult::Process => {}
            }
            
            // Add user message to session
            self.chat_session.add_message("user", input);
            
            // Build context-aware prompt
            let full_prompt = self.chat_session.build_prompt(input);
            
            // Generate response
            match self.inference_service.generate_streaming(
                &full_prompt,
                512,
                0.7,
                &mut self.model_service,
                &self.tokenizer_service,
                false,
            ) {
                Ok(response) => {
                    let cleaned_response = self.clean_response(&response);
                    self.chat_session.add_message("assistant", &cleaned_response);
                    println!(); // Add spacing
                },
                Err(e) => {
                    eprintln!("\n Generation error: {}", e);
                    eprintln!("Try a shorter message or use /reset to clear history.");
                }
            }
        }
        
        println!("\nThanks for chatting! Goodbye!");
        Ok(())
    }

    fn print_chat_header(&self) {
        println!("╭─────────────────────────────────────────────╮");
        println!("│                  PolyLLM                    │");
        println!("├─────────────────────────────────────────────┤");
        println!("│ Commands:                                   │");
        println!("│  /quit, /exit    - Exit chat                │");
        println!("│  /clear          - Clear screen             │");
        println!("│  /reset          - Reset conversation       │");
        println!("│  /help           - Show all commands        │");
        println!("│  /system <msg>   - Set system prompt        │");
        println!("│  /history        - Show conversation        │");
        println!("│  /save <file>    - Save conversation        │");
        println!("│  /stats          - Show model statistics    │");
        println!("╰─────────────────────────────────────────────╯");
        println!();
    }

    fn print_user_prompt(&self) {
        let msg_count = self.chat_session.messages.len() / 2 + 1;
        print!("[{}] You: ", msg_count);
        io::stdout().flush().unwrap();
    }

    fn handle_command(&mut self, input: &str) -> CommandResult {
        if !input.starts_with('/') {
            return CommandResult::Process;
        }

        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        let command = parts[0].to_lowercase();
        let args = parts.get(1).unwrap_or(&"");

        match command.as_str() {
            "/quit" | "/exit" => {
                println!("Goodbye!");
                CommandResult::Exit
            },
            "/clear" => {
                self.clear_screen();
                self.print_chat_header();
                CommandResult::Continue
            },
            "/reset" => {
                self.chat_session.clear_history();
                println!("Conversation history cleared!");
                CommandResult::Continue
            },
            "/help" => {
                self.show_help();
                CommandResult::Continue
            },
            "/system" => {
                self.handle_system_command(args);
                CommandResult::Continue
            },
            "/history" => {
                self.show_history();
                CommandResult::Continue
            },
            "/save" => {
                self.save_conversation(args);
                CommandResult::Continue
            },
            "/stats" => {
                self.show_stats();
                CommandResult::Continue
            },
            _ => {
                println!(" Unknown command: {}. Type /help for available commands.", command);
                CommandResult::Continue
            }
        }
    }

    fn clear_screen(&self) {
        if cfg!(windows) {
            std::process::Command::new("cmd")
                .args(["/c", "cls"])
                .status()
                .ok();
        } else {
            print!("\x1B[2J\x1B[1;1H");
        }
    }

    fn show_help(&self) {
        println!("- Available Commands:");
        println!("  /quit, /exit     - Exit the chat");
        println!("  /clear           - Clear screen");
        println!("  /reset           - Reset conversation history");
        println!("  /system <prompt> - Set new system prompt");
        println!("  /history         - Show conversation history");
        println!("  /save <filename> - Save conversation to file");
        println!("  /stats           - Show model statistics");
        println!("  /help            - Show this help message");
        println!();
    }

    fn handle_system_command(&mut self, args: &str) {
        if args.is_empty() {
            println!("Current system prompt: {}", self.chat_session.system_prompt);
        } else {
            self.chat_session.system_prompt = args.to_string();
            self.chat_session.clear_history();
            println!("System prompt updated and history cleared!");
        }
    }

    fn show_history(&self) {
        if self.chat_session.messages.is_empty() {
            println!("No conversation history yet.");
        } else {
            println!("Conversation History:");
            println!("   System: {}", self.chat_session.system_prompt);
            println!("   ─────────────────────────");
            
            for msg in &self.chat_session.messages {
                let role_emoji = match msg.role.as_str() {
                    "user" => "👤",
                    "assistant" => "🤖",
                    _ => "❓",
                };
                let preview = if msg.content.len() > 60 {
                    format!("{}...", &msg.content[..60])
                } else {
                    msg.content.clone()
                };
                println!("   {} {}: {}", role_emoji, msg.role, preview);
            }
            println!("   ─────────────────────────");
        }
    }

    fn save_conversation(&self, filename: &str) {
        if filename.is_empty() {
            println!(" Please specify a filename: /save <filename>");
            return;
        }

        match self.write_conversation_file(filename) {
            Ok(_) => println!("Conversation saved to: {}", filename),
            Err(e) => println!(" Failed to save: {}", e),
        }
    }

    fn write_conversation_file(&self, filename: &str) -> Result<()> {
        use std::fs::File;
        
        let mut file = File::create(filename)?;
        
        writeln!(file, "# AI Chat Conversation")?;
        writeln!(file, "Date: {:?}", std::time::SystemTime::now())?;
        writeln!(file, "System Prompt: {}", self.chat_session.system_prompt)?;
        writeln!(file, "Total Messages: {}", self.chat_session.messages.len())?;
        writeln!(file, "\n---\n")?;
        
        for msg in &self.chat_session.messages {
            writeln!(file, "## {}", 
                match msg.role.as_str() {
                    "user" => "User",
                    "assistant" => "Assistant", 
                    _ => &msg.role
                }
            )?;
            writeln!(file, "{}", msg.content)?;
            writeln!(file)?;
        }
        
        Ok(())
    }

    fn show_stats(&self) {
        println!("- Model Statistics:");
        println!("   Device: {:?}", self.model_service.device());
        println!("   Vocabulary size: {}", self.tokenizer_service.get_vocab_size());
        println!("   Conversation messages: {}", self.chat_session.messages.len());
        println!("   Max history: {}", self.chat_session.max_history);
        println!("   System prompt length: {} chars", self.chat_session.system_prompt.len());
    }

    fn clean_response(&self, response: &str) -> String {
        // First, stop at the first occurrence of <|user|> or <|assistant|>
        let cutoff = ["<|user|>", "<|assistant|>"]
            .iter()
            .filter_map(|tok| response.find(tok))
            .min()
            .unwrap_or(response.len());

        let cleaned = &response[..cutoff];

        cleaned
            .replace("<|system|>", "")
            .replace("<|end|>", "")
            .replace("</s>", "")
            .trim()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chat_session_creation() {
        let session = ChatSession::new(Some("Test prompt"), 10);
        assert_eq!(session.system_prompt, "Test prompt");
        assert_eq!(session.max_history, 10);
        assert_eq!(session.messages.len(), 0);
    }
    
    #[test]
    fn test_message_addition() {
        let mut session = ChatSession::new(None, 5);
        session.add_message("user", "Hello");
        session.add_message("assistant", "Hi there!");
        
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, "user");
        assert_eq!(session.messages[1].role, "assistant");
    }
}