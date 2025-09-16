use anyhow::Result;
use candle_core::Device;

pub struct DeviceManager;

impl DeviceManager {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn get_best_device(&self) -> Result<Device> {
        if candle_core::utils::cuda_is_available() {
            println!("Using CUDA GPU acceleration");
            Ok(Device::new_cuda(0)?)
        } else if candle_core::utils::metal_is_available() {
            println!("Using Metal GPU acceleration");
            Ok(Device::new_metal(0)?)
        } else {
            println!("Using CPU (consider GPU for better performance)");
            Ok(Device::Cpu)
        }
    }

    pub fn print_system_info(&self) {
        println!("╭──────────  System Information ────────────╮");
        println!("│ OS: {:20}                  │", std::env::consts::OS);
        println!("│ Architecture: {:11}                 │", std::env::consts::ARCH);
        
        let cuda_status = if candle_core::utils::cuda_is_available() { "Available" } else { "Not available" };
        let metal_status = if candle_core::utils::metal_is_available() { "Available" } else { "Not available" };
        
        println!("│ CUDA: {:19}                 │", cuda_status);
        println!("│ Metal: {:18}                 │", metal_status);
        println!("╰───────────────────────────────────────────╯\n");
    }
}