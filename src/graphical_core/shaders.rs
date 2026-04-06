use vulkan_rust::{cast_to_u32, vk, Device};

/// Wraps raw SPIR-V bytecode in a Vulkan shader module.
/// Handles unaligned `include_bytes!` data by copying to an aligned buffer when needed.
pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let code = match cast_to_u32(bytecode) {
        Ok(aligned) => {
            let info = vk::ShaderModuleCreateInfo::builder().code(aligned);
            return Ok(device.create_shader_module(&info, None)?);
        }
        Err(_) => {
            // include_bytes! pointer not 4-byte aligned — copy into aligned Vec
            assert!(bytecode.len().is_multiple_of(4), "SPIR-V size must be a multiple of 4");
            let mut aligned = vec![0u32; bytecode.len() / 4];
            std::ptr::copy_nonoverlapping(bytecode.as_ptr(), aligned.as_mut_ptr() as *mut u8, bytecode.len());
            aligned
        }
    };
    let info = vk::ShaderModuleCreateInfo::builder().code(&code);
    Ok(device.create_shader_module(&info, None)?)
}
