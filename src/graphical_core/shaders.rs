use vulkan_rust::{cast_to_u32, vk, Device};

/// Wraps raw SPIR-V bytecode in a Vulkan shader module.
pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let code = cast_to_u32(bytecode).map_err(|e| anyhow::anyhow!("Invalid shader bytecode: {}", e))?;
    let info = vk::ShaderModuleCreateInfo::builder().code(code);
    Ok(device.create_shader_module(&info, None)?)
}
