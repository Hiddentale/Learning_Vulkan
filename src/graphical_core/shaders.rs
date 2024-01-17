use vulkanalia::{Device, vk};
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::{DeviceV1_0, HasBuilder};

pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();
    let info = vk::ShaderModuleCreateInfo::builder().code_size(bytecode.code_size()).code(bytecode.code());
    Ok(device.create_shader_module(&info, None)?)
}