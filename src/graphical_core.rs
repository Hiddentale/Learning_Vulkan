pub mod buffers;
pub mod camera;
pub mod commands;
pub mod depth;
pub mod descriptors;
pub mod gpu;
pub mod input;
pub mod instance;
pub mod memory;
pub mod mesh;
pub mod pipeline;
pub mod queue_families;
pub mod render_pass;
mod shaders;
pub mod swapchain;
pub mod texture_mapping;
pub mod vulkan_object;

const MAX_FRAMES_IN_FLIGHT: usize = 2;
