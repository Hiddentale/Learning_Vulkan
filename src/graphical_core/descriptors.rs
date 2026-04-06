use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkan_rust::{vk, Device};

/// Defines the descriptor set layout: what resources shaders can access.
/// Used by the sky pipeline. The mesh shader pipeline has its own layout.
pub fn create_layout(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let sampler_binding = *vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let ubo_binding = *vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

    let palette_binding = *vk::DescriptorSetLayoutBinding::builder()
        .binding(2)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = [sampler_binding, ubo_binding, palette_binding];
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    data.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };
    Ok(())
}

/// Creates a descriptor pool sized for one set with a sampler and two uniform buffers.
pub fn create_pool(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let sampler_pool_size = *vk::DescriptorPoolSize::builder()
        .descriptor_count(1)
        .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);

    let ubo_pool_size = *vk::DescriptorPoolSize::builder()
        .descriptor_count(2)
        .r#type(vk::DescriptorType::UNIFORM_BUFFER);

    let pool_sizes = [sampler_pool_size, ubo_pool_size];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);

    data.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    Ok(())
}

/// Allocates a descriptor set from a pool, matching the provided layout.
pub fn allocate_set(device: &Device, descriptor_pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> anyhow::Result<Vec<vk::DescriptorSet>> {
    let layouts = [layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    Ok(unsafe { device.allocate_descriptor_sets(&allocate_info)? })
}

/// Writes actual resources (texture sampler + camera UBO + palette UBO) into a descriptor set.
pub fn update_set(
    device: &Device,
    descriptor_set: vk::DescriptorSet,
    image_view: vk::ImageView,
    sampler: vk::Sampler,
    uniform_buffer: vk::Buffer,
    palette_buffer: vk::Buffer,
) {
    // Bind arrays to locals so they outlive the WriteDescriptorSet structs.
    // The builder stores raw pointers — dereferencing (*) copies the struct but
    // not the pointed-to data. Temporaries would dangle in release mode.
    let image_infos = [*vk::DescriptorImageInfo::builder()
        .image_view(image_view)
        .sampler(sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

    let ubo_infos = [*vk::DescriptorBufferInfo::builder()
        .buffer(uniform_buffer)
        .offset(0)
        .range(std::mem::size_of::<crate::graphical_core::camera::UniformBufferObject>() as u64)];

    let palette_infos = [*vk::DescriptorBufferInfo::builder()
        .buffer(palette_buffer)
        .offset(0)
        .range(std::mem::size_of::<crate::voxel::material::MaterialPalette>() as u64)];

    let writes = [
        *vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos),
        *vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_infos),
        *vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&palette_infos),
    ];

    unsafe {
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}
