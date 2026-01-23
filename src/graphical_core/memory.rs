use anyhow;
use vulkanalia::vk;

/// Finds a memory type that satisfies both hardware requirements and desired properties.
/// # Parameters
/// - `memory_properties`: The GPU's available memory types and their properties
/// - `allowed_memory_types`: Bitmask of which memory types the buffer supports
/// - `desired_properties`: The properties we need
///
/// # Returns
/// The index of the first suitable memory type found.
///
/// # Errors
/// Returns an error if no memory type satisfies both requirements.
pub fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    allowed_memory_types: u32,
    requested_properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    let number_of_different_memory_types = memory_properties.memory_type_count;

    for memory_type_index in 0..number_of_different_memory_types {
        let memory_type_is_allowed = (allowed_memory_types & (1 << memory_type_index)) != 0;
        if memory_type_is_allowed {
            let memory_type_properties = memory_properties.memory_types[memory_type_index as usize].property_flags;

            let has_all_desired_properties = (memory_type_properties & requested_properties) == requested_properties;
            if has_all_desired_properties {
                return Ok(memory_type_index);
            }
        }
    }
    anyhow::bail!(
        "Failed to find a suitable memory type for requested properties: {:?}",
        requested_properties
    );
}
