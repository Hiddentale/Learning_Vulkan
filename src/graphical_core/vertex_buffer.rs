/*
Plan:
    Allocate a chunk of GPU memory (vertex buffer):
        1. Create a buffer with usage VERTEX_BUFFER
        2. Allocate memory with properties HOST_VISIBLE | HOST_COHERENT
        3. Bind buffer to memory

    Upload vertex data from CPU → GPU:
        4. Map the memory (get a raw pointer)
        5. memcpy your vertex data into it
        6. Unmap the memory

    Bind that buffer when  ready to draw
    Tell the GPU how to interpret the data
 */

use vulkanalia::{vk::DeviceV1_0, Device, vk};

//#[repr(C)]
//#[derive(Copy, Clone)]
unsafe fn temp(device: &Device) {
    let buffer_create_info = &vk::BufferCreateInfo {};
    let buffer = device.create_buffer(buffer_create_info,);
}
