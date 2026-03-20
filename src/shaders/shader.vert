#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(binding = 1) uniform UniformBufferObject {
    mat4 model_matrix;
    mat4 view_projection_matrix;
} ubo;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.view_projection_matrix * ubo.model_matrix * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
}