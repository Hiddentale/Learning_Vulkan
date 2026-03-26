#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inColor;

layout(push_constant) uniform PushConstants {
    mat4 model_matrix;
} push;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection_matrix;
} ubo;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragColor;

void main() {
    gl_Position = ubo.view_projection_matrix * push.model_matrix * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    fragColor = inColor;
}
