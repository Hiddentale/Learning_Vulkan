#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in uint inMaterialId;

layout(push_constant) uniform PushConstants {
    mat4 model_matrix;
} push;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection_matrix;
    vec3 light_direction;
    float ambient_strength;
} ubo;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragNormalWorld;
layout(location = 2) flat out uint fragMaterialId;

void main() {
    gl_Position = ubo.view_projection_matrix * push.model_matrix * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    // Chunk transforms are translation-only, so mat3(model_matrix) is safe
    // (no inverse-transpose needed for uniform scale)
    fragNormalWorld = mat3(push.model_matrix) * inNormal;
    fragMaterialId = inMaterialId;
}
