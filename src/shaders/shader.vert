#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in uint inMaterialId;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection_matrix;
    vec3 light_direction;
    float ambient_strength;
} ubo;

layout(binding = 3) readonly buffer TransformSSBO {
    mat4 model_matrices[];
} transforms;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragNormalWorld;
layout(location = 2) flat out uint fragMaterialId;

void main() {
    mat4 model = transforms.model_matrices[gl_InstanceIndex];
    gl_Position = ubo.view_projection_matrix * model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    // Chunk transforms are translation-only, so mat3(model) is safe
    fragNormalWorld = mat3(model) * inNormal;
    fragMaterialId = inMaterialId;
}
