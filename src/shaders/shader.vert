#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in uint inMaterialId;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection_matrix;
    mat4 inverse_view_projection;
    vec3 light_direction;
    float ambient_strength;
} ubo;

layout(binding = 3) readonly buffer TransformSSBO {
    mat4 model_matrices[];
} transforms;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec3 fragNormalWorld;
layout(location = 2) flat out uint fragMaterialId;
layout(location = 3) out vec3 fragWorldPos;

void main() {
    mat4 model = transforms.model_matrices[gl_InstanceIndex];
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = ubo.view_projection_matrix * worldPos;
    fragTexCoord = inTexCoord;
    // Chunk transforms are translation-only, so mat3(model) is safe
    fragNormalWorld = mat3(model) * inNormal;
    fragMaterialId = inMaterialId;
    fragWorldPos = worldPos.xyz;
}
