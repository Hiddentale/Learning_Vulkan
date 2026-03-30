#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragNormalWorld;
layout(location = 2) flat in uint fragMaterialId;

layout(binding = 0) uniform sampler2D texSampler;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection_matrix;
    vec3 light_direction;
    float ambient_strength;
} ubo;

struct MaterialEntry {
    vec3 color;
    float roughness;
    vec3 emissive;
    float _padding;
};

layout(binding = 2) uniform MaterialPalette {
    MaterialEntry entries[256];
} palette;

layout(location = 0) out vec4 outColor;

const float EDGE_WIDTH = 0.02;

void main() {
    MaterialEntry mat = palette.entries[fragMaterialId];

    // Material ID 1 = Grass: sample from texture atlas instead of palette color.
    // Atlas layout: left half = side face, right half = top face.
    vec3 baseColor;
    if (fragMaterialId == 1u) {
        bool isTop = fragNormalWorld.y > 0.5;
        vec2 atlasUV = vec2(fragTexCoord.x * 0.5 + (isTop ? 0.5 : 0.0), fragTexCoord.y);
        baseColor = texture(texSampler, atlasUV).rgb;
    } else {
        baseColor = mat.color;
    }

    vec3 N = normalize(fragNormalWorld);
    vec3 L = normalize(-ubo.light_direction);
    float diffuse = max(dot(N, L), 0.0);
    vec3 lighting = (ubo.ambient_strength + diffuse) * baseColor;
    vec3 finalColor = lighting + mat.emissive;

    float edgeX = min(fragTexCoord.x, 1.0 - fragTexCoord.x);
    float edgeY = min(fragTexCoord.y, 1.0 - fragTexCoord.y);
    float edge = min(edgeX, edgeY);

    if (edge < EDGE_WIDTH && fragMaterialId != 1u) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        outColor = vec4(finalColor, 1.0);
    }
}
