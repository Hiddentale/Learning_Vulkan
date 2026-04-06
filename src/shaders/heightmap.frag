#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) flat in uint fragMaterialId;
layout(location = 2) in vec3 fragWorldPos;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
} ubo;

struct MaterialEntry {
    vec3 color;
    float roughness;
    vec3 emissive;
    float _padding;
};

layout(set = 0, binding = 1) uniform MaterialPalette {
    MaterialEntry entries[256];
} palette;

layout(set = 0, binding = 2) uniform sampler2DArray texArray;

layout(location = 0) out vec4 outColor;

float hash1(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

vec3 stochastic_sample(uint layer, vec2 faceUV, vec2 worldCell, float atlasOffsetU) {
    float r = hash1(worldCell);
    bool flipX = r > 0.5;
    bool flipY = atlasOffsetU > 0.25 && fract(r * 3.7) > 0.5;
    vec2 uv = vec2(flipX ? 1.0 - faceUV.x : faceUV.x,
                   flipY ? 1.0 - faceUV.y : faceUV.y);
    uv.x = uv.x * 0.5 + atlasOffsetU;
    return texture(texArray, vec3(uv, float(layer))).rgb;
}

vec3 material_color(uint mat_id, vec3 world_pos, vec3 normal) {
    MaterialEntry mat = palette.entries[mat_id];
    vec3 texCenter = texture(texArray, vec3(0.25, 0.5, float(mat_id))).rgb;
    bool hasTexture = texCenter.r < 0.99 || texCenter.g < 0.99 || texCenter.b < 0.99;
    if (!hasTexture) return mat.color;

    bool isTop = normal.y > 0.5;
    float atlasOffset = isTop ? 0.5 : 0.0;
    vec2 faceUV;
    vec2 cell;
    if (isTop) {
        faceUV = fract(world_pos.xz);
        cell = floor(world_pos.xz);
    } else if (abs(normal.x) > 0.5) {
        faceUV = fract(world_pos.zy);
        cell = floor(world_pos.zy);
    } else {
        faceUV = fract(world_pos.xy);
        cell = floor(world_pos.xy);
    }
    return stochastic_sample(mat_id, faceUV, cell, atlasOffset);
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-ubo.light_direction);
    float diffuse = max(dot(N, L), 0.0);

    vec3 baseColor = material_color(fragMaterialId, fragWorldPos, N);
    MaterialEntry mat = palette.entries[fragMaterialId];
    vec3 color = (ubo.ambient_strength + diffuse) * baseColor + mat.emissive;
    outColor = vec4(color, 1.0);
}
