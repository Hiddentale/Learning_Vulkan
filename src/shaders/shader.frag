#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragNormalWorld;
layout(location = 2) flat in uint fragMaterialId;
layout(location = 3) in vec3 fragWorldPos;

layout(binding = 0) uniform sampler2D texSampler;

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection_matrix;
    mat4 inverse_view_projection;
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

// Hash a block-grid cell to a random value in [0, 1).
float hash1(vec2 cell) {
    return fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
}

// Stochastic tiling for pixel art: randomly flips UV axes per block cell.
// Flipping preserves pixel alignment (unlike rotation). Uses fragTexCoord
// (0-1 face UV) for sampling and floor(worldPos) for per-block randomisation.
vec3 stochastic_sample(sampler2D tex, vec2 faceUV, vec2 worldCell, float atlasOffsetU) {
    float r = hash1(worldCell);
    bool flipX = r > 0.5;
    // Only flip Y for the top face — side faces have a directional gradient
    // (grass at top, dirt at bottom) that must not be inverted.
    bool flipY = atlasOffsetU > 0.25 && fract(r * 3.7) > 0.5;
    vec2 uv = vec2(flipX ? 1.0 - faceUV.x : faceUV.x,
                   flipY ? 1.0 - faceUV.y : faceUV.y);
    // Scale x into the correct atlas half [0, 0.5] or [0.5, 1.0]
    uv.x = uv.x * 0.5 + atlasOffsetU;
    return texture(tex, uv).rgb;
}

void main() {
    MaterialEntry mat = palette.entries[fragMaterialId];

    // Material ID 1 = Grass: atlas sample with per-block stochastic flip.
    // Atlas layout: left half (U: 0.0-0.5) = side, right half (U: 0.5-1.0) = top.
    // floor(worldPos) gives a stable integer cell per block face for hashing.
    vec3 baseColor;
    if (fragMaterialId == 1u) {
        bool isTop = fragNormalWorld.y > 0.5;
        float atlasOffset = isTop ? 0.5 : 0.0;
        // Pick the two axes that vary across this face (not the normal axis).
        vec2 cell;
        if (isTop) {
            cell = floor(fragWorldPos.xz);
        } else if (abs(fragNormalWorld.x) > 0.5) {
            cell = floor(fragWorldPos.yz);
        } else {
            cell = floor(fragWorldPos.xy);
        }
        baseColor = stochastic_sample(texSampler, fragTexCoord, cell, atlasOffset);
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
