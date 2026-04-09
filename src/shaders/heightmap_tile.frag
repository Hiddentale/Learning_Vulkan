#version 450

// Heightmap tile fragment shader. Matches the voxel mesh visual style:
// stochastic per-block texture sampling from the same texture array and
// material palette that the voxel mesh pipeline uses. Material id is
// sampled from a parallel R8_UINT atlas at the same page layout as the
// height atlas.

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in float fragHeight;
layout(location = 2) in vec2 fragFaceUV;   // face-local (u, v) in block coords
layout(location = 3) in vec2 fragAtlasUV;  // atlas UV for material lookup

layout(set = 0, binding = 3) uniform CameraUBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
    float planet_radius;
    float cube_half;
} camera;

// Material atlas (R8_UINT, NEAREST): same page layout as the height atlas.
layout(set = 0, binding = 5) uniform usampler2D mat_atlas;

// Texture array shared with the voxel mesh pipeline.
layout(set = 0, binding = 6) uniform sampler2DArray texArray;

// Material palette shared with the voxel mesh pipeline.
struct MaterialEntry {
    vec3 color;
    float roughness;
    vec3 emissive;
    float _padding;
};
layout(set = 0, binding = 7) uniform MaterialPalette {
    MaterialEntry entries[256];
} palette;

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

    // Heightmap is always "looking down at the top", so use the top-face
    // atlas half. For steep slopes, fall back to the side-face half using
    // the world normal, exactly like heightmap.frag does.
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
    // Sample material id from the material atlas.
    uint mat_id = texture(mat_atlas, fragAtlasUV).r;

    // Screen-space derivatives give the geometric normal. The cross
    // product sign depends on screen-space winding, so flip to always
    // point outward (same hemisphere as the radial direction).
    vec3 N = normalize(cross(dFdx(fragWorldPos), dFdy(fragWorldPos)));
    if (dot(N, normalize(fragWorldPos)) < 0.0) N = -N;
    vec3 L = normalize(-camera.light_direction);
    float diffuse = max(dot(N, L), 0.0);

    vec3 baseColor = material_color(mat_id, fragWorldPos, N);
    MaterialEntry mat = palette.entries[mat_id];
    vec3 color = (camera.ambient_strength + diffuse) * baseColor + mat.emissive;
    outColor = vec4(color, 1.0);
}
