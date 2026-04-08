#version 450

// Heightmap tile fragment shader. Phase 3 minimal version: derive material
// color from height-vs-sea-level bands and apply lambert against the
// camera UBO's directional light. Phase 4 will add stochastic texturing
// to match the voxel mesh shader path.

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in float fragHeight;

layout(set = 0, binding = 3) uniform CameraUBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
    float planet_radius;
    float cube_half;
} camera;

layout(location = 0) out vec4 outColor;

const float SEA_LEVEL = 64.0;       // matches voxel::terrain::SEA_LEVEL
const float SAND_BAND = 4.0;
const float SNOW_HEIGHT = 200.0;

vec3 material_color(float h) {
    if (h < SEA_LEVEL - 0.5) return vec3(0.10, 0.30, 0.55);          // water
    if (h < SEA_LEVEL + SAND_BAND) return vec3(0.85, 0.78, 0.55);   // sand
    if (h > SNOW_HEIGHT) return vec3(0.95);                          // snow
    return vec3(0.30, 0.55, 0.20);                                   // grass
}

void main() {
    // Outward radial as a cheap normal for lambert. Phase 4 will compute
    // a real surface normal from finite-difference texture samples.
    vec3 N = normalize(fragWorldPos);
    vec3 L = normalize(-camera.light_direction);
    float diffuse = max(dot(N, L), 0.0);
    vec3 base = material_color(fragHeight);
    vec3 color = (camera.ambient_strength + diffuse) * base;
    outColor = vec4(color, 1.0);
}
