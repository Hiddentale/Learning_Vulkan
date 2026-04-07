#version 450
#extension GL_EXT_multiview : require

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in uint inMaterialId;
layout(location = 3) in float inMorphDeltaY;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
} ubo;

layout(push_constant) uniform PC {
    vec3 camera_pos;
    float morph_factor;
} pc;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) flat out uint fragMaterialId;
layout(location = 2) out vec3 fragWorldPos;

// Phase B2b: cube-to-sphere projection (must match voxel.mesh / sphere.rs).
const float PLANET_RADIUS = 48.0;
const float CUBE_HALF = 24.0;

vec3 cube_to_sphere_unit(vec3 v) {
    float x = v.x, y = v.y, z = v.z;
    return vec3(
        x * sqrt(max(1.0 - y*y*0.5 - z*z*0.5 + y*y*z*z/3.0, 0.0)),
        y * sqrt(max(1.0 - z*z*0.5 - x*x*0.5 + z*z*x*x/3.0, 0.0)),
        z * sqrt(max(1.0 - x*x*0.5 - y*y*0.5 + x*x*y*y/3.0, 0.0))
    );
}

vec3 project_posy(vec3 flat_pos) {
    vec3 cube = vec3((flat_pos.x - CUBE_HALF) / CUBE_HALF, 1.0, (flat_pos.z - CUBE_HALF) / CUBE_HALF);
    vec3 unit = cube_to_sphere_unit(cube);
    return unit * (PLANET_RADIUS + flat_pos.y);
}

void main() {
    vec3 pos = inPosition;
    pos.y -= inMorphDeltaY * pc.morph_factor;
    vec3 world_pos = project_posy(pos);

    gl_Position = ubo.view_projection[gl_ViewIndex] * vec4(world_pos, 1.0);
    fragNormal = inNormal;
    fragMaterialId = inMaterialId;
    fragWorldPos = world_pos;
}
