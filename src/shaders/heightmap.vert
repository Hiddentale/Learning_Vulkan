#version 450
#extension GL_EXT_multiview : require

// Heightmap tile vertex shader. Vertices are pre-projected to world cartesian
// on the CPU (the tile mesh is curved already), so this shader just applies
// the view-projection. The morph delta is a scalar radial offset along the
// outward direction; geomorphing toward the coarser LOD interpolates along
// `normalize(position) * morph_delta_r * morph_factor`.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in uint inMaterialId;
layout(location = 3) in float inMorphDeltaR;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
    float planet_radius;
    float cube_half;
} ubo;

layout(push_constant) uniform PC {
    vec3 camera_pos;
    float morph_factor;
} pc;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) flat out uint fragMaterialId;
layout(location = 2) out vec3 fragWorldPos;

void main() {
    vec3 pos = inPosition;
    vec3 radial = normalize(pos);
    pos -= radial * inMorphDeltaR * pc.morph_factor;

    gl_Position = ubo.view_projection[gl_ViewIndex] * vec4(pos, 1.0);
    fragNormal = inNormal;
    fragMaterialId = inMaterialId;
    fragWorldPos = pos;
}
