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

void main() {
    vec3 pos = inPosition;
    pos.y -= inMorphDeltaY * pc.morph_factor;

    gl_Position = ubo.view_projection[gl_ViewIndex] * vec4(pos, 1.0);
    fragNormal = inNormal;
    fragMaterialId = inMaterialId;
    fragWorldPos = pos;
}
