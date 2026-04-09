#version 450
#extension GL_EXT_multiview : enable

layout(binding = 1) uniform UniformBufferObject {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
} ubo;

layout(push_constant) uniform SkyPush {
    vec2 screen_size;
} push;

layout(location = 0) out vec4 outColor;

const vec3 ZENITH_COLOR  = vec3(0.15, 0.3, 0.65);
const vec3 HORIZON_COLOR = vec3(0.6, 0.75, 0.9);
const vec3 GROUND_COLOR  = vec3(0.2, 0.2, 0.2);
const vec3 SUN_COLOR     = vec3(1.0, 0.95, 0.8);
const float SUN_RADIUS   = 0.995;
const float SUN_GLOW     = 0.97;

void main() {
    // Convert pixel coords to NDC [-1, 1]. Reverse-Z: small ndc.z = far plane.
    // Use a tiny non-zero value so the inverse-VP divide stays well-conditioned
    // even with an infinite-far projection (where ndc.z=0 maps to w→0).
    vec2 ndc = (gl_FragCoord.xy / push.screen_size) * 2.0 - 1.0;
    vec4 clip = vec4(ndc, 1e-5, 1.0);

    // Unproject to world space using this eye's inverse VP
    vec4 world_pos = ubo.inverse_view_projection[gl_ViewIndex] * clip;
    vec3 dir = normalize(world_pos.xyz / world_pos.w);

    // Vertical gradient
    float y = dir.y;
    vec3 sky;
    if (y > 0.0) {
        float t = sqrt(y);
        sky = mix(HORIZON_COLOR, ZENITH_COLOR, t);
    } else {
        float t = sqrt(-y);
        sky = mix(HORIZON_COLOR, GROUND_COLOR, t);
    }

    // Sun disc and glow
    vec3 sun_dir = normalize(-ubo.light_direction);
    float sun_dot = dot(dir, sun_dir);
    if (sun_dot > SUN_RADIUS) {
        sky = SUN_COLOR;
    } else if (sun_dot > SUN_GLOW) {
        float glow = (sun_dot - SUN_GLOW) / (SUN_RADIUS - SUN_GLOW);
        sky = mix(sky, SUN_COLOR, glow * glow);
    }

    outColor = vec4(sky, 1.0);
}
