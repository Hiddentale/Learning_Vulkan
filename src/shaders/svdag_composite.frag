#version 450

layout(set = 0, binding = 0) uniform sampler2D svdag_color;
layout(set = 0, binding = 1) uniform sampler2D svdag_depth;

layout(location = 0) out vec4 out_color;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(svdag_color, 0));
    vec4 color = texture(svdag_color, uv);
    if (color.a < 0.01) discard;

    float depth = texture(svdag_depth, uv).r;
    gl_FragDepth = depth;
    out_color = color;
}
