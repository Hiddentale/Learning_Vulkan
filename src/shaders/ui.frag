#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D font_atlas;

layout(location = 0) out vec4 out_color;

void main() {
    float alpha = texture(font_atlas, frag_uv).r;
    // UV of (0,0) with zero alpha = solid colored quad (no font sampling)
    if (frag_uv == vec2(0.0)) {
        out_color = frag_color;
    } else {
        if (alpha < 0.5) discard;
        out_color = vec4(frag_color.rgb, frag_color.a * alpha);
    }
}
