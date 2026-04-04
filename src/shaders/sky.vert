#version 450
#extension GL_EXT_multiview : enable

// Fullscreen triangle — no vertex input needed.
// gl_VertexIndex 0,1,2 produces a triangle covering the entire screen.
void main() {
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
}
