#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec3 fragColor;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

const float EDGE_WIDTH = 0.02;

void main() {
    float edgeX = min(fragTexCoord.x, 1.0 - fragTexCoord.x);
    float edgeY = min(fragTexCoord.y, 1.0 - fragTexCoord.y);
    float edge = min(edgeX, edgeY);

    if (edge < EDGE_WIDTH) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        outColor = vec4(fragColor, 1.0);
    }
}
