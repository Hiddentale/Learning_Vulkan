#version 450

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view_projection[2];
    mat4 inverse_view_projection[2];
    vec3 light_direction;
    float ambient_strength;
} camera;

// SVDAG geometry data (materials embedded in 64-byte leaf nodes)
layout(set = 0, binding = 1) readonly buffer SvdagGeometry {
    uint geometry_data[];
};

struct SvdagChunkInfo {
    vec3 aabb_min;
    uint dag_offset;
    vec3 aabb_max;
    uint dag_size;
};

layout(set = 0, binding = 2) readonly buffer SvdagChunkInfoSSBO {
    SvdagChunkInfo chunks[];
};

layout(push_constant) uniform PushConstants {
    vec3 camera_pos;
    uint chunk_count;
    vec2 screen_size;
};

layout(location = 0) out vec4 out_color;

// --- Ray-AABB intersection ---
bool ray_aabb(vec3 ro, vec3 inv_rd, vec3 aabb_min, vec3 aabb_max, out float t_near, out float t_far) {
    vec3 t0 = (aabb_min - ro) * inv_rd;
    vec3 t1 = (aabb_max - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far = min(min(tmax.x, tmax.y), tmax.z);
    return t_near <= t_far && t_far > 0.0;
}

// --- Read a byte from the geometry SSBO ---
uint read_byte(uint byte_offset) {
    uint word = geometry_data[byte_offset >> 2];
    uint shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// --- Read a u32 from the geometry SSBO ---
uint read_u32(uint byte_offset) {
    uint b0 = read_byte(byte_offset);
    uint b1 = read_byte(byte_offset + 1u);
    uint b2 = read_byte(byte_offset + 2u);
    uint b3 = read_byte(byte_offset + 3u);
    return b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
}

// --- Compute octant index ---
uint octant_index(uvec3 pos, uint half_size) {
    uint xi = pos.x >= half_size ? 1u : 0u;
    uint yi = pos.y >= half_size ? 1u : 0u;
    uint zi = pos.z >= half_size ? 1u : 0u;
    return xi | (yi << 1u) | (zi << 2u);
}

// --- SVDAG lookup: returns material ID (0 = air) ---
uint svdag_lookup(uint dag_base, uint x, uint y, uint z) {
    uint offset = dag_base + read_u32(dag_base);
    uint size = 16u;

    while (size > 4u) {
        uint child_mask = read_byte(offset);
        uint half_size = size >> 1u;
        uint octant = octant_index(uvec3(x, y, z), half_size);

        if ((child_mask & (1u << octant)) == 0u) {
            return 0u; // air
        }

        uint bits_below = child_mask & ((1u << octant) - 1u);
        uint child_rank = bitCount(bits_below);
        uint child_offset = read_u32(offset + 1u + child_rank * 4u);

        offset = dag_base + child_offset;
        x = x % half_size;
        y = y % half_size;
        z = z % half_size;
        size = half_size;
    }

    // Leaf: 64 bytes, one u8 per voxel. Layout: x + z*4 + y*16
    uint idx = x + z * 4u + y * 16u;
    return read_byte(offset + idx);
}

// --- DDA through a chunk's SVDAG ---
bool trace_chunk(vec3 ro, vec3 rd, vec3 aabb_min, float t_enter, float t_exit,
                 uint dag_base, out float t_hit, out vec3 hit_normal, out uint hit_material) {
    vec3 pos = ro + rd * (t_enter + 0.001);
    vec3 local = pos - aabb_min;
    ivec3 voxel = clamp(ivec3(floor(local)), ivec3(0), ivec3(15));

    ivec3 step_dir = ivec3(sign(rd));
    vec3 t_delta = abs(1.0 / rd);
    vec3 t_max_val;
    for (int i = 0; i < 3; i++) {
        if (rd[i] > 0.0) {
            t_max_val[i] = t_enter + (float(voxel[i] + 1) - local[i]) / rd[i];
        } else if (rd[i] < 0.0) {
            t_max_val[i] = t_enter + (float(voxel[i]) - local[i]) / rd[i];
        } else {
            t_max_val[i] = 1e30;
        }
    }

    for (int i = 0; i < 64; i++) {
        if (voxel.x < 0 || voxel.x > 15 || voxel.y < 0 || voxel.y > 15 || voxel.z < 0 || voxel.z > 15) {
            return false;
        }

        uint mat = svdag_lookup(dag_base, uint(voxel.x), uint(voxel.y), uint(voxel.z));
        if (mat != 0u) {
            t_hit = min(min(t_max_val.x, t_max_val.y), t_max_val.z);
            if (t_max_val.x < t_max_val.y && t_max_val.x < t_max_val.z) {
                hit_normal = vec3(-float(step_dir.x), 0.0, 0.0);
            } else if (t_max_val.y < t_max_val.z) {
                hit_normal = vec3(0.0, -float(step_dir.y), 0.0);
            } else {
                hit_normal = vec3(0.0, 0.0, -float(step_dir.z));
            }
            hit_material = mat;
            return true;
        }

        if (t_max_val.x < t_max_val.y) {
            if (t_max_val.x < t_max_val.z) {
                voxel.x += step_dir.x;
                t_max_val.x += t_delta.x;
            } else {
                voxel.z += step_dir.z;
                t_max_val.z += t_delta.z;
            }
        } else {
            if (t_max_val.y < t_max_val.z) {
                voxel.y += step_dir.y;
                t_max_val.y += t_delta.y;
            } else {
                voxel.z += step_dir.z;
                t_max_val.z += t_delta.z;
            }
        }

        if (min(t_max_val.x, min(t_max_val.y, t_max_val.z)) > t_exit) {
            return false;
        }
    }
    return false;
}

vec3 material_color(uint mat_id) {
    if (mat_id == 1u) return vec3(0.24, 0.60, 0.15); // Grass
    if (mat_id == 2u) return vec3(0.55, 0.35, 0.17); // Dirt
    if (mat_id == 3u) return vec3(0.50, 0.50, 0.50); // Stone
    if (mat_id == 4u) return vec3(0.20, 0.50, 0.95); // Water
    if (mat_id == 5u) return vec3(0.82, 0.75, 0.52); // Sand
    if (mat_id == 6u) return vec3(0.90, 0.92, 0.95); // Snow
    if (mat_id == 7u) return vec3(0.55, 0.55, 0.55); // Gravel
    return vec3(0.50, 0.50, 0.50);
}

// --- Simple frustum check: is an AABB at least partially in front of the camera? ---
bool aabb_in_front(vec3 aabb_min, vec3 aabb_max, vec3 cam_pos, vec3 cam_fwd) {
    // Test the AABB corner most in the forward direction
    vec3 p = mix(aabb_min, aabb_max, vec3(greaterThan(cam_fwd, vec3(0.0))));
    return dot(p - cam_pos, cam_fwd) > 0.0;
}

void main() {
    vec2 uv = gl_FragCoord.xy / screen_size;
    uint view_index = 0;

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world_near = camera.inverse_view_projection[view_index] * vec4(ndc, 0.0, 1.0);
    vec4 world_far = camera.inverse_view_projection[view_index] * vec4(ndc, 1.0, 1.0);
    world_near /= world_near.w;
    world_far /= world_far.w;

    vec3 ro = world_near.xyz;
    vec3 rd = normalize(world_far.xyz - world_near.xyz);
    vec3 inv_rd = 1.0 / rd;

    // Camera forward from inverse VP (center pixel direction)
    vec4 center_far = camera.inverse_view_projection[view_index] * vec4(0.0, 0.0, 1.0, 1.0);
    center_far /= center_far.w;
    vec3 cam_fwd = normalize(center_far.xyz - ro);

    float best_t = 1e30;
    vec3 best_normal = vec3(0.0);
    uint best_material = 0u;

    for (uint ci = 0; ci < chunk_count; ci++) {
        SvdagChunkInfo chunk = chunks[ci];

        // Frustum cull: skip chunks entirely behind the camera
        if (!aabb_in_front(chunk.aabb_min, chunk.aabb_max, camera_pos, cam_fwd)) {
            continue;
        }

        float t_near, t_far;
        if (!ray_aabb(ro, inv_rd, chunk.aabb_min, chunk.aabb_max, t_near, t_far)) {
            continue;
        }
        if (t_near > best_t) continue;

        float t_hit;
        vec3 hit_normal;
        uint hit_mat;
        if (trace_chunk(ro, rd, chunk.aabb_min, max(t_near, 0.0), t_far,
                        chunk.dag_offset, t_hit, hit_normal, hit_mat)) {
            if (t_hit < best_t) {
                best_t = t_hit;
                best_normal = hit_normal;
                best_material = hit_mat;
            }
        }
    }

    if (best_t >= 1e29) {
        discard;
    }

    // Project hit position to clip space for depth write
    vec3 hit_world = ro + rd * best_t;
    vec4 hit_clip = camera.view_projection[view_index] * vec4(hit_world, 1.0);
    gl_FragDepth = hit_clip.z / hit_clip.w;

    vec3 base_color = material_color(best_material);
    float ndl = max(dot(best_normal, camera.light_direction), 0.0);
    vec3 color = base_color * (camera.ambient_strength + ndl * (1.0 - camera.ambient_strength));
    out_color = vec4(color, 1.0);
}
