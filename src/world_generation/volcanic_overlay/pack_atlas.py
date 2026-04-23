"""
Pack volcanic GeoTIFF patches into a single binary atlas for Rust.

Reads all .tif files from data/volcanic_patches/<category>/, resamples
each to 256x256, and writes a single binary atlas file.

Format:
    Header (20 bytes):
        magic:       4 bytes  "VOLC"
        version:     u32      1
        n_categories: u32     4
        patches_per_cat: u32  max patches in any category (padded with zeros)
        patch_size:  u32      256

    Per-patch metadata (16 bytes each), n_categories * patches_per_cat entries:
        min_elev:    i16      minimum elevation in meters
        max_elev:    i16      maximum elevation in meters
        mean_elev:   i16      mean elevation in meters
        land_pct:    u8       percentage of pixels above sea level (0-100)
        flags:       u8       bit 0: valid (1) or padding (0)
        base_level:  i16      median elevation along outer edge (seafloor baseline)
        peak_height: i16      max elevation relative to base_level
        _reserved:   4 bytes

    Pixel data: n_categories * patches_per_cat * 256 * 256 * i16
        Row-major elevation in meters. Invalid patches are all zeros.

Usage:
    pip install rasterio numpy
    python src/world_generation/volcanic_overlay/pack_atlas.py

Output:
    data/volcanic_atlas.bin
    data/volcanic_atlas.ply  (3D visualization)
"""

import os
import struct
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PATCHES_DIR = os.path.join(PROJECT_ROOT, "data", "volcanic_patches")
OUTPUT_BIN = os.path.join(PROJECT_ROOT, "data", "volcanic_atlas.bin")
OUTPUT_PLY = os.path.join(PROJECT_ROOT, "data", "volcanic_atlas.ply")

PATCH_SIZE = 256

CATEGORIES = [
    "shield_volcanic_islands",
    "stratovolcanic_arcs",
    "seamounts_and_atolls",
    "archipelago_clusters",
]


def load_and_resample(path):
    """Load a GeoTIFF and resample to PATCH_SIZE x PATCH_SIZE."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as ds:
        data = ds.read(
            1,
            out_shape=(PATCH_SIZE, PATCH_SIZE),
            resampling=Resampling.bilinear,
        )
    return data.astype(np.int16)


def compute_base_level(patch):
    """Median elevation along the outer 2-pixel border."""
    border = np.concatenate([
        patch[0, :], patch[-1, :],        # top and bottom rows
        patch[1:-1, 0], patch[1:-1, -1],  # left and right columns
    ])
    return int(np.median(border))


def export_ply(all_patches, categories_info):
    """Write a PLY file showing all patches as 3D surfaces laid out in a grid."""
    valid_patches = []
    for cat_idx, (cat_name, patches) in enumerate(categories_info):
        for patch_idx, (name, patch) in enumerate(patches):
            valid_patches.append((cat_name, name, cat_idx, patch_idx, patch))

    if not valid_patches:
        print("No patches to export.")
        return

    spacing = PATCH_SIZE + 20
    step = 4  # sample every 4th pixel for manageable file size
    sampled = PATCH_SIZE // step

    all_verts = []
    all_faces = []
    vert_offset = 0

    for cat_name, name, cat_idx, patch_idx, patch in valid_patches:
        col = patch_idx
        row = cat_idx
        ox = col * spacing
        oy = row * spacing

        base = compute_base_level(patch)

        verts = []
        for r in range(sampled):
            for c in range(sampled):
                pr, pc = r * step, c * step
                elev = float(patch[pr, pc] - base)
                x = ox + c * step
                z = oy + r * step
                y = elev * 0.02  # vertical exaggeration

                # Color by elevation
                if patch[pr, pc] > 0:
                    # Land: green to brown
                    t = min(1.0, max(0.0, elev / 4000.0))
                    cr = int(80 + t * 140)
                    cg = int(160 - t * 80)
                    cb = int(60 + t * 40)
                else:
                    # Ocean: blue shades
                    t = min(1.0, max(0.0, -patch[pr, pc] / 6000.0))
                    cr = int(30 - t * 20)
                    cg = int(80 - t * 40)
                    cb = int(200 - t * 80)

                verts.append((x, y, z, cr, cg, cb))

        faces = []
        for r in range(sampled - 1):
            for c in range(sampled - 1):
                i = r * sampled + c
                faces.append((
                    vert_offset + i,
                    vert_offset + i + 1,
                    vert_offset + i + sampled,
                ))
                faces.append((
                    vert_offset + i + 1,
                    vert_offset + i + sampled + 1,
                    vert_offset + i + sampled,
                ))

        all_verts.extend(verts)
        all_faces.extend(faces)
        vert_offset += len(verts)

    with open(OUTPUT_PLY, "wb") as f:
        header = (
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {len(all_verts)}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"element face {len(all_faces)}\n"
            f"property list uchar int vertex_indices\n"
            f"end_header\n"
        )
        f.write(header.encode("ascii"))

        for x, y, z, r, g, b in all_verts:
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))

        for a, b, c in all_faces:
            f.write(struct.pack("<Biii", 3, a, b, c))

    print(f"  PLY: {OUTPUT_PLY} ({len(all_verts)} verts, {len(all_faces)} faces)")


def main():
    categories_info = []
    max_patches = 0

    # First pass: load all patches
    for cat_name in CATEGORIES:
        cat_dir = os.path.join(PATCHES_DIR, cat_name)
        if not os.path.isdir(cat_dir):
            print(f"  [warn] category dir not found: {cat_dir}")
            categories_info.append((cat_name, []))
            continue

        tifs = sorted(f for f in os.listdir(cat_dir) if f.endswith(".tif"))
        patches = []
        for tif in tifs:
            path = os.path.join(cat_dir, tif)
            name = os.path.splitext(tif)[0]
            patch = load_and_resample(path)
            patches.append((name, patch))
            print(f"  [load] {cat_name}/{name}: {patch.min()}m to {patch.max()}m")

        categories_info.append((cat_name, patches))
        max_patches = max(max_patches, len(patches))

    if max_patches == 0:
        print("No patches found!")
        return

    print(f"\n{len(CATEGORIES)} categories, max {max_patches} patches per category")

    # Write binary atlas
    n_cats = len(CATEGORIES)
    meta_size = 16  # bytes per patch metadata
    header_size = 20
    total_meta = n_cats * max_patches * meta_size
    total_pixels = n_cats * max_patches * PATCH_SIZE * PATCH_SIZE

    with open(OUTPUT_BIN, "wb") as f:
        # Header
        f.write(b"VOLC")
        f.write(struct.pack("<IIII", 1, n_cats, max_patches, PATCH_SIZE))

        # Metadata for all slots
        for cat_name, patches in categories_info:
            for i in range(max_patches):
                if i < len(patches):
                    name, patch = patches[i]
                    base = compute_base_level(patch)
                    mn = int(patch.min())
                    mx = int(patch.max())
                    mean = int(patch.mean())
                    land = int(np.sum(patch > 0) / patch.size * 100)
                    peak = int(mx - base)
                    f.write(struct.pack("<hhhBBhh4x",
                        mn, mx, mean, land, 1, base, peak))
                else:
                    # Invalid padding slot
                    f.write(b"\x00" * meta_size)

        # Pixel data for all slots
        for cat_name, patches in categories_info:
            for i in range(max_patches):
                if i < len(patches):
                    name, patch = patches[i]
                    f.write(patch.tobytes())
                else:
                    f.write(b"\x00" * (PATCH_SIZE * PATCH_SIZE * 2))

    file_size = os.path.getsize(OUTPUT_BIN)
    print(f"\n  Atlas: {OUTPUT_BIN} ({file_size} bytes, {file_size/1024/1024:.1f} MB)")

    # Export PLY visualization
    export_ply(all_patches=None, categories_info=categories_info)

    print("\nDone!")


if __name__ == "__main__":
    main()
