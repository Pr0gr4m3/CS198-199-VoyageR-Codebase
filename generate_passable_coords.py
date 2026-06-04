"""
Generate a txt file listing all passable node coordinates at time step 1.
Outputs center (x, y) of each passable quadtree leaf within traversable bounds.
"""

import sys
import numpy as np
from quadtree_waves import WaveDataLoader, QuadtreeDecomposer

# Configuration (matching main script defaults)
wave_threshold = 6.0
max_depth = 7
max_traversable_x = 240
max_traversable_y = 240
time_step = 1  # time step index (0-based)

source = sys.argv[1] if len(sys.argv) > 1 else '11072025-11122025.nc'
print(f"Loading wave data from {source}...")
loader = WaveDataLoader(source)
loader.load()

total_time_steps = loader.wave_data.shape[0]
if time_step >= total_time_steps:
    print(f"WARNING: time_step={time_step} out of range [0, {total_time_steps-1}]. Using 0.")
    time_step = 0

grid = loader.get_grid_at_time(time_step)
print(f"Grid shape: {grid.shape}, range: [{grid.min():.2f}, {grid.max():.2f}]")

# Build quadtree
decomposer = QuadtreeDecomposer(grid, wave_threshold, max_depth)
root = decomposer.build()
leaves = decomposer.get_all_leaves()

# Filter: passable + within traversable bounds (same logic as build_quadtree_graph)
passable_coords = []
for n in leaves:
    if not n.is_passable:
        continue
    cx = n.x + n.size / 2.0
    cy = n.y + n.size / 2.0
    if cx > max_traversable_x or cy > max_traversable_y:
        continue
    passable_coords.append((cx, cy, n.size))

# Sort by (y, x) for readability
passable_coords.sort(key=lambda c: (c[1], c[0]))

output_file = "passable_nodes_timestep1.txt"
with open(output_file, "w") as f:
    f.write(f"# Passable node coordinates at time step {time_step}\n")
    f.write(f"# wave_threshold={wave_threshold}, max_depth={max_depth}\n")
    f.write(f"# traversable bounds: x=[0, {max_traversable_x}], y=[0, {max_traversable_y}]\n")
    f.write(f"# Total passable nodes: {len(passable_coords)}\n")
    f.write(f"# Format: center_x, center_y  (node_size)\n")
    f.write(f"# X range: [{min(c[0] for c in passable_coords):.1f}, {max(c[0] for c in passable_coords):.1f}]\n")
    f.write(f"# Y range: [{min(c[1] for c in passable_coords):.1f}, {max(c[1] for c in passable_coords):.1f}]\n")
    f.write("#\n")
    for cx, cy, size in passable_coords:
        f.write(f"{cx:.1f}, {cy:.1f}  (size={size})\n")

print(f"\nWrote {len(passable_coords)} passable node centers to {output_file}")
print(f"X range: [{min(c[0] for c in passable_coords):.1f}, {max(c[0] for c in passable_coords):.1f}]")
print(f"Y range: [{min(c[1] for c in passable_coords):.1f}, {max(c[1] for c in passable_coords):.1f}]")

# --- Second file: every individual integer coordinate inside passable nodes ---
all_pixels = set()
for n in leaves:
    if not n.is_passable:
        continue
    cx = n.x + n.size / 2.0
    cy = n.y + n.size / 2.0
    if cx > max_traversable_x or cy > max_traversable_y:
        continue
    for px in range(n.x, n.x + n.size):
        for py in range(n.y, n.y + n.size):
            if px <= max_traversable_x and py <= max_traversable_y:
                all_pixels.add((px, py))

all_pixels = sorted(all_pixels, key=lambda c: (c[1], c[0]))

output_file2 = "passable_pixels_timestep1.txt"
with open(output_file2, "w") as f:
    f.write(f"# Every passable integer coordinate at time step {time_step}\n")
    f.write(f"# wave_threshold={wave_threshold}, max_depth={max_depth}\n")
    f.write(f"# traversable bounds: x=[0, {max_traversable_x}], y=[0, {max_traversable_y}]\n")
    f.write(f"# Total passable pixels: {len(all_pixels)}\n")
    f.write(f"# Format: x, y\n")
    f.write(f"# X range: [{min(c[0] for c in all_pixels)}, {max(c[0] for c in all_pixels)}]\n")
    f.write(f"# Y range: [{min(c[1] for c in all_pixels)}, {max(c[1] for c in all_pixels)}]\n")
    f.write("#\n")
    for px, py in all_pixels:
        f.write(f"{px}, {py}\n")

print(f"\nWrote {len(all_pixels)} individual passable pixel coordinates to {output_file2}")
print(f"X range: [{min(c[0] for c in all_pixels)}, {max(c[0] for c in all_pixels)}]")
print(f"Y range: [{min(c[1] for c in all_pixels)}, {max(c[1] for c in all_pixels)}]")
loader.close()
