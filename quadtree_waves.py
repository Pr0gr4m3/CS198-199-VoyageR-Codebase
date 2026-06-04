"""
Quadtree decomposition module for wave data from Waves.nc file.

This module handles:
- Loading wave data from Waves.nc (VHM0 variable)
- Treating missing/invalid data as land (unpassable)
- Building a quadtree decomposition based on wave thresholds
- Classifying nodes as passable (low waves) or unpassable (high waves or land)
- Supporting max depth to prevent over-decomposition
"""

import numpy as np
import netCDF4
import os
import glob
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class QuadtreeNode:
    """Represents a single quadtree node with position, size, and passability."""
    x: int                          # x-coordinate of top-left corner
    y: int                          # y-coordinate of top-left corner
    size: int                       # side length of square region
    depth: int = 0                  # current depth in tree
    is_passable: bool = True        # True if passable (low waves), False if unpassable
    is_uniform: bool = False        # True if region is uniform in wave classification
    children: List['QuadtreeNode'] = None  # 4 children if not leaf
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center coordinates of this node."""
        return (self.x + self.size / 2.0, self.y + self.size / 2.0)


class WaveDataLoader:
    """Loads and preprocesses wave data from Waves.nc file."""
    
    def __init__(self, filepath: str):
        """Initialize wave data loader.
        
        Args:
            filepath: Path to Waves.nc file
        """
        self.filepath = filepath
        self.ds = None
        self.wave_data = None  # Will store the full 3D array
        self.time = None
        self.grid_height = None
        self.grid_width = None
        # For real-time monitoring of a directory for new .nc files
        self._monitor_thread = None
        self._stop_monitor = False
        self._monitor_poll_interval = 2.0
        self.known_files = []
        
    def load(self) -> np.ndarray:
        """Load wave data from netCDF file.
        
        Returns:
            Preprocessed 2D wave grid (single time slice or aggregated)
        """
        # Support either a single netCDF file or a directory containing many .nc files
        if os.path.isdir(self.filepath):
            # Collect .nc files in sorted order
            pattern = os.path.join(self.filepath, "*.nc")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No .nc files found in directory: {self.filepath}")

            grids = []
            for f in files:
                dsf = netCDF4.Dataset(f)
                # try to find VHM0 variable
                if 'VHM0' not in dsf.variables:
                    dsf.close()
                    raise KeyError(f"VHM0 variable not found in file: {f}")
                var = dsf.variables['VHM0'][:]
                # Normalize to 2D grid: if var has time dim, take first time index
                if var.ndim == 3:
                    # shape: (time, lat, lon) -> take first time step
                    grid2d = var[0, :, :]
                elif var.ndim == 2:
                    grid2d = var
                else:
                    dsf.close()
                    raise ValueError(f"Unsupported VHM0 shape in file {f}: {var.shape}")

                # Handle masked arrays and convert to float32
                if np.ma.is_masked(grid2d):
                    grid2d = np.ma.filled(grid2d, fill_value=np.nan)
                grid2d = grid2d.astype(np.float32)

                grids.append(grid2d)
                dsf.close()

            # Validate that all grids have the same spatial shape
            shapes = {g.shape for g in grids}
            if len(shapes) != 1:
                raise ValueError(f"Inconsistent grid shapes among files in {self.filepath}: {shapes}")

            # Stack into (time, lat, lon)
            self.wave_data = np.stack(grids, axis=0)
            # remember which files were loaded
            self.known_files = list(files)
            self.time = np.arange(self.wave_data.shape[0])
            self.grid_height = self.wave_data.shape[1]
            self.grid_width = self.wave_data.shape[2]
            print(f"Loaded {len(files)} .nc files from {self.filepath}, stacked to shape={self.wave_data.shape}")
            return self.wave_data

        # Fallback: single file path
        self.ds = netCDF4.Dataset(self.filepath)
        
        # Extract dimensions
        if 'time' in self.ds.variables:
            self.time = self.ds.variables['time'][:]
        else:
            # create synthetic time index if missing
            self.time = np.arange(1)

        # Extract wave heights (VHM0)
        self.wave_data = self.ds.variables['VHM0'][:]  # shape: (time, lat, lon)
        
        # If wave_data is 2D (no time dim), expand to (1, lat, lon)
        if self.wave_data.ndim == 2:
            self.wave_data = np.expand_dims(self.wave_data, axis=0)

        self.grid_height = self.wave_data.shape[1]
        self.grid_width = self.wave_data.shape[2]

        print(f"Loaded wave data: shape={self.wave_data.shape}")
        print(f"Grid dimensions: {self.grid_height} x {self.grid_width}")
        print(f"Time steps: {self.wave_data.shape[0]}")
        
        return self.wave_data
    
    def _process_grid(self, grid: np.ndarray) -> np.ndarray:
        """Apply standard processing to a grid: convert type, handle masks, rotate, replace invalid.
        
        Args:
            grid: 2D numpy array to process
            
        Returns:
            Processed 2D array
        """
        # Convert masked array to regular array
        if np.ma.is_masked(grid):
            grid = np.ma.filled(grid, fill_value=np.nan)
        
        grid = grid.astype(np.float32)
        
        # Fix orientation: rotate 180 degrees and flip horizontally
        grid = np.rot90(grid, 2)
        grid = np.fliplr(grid)
        
        # Replace invalid/missing data with high values (unpassable land)
        invalid_mask = np.isnan(grid) | (grid < 0)
        grid[invalid_mask] = 10.0
        
        return grid
    
    def get_grid_at_time(self, time_idx: int = 0) -> np.ndarray:
        """Get 2D wave grid at a specific time index.
        
        Args:
            time_idx: Index into time dimension
            
        Returns:
            2D array of wave heights, missing data replaced with high values (land)
        """
        if self.wave_data is None:
            self.load()
        
        grid = self.wave_data[time_idx, :, :]
        return self._process_grid(grid)

    def get_grid_at_time_with_land_mask(self, time_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get a processed grid plus a boolean mask for land cells.

        Returns:
            Tuple of (processed grid, land mask) where land mask is True for
            missing or invalid cells before land replacement.
        """
        if self.wave_data is None:
            self.load()

        grid = self.wave_data[time_idx, :, :]

        if np.ma.is_masked(grid):
            land_mask = np.ma.getmaskarray(grid)
            grid = np.ma.filled(grid, fill_value=np.nan)
        else:
            land_mask = np.isnan(grid) | (grid < 0)

        grid = grid.astype(np.float32)
        grid = np.rot90(grid, 2)
        grid = np.fliplr(grid)
        land_mask = np.rot90(land_mask, 2)
        land_mask = np.fliplr(land_mask)

        grid[land_mask] = 10.0
        return grid, land_mask

    def _load_grid_from_file(self, fpath: str) -> np.ndarray:
        """Load a single .nc file and return a 2D grid (first time index if present)."""
        dsf = netCDF4.Dataset(fpath)
        try:
            if 'VHM0' not in dsf.variables:
                raise KeyError(f"VHM0 variable not found in file: {fpath}")
            var = dsf.variables['VHM0'][:]
            if var.ndim == 3:
                grid2d = var[0, :, :]
            elif var.ndim == 2:
                grid2d = var
            else:
                raise ValueError(f"Unsupported VHM0 shape in file {fpath}: {var.shape}")

            if np.ma.is_masked(grid2d):
                grid2d = np.ma.filled(grid2d, fill_value=np.nan)
            grid2d = grid2d.astype(np.float32)
            return grid2d
        finally:
            dsf.close()

    def start_monitor(self, poll_interval: float = 2.0):
        """Start a background thread to monitor the target directory for new .nc files.

        This will append newly-detected files to `self.wave_data` (if shapes match).
        """
        if not os.path.isdir(self.filepath):
            print("start_monitor: filepath is not a directory; monitoring skipped.")
            return
        self._monitor_poll_interval = float(poll_interval)
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_monitor = False
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print(f"Started monitoring directory for new .nc files: {self.filepath}")

    def _monitor_loop(self):
        """Background loop that polls the directory for new .nc files."""
        while not self._stop_monitor:
            try:
                pattern = os.path.join(self.filepath, "*.nc")
                files = sorted(glob.glob(pattern))
                # Detect new files
                new_files = [f for f in files if f not in self.known_files]
                for f in new_files:
                    try:
                        grid2d = self._load_grid_from_file(f)
                    except Exception as e:
                        print(f"Warning: failed to load new file {f}: {e}")
                        continue

                    # If wave_data is None, initialize
                    if self.wave_data is None:
                        self.wave_data = np.expand_dims(grid2d, axis=0)
                        self.grid_height, self.grid_width = grid2d.shape
                        self.time = np.arange(self.wave_data.shape[0])
                        self.known_files.append(f)
                        print(f"Appended first .nc file from monitor: {f}")
                        continue

                    # Validate shape
                    if grid2d.shape != (self.grid_height, self.grid_width):
                        print(f"Skipping {f}: grid shape {grid2d.shape} does not match existing shape {(self.grid_height, self.grid_width)}")
                        self.known_files.append(f)
                        continue

                    # Append along time axis
                    self.wave_data = np.concatenate([self.wave_data, np.expand_dims(grid2d, axis=0)], axis=0)
                    self.time = np.arange(self.wave_data.shape[0])
                    self.known_files.append(f)
                    print(f"Detected and appended new .nc file: {f} (total time steps: {self.wave_data.shape[0]})")
            except Exception:
                pass
            time.sleep(self._monitor_poll_interval)
    
    def get_aggregated_grid(self, aggregation_fn: str = "mean") -> np.ndarray:
        """Get aggregated 2D grid across all time steps.
        
        Args:
            aggregation_fn: "mean", "max", or "min"
            
        Returns:
            2D array aggregated across time
        """
        if self.wave_data is None:
            self.load()
        
        # Use masked array operations to handle masked data properly
        if aggregation_fn == "mean":
            grid = np.ma.mean(self.wave_data, axis=0)
        elif aggregation_fn == "max":
            grid = np.ma.max(self.wave_data, axis=0)
        elif aggregation_fn == "min":
            grid = np.ma.min(self.wave_data, axis=0)
        else:
            raise ValueError(f"Unknown aggregation function: {aggregation_fn}")
        
        return self._process_grid(grid)
    
    def close(self):
        """Close the netCDF file."""
        # Stop background monitor if running
        try:
            self._stop_monitor = True
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=1.0)
        except Exception:
            pass
        if self.ds:
            self.ds.close()


class QuadtreeDecomposer:
    """Builds a quadtree decomposition of a wave grid.
    
    Hierarchically partitions a wave field based on uniform wave height regions,
    creating a spatial index that enables efficient pathfinding.
    """
    
    def __init__(self, grid: np.ndarray, wave_threshold: float = 0.5, 
                 max_depth: int = 8, min_size: int = 1):
        """Initialize quadtree decomposer.
        
        Args:
            grid: 2D numpy array of wave heights (meters)
            wave_threshold: Wave height threshold for passability
                           (values <= threshold are passable)
            max_depth: Maximum decomposition depth to prevent over-subdivision
            min_size: Minimum size of a node (prevents infinite recursion)
        """
        self.grid = grid
        self.wave_threshold = wave_threshold
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        
    def build(self) -> QuadtreeNode:
        """Build the complete quadtree.
        
        Returns:
            Root node of the quadtree
        """
        height, width = self.grid.shape
        # Round to nearest power of 2 for clean quadtree structure
        size = self._next_power_of_2(max(height, width))
        
        # Pad grid if necessary
        padded_grid = np.zeros((size, size), dtype=np.float32)
        padded_grid[:height, :width] = self.grid
        # Match legacy 03-06/03-08 behavior.
        # With higher thresholds (for example 6.0), this can leave padded areas passable.
        padded_grid[height:, :] = 1.0
        padded_grid[:, width:] = 1.0
        
        # Store padded grid so refine_goal_region can reuse it
        self._padded_grid = padded_grid
        
        self.root = self._build_recursive(padded_grid, 0, 0, size, 0)
        return self.root
    
    


    def _build_recursive(self, grid: np.ndarray, x: int, y: int, size: int, 
                        depth: int) -> QuadtreeNode:
        """Recursively build quadtree node.
        
        Args:
            grid: Full 2D grid data
            x: x-coordinate of this region
            y: y-coordinate of this region
            size: Size of this region
            depth: Current depth
            
        Returns:
            QuadtreeNode for this region
        """
        # Extract subgrid
        subgrid = grid[y:y+size, x:x+size]
        
        # Determine passability: region is passable if most cells <= threshold
        passable_ratio = np.mean(subgrid <= self.wave_threshold)
        is_passable = passable_ratio >= 0.5
        
        # Check if region is uniform (all cells same classification)
        is_uniform = np.all(subgrid <= self.wave_threshold) or np.all(subgrid > self.wave_threshold)
        
        # Create node
        node = QuadtreeNode(
            x=x, y=y, size=size, depth=depth,
            is_passable=is_passable,
            is_uniform=is_uniform
        )
        
        # Stopping conditions: leaf node if uniform, at max depth, or minimum size reached
        if is_uniform or depth >= self.max_depth or size <= self.min_size:
            return node
        
        # Recursively subdivide into 4 quadrants
        half_size = size // 2
        node.children = [
            self._build_recursive(grid, x,           y,           half_size, depth + 1),  # TL
            self._build_recursive(grid, x + half_size, y,           half_size, depth + 1),  # TR
            self._build_recursive(grid, x,           y + half_size, half_size, depth + 1),  # BL
            self._build_recursive(grid, x + half_size, y + half_size, half_size, depth + 1),  # BR
        ]
        
        return node
    
    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Find next power of 2 >= n."""
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()
    
    def get_all_leaves(self, node: Optional[QuadtreeNode] = None) -> List[QuadtreeNode]:
        """Get all leaf nodes in the quadtree.
        
        Args:
            node: Starting node (uses root if None)
            
        Returns:
            List of all leaf nodes
        """
        if node is None:
            node = self.root
        
        if node is None:
            return []
        
        if node.is_leaf():
            return [node]
        
        leaves = []
        for child in node.children:
            leaves.extend(self.get_all_leaves(child))
        return leaves
    
    def find_leaf_containing(self, px: float, py: float, 
                            node: Optional[QuadtreeNode] = None) -> Optional[QuadtreeNode]:
        """Find leaf node containing point (px, py).
        
        Args:
            px: x-coordinate of point
            py: y-coordinate of point
            node: Starting node (uses root if None)
            
        Returns:
            Leaf node containing the point, or None if outside bounds
        """
        if node is None:
            node = self.root
        
        if node is None:
            return None
        
        # Check if point is in this node's region
        if not (node.x <= px < node.x + node.size and 
                node.y <= py < node.y + node.size):
            return None
        
        # If leaf, return it
        if node.is_leaf():
            return node
        
        # Otherwise search children
        for child in node.children:
            found = self.find_leaf_containing(px, py, child)
            if found:
                return found
        
        return None
    
    def refine_goal_region(self, goal_x: float, goal_y: float) -> None:
        """Force subdivision around the goal until max depth for precise arrival.
        
        This method finds the leaf containing the goal and forces it to subdivide
        recursively until max_depth is reached, allowing the agent to arrive
        at higher precision.
        
        Uses the same padded grid and _build_recursive logic as build() so that
        passability classification is identical.
        
        Args:
            goal_x: x-coordinate of goal
            goal_y: y-coordinate of goal
        """
        # Need the padded grid that build() used for consistent passability
        padded = getattr(self, '_padded_grid', None)
        if padded is None:
            return
        
        def refine_recursive(node: QuadtreeNode) -> None:
            """Recursively refine a node containing the goal."""
            if node.depth >= self.max_depth or node.size <= self.min_size:
                return
            
            # Check if goal is in this node's region
            if not (node.x <= goal_x < node.x + node.size and 
                    node.y <= goal_y < node.y + node.size):
                return
            
            # If this is a leaf and we haven't reached max depth, subdivide it
            # using _build_recursive so passability + is_uniform match exactly
            if node.is_leaf() and node.depth < self.max_depth:
                half_size = node.size // 2
                node.children = [
                    self._build_recursive(padded, node.x,             node.y,             half_size, node.depth + 1),
                    self._build_recursive(padded, node.x + half_size, node.y,             half_size, node.depth + 1),
                    self._build_recursive(padded, node.x,             node.y + half_size, half_size, node.depth + 1),
                    self._build_recursive(padded, node.x + half_size, node.y + half_size, half_size, node.depth + 1),
                ]
            
            # Continue refinement only in the child that contains the goal
            if not node.is_leaf():
                for child in node.children:
                    refine_recursive(child)
        
        if self.root is not None:
            refine_recursive(self.root)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the quadtree.
        
        Returns:
            Dictionary with tree statistics
        """
        leaves = self.get_all_leaves()
        passable_leaves = [n for n in leaves if n.is_passable]
        unpassable_leaves = [n for n in leaves if not n.is_passable]
        
        depths = [n.depth for n in leaves]
        sizes = [n.size for n in leaves]
        
        return {
            "total_leaves": len(leaves),
            "passable_leaves": len(passable_leaves),
            "unpassable_leaves": len(unpassable_leaves),
            "max_depth": max(depths) if depths else 0,
            "min_depth": min(depths) if depths else 0,
            "avg_depth": np.mean(depths) if depths else 0,
            "unique_sizes": sorted(set(sizes)),
            "total_area_passable": sum(n.size * n.size for n in passable_leaves),
            "total_area_unpassable": sum(n.size * n.size for n in unpassable_leaves),
        }

        

def test_wave_quadtree():
    """Test the quadtree decomposition on wave data."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Load wave data
    loader = WaveDataLoader('Waves.nc')
    grid = loader.get_aggregated_grid(aggregation_fn="mean")
    loader.close()
    
    print(f"Original grid range: [{grid.min():.3f}, {grid.max():.3f}]")
    print(f"Original grid mean: {grid.mean():.3f}, std: {grid.std():.3f}")
    
    # Use a reasonable threshold based on the actual data range
    # A threshold of 2.0 means waves <= 2.0 meters are passable
    wave_threshold = 2.0
    
    # Decompose quadtree (no normalization, work with actual wave heights)
    decomposer = QuadtreeDecomposer(
        grid=grid,
        wave_threshold=wave_threshold,  # Waves <= 2.0 m are passable
        max_depth=7
    )
    root = decomposer.build()
    
    # Print statistics
    stats = decomposer.get_statistics()
    print("\nQuadtree Statistics:")
    print(f"  Wave threshold: {wave_threshold} m")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Original wave data
    im1 = ax1.imshow(grid, cmap='viridis', origin='upper')
    ax1.set_title('Wave Height Data')
    plt.colorbar(im1, ax=ax1, label='Wave Height (m)')
    
    # Plot 2: Quadtree decomposition
    ax2.imshow(grid, cmap='viridis', origin='upper', alpha=0.3)
    
    # Draw quadtree rectangles
    leaves = decomposer.get_all_leaves()
    for leaf in leaves:
        color = 'green' if leaf.is_passable else 'red'
        alpha = 0.3 if leaf.is_passable else 0.5
        rect = patches.Rectangle(
            (leaf.x, leaf.y), leaf.size, leaf.size,
            linewidth=0.5, edgecolor='black', facecolor=color, alpha=alpha
        )
        ax2.add_patch(rect)
    
    ax2.set_title(f'Quadtree Decomposition (Green=Passable, Red=Unpassable)')
    ax2.set_xlim(0, grid.shape[1])
    ax2.set_ylim(grid.shape[0], 0)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_wave_quadtree()
