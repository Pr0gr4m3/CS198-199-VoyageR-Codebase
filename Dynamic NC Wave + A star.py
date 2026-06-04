"""
Integrated pathfinding system using quadtree decomposition on real wave data.

Features:
- Loads wave data from single .nc file or directory of .nc files
- Automatically monitors directory for new files (real-time streaming)
- Treats missing data (masked areas) as land (unpassable)
- Builds a quadtree decomposition based on wave height thresholds
- Uses predictive time-aware A* or greedy local-step pathfinding
- Visualizes agent movement on wave field with quadtree overlay
- Records complete trajectory and generates final path visualization
"""

import heapq
import sys
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import itertools
from matplotlib.colors import ListedColormap
from quadtree_waves import WaveDataLoader, QuadtreeDecomposer


# ==================== Agent Class ====================
class Agent:
    """Agent that navigates through quadtree-decomposed space."""
    
    def __init__(self, x, y, speed=0.5, bounds=None):
        self.x = x
        self.y = y
        self.speed = speed  # units per frame
        self.path = []
        self.path_index = 0
        self.id_map = {}  # Maps node ID to QuadtreeNode objects
        self.position_history = [(x, y)]  # Track all positions visited
        self.previous_node_id = None  # Track previous node to avoid backtracking
        self.recent_node_ids = deque(maxlen=12)  # Rolling window of recent nodes for anti-oscillation
        self.bounds = bounds  # Optional bounds (x_min, x_max, y_min, y_max)

    def update(self, new_path, id_map):
        """Update agent with new path and store id_map."""
        # Only reset progress when the path actually changes
        if new_path is None:
            return
        if self.path != new_path:
            self.path = new_path
            self.path_index = 0
        self.id_map = id_map

    def path_complete(self):
        """Return True if the agent has finished walking its current path."""
        if not self.path:
            return True
        return self.path_index >= len(self.path)

    def path_valid(self, id_map=None):
        """Return True if remaining path nodes all exist in the given id_map."""
        check_map = id_map if id_map is not None else self.id_map
        if not check_map or not self.path:
            return False
        for nid in self.path[self.path_index:]:
            if nid not in check_map:
                return False
        return True

    def step(self):
        """Advance the agent along its current path by one movement step."""
        self.move_along_path()

    def move_along_path(self):
        """Move agent along current path by speed amount."""
        if not self.path or self.path_index >= len(self.path):
            return

        current_id = self.path[self.path_index]
        if current_id not in self.id_map:
            return

        current_node = self.id_map[current_id]
        current_center = np.array([current_node.x + current_node.size/2.0,
                                   current_node.y + current_node.size/2.0])

        # Determine target
        if self.path_index + 1 < len(self.path):
            next_id = self.path[self.path_index + 1]
            if next_id not in self.id_map:
                return
            next_node = self.id_map[next_id]
            target = np.array([next_node.x + next_node.size/2.0,
                             next_node.y + next_node.size/2.0])
        else:
            target = current_center

        # Move towards target
        agent_pos = np.array([self.x, self.y])
        direction = target - agent_pos
        distance = np.linalg.norm(direction)

        if distance > self.speed:
            direction_norm = direction / distance
            self.x += direction_norm[0] * self.speed
            self.y += direction_norm[1] * self.speed
        else:
            self.x, self.y = target[0], target[1]
            # Record the node we just left to avoid immediate backtracking
            try:
                self.previous_node_id = current_id
                self.recent_node_ids.append(current_id)
            except Exception:
                pass
            self.path_index += 1
        
        # Clamp position within bounds if bounds are set
        if self.bounds is not None:
            x_min, x_max, y_min, y_max = self.bounds
            self.x = max(x_min, min(x_max, self.x))
            self.y = max(y_min, min(y_max, self.y))
        
        # Record position in history
        self.position_history.append((self.x, self.y))


# ==================== Graph Building ====================
def quadtree_rects_touch(a, b):
    """Check if two quadtree nodes touch orthogonally (4-neighbors)."""
    ax0, ax1 = a.x, a.x + a.size
    ay0, ay1 = a.y, a.y + a.size
    bx0, bx1 = b.x, b.x + b.size
    by0, by1 = b.y, b.y + b.size

    # Horizontal adjacency
    overlap_y = min(ay1, by1) - max(ay0, by0)
    if overlap_y > 0 and (ax1 == bx0 or bx1 == ax0):
        return True
    
    # Vertical adjacency
    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    if overlap_x > 0 and (ay1 == by0 or by1 == ay0):
        return True
    return False


def build_quadtree_graph(leaves, grid=None, max_x=None, max_y=None):
    """Build adjacency graph from quadtree leaves.
    
    Args:
        leaves: List of leaf QuadtreeNode objects
        grid: Optional current wave grid for storing wave information
        max_x: Maximum x coordinate for node centers (nodes beyond this are excluded)
        max_y: Maximum y coordinate for node centers (nodes beyond this are excluded)
        
    Returns:
        Tuple of (adjacency dict, id_map)
    """
    # Only passable nodes are in the graph
    passable_nodes = [n for n in leaves if n.is_passable]
    
    # Filter nodes by center position bounds if specified
    if max_x is not None or max_y is not None:
        filtered_nodes = []
        for n in passable_nodes:
            cx = n.x + n.size / 2.0
            cy = n.y + n.size / 2.0
            if max_x is not None and cx > max_x:
                continue
            if max_y is not None and cy > max_y:
                continue
            filtered_nodes.append(n)
        passable_nodes = filtered_nodes
    
    # Create ID mapping: (x, y, size) -> QuadtreeNode
    id_map = {(n.x, n.y, n.size): n for n in passable_nodes}
    
    # Store current wave heights in nodes for use in pathfinding
    if grid is not None:
        for node_id, node in id_map.items():
            x, y, size = node_id
            cx, cy = int(x + size/2.0), int(y + size/2.0)
            if 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0]:
                node._current_waves = grid[cy, cx]
            else:
                node._current_waves = 0.0
    
    # Build adjacency list
    adj = {key: [] for key in id_map.keys()}
    keys = list(id_map.keys())
    
    for i, ka in enumerate(keys):
        node_a = id_map[ka]
        for kb in keys[i+1:]:
            node_b = id_map[kb]
            if quadtree_rects_touch(node_a, node_b):
                # Weight is Euclidean distance between centers
                ca = (node_a.x + node_a.size/2.0, node_a.y + node_a.size/2.0)
                cb = (node_b.x + node_b.size/2.0, node_b.y + node_b.size/2.0)
                w = np.hypot(ca[0]-cb[0], ca[1]-cb[1])
                adj[ka].append((kb, w))
                adj[kb].append((ka, w))
    
    return adj, id_map


def _compute_distance_scale(id_map):
    """Compute a stable distance scale from quadtree node extents."""
    if not id_map:
        return 1.0

    nodes = list(id_map.values())
    min_x = min(n.x for n in nodes)
    min_y = min(n.y for n in nodes)
    max_x = max(n.x + n.size for n in nodes)
    max_y = max(n.y + n.size for n in nodes)
    diag = float(np.hypot(max_x - min_x, max_y - min_y))
    return max(diag, 1.0)


def _compute_wave_scale(id_map=None, wave_threshold=None, grid=None):
    """Compute a stable wave scale from threshold and observed wave heights."""
    candidates = []

    if wave_threshold is not None and np.isfinite(wave_threshold):
        candidates.append(float(wave_threshold))

    if id_map:
        for node in id_map.values():
            if hasattr(node, 'average_wave_height') and np.isfinite(node.average_wave_height):
                candidates.append(float(node.average_wave_height))
            elif hasattr(node, '_current_waves') and np.isfinite(node._current_waves):
                candidates.append(float(node._current_waves))

    if grid is not None and grid.size > 0:
        try:
            gmax = float(np.nanmax(grid))
            if np.isfinite(gmax):
                candidates.append(gmax)
        except Exception:
            pass

    if not candidates:
        return 1.0
    return max(max(candidates), 1.0)


def _normalize_cost(value, scale):
    """Normalize a scalar cost into [0, 1] with robust guards."""
    if scale is None or not np.isfinite(scale) or scale <= 0.0:
        return 0.0
    if value is None or not np.isfinite(value):
        return 0.0
    return float(np.clip(value / scale, 0.0, 1.0))


def heuristic(node_a_id, node_b_id, id_map=None, alpha=0.3, distance_scale=None, wave_scale=None):
    """Heuristic: Weighted combination of distance and wave avoidance.
    
    Args:
        node_a_id: Starting node ID (x, y, size)
        node_b_id: Goal node ID (x, y, size)
        id_map: Dictionary mapping node IDs to QuadtreeNode objects (optional)
        alpha: Weight for wave avoidance (0.0 = distance only, 1.0 = wave heavy)
        
    Returns:
        Heuristic cost value
    """
    ax, ay, asz = node_a_id
    bx, by, bsz = node_b_id
    ac = (ax + asz/2.0, ay + asz/2.0)
    bc = (bx + bsz/2.0, by + bsz/2.0)
    
    distance = np.hypot(ac[0]-bc[0], ac[1]-bc[1])
    
    # Wave avoidance component
    wave_cost = 0.0
    if id_map and node_a_id in id_map:
        node_a = id_map[node_a_id]
        if hasattr(node_a, 'average_wave_height'):
            wave_cost = node_a.average_wave_height

    # Keep 03-08 fixed normalization so costs stay comparable with legacy runs.
    distance_norm = distance / 300.0
    wave_norm = wave_cost / 10.0
    
    # Additive weighted combination - alpha truly controls the tradeoff
    return (1.0 - alpha) * distance_norm + alpha * wave_norm


def astar_quadtree(start_id, goal_id, graph, id_map, alpha=0.5, current_time=0, loader=None, wave_threshold=6.0, max_depth=7, future_steps=5, distance_scale=None, wave_scale=None):
    """Predictive time-aware A* pathfinding on quadtree graph.
    
    This version looks ahead in time to predict future wave conditions and incorporates
    wave heights into the heuristic to favor routes through calmer areas.
    
    Args:
        start_id: Starting node ID (x, y, size)
        goal_id: Goal node ID (x, y, size)
        graph: Adjacency dict from build_quadtree_graph
        id_map: Dictionary mapping node IDs to QuadtreeNode objects
        alpha: Weight for wave avoidance in heuristic (0.0 = distance only, 1.0 = wave heavy)
        current_time: Current time index in the simulation
        loader: WaveDataLoader instance for predictive future grids
        wave_threshold: Wave height threshold for passability
        max_depth: Max depth for quadtree decomposition
        future_steps: Number of time steps to look ahead for predictions
        
    Returns:
        List of node IDs forming the path, or None if no path exists
    """
    from quadtree_waves import QuadtreeDecomposer
    
    # Compute predicted future grids (look ahead by future_steps)
    future_grids = []
    if loader is not None:
        total_time_steps = loader.wave_data.shape[0]
        for offset in range(1, future_steps + 1):
            future_t = (current_time + offset) % total_time_steps
            future_grids.append(loader.get_grid_at_time(future_t))
    
    
    openq = []
    start_h = heuristic(start_id, goal_id, id_map, alpha, distance_scale, wave_scale)
    heapq.heappush(openq, (start_h, 0.0, start_id))
    
    came_from = {}
    g_cost = {start_id: 0.0}
    closed = set()

    while openq:
        _, gcur, current = heapq.heappop(openq)
        
        if current == goal_id:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        if current in closed:
            continue
        closed.add(current)

        for neighbor, weight in graph.get(current, []):
            # Apply wave-based cost as additive component
            # Alpha controls the distance vs wave tradeoff
            distance_cost = weight / 300.0
            wave_cost = 0.0
            
            if neighbor in id_map:
                neighbor_node = id_map[neighbor]
                if hasattr(neighbor_node, 'average_wave_height'):
                    wave_cost = neighbor_node.average_wave_height / 10.0
            
            # Additive weighted combination - alpha truly controls the tradeoff
            edge_cost = (1.0 - alpha) * distance_cost + alpha * wave_cost
            tentative = gcur + edge_cost

            if neighbor in g_cost and tentative >= g_cost[neighbor]:
                continue
            
            came_from[neighbor] = current
            g_cost[neighbor] = tentative
            h = heuristic(neighbor, goal_id, id_map, alpha, distance_scale, wave_scale)
            f = tentative + h
            heapq.heappush(openq, (f, tentative, neighbor))

    return None


def find_nearest_passable_node(agent_x, agent_y, id_map, goal_x=None, goal_y=None, alpha=0.3, reachable_set=None, distance_scale=None, wave_scale=None):
    """Find the best passable node considering both distance and wave height.
    
    Uses the same weighted distance + wave cost evaluation as the A* heuristic
    so that the selected node favors calmer waters, not just proximity.
    
    Args:
        agent_x: Agent's current X position
        agent_y: Agent's current Y position
        id_map: Dictionary mapping node IDs to QuadtreeNode objects
        goal_x: Optional goal X position (if provided, considers distance to goal)
        goal_y: Optional goal Y position (if provided, considers distance to goal)
        alpha: Weight for wave avoidance (0.0 = distance only, 1.0 = wave heavy)
        reachable_set: Optional set of node IDs reachable from the agent.
                       If provided, only nodes in this set are considered.
        
    Returns:
        Best node ID or None if no passable nodes exist
    """
    if not id_map:
        return None

    # Determine which nodes to evaluate
    candidate_ids = reachable_set if reachable_set is not None else id_map.keys()
    
    best_node_id = None
    best_score = float('inf')
    
    for node_id in candidate_ids:
        if node_id not in id_map:
            continue
        node = id_map[node_id]
        cx = node.x + node.size / 2.0
        cy = node.y + node.size / 2.0
        dist_from_agent = np.hypot(agent_x - cx, agent_y - cy)
        
        # Wave cost for this candidate node
        wave_cost = 0.0
        if hasattr(node, 'average_wave_height'):
            wave_cost = node.average_wave_height
        elif hasattr(node, '_current_waves'):
            wave_cost = node._current_waves
        
        if goal_x is not None and goal_y is not None:
            dist_to_goal = np.hypot(goal_x - cx, goal_y - cy)
            total_dist = dist_from_agent + dist_to_goal
        else:
            total_dist = dist_from_agent
        
        # Weighted combination matching the 03-08 heuristic normalization
        dist_norm = total_dist / 300.0
        wave_norm = wave_cost / 10.0
        score = (1.0 - alpha) * dist_norm + alpha * wave_norm
        
        if score < best_score:
            best_score = score
            best_node_id = node_id
    
    return best_node_id


def compute_path(start_id, goal_id, graph, id_map, alpha, wave_threshold, 
                 local_step_planning, current_time, loader, max_depth, future_steps, 
                 prev_node_id=None, recent_node_ids=None):
    """Compute the next path using either local-step greedy or full A*.
    
    Args:
        start_id: Current node ID
        goal_id: Goal node ID
        graph: Adjacency graph
        id_map: Node mapping
        alpha: Wave avoidance weight
        wave_threshold: Wave height threshold
        local_step_planning: Use greedy planning if True
        current_time: Current time index
        loader: Wave data loader
        max_depth: Max quadtree depth
        future_steps: Look-ahead horizon
        prev_node_id: Previous node to avoid (for backtracking avoidance)
        recent_node_ids: Deque of recently visited nodes (for anti-oscillation)
        
    Returns:
        Path (list of node IDs) or None if no path found
    """
    # Pre-calculate average wave height for all nodes (needed for both planning modes)
    future_grids = []
    if loader is not None:
        total_time_steps = loader.wave_data.shape[0]
        for offset in range(1, future_steps + 1):
            future_t = (current_time + offset) % total_time_steps
            future_grids.append(loader.get_grid_at_time(future_t))
    
    # Compute average wave height for each node
    for node_id, node in id_map.items():
        wave_heights = []
        x, y, size = node_id
        
        # Include current waves if available
        if hasattr(node, '_current_waves'):
            wave_heights.append(node._current_waves)
        
        # Include future predictions
        for future_grid in future_grids:
            # Sample wave height at node center
            cx, cy = int(x + size/2.0), int(y + size/2.0)
            if 0 <= cx < future_grid.shape[1] and 0 <= cy < future_grid.shape[0]:
                wave_heights.append(future_grid[cy, cx])
        
        # Use max to be conservative (highest wave height encountered)
        if wave_heights:
            node.average_wave_height = max(wave_heights)
        else:
            node.average_wave_height = 0.0

    if local_step_planning:
        # Radius-bounded A* — same algorithm as global A* but only explores
        # nodes within LOOKAHEAD_RADIUS of the agent (direct neighbors always
        # included). Finds the optimal A* path to the best reachable node
        # within the bounded area.fstar
        LOOKAHEAD_RADIUS = 60.0  # Euclidean distance from agent to explore
        REVISIT_PENALTY = 1000.0    # Extra cost for recently visited nodes
        
        # Build a set of recently visited nodes for O(1) lookup
        visited_set = set(recent_node_ids) if recent_node_ids else set()
        
        # Already at the goal
        if start_id == goal_id:
            return [start_id]
        
        # Goal is a direct neighbor — go straight there
        for neighbor, weight in graph.get(start_id, []):
            if neighbor == goal_id:
                return [start_id, goal_id]
        
        sx, sy, ssz = start_id
        sc = (sx + ssz/2.0, sy + ssz/2.0)
        
        def _center(node_id):
            nx, ny, nsz = node_id
            return (nx + nsz/2.0, ny + nsz/2.0)
        
        def _in_radius(node_id):
            """Check if node is within lookahead radius of agent."""
            nc = _center(node_id)
            return np.hypot(nc[0]-sc[0], nc[1]-sc[1]) <= LOOKAHEAD_RADIUS
        
        # Build radius-bounded subgraph: only nodes within radius + direct neighbors
        direct_neighbors = {n for n, _ in graph.get(start_id, [])}
        bounded_nodes = set()
        bounded_nodes.add(start_id)
        for nid in graph:
            if nid in direct_neighbors or _in_radius(nid):
                bounded_nodes.add(nid)
        
        # Run standard A* on this bounded subgraph
        # (same logic as astar_quadtree but restricted to bounded_nodes)
        openq = []
        start_h = heuristic(start_id, goal_id, id_map, alpha)
        heapq.heappush(openq, (start_h, 0.0, start_id))
        
        came_from = {}
        g_cost_local = {start_id: 0.0}
        closed = set()
        
        # Track the best frontier node (closest to goal by heuristic)
        # in case A* can't reach the goal within the radius
        best_frontier_id = None
        best_frontier_h = float('inf')
        
        while openq:
            f_val, gcur, current = heapq.heappop(openq)
            
            if current == goal_id:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            if current in closed:
                continue
            closed.add(current)
            
            # Track best frontier node by DISTANCE to goal only
            # (wave avoidance is handled by A* edge costs; frontier selection
            # must prioritize progress toward the goal)
            # Exclude recently visited nodes from frontier selection to prevent oscillation
            if current != start_id:
                nc = _center(current)
                gc = _center(goal_id)
                dist_to_goal = np.hypot(nc[0]-gc[0], nc[1]-gc[1])
                # Prefer unvisited nodes; track visited ones as secondary fallback
                if current != prev_node_id and current not in visited_set:
                    if dist_to_goal < best_frontier_h:
                        best_frontier_h = dist_to_goal
                        best_frontier_id = current
                elif best_frontier_id is None:
                    # All candidates so far are visited — keep the closest as fallback
                    if dist_to_goal < best_frontier_h:
                        best_frontier_h = dist_to_goal
                        best_frontier_id = current
            
            for neighbor, weight in graph.get(current, []):
                # Only expand into bounded nodes
                if neighbor not in bounded_nodes:
                    continue
                if neighbor in closed:
                    continue
                
                # Same edge cost as global A*
                distance_cost = weight / 300.0
                wave_cost = 0.0
                if neighbor in id_map:
                    neighbor_node = id_map[neighbor]
                    if hasattr(neighbor_node, 'average_wave_height'):
                        wave_cost = neighbor_node.average_wave_height / 10.0
                edge_cost = (1.0 - alpha) * distance_cost + alpha * wave_cost
                
                # Penalize revisiting recently visited nodes to break oscillation
                if neighbor in visited_set and neighbor != goal_id:
                    edge_cost += REVISIT_PENALTY
                
                tentative = gcur + edge_cost
                
                if neighbor in g_cost_local and tentative >= g_cost_local[neighbor]:
                    continue
                
                came_from[neighbor] = current
                g_cost_local[neighbor] = tentative
                h = heuristic(neighbor, goal_id, id_map, alpha)
                heapq.heappush(openq, (tentative + h, tentative, neighbor))
        
        # Goal not reachable within radius — return path to best frontier node
        if best_frontier_id is not None:
            path = [best_frontier_id]
            current = best_frontier_id
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        # Fallback: try any direct neighbor (even prev_node)
        neighbors = graph.get(start_id, [])
        if neighbors:
            best_n = min(neighbors, key=lambda nw: heuristic(nw[0], goal_id, id_map, alpha))
            return [start_id, best_n[0]]
        
        return None
    else:
        # Full predictive A*
        return astar_quadtree(start_id, goal_id, graph, id_map,
                            alpha=alpha, current_time=current_time, loader=loader,
                            wave_threshold=wave_threshold, max_depth=max_depth,
                            future_steps=future_steps)


# ==================== Visualization ====================
def draw_quadtree(ax, node):
    """Draw quadtree nodes as colored rectangles."""
    if node is None:
        return
    
    color = '#2ecc71' if node.is_passable else '#c0392b'
    alpha = 0.14 if node.is_passable else 0.22
    
    rect = patches.Rectangle(
        (node.x, node.y), node.size, node.size,
        linewidth=0.5,
        edgecolor='black',
        facecolor=color,
        alpha=alpha
    )
    ax.add_patch(rect)
    
    for child in node.children:
        draw_quadtree(ax, child)


def _make_wave_display_grid(grid, land_mask=None):
    """Return a masked array where land_mask True hides land for cmap masking."""
    if land_mask is None:
        land_mask = np.zeros_like(grid, dtype=bool)
    return np.ma.array(grid, mask=land_mask)


def _get_wave_cmap():
    """Return wave colormap; masked (land) color will be set by overlay instead."""
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad(color='white')
    return cmap


def _make_land_overlay(grid, land_mask):
    """Create a land-only overlay mask (1.0 on land, masked elsewhere)."""
    return np.ma.masked_where(~land_mask, np.ones_like(grid, dtype=float))


def _get_land_cmap():
    """Return a solid dark-blue colormap for land overlay."""
    cmap = ListedColormap(['#0b3d91'])
    cmap.set_bad(color=(0, 0, 0, 0))
    return cmap


def _get_right_cmap():
    """Return a distinct cmap for the right-hand visualization (avoid turbo)."""
    # Use same base cmap as left pane for a similar palette
    return _get_wave_cmap()


def draw_path(ax, path_ids, id_map):
    """Draw path as polyline connecting node centers."""
    if not path_ids:
        return
    
    centers_x = []
    centers_y = []
    for nid in path_ids:
        if nid not in id_map:
            continue
        n = id_map[nid]
        cx = n.x + n.size/2.0
        cy = n.y + n.size/2.0
        centers_x.append(cx)
        centers_y.append(cy)
    
    if centers_x:
        ax.plot(centers_x, centers_y, color='red', linewidth=2, zorder=4, marker='o', markersize=3)


def save_agent_path_png(grid, agent, start_pos, goal_pos, land_mask=None, output_filename='agent_final_path.png'):
    """Generate and save a visualization of the agent's complete path.

    Args:
        grid: The wave height grid
        agent: The Agent object containing position_history
        start_pos: Tuple (x, y) of start position
        goal_pos: Tuple (x, y) of goal position
        land_mask: Optional boolean mask for land cells (display-only)
        output_filename: Name of the output PNG file
    """
    fig, ax = plt.subplots(figsize=(22, 16))

    # Show wave field as background (mask land so it can be overlaid)
    display_grid = _make_wave_display_grid(grid, land_mask)
    im = ax.imshow(display_grid, cmap=_get_wave_cmap(), origin='upper', alpha=0.95)
    # Overlay dark-blue land so land is visually distinct in saved image
    if land_mask is not None:
        ax.imshow(_make_land_overlay(grid, land_mask), cmap=_get_land_cmap(),
                  origin='upper', alpha=1.0, vmin=0.0, vmax=1.0, zorder=3)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wave Height (m)', fontsize=36)
    cbar.ax.tick_params(labelsize=24)
    
    # Plot agent's path history
    if len(agent.position_history) > 1:
        positions = agent.position_history
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Draw path as a solid black line for final PNG
        for i in range(len(positions) - 1):
            ax.plot(x_coords[i:i+2], y_coords[i:i+2],
                color='black', linewidth=3.5, alpha=0.98, zorder=3)
        
        # Mark start point
        ax.plot(x_coords[0], y_coords[0], 'go', markersize=30, 
               label='Start', zorder=5, markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Mark end point
        ax.plot(x_coords[-1], y_coords[-1], 'r*', markersize=30, 
               label=f'Agent Final Position', zorder=5, markeredgecolor='darkred', markeredgewidth=1)
    
    # Mark goal position
    ax.plot(goal_pos[0], goal_pos[1], 'y*', markersize=30, 
           label='Goal Target', zorder=5, markeredgecolor='orange', markeredgewidth=1)
    
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(grid.shape[0], 0)
    ax.set_xlabel('X Position (grid units)', fontsize=36)
    ax.set_ylabel('Y Position (grid units)', fontsize=36)
    ax.set_title('Agent Final Path - Complete Trajectory', 
                fontsize=35, fontweight='bold')
    ax.legend(fontsize=20, loc='best')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)
    
    # Calculate total distance traveled
    total_distance = 0.0
    if len(agent.position_history) > 1:
        for i in range(len(agent.position_history) - 1):
            pos1 = agent.position_history[i]
            pos2 = agent.position_history[i + 1]
            distance = np.hypot(pos2[0] - pos1[0], pos2[1] - pos1[1])
            total_distance += distance
    
    # Save to file
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Agent final path saved to: {output_filename}")
    print(f"  Total distance traveled: {total_distance:.2f} grid units")
    print(f"  Number of position samples: {len(agent.position_history)}")
    plt.close(fig)


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Start timer
    start_time_actual = time.time()
    
    print("=" * 60)
    print("QUADTREE PATHFINDING WITH REAL WAVE DATA")
    print("=" * 60)
    
    # Configuration
    wave_threshold = 6.0  # Waves <= 3.0 m are passable (more lenient threshold)
    max_depth = 7 # How deep the quadtree can go (lower = more coarse, higher = more fine)
    start_time = 0  # Which time step to start at (0-based index)
    # Heuristic weight: 0.0 = distance only, 1.0 = wave avoidance only
    # Use values in [0.0, 1.0]. 
    alpha = 0.1  # sensible default for balanced behavior
    future_steps = 3  # Number of time steps to look ahead for predictions
    # If True, select the next move by evaluating neighbors only (local step),
    # instead of running full A* from start -> goal. Much faster but greedy.
    local_step_planning = False
    min_distance_per_timestep = 20.0  # Minimum distance before advancing time step
    
    # Load wave data (time series) -- allow passing file or folder as CLI arg
    source = sys.argv[1] if len(sys.argv) > 1 else '11072025-11122025.nc'
    print(f"\n1. Loading wave data (time series) from {source}...")
    loader = WaveDataLoader(source)
    # ensure wave_data/time are loaded
    loader.load()
    # Start background monitor to detect new .nc files in the source dir (if source is a dir)
    try:
        loader.start_monitor(poll_interval=2.0)
    except Exception:
        pass

    total_time_steps = loader.wave_data.shape[0]

    # Validate start_time
    if start_time < 0 or start_time >= total_time_steps:
        print(f"   WARNING: start_time={start_time} out of range [0, {total_time_steps-1}]. Clamping to 0.")
        start_time = max(0, min(start_time, total_time_steps - 1))

    print(f"   Time steps available: {total_time_steps}. Animating in real-time (append-enabled) starting from t={start_time}.")

    # Initial grid (at start_time)
    grid, land_mask = loader.get_grid_at_time_with_land_mask(start_time)
    print(f"   Initial grid: shape={grid.shape}, range=[{grid.min():.2f}, {grid.max():.2f}]")

    # Setup agent and static goal coordinates (pixel indices)
    grid_h, grid_w = grid.shape
    start_pos = (190,10)
    goal_pos = (220,220)
    
    # Define traversable bounds (quadtree nodes beyond these are excluded)
    max_traversable_x = 240
    max_traversable_y = 240
    
    # Validate and clamp positions to be within traversable bounds
    start_pos = (max(0, min(max_traversable_x, start_pos[0])), 
                 max(0, min(max_traversable_y, start_pos[1])))
    goal_pos = (max(0, min(max_traversable_x, goal_pos[0])), 
                max(0, min(max_traversable_y, goal_pos[1])))
    
    print(f"   Start position: {start_pos}, Goal position: {goal_pos}")
    print(f"   Grid bounds: x=[0, {grid_w}], y=[0, {grid_h}]")
    print(f"   Traversable bounds: x=[0, {max_traversable_x}], y=[0, {max_traversable_y}]")
    
    agent = Agent(start_pos[0], start_pos[1], speed=9, bounds=(0, max_traversable_x, 0, max_traversable_y))
    reached_goal = False
    total_path_cost = 0.0  # Accumulated weighted path cost
    temp_goal_active = False   # True when routing to a temporary "rest stop" goal
    temp_goal_node_id = None   # Node ID of the current temporary goal

    # Prepare matplotlib figure and axes
    print("\n2. Preparing animation...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    im1 = ax1.imshow(_make_wave_display_grid(grid, land_mask), cmap=_get_wave_cmap(), origin='upper')
    ax1.set_title(f'Wave Heights (time-varying)\nThreshold: {wave_threshold} m', fontsize=20, fontweight='bold')
    ax1.axvline(start_pos[0], color='red', linewidth=1.5)
    ax1.axhline(start_pos[1], color='red', linewidth=1.5)
    ax1.axvline(goal_pos[0], color='green', linewidth=1.5)
    ax1.axhline(goal_pos[1], color='green', linewidth=1.5)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Wave Height (m)')
    cbar1.set_label('Wave Height (m)', fontsize=12)
    cbar1.ax.tick_params(labelsize=24)
    ax1.tick_params(axis='both', which='major', labelsize=24)

    # Right axis will be redrawn each frame (quadtree + path + agent)
    ax2.set_title('Quadtree Decomposition + A* Path', fontsize=20, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # Time-step control (advance only after enough movement)
    current_time_index = start_time
    distance_since_time_step = 0.0

    # Animation update function
    def update_frame(frame):
        global current_time_index, distance_since_time_step
        global temp_goal_active, temp_goal_node_id
        global total_path_cost

        # Determine current available total time steps (supports real-time additions)
        total_time_steps = loader.wave_data.shape[0]
        if total_time_steps == 0:
            return [im1]

        # Use current time index (advance only after enough movement)
        if current_time_index < total_time_steps:
            t = current_time_index
        else:
            t = total_time_steps - 1

        grid_t, land_mask_t = loader.get_grid_at_time_with_land_mask(t)

        # Build quadtree for this time step
        decomposer_t = QuadtreeDecomposer(grid_t, wave_threshold, max_depth)
        root_t = decomposer_t.build()
        
        # Refine quadtree around goal for precise arrival
        decomposer_t.refine_goal_region(goal_pos[0], goal_pos[1])
        
        leaves_t = decomposer_t.get_all_leaves()
        adj_t, id_map_t = build_quadtree_graph(leaves_t, grid_t, max_x=max_traversable_x, max_y=max_traversable_y)

        # Find leaf containing agent (current position) and goal
        start_leaf_t = decomposer_t.find_leaf_containing(agent.x, agent.y)
        goal_leaf_t = decomposer_t.find_leaf_containing(goal_pos[0], goal_pos[1])
        waiting_at_temp = False  # True when agent is idling at a temp rest-stop

        if goal_leaf_t is not None:
            goal_id_t = (goal_leaf_t.x, goal_leaf_t.y, goal_leaf_t.size)
            goal_is_passable = goal_id_t in adj_t

            # --- Determine effective goal (actual vs temporary rest stop) ---
            if goal_is_passable:
                # Actual goal is reachable - clear any temp goal and route normally
                if temp_goal_active:
                    print(f"  [t={t}] Goal is passable again! Resuming normal routing.")
                temp_goal_active = False
                temp_goal_node_id = None
                effective_goal_id = goal_id_t
            else:
                # Actual goal is unpassable - manage temporary rest-stop goal
                if (temp_goal_active and temp_goal_node_id is not None
                        and temp_goal_node_id in adj_t):
                    # Current temp goal still passable
                    if start_leaf_t is not None:
                        sid = (start_leaf_t.x, start_leaf_t.y, start_leaf_t.size)
                        if sid == temp_goal_node_id:
                            # Already at the rest stop - idle here
                            waiting_at_temp = True
                            effective_goal_id = None
                        else:
                            # Still en route to temp goal
                            effective_goal_id = temp_goal_node_id
                    else:
                        effective_goal_id = temp_goal_node_id
                else:
                    # Need a new temp goal (first time, or old one became unpassable)
                    if temp_goal_active:
                        print(f"  [t={t}] Temp goal became unpassable. Seeking new rest stop...")
                    # BFS from agent to find all nodes reachable via the graph
                    reachable = set()
                    if start_leaf_t is not None:
                        agent_id = (start_leaf_t.x, start_leaf_t.y, start_leaf_t.size)
                        if agent_id in adj_t:
                            bfs_queue = deque([agent_id])
                            reachable.add(agent_id)
                            while bfs_queue:
                                cur = bfs_queue.popleft()
                                for nbr, _ in adj_t[cur]:
                                    if nbr not in reachable:
                                        reachable.add(nbr)
                                        bfs_queue.append(nbr)
                    # Use alpha=0 so proximity to goal dominates; only consider reachable nodes.
                    # If agent is also unpassable (reachable is empty), skip the
                    # reachability filter so we can still find a temp goal to beeline to.
                    new_temp = find_nearest_passable_node(
                        goal_pos[0], goal_pos[1], id_map_t, alpha=0.0,
                        reachable_set=reachable if reachable else None)
                    if new_temp:
                        temp_goal_active = True
                        temp_goal_node_id = new_temp
                        effective_goal_id = new_temp
                        tn = id_map_t[new_temp]
                        print(f"  [t={t}] Goal unpassable. Routing to rest stop at "
                              f"({tn.x + tn.size/2:.0f}, {tn.y + tn.size/2:.0f})")
                    else:
                        # No passable nodes near goal at all - wait in place
                        waiting_at_temp = True
                        effective_goal_id = None

            # --- Route to the effective goal ---
            if effective_goal_id is not None:
                if start_leaf_t is not None:
                    start_id_t = (start_leaf_t.x, start_leaf_t.y, start_leaf_t.size)

                    if start_id_t in adj_t and effective_goal_id in adj_t:
                        # Normal A* / local-step routing
                        path_t = compute_path(start_id_t, effective_goal_id, adj_t,
                                              id_map_t, alpha, wave_threshold,
                                              local_step_planning, t, loader,
                                              max_depth, future_steps,
                                              agent.previous_node_id,
                                              agent.recent_node_ids)
                        if path_t:
                            agent.update(path_t, id_map_t)

                    elif start_id_t not in adj_t:
                        # Agent stuck in unpassable node.
                        if temp_goal_active and temp_goal_node_id is not None and temp_goal_node_id in id_map_t:
                            # Both agent and goal are unpassable — beeline to the
                            # already-established temp goal (nearest passable to goal)
                            beeline_target = temp_goal_node_id
                        else:
                            # Only agent is unpassable (goal is fine) — find
                            # nearest passable node from the agent toward the goal
                            beeline_target = find_nearest_passable_node(
                                agent.x, agent.y, id_map_t,
                                goal_x=goal_pos[0], goal_y=goal_pos[1], alpha=0.0)
                        if beeline_target and beeline_target in id_map_t:
                            temp_goal_active = True
                            temp_goal_node_id = beeline_target
                            effective_goal_id = beeline_target
                            tn = id_map_t[beeline_target]
                            print(f"  [t={t}] Agent in unpassable node. Beelining to "
                                  f"temp goal at "
                                  f"({tn.x + tn.size/2:.0f}, {tn.y + tn.size/2:.0f})")
                            # Agent can't pathfind from an unpassable area;
                            # walk directly toward the temp goal
                            agent.path = [beeline_target]
                            agent.path_index = 0
                            agent.id_map = id_map_t
                        else:
                            # No passable nodes found at all - wait in place
                            waiting_at_temp = True

                else:
                    # EMERGENCY FALLBACK: no valid leaf for agent position
                    # Same approach as agent-in-unpassable: find nearest passable
                    # node from agent closest to goal, set as temporary goal
                    nearest = find_nearest_passable_node(
                        agent.x, agent.y, id_map_t,
                        goal_x=goal_pos[0], goal_y=goal_pos[1], alpha=0.0)
                    if nearest:
                        temp_goal_active = True
                        temp_goal_node_id = nearest
                        tn = id_map_t[nearest]
                        print(f"  [t={t}] Agent position invalid. Routing to nearest "
                              f"passable node at "
                              f"({tn.x + tn.size/2:.0f}, {tn.y + tn.size/2:.0f})")
                        agent.path = [nearest]
                        agent.path_index = 0
                        agent.id_map = id_map_t
                    else:
                        waiting_at_temp = True

        # Move agent one step
        prev_x, prev_y = agent.x, agent.y
        agent.step()
        moved_distance = np.hypot(agent.x - prev_x, agent.y - prev_y)
        distance_since_time_step += moved_distance

        # Accumulate weighted path cost (same formula as A* edge cost)
        wave_at_agent = 0.0
        ay, ax = int(round(agent.y)), int(round(agent.x))
        if 0 <= ay < grid_t.shape[0] and 0 <= ax < grid_t.shape[1]:
            wave_at_agent = grid_t[ay, ax]
        # Alpha-independent cost: pure quality metric (distance + wave exposure).
        # Routing decisions still use alpha; only reporting is equalized.
        cost_increment = (moved_distance / 300.0) + (wave_at_agent / 10.0)
        total_path_cost += cost_increment

        if distance_since_time_step >= min_distance_per_timestep:
            if current_time_index < total_time_steps - 1:
                current_time_index += 1
            distance_since_time_step -= min_distance_per_timestep

        # If agent is waiting at a temp rest stop, still advance time so conditions change
        if waiting_at_temp and current_time_index < total_time_steps - 1:
            current_time_index += 1

        # Check if the agent has arrived into the same quadtree node as the goal
        # If so, stop the animation.
        global reached_goal
        agent_leaf_after = decomposer_t.find_leaf_containing(agent.x, agent.y)
        if agent_leaf_after is not None and goal_leaf_t is not None and not reached_goal:
            agent_id_after = (agent_leaf_after.x, agent_leaf_after.y, agent_leaf_after.size)
            goal_id_now = (goal_leaf_t.x, goal_leaf_t.y, goal_leaf_t.size)
            if agent_id_after == goal_id_now and goal_id_now in adj_t:
                reached_goal = True
                print(f"Agent reached goal node at time step {t}. Stopping animation.")
                try:
                    ani.event_source.stop()
                except Exception:
                    pass
                # Close animation window and save final visualization
                plt.close(fig)
                
                # Generate final path visualization immediately
                final_grid_at_goal, final_land_mask = loader.get_grid_at_time_with_land_mask(t)
                save_agent_path_png(final_grid_at_goal, agent, start_pos, goal_pos, final_land_mask)
                
                # Calculate and print runtime
                end_time_actual = time.time()
                runtime = end_time_actual - start_time_actual
                print(f"\n" + "=" * 60)
                print(f"TOTAL PATH COST: {total_path_cost:.4f}")
                print(f"PROGRAM RUNTIME: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
                print("=" * 60)
                
                # Close loader and exit
                loader.close()
                sys.exit(0)

        # Update left image
        im1.set_data(_make_wave_display_grid(grid_t, land_mask_t))

        # Redraw right axis contents (waves + dark-blue land overlay)
        ax2.clear()
        ax2.imshow(np.ma.array(grid_t, mask=land_mask_t), cmap=_get_right_cmap(), origin='upper', alpha=0.95)
        ax2.imshow(
            _make_land_overlay(grid_t, land_mask_t),
            cmap=_get_land_cmap(),
            origin='upper',
            alpha=1.0,
            vmin=0.0,
            vmax=1.0,
            zorder=3,
        )
        draw_quadtree(ax2, root_t)
        # draw current planned path (if any)
        draw_path(ax2, agent.path, agent.id_map)
        ax2.plot(agent.x, agent.y, 'ro', markersize=6, label='Agent', zorder=5)
        ax2.plot(goal_pos[0], goal_pos[1], 'g*', markersize=12, label='Goal', zorder=5)
        # Show temporary rest-stop goal if active
        if temp_goal_active and temp_goal_node_id is not None and temp_goal_node_id in id_map_t:
            tn = id_map_t[temp_goal_node_id]
            ax2.plot(tn.x + tn.size/2.0, tn.y + tn.size/2.0, 'y^', markersize=10,
                     label='Rest Stop', zorder=5, markeredgecolor='orange', markeredgewidth=1)
        ax2.set_xlim(0, grid_w)
        ax2.set_ylim(grid_h, 0)
        ax2.set_title(f'Time step: {t+1}/{total_time_steps} | Frame: {frame+1} | Loaded files: {len(getattr(loader, "known_files", []))}', fontsize=24)
        ax2.legend(fontsize=24)

        return [im1]

    ani = animation.FuncAnimation(fig, update_frame, frames=itertools.count(), interval=100, blit=False)

    plt.tight_layout()
    print("   Starting animation (close the plot window to finish)...")
    plt.show()

    # Only reach here if animation completes without reaching goal
    print("\n" + "=" * 60)
    print("PATHFINDING ANIMATION COMPLETE")
    print("=" * 60)
    
    loader.close()
    
    # Calculate and print runtime
    end_time = time.time()
    runtime = end_time - start_time_actual
    print(f"\n" + "=" * 60)
    print(f"TOTAL PATH COST: {total_path_cost:.4f}")
    print(f"PROGRAM RUNTIME: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    print("=" * 60)
