"""
Batch runner for quadtree pathfinding test cases.

Reads start/goal pairs from test_cases.txt, executes each case without visualization,
and writes a report matching the existing results text layout.
"""

import argparse
import importlib.util
import math
import os
import re
import statistics
import sys
import time
from datetime import datetime

import numpy as np

from quadtree_waves import QuadtreeDecomposer, WaveDataLoader


def load_main_module(script_path):
    """Load the main pathfinding script as a Python module."""
    spec = importlib.util.spec_from_file_location("dynamic_nc_wave_astar", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_test_cases(path):
    """Parse test cases from test_cases.txt."""
    pattern = re.compile(
        r"start_pos\s*=\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*"
        r"goal_pos\s*=\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
        re.IGNORECASE,
    )

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            sx, sy, gx, gy = map(int, match.groups())
            cases.append(((sx, sy), (gx, gy)))
    return cases


def clamp_pos(pos, max_x, max_y):
    x, y = pos
    return (max(0, min(max_x, x)), max(0, min(max_y, y)))


def distance_traveled(history):
    if len(history) < 2:
        return 0.0
    dist = 0.0
    for i in range(len(history) - 1):
        x1, y1 = history[i]
        x2, y2 = history[i + 1]
        dist += float(np.hypot(x2 - x1, y2 - y1))
    return dist


def run_single_case(mod, loader, cfg, start_pos, goal_pos):
    """Run one start->goal simulation using the same logic as the animation script."""
    start_case = time.time()

    agent = mod.Agent(
        start_pos[0],
        start_pos[1],
        speed=cfg["agent_speed"],
        bounds=(0, cfg["max_traversable_x"], 0, cfg["max_traversable_y"]),
    )

    current_time_index = cfg["start_time"]
    distance_since_time_step = 0.0
    total_path_cost = 0.0

    temp_goal_active = False
    temp_goal_node_id = None

    reached_goal = False
    collisions = 0
    frames_used = 0

    # Cache expensive quadtree/graph builds per time-step, matching old runner behavior.
    cached_time = -1
    decomposer_t = None
    adj_t = None
    id_map_t = None
    grid_t = None

    for frame in range(cfg["max_frames"]):
        frames_used = frame + 1

        total_time_steps = loader.wave_data.shape[0]
        if total_time_steps <= 0:
            break

        t = current_time_index if current_time_index < total_time_steps else total_time_steps - 1

        if t != cached_time:
            cached_time = t
            grid_t = loader.get_grid_at_time(t)

            decomposer_t = QuadtreeDecomposer(grid_t, cfg["wave_threshold"], cfg["max_depth"])
            decomposer_t.build()
            decomposer_t.refine_goal_region(goal_pos[0], goal_pos[1])

            leaves_t = decomposer_t.get_all_leaves()
            adj_t, id_map_t = mod.build_quadtree_graph(
                leaves_t,
                grid_t,
                max_x=cfg["max_traversable_x"],
                max_y=cfg["max_traversable_y"],
            )

        start_leaf_t = decomposer_t.find_leaf_containing(agent.x, agent.y)
        goal_leaf_t = decomposer_t.find_leaf_containing(goal_pos[0], goal_pos[1])
        waiting_at_temp = False

        if goal_leaf_t is not None:
            goal_id_t = (goal_leaf_t.x, goal_leaf_t.y, goal_leaf_t.size)
            goal_is_passable = goal_id_t in adj_t

            if goal_is_passable:
                temp_goal_active = False
                temp_goal_node_id = None
                effective_goal_id = goal_id_t
            else:
                if (
                    temp_goal_active
                    and temp_goal_node_id is not None
                    and temp_goal_node_id in adj_t
                ):
                    if start_leaf_t is not None:
                        sid = (start_leaf_t.x, start_leaf_t.y, start_leaf_t.size)
                        if sid == temp_goal_node_id:
                            waiting_at_temp = True
                            effective_goal_id = None
                        else:
                            effective_goal_id = temp_goal_node_id
                    else:
                        effective_goal_id = temp_goal_node_id
                else:
                    reachable = set()
                    if start_leaf_t is not None:
                        agent_id = (start_leaf_t.x, start_leaf_t.y, start_leaf_t.size)
                        if agent_id in adj_t:
                            queue = [agent_id]
                            reachable.add(agent_id)
                            qi = 0
                            while qi < len(queue):
                                cur = queue[qi]
                                qi += 1
                                for nbr, _ in adj_t[cur]:
                                    if nbr not in reachable:
                                        reachable.add(nbr)
                                        queue.append(nbr)

                    new_temp = mod.find_nearest_passable_node(
                        goal_pos[0],
                        goal_pos[1],
                        id_map_t,
                        alpha=0.0,
                        reachable_set=reachable if reachable else None,
                    )

                    if new_temp:
                        temp_goal_active = True
                        temp_goal_node_id = new_temp
                        effective_goal_id = new_temp
                    else:
                        waiting_at_temp = True
                        effective_goal_id = None

            if effective_goal_id is not None:
                if start_leaf_t is not None:
                    start_id_t = (start_leaf_t.x, start_leaf_t.y, start_leaf_t.size)

                    if start_id_t in adj_t and effective_goal_id in adj_t:
                        path_t = mod.compute_path(
                            start_id_t,
                            effective_goal_id,
                            adj_t,
                            id_map_t,
                            cfg["alpha"],
                            cfg["wave_threshold"],
                            cfg["local_step_planning"],
                            t,
                            loader,
                            cfg["max_depth"],
                            cfg["future_steps"],
                            agent.previous_node_id,
                            agent.recent_node_ids,
                        )
                        if path_t:
                            agent.update(path_t, id_map_t)
                    elif start_id_t not in adj_t:
                        collisions += 1
                        if (
                            temp_goal_active
                            and temp_goal_node_id is not None
                            and temp_goal_node_id in id_map_t
                        ):
                            beeline_target = temp_goal_node_id
                        else:
                            beeline_target = mod.find_nearest_passable_node(
                                agent.x,
                                agent.y,
                                id_map_t,
                                goal_x=goal_pos[0],
                                goal_y=goal_pos[1],
                                alpha=0.0,
                            )

                        if beeline_target and beeline_target in id_map_t:
                            temp_goal_active = True
                            temp_goal_node_id = beeline_target
                            agent.path = [beeline_target]
                            agent.path_index = 0
                            agent.id_map = id_map_t
                        else:
                            waiting_at_temp = True
                else:
                    collisions += 1
                    nearest = mod.find_nearest_passable_node(
                        agent.x,
                        agent.y,
                        id_map_t,
                        goal_x=goal_pos[0],
                        goal_y=goal_pos[1],
                        alpha=0.0,
                    )
                    if nearest:
                        temp_goal_active = True
                        temp_goal_node_id = nearest
                        agent.path = [nearest]
                        agent.path_index = 0
                        agent.id_map = id_map_t
                    else:
                        waiting_at_temp = True

        prev_x, prev_y = agent.x, agent.y
        agent.step()

        moved_distance = float(np.hypot(agent.x - prev_x, agent.y - prev_y))
        distance_since_time_step += moved_distance

        wave_at_agent = 0.0
        ay, ax = int(round(agent.y)), int(round(agent.x))
        if 0 <= ay < grid_t.shape[0] and 0 <= ax < grid_t.shape[1]:
            wave_at_agent = float(grid_t[ay, ax])

        # Alpha-independent cost: pure quality metric (distance + wave exposure).
        # Routing decisions still use alpha; only reporting is equalized.
        cost_increment = (moved_distance / 300.0) + (wave_at_agent / 10.0)
        total_path_cost += cost_increment

        if distance_since_time_step >= cfg["min_distance_per_timestep"]:
            if current_time_index < total_time_steps - 1:
                current_time_index += 1
            distance_since_time_step -= cfg["min_distance_per_timestep"]

        if waiting_at_temp and current_time_index < total_time_steps - 1:
            current_time_index += 1

        agent_leaf_after = decomposer_t.find_leaf_containing(agent.x, agent.y)
        if agent_leaf_after is not None and goal_leaf_t is not None:
            agent_id_after = (
                agent_leaf_after.x,
                agent_leaf_after.y,
                agent_leaf_after.size,
            )
            goal_id_now = (goal_leaf_t.x, goal_leaf_t.y, goal_leaf_t.size)
            if agent_id_after == goal_id_now and goal_id_now in adj_t:
                reached_goal = True
                break

    runtime = time.time() - start_case
    dist = distance_traveled(agent.position_history)

    status = "REACHED" if reached_goal else "TIMEOUT"

    return {
        "status": status,
        "path_cost": float(total_path_cost),
        "distance": dist,
        "runtime": runtime,
        "samples": len(agent.position_history),
        "frames": frames_used,
        "collisions": collisions,
        "reached": reached_goal,
    }


def fmt_pos(pos):
    return f"({pos[0]:3d},{pos[1]:3d})"


def make_report(cases, results, cfg, total_batch_time, total_time_steps):
    lines = []
    sep = "=" * 105
    dash = "-" * 105

    reached_results = [r for r in results if r["reached"]]
    reached_count = len(reached_results)
    timeout_count = len(results) - reached_count
    total_collisions = sum(r["collisions"] for r in results)
    collision_cases = sum(1 for r in results if r["collisions"] > 0)

    lines.append(sep)
    lines.append("BATCH TEST RESULTS - Quadtree Pathfinding")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"  wave_threshold: {cfg['wave_threshold']}")
    lines.append(f"  max_depth: {cfg['max_depth']}")
    lines.append(f"  alpha: {cfg['alpha']}")
    lines.append(f"  future_steps: {cfg['future_steps']}")
    lines.append(f"  local_step_planning: {cfg['local_step_planning']}")
    lines.append(f"  min_distance_per_timestep: {cfg['min_distance_per_timestep']}")
    lines.append(f"  max_traversable_x: {cfg['max_traversable_x']}")
    lines.append(f"  max_traversable_y: {cfg['max_traversable_y']}")
    lines.append(f"  max_frames: {cfg['max_frames']}")
    lines.append(f"  wave_data_source: {cfg['wave_data_source']}")
    lines.append(f"  total_time_steps: {total_time_steps}")
    lines.append(f"  total_test_cases: {len(cases)}")
    lines.append("")
    lines.append(dash)
    lines.append("  #         Start          Goal    Status   Path Cost    Distance     Runtime   Samples   Frames   Collisions")
    lines.append(dash)

    for idx, (case, r) in enumerate(zip(cases, results), start=1):
        start_pos, goal_pos = case
        lines.append(
            f"{idx:3d}  {fmt_pos(start_pos)}  {fmt_pos(goal_pos)}   "
            f"{r['status']:<7s}  "
            f"{r['path_cost']:10.4f}  "
            f"{r['distance']:10.1f}  "
            f"{r['runtime']:9.2f}s  "
            f"{r['samples']:8d}  "
            f"{r['frames']:7d}  "
            f"{r['collisions']:11d}"
        )

    lines.append(dash)
    lines.append("")
    lines.append(sep)
    lines.append("SUMMARY")
    lines.append(sep)
    lines.append(f"  Tests run:          {len(results)}")
    lines.append(
        f"  Goals reached:      {reached_count}/{len(results)} ({(100.0 * reached_count / len(results)) if results else 0.0:.1f}%)"
    )
    lines.append(f"  Timeouts:           {timeout_count}")
    lines.append(f"  Total batch time:   {total_batch_time:.2f}s ({total_batch_time/60.0:.2f} min)")
    lines.append(f"  Total collisions:   {total_collisions} across {collision_cases} test case(s)")
    lines.append("")

    lines.append("  Statistics (reached cases only):")
    lines.append("  Metric                        Min        Max       Mean     Median")
    lines.append("  --------------------------------------------------------------")

    if reached_results:
        costs = [r["path_cost"] for r in reached_results]
        dists = [r["distance"] for r in reached_results]
        runtimes = [r["runtime"] for r in reached_results]
        samples = [float(r["samples"]) for r in reached_results]

        lines.append(
            f"  Path Cost              {min(costs):10.2f}{max(costs):10.2f}{statistics.mean(costs):11.2f}{statistics.median(costs):11.2f}"
        )
        lines.append(
            f"  Distance Traveled      {min(dists):10.2f}{max(dists):10.2f}{statistics.mean(dists):11.2f}{statistics.median(dists):11.2f}"
        )
        lines.append(
            f"  Runtime (s)            {min(runtimes):10.2f}{max(runtimes):10.2f}{statistics.mean(runtimes):11.2f}{statistics.median(runtimes):11.2f}"
        )
        lines.append(
            f"  Position Samples       {min(samples):10.2f}{max(samples):10.2f}{statistics.mean(samples):11.2f}{statistics.median(samples):11.2f}"
        )
    else:
        lines.append("  Path Cost                      n/a       n/a        n/a        n/a")
        lines.append("  Distance Traveled              n/a       n/a        n/a        n/a")
        lines.append("  Runtime (s)                    n/a       n/a        n/a        n/a")
        lines.append("  Position Samples               n/a       n/a        n/a        n/a")

    lines.append("")
    lines.append("  Path Efficiency (distance_traveled / straight_line):")

    efficiencies = []
    for idx, ((start_pos, goal_pos), r) in enumerate(zip(cases, results), start=1):
        straight = math.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
        if straight > 0 and r["distance"] > 0:
            eff = r["distance"] / straight
            efficiencies.append(eff)
            lines.append(f"    Test {idx:2d}: {eff:.3f}x")
        else:
            lines.append(f"    Test {idx:2d}: n/a")

    if efficiencies:
        lines.append(f"    Average efficiency: {statistics.mean(efficiencies):.3f}x  (1.0 = perfect straight line)")
    else:
        lines.append("    Average efficiency: n/a  (1.0 = perfect straight line)")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Run test_cases.txt batch simulation and write formatted report.")
    parser.add_argument("--source", default="11072025-11122025.nc", help="Wave data source (.nc file or folder).")
    parser.add_argument("--test-cases", default="test_cases.txt", help="Path to test cases file.")
    parser.add_argument("--output", default="test_results.txt", help="Output report file.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 = all).")

    parser.add_argument("--wave-threshold", type=float, default=6.0)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--future-steps", type=int, default=3)
    parser.add_argument(
        "--local-step-planning",
        dest="local_step_planning",
        action="store_true",
        default=False,
        help="Use local-step planning (default).",
    )
    parser.add_argument(
        "--global-planning",
        dest="local_step_planning",
        action="store_false",
        help="Use full global predictive A* (slower).",
    )
    parser.add_argument("--min-distance-per-timestep", type=float, default=20.0)
    parser.add_argument("--max-traversable-x", type=int, default=240)
    parser.add_argument("--max-traversable-y", type=int, default=240)
    parser.add_argument("--max-frames", type=int, default=5000)
    parser.add_argument("--agent-speed", type=float, default=9.0)
    parser.add_argument("--start-time", type=int, default=0)

    args = parser.parse_args()

    cfg = {
        "wave_data_source": args.source,
        "wave_threshold": args.wave_threshold,
        "max_depth": args.max_depth,
        "alpha": args.alpha,
        "future_steps": args.future_steps,
        "local_step_planning": args.local_step_planning,
        "min_distance_per_timestep": args.min_distance_per_timestep,
        "max_traversable_x": args.max_traversable_x,
        "max_traversable_y": args.max_traversable_y,
        "max_frames": args.max_frames,
        "agent_speed": args.agent_speed,
        "start_time": args.start_time,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(script_dir, "Dynamic NC Wave + A star.py")
    mod = load_main_module(main_script_path)

    cases = parse_test_cases(args.test_cases)
    if not cases:
        raise RuntimeError(f"No test cases found in: {args.test_cases}")

    if args.max_cases and args.max_cases > 0:
        cases = cases[: args.max_cases]

    loader = WaveDataLoader(args.source)
    loader.load()

    total_time_steps = int(loader.wave_data.shape[0])
    if total_time_steps <= 0:
        loader.close()
        raise RuntimeError("No wave time steps loaded.")

    cfg["start_time"] = max(0, min(cfg["start_time"], total_time_steps - 1))

    clamped_cases = []
    for start_pos, goal_pos in cases:
        clamped_start = clamp_pos(start_pos, cfg["max_traversable_x"], cfg["max_traversable_y"])
        clamped_goal = clamp_pos(goal_pos, cfg["max_traversable_x"], cfg["max_traversable_y"])
        clamped_cases.append((clamped_start, clamped_goal))

    print(f"Running {len(clamped_cases)} test case(s)...")
    batch_start = time.time()

    results = []
    try:
        for idx, (start_pos, goal_pos) in enumerate(clamped_cases, start=1):
            result = run_single_case(mod, loader, cfg, start_pos, goal_pos)
            results.append(result)

            straight = math.hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
            collision_text = (
                "no collisions"
                if result["collisions"] == 0
                else f"collisions={result['collisions']}"
            )
            print(
                f"Test {idx:3d}/{len(clamped_cases)}: ({start_pos[0]:3d}, {start_pos[1]:3d}) -> "
                f"({goal_pos[0]:3d},{goal_pos[1]:3d}) [straight-line: {straight:.1f}] ... "
                f"{result['status']} | cost={result['path_cost']:.4f} | dist={result['distance']:.1f} | "
                f"time={result['runtime']:.2f}s | samples={result['samples']} | {collision_text}"
            )
    finally:
        loader.close()

    total_batch_time = time.time() - batch_start
    report = make_report(clamped_cases, results, cfg, total_batch_time, total_time_steps)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
