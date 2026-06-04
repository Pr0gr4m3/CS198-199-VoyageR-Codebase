import random
import math

def generate_test_cases(passable_file, output_file, num_cases=30, min_distance=180, seed=42):
    random.seed(seed)
    
    # Load all passable pixels
    passable_pixels = []
    with open(passable_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) == 2:
                x, y = int(parts[0].strip()), int(parts[1].strip())
                passable_pixels.append((x, y))
    
    print(f"Loaded {len(passable_pixels)} passable pixels")
    
    # Generate random pairs with minimum distance
    test_cases = []
    attempts = 0
    max_attempts = 1000000
    
    while len(test_cases) < num_cases and attempts < max_attempts:
        start = random.choice(passable_pixels)
        goal = random.choice(passable_pixels)
        
        if start == goal:
            attempts += 1
            continue
        
        dist = math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
        
        if dist >= min_distance:
            pair = (start, goal)
            reverse_pair = (goal, start)
            if pair not in test_cases and reverse_pair not in test_cases:
                test_cases.append(pair)
                print(f"  Found case {len(test_cases):2d}: {start} -> {goal}  dist = {dist:.1f}")
        
        attempts += 1
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(f"# {num_cases} Random Test Cases (min distance {min_distance})\n")
        f.write(f"# Uniform random sampling from {passable_file}\n")
        f.write(f"# Random seed: {seed}\n")
        f.write(f"# Total passable pixels: {len(passable_pixels)}\n")
        f.write(f"# Format: start_pos -> goal_pos (distance)\n")
        f.write(f"#\n")
        for i, (start, goal) in enumerate(test_cases, 1):
            dist = math.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
            f.write(f"{i:2d}. start_pos = ({start[0]:3d}, {start[1]:3d})  "
                    f"goal_pos = ({goal[0]:3d}, {goal[1]:3d})  "
                    f"dist = {dist:.1f}\n")
    
    print(f"\nWrote {len(test_cases)} test cases to {output_file}")

if __name__ == "__main__":
    generate_test_cases(
        passable_file="passable_pixels_timestep1.txt",
        output_file="test_cases.txt",
        num_cases=100,
        min_distance=180,
        seed=42
    )
