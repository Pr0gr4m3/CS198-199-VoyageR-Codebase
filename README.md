# Quadtree Pathfinding with Real Wave Data

A quadtree-based navigation system performing predictive, wave-aware A* planning on NetCDF VHM0 data. Includes code to reproduce experiments and export agent trajectories.

## Summary
- Main simulation + visualization: main.py
- Quadtree utilities & NetCDF loader: quadtree_waves.py
- Batch runner: run_test_cases.py
- Test generators: generate_test_cases.py, generate_passable_coords.py
- Example data: 11072025-11122025.nc (or files in NC/)
- Reports: test_cases.txt, test_results.txt
- Output: agent_final_path.png (sample)

## Requirements
- Python 3.7+
- numpy
- matplotlib
- netCDF4

Install:
```bash
pip install numpy matplotlib netcdf4