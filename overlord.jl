"""
spawns julia slaves by:
- writes sbatch script in which it calls caller.jl which contains the simulator specifications
- sbatch the batch script
- repeat for all slaves
"""