#!/bin/bash
#SBATCH --job-name=active_learning
#SBATCH --output=AL_%j.out
#SBATCH --error=AL_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load modules (adjust for your cluster)
module load python/3.9

# Activate virtual environment (adjust path as needed)
# source ~/venv/bin/activate

# Run with different ensemble sizes
python AL_script.py --num_ensemble 50

echo "Job completed!"