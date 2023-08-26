#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --output=..../trcmed_submission/triton_outputs_parametric/triton_output_%a.out
#SBATCH --array=0

module load r/4.1.1-python3
srun Rscript --vanilla ./src/run/parametric/PResp/SimpleModelHier.R operation
