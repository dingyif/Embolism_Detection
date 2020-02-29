#! /bin/bash
#SBATCH -c 1
#SBATCH --mem 200GB
#SBATCH --time 35:00
#SBATCH -A stats
#SBATCH --array 10,16,23,25,27
#SBATCH --output=s-%A_%a.out

module load anaconda
source activate <path-to-your-env>
python -u AP_v3.py --folder_idx $SLURM_ARRAY_TASK_ID --has_proc 1 --chunk_size 4000 --chunk_idx 0 --version 9.83
