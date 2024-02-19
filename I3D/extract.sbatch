#!/bin/bash
#SBATCH --job-name i3d-extract
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1
#SBATCH --time 24:00:00

date
singularity exec --nv i3d-extractor-ubt22.sif python3 extract_i3d.py PATH_TO_CSV.csv PATH_TO_VIDEO.mp4 ./tmp/ ./tmp/

date
