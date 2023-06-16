#!/bin/bash
#SBATCH --job-name=img_cap
#SBATCH --account=sail
#SBATCH --partition=sail
#SBATCH --gres=gpu:1
#SBATCH --output=imgcap.log
pwd
hostname
date
module load slurm
module load python3/3.8
module load cuda
module load gcc
module load cmake
source /data/quantizeVL/quantenv/bin/activate
export PATH=$PATH:/data/quantizeVL/quantenv/bin
export PATH=$PATH:/nfs/stak/users/kannegaa/.local/bin
date
python3 run.py \
    --beam_size 1 \
    --target_seq_length 16 \
    --reset_context_delta \
    --lm_model cambridgeltl/magic_mscoco \
    --test_image_prefix_path ../data/mscoco/test_images \
    --test_path ../data/mscoco/mscoco_test.json \
    --save_path_prefix ../inference_result/mscoco/baselines/ \
    --save_name zerocap_result.json
