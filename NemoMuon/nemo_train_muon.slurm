#!/bin/bash
#SBATCH --job-name=nemo_muon_pretrain
#SBATCH --output=nemo_muon_pretrain_%j.out
#SBATCH --error=nemo_muon_pretrain_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=compute_bound


# Set root directory (modify this to your path)
export ROOT=/scratch/ldls-iiitd/group1_dir/NemoYogi
export ENROOT_DATA_PATH=$ROOT
export ENROOT_ROOTFS_WRITABLE=1
export PROJECT_ROOT="/scratch/ldls-iiitd/group1_dir/Nemo-optimizers"


# Set master address for distributed training
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1 | awk '{print $1}')
export MASTER_PORT=29500


# SSL certificate fix for dataset downloads
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
export SSL_CERT_DIR=/etc/pki/ca-trust/extracted/pem/


# Create necessary directories
mkdir -p $ROOT/results
mkdir -p $ROOT/logs


# Print job information
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Master address: $MASTER_ADDR"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per task: $SLURM_GPUS_PER_TASK"
echo "=========================================="
echo "Muon+AdamW Hybrid Optimizer Configuration"
echo "=========================================="


set -x


# Using NeMo container with Muon+AdamW optimizer
srun \
  --container-image="$PROJECT_ROOT/NeMo/nvidia+nemo+25.09.sqsh" \
  --container-mounts=/var/share/slurm:/var/share/slurm,/scratch:/scratch,/etc/pki/ca-trust:/etc/pki/ca-trust:ro \
  --no-container-mount-home \
  --container-remap-root \
  --export=ALL,MASTER_ADDR,MASTER_PORT,SSL_CERT_FILE,SSL_CERT_DIR \
  bash -c "pip install -q git+https://github.com/KellerJordan/Muon && python $ROOT/train.py \
    --name gpt_muon_pretrain \
    --dir $ROOT/results \
    --num_nodes 1 \
    --num_gpus_per_node 1 \
    --num_layers 12 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --seq_length 2048 \
    --max_steps 1000 \
    --global_batch_size 32 \
    --micro_batch_size 4 \
    --lr_muon 0.02 \
    --lr_adamw 3e-4 \
    --momentum 0.95 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.01"


echo "=========================================="
echo "Job finished at: $(date)"
echo "Results saved to: $ROOT/results"
