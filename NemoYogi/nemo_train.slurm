#!/bin/bash
#SBATCH --job-name=nemo_yogi_pretrain
#SBATCH --output=nemo_yogi_pretrain_%j.out
#SBATCH --error=nemo_yogi_pretrain_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=2
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

set -x

# Option 1: Using NeMo container (RECOMMENDED)
srun \
  --container-image="$PROJECT_ROOT/NeMo/nvidia+nemo+25.09.sqsh" \
  --container-mounts=/var/share/slurm:/var/share/slurm,/scratch:/scratch,/etc/pki/ca-trust:/etc/pki/ca-trust:ro \
  --no-container-mount-home \
  --container-remap-root \
  --export=ALL,MASTER_ADDR,MASTER_PORT,SSL_CERT_FILE,SSL_CERT_DIR \
  python $ROOT/nemo_pretrain_yogi2.py \
    --name gpt_yogi_pretrain \
    --dir $ROOT/results \
    --num_nodes 2 \
    --num_gpus_per_node 1 \
    --num_layers 12 \
    --hidden_size 768 \
    --num_attention_heads 12 \
    --seq_length 2048 \
    --max_steps 1000 \
    --global_batch_size 32 \
    --micro_batch_size 4 \
    --lr 1e-3 \
    --beta1 0.9 \
    --beta2 0.999 \
    --weight_decay 0.01

echo "Job finished at: $(date)"
