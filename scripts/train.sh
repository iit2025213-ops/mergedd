#!/bin/bash

# ==============================================================================
# SUPREME TRAINING ORCHESTRATOR (A100 80GB OPTIMIZED)
# Aim: WRMSSE < 0.5 | Resilience: High | Hardware: Ampere
# ==============================================================================

# 1. Environment Optimization for Ampere/A100
# Ensures the A100's Tensor Cores are fully utilized for FP16/BF16.
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=0       # Enable Peer-to-Peer for fast VRAM access
export NCCL_IB_DISABLE=1        # Disable InfiniBand if running on a single node to prevent overhead
export TF_CPP_MIN_LOG_LEVEL=3   # Suppress non-critical hardware warnings
export PYTHONUNBUFFERED=1       # Real-time logging for HFT-grade monitoring

# 2. Project Infrastructure
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/decagon_v1"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "----------------------------------------------------------------"
echo "LAUNCHING SUPREME M5 PIPELINE"
echo "Target: sub-0.5 WRMSSE"
echo "Log Directory: $LOG_DIR"
echo "----------------------------------------------------------------"

# 3. Execution Phase 1: Pre-processing & Graph Construction
# Only run if the artifacts don't exist to save time on the A100 duty cycle.
if [ ! -f "data/processed/m5_meta.pt" ]; then
    echo "[PHASE 0] Starting Information Alignment (Preprocessing)..."
    python scripts/preprocess.py --config configs/data_config.yaml
    
    echo "[PHASE 0] Constructing Decagon Topology (10 Graph Views)..."
    python scripts/generate_graphs.py --config configs/data_config.yaml
fi

# 4. Execution Phase 2: Decagon Ensemble Training
# We use 'nohup' to ensure the 50-epoch run isn't killed by a terminal timeout.
echo "[PHASE 1] Training 10-GNN Engine (VAT + EMA Enabled)..."
nohup python main.py \
    --config configs/supreme_config.yaml \
    --mode train \
    --log_dir "$LOG_DIR" \
    > "$LOG_DIR/gnn_train.log" 2>&1 &

GNN_PID=$!
echo "GNN Engine running in background (PID: $GNN_PID). Monitoring gradients..."

# 5. Execution Phase 3: Boosting Expert Calibration
# In research, we wait for the GNN embeddings to stabilize before fitting the trees.
# Here we wait for the GNN process to finish.
wait $GNN_PID

echo "[PHASE 2] Training Hybrid Correctors (LGBM & XGB Experts)..."
python main.py \
    --config configs/boosting_config.yaml \
    --mode boost_train \
    >> "$LOG_DIR/boosting.log" 2>&1

echo "----------------------------------------------------------------"
echo "PIPELINE COMPLETE. System idle. Ready for Predict/Audit."
echo "----------------------------------------------------------------"