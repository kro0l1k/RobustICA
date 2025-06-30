# RICA Installation Guide


## Quick Installation

### For CPU-only systems:
```bash
pip install -e .
```

### For NVIDIA GPU systems (CUDA):
```bash
pip install -e ".[cuda]"
```

### For Apple Silicon (M1/M2/M3) systems:
```bash
pip install -e ".[metal]"
```

## Manual Installation

### 1. Basic Dependencies
```bash
pip install numpy>=1.26.4 scikit-learn>=1.3.2 signax==0.2.1 statsmodels
```

### 2. Choose JAX Backend

#### For NVIDIA GPU (CUDA 12):
```bash
pip install "jax[cuda12_pip]>=0.4.25" "jaxlib[cuda12_pip]>=0.4.25"
```

#### For Apple Silicon (Metal):
```bash
pip install jax-metal>=0.0.6 jax==0.4.24 jaxlib==0.4.23
```

#### For CPU only:
```bash
pip install jax>=0.4.25 jaxlib>=0.4.25
```

## Hardware Detection

The updated code now automatically detects available hardware and uses the best option:

1. **Metal** - Apple Silicon GPUs (M1/M2/M3/M4)
2. **CUDA** - NVIDIA GPUs
3. **TPU** - Google TPUs (if available)
4. **CPU** - Fallback option

## Verification

After installation, run:
```bash
python -c "import jax; print('Available devices:', jax.devices())"
```

You should see your GPU/accelerator listed if properly configured.

## Troubleshooting

### CUDA Issues
- Ensure NVIDIA drivers are installed
- Verify CUDA 12 compatibility
- Try: `nvidia-smi` to check GPU status

### Metal Issues  
- Only works on Apple Silicon Macs (M1/M2/M3/M4)
- Ensure macOS is up to date
- Implement a fallback to cpu.
