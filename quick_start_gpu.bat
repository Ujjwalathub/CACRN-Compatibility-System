@echo off
REM Quick Start Script for GPU-Accelerated Training
REM For NVIDIA RTX 3050 (6GB VRAM)

echo ========================================
echo CACRN - GPU Accelerated Training
echo ========================================
echo.

REM Check if verify_gpu.py exists
if not exist verify_gpu.py (
    echo ERROR: verify_gpu.py not found
    echo Please run this script from the Model directory
    pause
    exit /b 1
)

echo Step 1: Verifying GPU setup...
echo.
python verify_gpu.py

echo.
echo ========================================
echo.
echo If GPU verification passed, press any key to start training...
echo Otherwise, press Ctrl+C to cancel and fix GPU issues
pause >nul

echo.
echo Step 2: Starting GPU-accelerated training...
echo.
echo Configuration:
echo - Batch Size: 128 (optimized for RTX 3050)
echo - Epochs: 100
echo - Mixed Precision: ENABLED
echo - Expected time: 5-10 minutes
echo.

REM Run training with GPU-optimized settings
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128 --epochs 100

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Check models/ directory for best_model.h5
echo 2. Review logs/ directory for TensorBoard logs
echo 3. Generate predictions with:
echo    python main.py --mode predict --test-profiles data/test.xlsx
echo.
pause
