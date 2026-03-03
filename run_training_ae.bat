@echo off
setlocal enabledelayedexpansion

REM ==============================
REM Autoencoder Hyperparameter Grid
REM ==============================

REM ---- Dataset ----
set DATA_PATH=diseases.csv
set INDICES_FILE=split_indices_full_80_10_10.npz

REM ---- Hyperparameter Grid ----
set LRS=0.0005 0.0001
set LATENTS=32 64 128
set BATCHES=32 64
set EPOCHS=100
set PATIENCE=15
set MIN_DELTA=0.0005

REM ---- Seeds ----
set SEEDS=0

for %%L in (%LRS%) do (
    for %%Z in (%LATENTS%) do (
        for %%B in (%BATCHES%) do (
            for %%S in (%SEEDS%) do (
                python train_auto_encoder_full.py ^
                  --data_path %DATA_PATH% ^
                  --indices_file %INDICES_FILE% ^
                  --lr %%L ^
                  --latent_dim %%Z ^
                  --batch_size %%B ^
                  --epochs %EPOCHS% ^
                  --patience %PATIENCE% ^
                  --min_delta %MIN_DELTA% ^
                  --seed %%S
            )
        )
    )
)
