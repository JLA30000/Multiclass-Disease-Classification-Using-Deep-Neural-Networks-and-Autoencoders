@echo off
setlocal enabledelayedexpansion

REM ---- Edit these if you want ----
set LR=0.001
set BS=32
set EPOCHS=50
set PATIENCE=10
set MIN_EPOCHS=10
set MIN_DELTA=0.001

REM Block size for aggregated confusion heatmap (e.g., 10, 25, 50)
set GROUP_SIZE=25

REM Normalization for the heatmap: true | pred | all | none
set NORMALIZE=true

REM Seeds 0..9
set SEEDS=0,1,2,3,4,5,6,7,8,9

echo Running 10-seed training (seeds: %SEEDS%)
python train.py --lr %LR% --batch_size %BS% --epochs %EPOCHS% --patience %PATIENCE% --min_epochs %MIN_EPOCHS% --min_delta %MIN_DELTA% --seeds %SEEDS% --group_size %GROUP_SIZE% --normalize %NORMALIZE%

echo.
echo Done.
pause
