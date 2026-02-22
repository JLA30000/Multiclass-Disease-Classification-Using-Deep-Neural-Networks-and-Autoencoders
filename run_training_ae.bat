@echo off
setlocal enabledelayedexpansion

REM ---- Hyperparams for the classifier-on-bottleneck ----
set LR=0.0001
set BS=64
set EPOCHS=50
set PATIENCE=10
set MIN_EPOCHS=10
set MIN_DELTA=0.001

REM ---- Confusion matrix options ----
set GROUP_SIZE=25
set NORMALIZE=true

REM ---- Seeds 0..9 ----
set SEEDS=0,1,2,3,4,5,6,7,8,9

echo Running AE-classifier for 10 seeds (seeds: %SEEDS%)
python auto_encoder_classification_train.py --lr %LR% --batch_size %BS% --epochs %EPOCHS% ^
  --patience %PATIENCE% --min_epochs %MIN_EPOCHS% --min_delta %MIN_DELTA% ^
  --seeds %SEEDS% --group_size %GROUP_SIZE% --normalize %NORMALIZE%

echo.
echo Done.
pause
