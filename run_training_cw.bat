@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Class-Weighted CE Classifier — Grid Search (full unbalanced)
REM ============================================================

set "PY=python"
set "SCRIPT=class_weighted_classifier.py"

set "DATA_PATH=diseases.csv"
set "INDICES_FILE=split_indices_full_80_10_10.npz"

set "SEEDS=1"
set "EPOCHS=50"
set "PATIENCE=10"
set "MIN_DELTA=1e-4"
set "MIN_EPOCHS=10"

REM ---- Output directory ----
set "OUTDIR=runs_cw_tuning"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "SUMMARY=%OUTDIR%\summary.csv"
> "%SUMMARY%" echo run_id,learning_rate,batch_size,arch,hidden_dims,val_macro_f1,val_acc,val_top3,val_top5,log_file

set "BEST_METRIC=-1"
set "BEST_RUN_ID="
set "BEST_LR="
set "BEST_BS="
set "BEST_ARCH="
set "BEST_HDIMS="
set "BEST_LOG="

REM ---- Hyperparameter Grid ----
set "LRS=0.001 0.0003 0.0001"
set "BATCHES=32 64"

REM Architecture configs (iterated by ID)
set "ARCH_COUNT=2"
set "ARCH1_NAME=256_128_64"
set "ARCH1_DIMS=256 128 64"
set "ARCH2_NAME=512_256_128_64"
set "ARCH2_DIMS=512 256 128 64"

REM ---- Grid Search ----
set RUN_NUM=0

for %%L in (%LRS%) do (
  for %%B in (%BATCHES%) do (
    for /L %%A in (1,1,%ARCH_COUNT%) do (

      set "LR=%%L"
      set "BS=%%B"
      set "ARCH_ID=%%A"

      if "%%A"=="1" ( set "ARCH_NAME=%ARCH1_NAME%" & set "HDIMS=%ARCH1_DIMS%" )
      if "%%A"=="2" ( set "ARCH_NAME=%ARCH2_NAME%" & set "HDIMS=%ARCH2_DIMS%" )

      set "LR_ID=!LR:.=p!"
      set /a RUN_NUM+=1

      set "RUN_ID=lr!LR_ID!_bs!BS!_h!ARCH_NAME!"
      set "OUT_PREFIX=%OUTDIR%\CW_!RUN_ID!"
      set "LOG_FILE=%OUTDIR%\!RUN_ID!.log"

      echo.
      echo ============================================================
      echo  [Run !RUN_NUM!] !RUN_ID!
      echo  lr=!LR!  bs=!BS!  hidden=!HDIMS!
      echo ============================================================

      "%PY%" "%SCRIPT%" ^
        --data_path "%DATA_PATH%" ^
        --indices_file "%INDICES_FILE%" ^
        --lr !LR! ^
        --batch_size !BS! ^
        --epochs %EPOCHS% ^
        --seeds "%SEEDS%" ^
        --patience %PATIENCE% ^
        --min_delta %MIN_DELTA% ^
        --min_epochs %MIN_EPOCHS% ^
        --hidden_dims !HDIMS! ^
        --out_prefix "!OUT_PREFIX!" > "!LOG_FILE!" 2>&1

      REM ---- Extract metrics from log ----
      set "F1="
      set "ACC="
      set "TOP3="
      set "TOP5="
      for /f "tokens=3" %%m in ('findstr /C:"GRID_METRIC val_macro_f1 " "!LOG_FILE!"') do set "F1=%%m"
      for /f "tokens=3" %%m in ('findstr /C:"GRID_METRIC val_acc " "!LOG_FILE!"') do set "ACC=%%m"
      for /f "tokens=3" %%m in ('findstr /C:"GRID_METRIC val_top3 " "!LOG_FILE!"') do set "TOP3=%%m"
      for /f "tokens=3" %%m in ('findstr /C:"GRID_METRIC val_top5 " "!LOG_FILE!"') do set "TOP5=%%m"

      if defined F1 (
        echo   macro_f1=!F1!  acc=!ACC!  top3=!TOP3!  top5=!TOP5!
      ) else (
        echo   WARNING: metric not found in !LOG_FILE!
      )

      >> "%SUMMARY%" echo !RUN_ID!,!LR!,!BS!,!ARCH_NAME!,!HDIMS!,!F1!,!ACC!,!TOP3!,!TOP5!,!LOG_FILE!

      REM ---- Track best (by macro F1) ----
      if defined F1 (
        for /f %%b in ('python -c "import sys;print(int(float(sys.argv[1])>float(sys.argv[2])))" !F1! !BEST_METRIC!') do (
          if "%%b"=="1" (
            set "BEST_METRIC=!F1!"
            set "BEST_RUN_ID=!RUN_ID!"
            set "BEST_LR=!LR!"
            set "BEST_BS=!BS!"
            set "BEST_ARCH=!ARCH_NAME!"
            set "BEST_HDIMS=!HDIMS!"
            set "BEST_LOG=!LOG_FILE!"
          )
        )
      )

    )
  )
)

REM ============================================================
REM  Print Winner
REM ============================================================
echo.
echo.
if "!BEST_RUN_ID!"=="" (
  echo No metric was parsed from any log.
  echo Check logs in %OUTDIR% for errors.
  echo See: "%SUMMARY%"
  exit /b 1
)

echo ============================================================
echo  BEST HYPERPARAMETERS FOUND
echo ============================================================
echo  val_macro_f1:   !BEST_METRIC!
echo  run_id:         !BEST_RUN_ID!
echo  learning_rate:  !BEST_LR!
echo  batch_size:     !BEST_BS!
echo  architecture:   !BEST_ARCH!
echo  hidden_dims:    !BEST_HDIMS!
echo  log:            !BEST_LOG!
echo  summary csv:    %SUMMARY%
echo ============================================================
echo.
echo Total runs: !RUN_NUM!
echo.
exit /b 0
