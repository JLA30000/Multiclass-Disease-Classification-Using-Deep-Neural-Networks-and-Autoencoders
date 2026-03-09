@echo off
setlocal EnableExtensions EnableDelayedExpansion


set "PY=python"
set "SCRIPT=auto_encoder_classification_train_full.py"

set "DATA_PATH=diseases.csv"
set "IF=split_indices_full_80_10_10.npz"

set "SEEDS=0,1,2,3,4,5,6,7,8,9"
set "EPOCHS=50"
set "PATIENCE=10"
set "MIN_DELTA=1e-4"
set "MIN_EPOCHS=10"
set "LATENT_DIM=64"
set "AE_CKPT=auto"

set "OUTDIR=runs_ae_clf_tuning"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "SUMMARY=%OUTDIR%\summary.csv"
> "%SUMMARY%" echo run_id,learning_rate,encoder_lr,batch_size,arch,hidden_dims,val_macro_f1,val_acc,val_top3,val_top5,log_file

set "MP=GRID_METRIC val_macro_f1"

set "BM=-1"
set "BRI="
set "BEST_LR="
set "BEST_ELR="
set "BEST_BS="
set "BEST_ARCH="
set "BEST_HDIMS="
set "BEST_LOG="

set "LRS=0.0001"
set "EL=1e-4"
set "BATCHES=64"

set "ARCH_COUNT=1"
set "ARCH1_NAME=256_128_64"
set "ARCH1_DIMS=256 128 64"
set "ARCH2_NAME=512_256_128_64"
set "ARCH2_DIMS=512 256 128 64"

set RUN_NUM=0

for %%L in (%LRS%) do (
  for %%E in (%EL%) do (
    for %%B in (%BATCHES%) do (
      for /L %%A in (1,1,%ARCH_COUNT%) do (

        set "LR=%%L"
        set "ELR=%%E"
        set "BS=%%B"
        set "ARCH_ID=%%A"

        if "%%A"=="1" ( set "ARCH_NAME=%ARCH1_NAME%" & set "HDIMS=%ARCH1_DIMS%" )
        if "%%A"=="2" ( set "ARCH_NAME=%ARCH2_NAME%" & set "HDIMS=%ARCH2_DIMS%" )

        set "LR_ID=!LR:.=p!"
        set "ELR_ID=!ELR!"
        set "ELR_ID=!ELR_ID:.=p!"
        set "ELR_ID=!ELR_ID:-=m!"
        set /a RUN_NUM+=1

        set "RUN_ID=lr!LR_ID!_elr!ELR_ID!_bs!BS!_h!ARCH_NAME!"
        set "OUT_PREFIX=%OUTDIR%\AEclf_!RUN_ID!"
        set "LOG_FILE=%OUTDIR%\!RUN_ID!.log"

        echo.
        echo ============================================================
        echo  [Run !RUN_NUM!] !RUN_ID!
        echo  lr=!LR!  encoder_lr=!ELR!  bs=!BS!  hidden=!HDIMS!
        echo ============================================================

        "%PY%" "%SCRIPT%" ^
          --data_path "%DATA_PATH%" ^
          --indices_file "%IF%" ^
          --lr !LR! ^
          --encoder_lr !ELR! ^
          --batch_size !BS! ^
          --epochs %EPOCHS% ^
          --seeds "%SEEDS%" ^
          --patience %PATIENCE% ^
          --min_delta %MIN_DELTA% ^
          --min_epochs %MIN_EPOCHS% ^
          --hidden_dims !HDIMS! ^
          --latent_dim %LATENT_DIM% ^
          --ae_checkpoint "%AE_CKPT%" ^
          --out_prefix "!OUT_PREFIX!" > "!LOG_FILE!" 2>&1

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
          echo.
          type "!LOG_FILE!"
        ) else (
          echo   WARNING: metric not found in !LOG_FILE!
        )

        >> "%SUMMARY%" echo !RUN_ID!,!LR!,!ELR!,!BS!,!ARCH_NAME!,!HDIMS!,!F1!,!ACC!,!TOP3!,!TOP5!,!LOG_FILE!

        if defined F1 (
          for /f %%b in ('python -c "import sys;print(int(float(sys.argv[1])>float(sys.argv[2])))" !F1! !BM!') do (
            if "%%b"=="1" (
              set "BM=!F1!"
              set "BRI=!RUN_ID!"
              set "BEST_LR=!LR!"
              set "BEST_ELR=!ELR!"
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
)

echo.
echo.
if "!BRI!"=="" (
  echo No metric was parsed from any log.
  echo Check logs in %OUTDIR% for errors.
  echo See: "%SUMMARY%"
  exit /b 1
)

echo ============================================================
echo  BEST HYPERPARAMETERS FOUND
echo ============================================================
echo  val_macro_f1:   !BM!
echo  run_id:         !BRI!
echo  learning_rate:  !BEST_LR!
echo  encoder_lr:     !BEST_ELR!
echo  batch_size:     !BEST_BS!
echo  architecture:   !BEST_ARCH!
echo  hidden_dims:    !BEST_HDIMS!
echo  log:            !BEST_LOG!
echo  confusion mat:  %OUTDIR%\AEclf_!BRI!.png
echo  summary csv:    %SUMMARY%
echo ============================================================
echo.
echo Total runs: !RUN_NUM!
echo.
exit /b 0
