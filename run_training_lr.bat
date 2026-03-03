@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Optional: activate venv
REM call .venv\Scripts\activate

REM =========================
REM Fixed config (edit if needed)
REM =========================
set "PY=python"
set "SCRIPT=logistic_regression_weighted.py"

set "DATA_PATH=diseases.csv"
set "INDICES_FILE=split_indices_full_80_10_10.npz"

set "SEEDS=0,1,2,3,4,5,6,7,8,9"
set "EPOCHS=100"
set "MIN_EPOCHS=10"
set "MIN_DELTA=1e-4"

set "GROUP_SIZE=25"
set "NORMALIZE=true"
set "TOP_CONFUSIONS=5"
set "REP_CLASSES=12"

REM output folder for logs / artifacts
set "OUTDIR=runs_lr_tuning"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "RESULTS_CSV=%OUTDIR%\results.csv"
set "BEST_TXT=%OUTDIR%\best_run.txt"

REM This must match the python "machine-readable" printed key
set "METRIC_KEY=best_val_macro_f1_mean"

REM Fresh results file each time
echo run_id,lr,batch_size,weight_decay,patience,metric,log_file> "%RESULTS_CSV%"

echo.
echo ================================
echo Logistic Regression Grid Search
echo Output dir: %OUTDIR%
echo Results: %RESULTS_CSV%
echo ================================
echo.

REM Track best in-memory
set "BEST_METRIC="
set "BEST_RUN_ID="
set "BEST_LR="
set "BEST_BS="
set "BEST_WD="
set "BEST_PAT="
set "BEST_LOG="

REM =========================
REM Grid search loops
REM =========================
for %%L in (0.005) do (
  for %%B in (32) do (
    for %%W in (1e-6) do (
      for %%P in (15) do (

        set "RUN_ID=lr%%L_bs%%B_wd%%W_pat%%P"
        set "OUT_PREFIX=%OUTDIR%\LR_!RUN_ID!"
        set "LOG_FILE=%OUTDIR%\!RUN_ID!.log"

        echo ------------------------------------------------
        echo Running: !RUN_ID!
        echo Log: !LOG_FILE!
        echo ------------------------------------------------

        "%PY%" "%SCRIPT%" ^
          --data_path "%DATA_PATH%" ^
          --indices_file "%INDICES_FILE%" ^
          --lr %%L ^
          --batch_size %%B ^
          --weight_decay %%W ^
          --epochs "%EPOCHS%" ^
          --seeds "%SEEDS%" ^
          --patience %%P ^
          --min_delta "%MIN_DELTA%" ^
          --min_epochs "%MIN_EPOCHS%" ^
          --group_size "%GROUP_SIZE%" ^
          --normalize "%NORMALIZE%" ^
          --top_confusions "%TOP_CONFUSIONS%" ^
          --rep_classes "%REP_CLASSES%" ^
          --out_prefix "!OUT_PREFIX!" > "!LOG_FILE!" 2>&1

        if errorlevel 1 (
          echo [ERROR] Run failed: !RUN_ID!  ^(see !LOG_FILE!^)
          echo !RUN_ID!,%%L,%%B,%%W,%%P,,!LOG_FILE!>> "%RESULTS_CSV%"
        ) else (
          REM =========================
          REM Extract metric from log:
          REM expects a line like: best_val_macro_f1_mean=0.842137
          REM =========================
          set "METRIC="
          for /f "tokens=2 delims==" %%A in ('findstr /C:"!METRIC_KEY!=" "!LOG_FILE!"') do set "METRIC=%%A"

          echo [OK] Finished: !RUN_ID!  metric=!METRIC!
          echo !RUN_ID!,%%L,%%B,%%W,%%P,!METRIC!,!LOG_FILE!>> "%RESULTS_CSV%"

          if not defined METRIC (
            echo [WARN] Could not find "!METRIC_KEY!" in log: !LOG_FILE!
          ) else (
            REM =========================
            REM Update best (numeric compare)
            REM =========================
            if not defined BEST_METRIC (
              set "BEST_METRIC=!METRIC!"
              set "BEST_RUN_ID=!RUN_ID!"
              set "BEST_LR=%%L"
              set "BEST_BS=%%B"
              set "BEST_WD=%%W"
              set "BEST_PAT=%%P"
              set "BEST_LOG=!LOG_FILE!"
            ) else (
              for /f %%C in ('python -c "import sys;print(int(float(sys.argv[1]).__gt__(float(sys.argv[2]))))" !METRIC! !BEST_METRIC!') do (
                if "%%C"=="1" (
                  set "BEST_METRIC=!METRIC!"
                  set "BEST_RUN_ID=!RUN_ID!"
                  set "BEST_LR=%%L"
                  set "BEST_BS=%%B"
                  set "BEST_WD=%%W"
                  set "BEST_PAT=%%P"
                  set "BEST_LOG=!LOG_FILE!"
                )
              )
            )
          )
        )

        echo.
      )
    )
  )
)

REM =========================
REM Print final best
REM =========================
echo ================================
echo DONE. Best run (by %METRIC_KEY%):
echo   RUN_ID:    !BEST_RUN_ID!
echo   metric:    !BEST_METRIC!
echo   lr:        !BEST_LR!
echo   batch:     !BEST_BS!
echo   wd:        !BEST_WD!
echo   patience:  !BEST_PAT!
echo   log:       !BEST_LOG!
echo ================================

(
  echo Best run summary
  echo RUN_ID=!BEST_RUN_ID!
  echo metric=!BEST_METRIC!
  echo lr=!BEST_LR!
  echo batch_size=!BEST_BS!
  echo weight_decay=!BEST_WD!
  echo patience=!BEST_PAT!
  echo log_file=!BEST_LOG!
) > "%BEST_TXT%"

echo Wrote:
echo   %RESULTS_CSV%
echo   %BEST_TXT%

endlocal
exit /b 0
