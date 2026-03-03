@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ----------------------------
REM Settings
REM ----------------------------
set "PY=python"
set "SCRIPT=xgboost_weighted.py"

set "DATA_PATH=diseases.csv"
set "INDICES_FILE=split_indices_full_80_10_10.npz"

set "SEEDS=0,1,2,3,4,5,6,7,8,9"
set "EARLY=50"
set "DEVICE=cuda"

set "NO_PLOTS=0"

set "OUTDIR=runs_xgb_tuning"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set "SUMMARY=%OUTDIR%\summary.csv"
> "%SUMMARY%" echo run_id,learning_rate,max_depth,subsample,colsample_bytree,reg_lambda,val_macro_f1,log_file

set "METRIC_PATTERN=GRID_METRIC val_macro_f1"

set "BEST_METRIC=-1"
set "BEST_RUN_ID="
set "BEST_LR="
set "BEST_MD="
set "BEST_SUB="
set "BEST_COL="
set "BEST_LAM="
set "BEST_LOG="

REM ----------------------------
REM Grid Search
REM ----------------------------
for %%R in (0.1) do (
  for %%D in (5) do (
    for %%S in (0.75) do (
      for %%C in (0.9) do (
        for %%A in (0.5) do (

          set "LR=%%R"
          set "MD=%%D"
          set "SUB=%%S"
          set "COL=%%C"
          set "LAM=%%A"

          set "LR_ID=!LR:.=p!"
          set "SUB_ID=!SUB:.=p!"
          set "COL_ID=!COL:.=p!"
          set "LAM_ID=!LAM:.=p!"

          set "RUN_ID=lr!LR_ID!_md!MD!_sub!SUB_ID!_col!COL_ID!_lam!LAM_ID!"
          set "OUT_PREFIX=%OUTDIR%\XGB_!RUN_ID!"
          set "LOG_FILE=%OUTDIR%\!RUN_ID!.log"

          echo Running !RUN_ID! ...

          if "!NO_PLOTS!"=="1" (
            "%PY%" "%SCRIPT%" ^
              --data_path "%DATA_PATH%" ^
              --indices_file "%INDICES_FILE%" ^
              --seeds "%SEEDS%" ^
              --early_stopping_rounds "%EARLY%" ^
              --n_estimators 4000 ^
              --learning_rate !LR! ^
              --max_depth !MD! ^
              --min_child_weight 1 ^
              --gamma 0 ^
              --subsample !SUB! ^
              --colsample_bytree !COL! ^
              --reg_lambda !LAM! ^
              --reg_alpha 0 ^
              --tree_method hist ^
              --device %DEVICE% ^
              --n_jobs 8 ^
              --out_prefix "!OUT_PREFIX!" ^
              --no_plots > "!LOG_FILE!" 2>&1
          ) else (
            "%PY%" "%SCRIPT%" ^
              --data_path "%DATA_PATH%" ^
              --indices_file "%INDICES_FILE%" ^
              --seeds "%SEEDS%" ^
              --early_stopping_rounds "%EARLY%" ^
              --n_estimators 4000 ^
              --learning_rate !LR! ^
              --max_depth !MD! ^
              --min_child_weight 1 ^
              --gamma 0 ^
              --subsample !SUB! ^
              --colsample_bytree !COL! ^
              --reg_lambda !LAM! ^
              --reg_alpha 0 ^
              --tree_method hist ^
              --device %DEVICE% ^
              --n_jobs 8 ^
              --out_prefix "!OUT_PREFIX!" > "!LOG_FILE!" 2>&1
          )

          REM Extract metric: match ONLY the mean line, not *_std
          set "METRIC="
          for /f "tokens=3" %%m in ('findstr /C:"!METRIC_PATTERN! " "!LOG_FILE!"') do set "METRIC=%%m"

          >> "%SUMMARY%" echo !RUN_ID!,!LR!,!MD!,!SUB!,!COL!,!LAM!,!METRIC!,!LOG_FILE!

          if defined METRIC (
            for /f %%b in ('python -c "import sys;print(int(float(sys.argv[1]).__gt__(float(sys.argv[2]))))" !METRIC! !BEST_METRIC!') do (
              if "%%b"=="1" (
                set "BEST_METRIC=!METRIC!"
                set "BEST_RUN_ID=!RUN_ID!"
                set "BEST_LR=!LR!"
                set "BEST_MD=!MD!"
                set "BEST_SUB=!SUB!"
                set "BEST_COL=!COL!"
                set "BEST_LAM=!LAM!"
                set "BEST_LOG=!LOG_FILE!"
              )
            )
          ) else (
            echo   WARNING: metric not found in !LOG_FILE!
          )

        )
      )
    )
  )
)

REM ----------------------------
REM Print winner
REM ----------------------------
echo.
if "!BEST_RUN_ID!"=="" (
  echo No metric was parsed from logs.
  echo Fix METRIC_PATTERN to match what your script prints.
  echo See: "%SUMMARY%"
  exit /b 1
)

echo ============================
echo BEST HYPERPARAMETERS FOUND
echo ============================
echo val_macro_f1:     !BEST_METRIC!
echo run_id:           !BEST_RUN_ID!
echo learning_rate:    !BEST_LR!
echo max_depth:        !BEST_MD!
echo subsample:        !BEST_SUB!
echo colsample_bytree: !BEST_COL!
echo reg_lambda:       !BEST_LAM!
echo log:              !BEST_LOG!
echo summary csv:      %SUMMARY%
echo.
exit /b 0
