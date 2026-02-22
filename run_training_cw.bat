@echo off
setlocal

REM Optional: activate venv
REM call .venv\Scripts\activate

python class_weighted_classifier.py ^
  --data_path diseases.csv ^
  --indices_file split_indices_full_80_10_10.npz ^
  --lr 0.0001 ^
  --batch_size 32 ^
  --epochs 50 ^
  --seeds 0,1,2,3,4,5,6,7,8,9 ^
  --patience 10 ^
  --min_delta 1e-4 ^
  --min_epochs 10 ^
  --hidden_dims 512 256 128 64 ^
  --group_size 25 ^
  --normalize true ^
  --top_confusions 5 ^
  --rep_classes 12 ^
  --out_prefix CW_block_confusion_g25_seeds0-9

echo.
echo Finished class-weighted runs.
pause
