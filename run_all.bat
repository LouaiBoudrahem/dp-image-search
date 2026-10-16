@echo off
REM
cd /d "%~dp0"
for %%s in (0 1 2) do python train.py --seed %%s
for %%s in (0 1 2) do python train.py --scratch --seed %%s
echo ALL DONE. Results are in results.csv
pause
