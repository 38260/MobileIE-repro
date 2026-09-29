@echo off
REM Run both LOLv1 stages back to back, detached, so an ~18h chain survives the
REM session AND survives being killed: a stage that dies is restarted from
REM runs\<exp_name>\ckpt\state_last.pt (up to %MAXRETRY% times).
REM   logs\<config>.log      per-stage stdout
REM   logs\runner.log        lifecycle + retries
REM   runs\<name>\metrics.jsonl   per-epoch loss / val psnr
REM
REM   scripts\run_two_stage.cmd
REM   scripts\run_two_stage.cmd lolv1_stage1 lolv1_stage2_iwo
setlocal enabledelayedexpansion
set REPRO=%~dp0..
cd /d %REPRO%
set S1=%~1
set S2=%~2
if "%S1%"=="" set S1=lolv1_stage1
if "%S2%"=="" set S2=lolv1_stage2_iwo
set MAXRETRY=4
set TQDM_DISABLE=1
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PY=%REPRO%\..\.venv\Scripts\python.exe

call :stage %S1%
if errorlevel 1 exit /b 1
call :stage %S2%
if errorlevel 1 exit /b 1
echo [runner] both stages done %date% %time% >> logs\runner.log
endlocal
exit /b 0

:stage
set CFG=%~1
set TRY=0
:attempt
set /a TRY+=1
if %TRY% gtr %MAXRETRY% (
  echo [runner] %CFG% gave up after %MAXRETRY% retries %date% %time% >> logs\runner.log
  exit /b 1
)
echo [runner] %CFG% attempt %TRY% start %date% %time% >> logs\runner.log
REM train.py auto-resumes from the newest runs\<exp>*\ckpt\state_last.pt, so a retry
REM continues the same run instead of starting a fresh one.
"%PY%" scripts\train.py --config %CFG% >> "logs\%CFG%.log" 2>&1
set RC=%ERRORLEVEL%
if not %RC% equ 0 goto retry
echo [runner] %CFG% done rc=0 %date% %time% >> logs\runner.log
exit /b 0
:retry
echo [runner] %CFG% exited rc=%RC% %date% %time% -- retry in 20s >> logs\runner.log
timeout /t 20 /nobreak >nul
goto attempt
