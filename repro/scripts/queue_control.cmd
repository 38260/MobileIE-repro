@echo off
REM Queue the no-IWO control run behind the chain that is training right now.
REM Polls every 60s; as soon as no train.py / run_two_stage process is left it
REM starts configs\lolv1_ctrl_noiwo.yaml (ep1001-2000, resumed from the seeded
REM runs\lolv1_ctrl_noiwo\ckpt\state_last.pt) with the same retry rules as
REM run_two_stage.cmd.
REM
REM   logs\queue.log              watcher heartbeat
REM   logs\lolv1_ctrl_noiwo.log   control stdout
REM
REM Start it detached:
REM   powershell -NoProfile -Command "Start-Process cmd -ArgumentList '/c','scripts\queue_control.cmd' -WindowStyle Hidden"
REM Cancel while it is still waiting:  echo.> logs\queue.cancel
setlocal enabledelayedexpansion
set REPRO=%~dp0..
cd /d %REPRO%
set CFG=lolv1_ctrl_noiwo
set MAXRETRY=4
set TQDM_DISABLE=1
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set PY=%REPRO%\..\.venv\Scripts\python.exe
set BAD=0

echo [queue] watcher start %time% >> logs\queue.log

:wait
if exist logs\queue.cancel (
  echo [queue] cancelled by logs\queue.cancel %time%, control run not started >> logs\queue.log
  exit /b 3
)
set N=ERR
for /f "delims=" %%C in ('powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train_procs.ps1 2^>nul') do set N=%%C
if "%N%"=="ERR" goto probe_failed
set BAD=0
if not "%N%"=="0" goto busy
goto start

:busy
echo [queue] %N% training process(es) alive, poll again in 60s %time% >> logs\queue.log
ping -n 61 127.0.0.1 >nul
goto wait

REM A failed probe keeps us waiting: double-booking the GPU would kill both runs.
:probe_failed
set /a BAD+=1
echo [queue] probe failed (!BAD!) %time% >> logs\queue.log
if !BAD! leq 10 (
  ping -n 61 127.0.0.1 >nul
  goto wait
)
echo [queue] process probe broken, giving up %time% >> logs\queue.log
exit /b 2

:start
set TRY=0
echo [queue] GPU free, launching %CFG% %time% >> logs\queue.log
REM runner.log is what the board scans for stage names, so the control only shows up
REM as a third stage card once it is really running.
echo [queue] %CFG% queued start %time% >> logs\runner.log

:attempt
set /a TRY+=1
if %TRY% gtr %MAXRETRY% (
  echo [queue] %CFG% gave up after %MAXRETRY% retries %time% >> logs\queue.log
  exit /b 1
)
echo [queue] %CFG% attempt %TRY% start %time% >> logs\queue.log
"%PY%" scripts\train.py --config %CFG% >> "logs\%CFG%.log" 2>&1
set RC=%ERRORLEVEL%
if not %RC% equ 0 goto retry
echo [queue] %CFG% done rc=0 %time% >> logs\queue.log
endlocal
exit /b 0

:retry
echo [queue] %CFG% exited rc=%RC% %time% -- retry in 20s >> logs\queue.log
ping -n 21 127.0.0.1 >nul
goto attempt
