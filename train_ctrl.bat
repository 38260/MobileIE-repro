@echo off
REM ===========================================================================
REM  Run the LOLv1 no-IWO control run in THIS terminal, printing one line per
REM  epoch plus a live per-step progress bar.
REM
REM    train_ctrl.bat        resume from ckpt\state_last.pt and keep printing
REM
REM  * Resumes automatically: runs\lolv1_ctrl_noiwo\ckpt\state_last.pt is read by
REM    train.py itself, so a re-run continues instead of starting over.
REM  * Ctrl+C stops it.  state_last.pt is written every epoch, so at most the
REM    current epoch is lost; the next run picks up from the last completed one.
REM  * KEEP THIS WINDOW OPEN while it runs - closing it kills the training.
REM  * Do not also press "start" on the control board: it will refuse, because a
REM    train.py process is already alive.
REM
REM  Same numbers also land in runs\lolv1_ctrl_noiwo\metrics.jsonl and
REM  runs\lolv1_ctrl_noiwo\train.log, so the board can watch it at the same time.
REM
REM  ASCII only on purpose: cmd reads this file in the console codepage.
REM ===========================================================================
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [!] no virtualenv at "%PY%"
  echo     create it first:  powershell -ExecutionPolicy Bypass -File "%ROOT%repro\setup.ps1"
  pause
  exit /b 1
)

cd /d "%ROOT%repro"
echo [*] control run (lolv1_ctrl_noiwo) - resuming from the last checkpoint
echo [*] one line per epoch, about 27 s each; Ctrl+C to stop
echo.
"%PY%" scripts\train.py --config lolv1_ctrl_noiwo

echo.
echo [*] training process exited with code %ERRORLEVEL%
pause
