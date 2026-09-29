@echo off
REM ===========================================================================
REM  MobileIE - start the live control board (progress + manual trigger).
REM  ASCII only: cmd reads this file in the console codepage, Chinese here
REM  breaks parsing. The board itself prints Chinese fine.
REM
REM    board.bat          start the board on port 8765 and open the browser
REM    board.bat 8899     use another port
REM
REM  Closing the board window stops the board only - training keeps running.
REM  Use the STOP button on the page (or run.cmd stop) to stop training.
REM ===========================================================================
setlocal
set "ROOT=%~dp0"
set "REPRO=%ROOT%repro"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8765"
set "URL=http://127.0.0.1:%PORT%/"

if not exist "%PY%" (
  echo [!] no virtualenv at "%PY%"
  echo     create it first:  powershell -ExecutionPolicy Bypass -File "%REPRO%\setup.ps1"
  pause
  exit /b 1
)

REM is this exact port already taken? then just point at the live board
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo [=] something already listens on port %PORT% - opening %URL%
  start "" "%URL%"
  exit /b 0
)

cd /d "%REPRO%"
echo [*] starting control board on %URL%
start "" "%PY%" scripts\board.py --port %PORT%
REM give the server a moment to bind, then open the page
ping -n 4 127.0.0.1 >/dev/null
start "" "%URL%"
echo [*] opened %URL%  -  curves refresh every 3 s
echo [*] training logs: repro\logs\runner.log  and  repro\runs\*\train.log
echo [*] text status instead of the page:  run.cmd status   or   run.cmd watch
timeout /t 3 /nobreak >/dev/null 2>&1
exit /b 0
