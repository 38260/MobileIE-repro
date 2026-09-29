@echo off
REM ===========================================================================
REM  One-click LAN distribution for the tablet benchmark (ASCII only: cmd reads
REM  this file in the console codepage).
REM
REM    serve.cmd          serve on port 8770
REM    serve.cmd 8899     serve on another port
REM
REM  The tablet and this PC must be on the same Wi-Fi. serve.py prints the exact
REM  one-line command to paste into the tablet's Ubuntu (proot) shell, and it
REM  receives the benchmark JSON back into deployesults_tablet\.
REM ===========================================================================
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" deploy\serve.py %1
exit /b 0
