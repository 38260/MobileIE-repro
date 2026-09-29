@echo off
REM ===========================================================================
REM  MobileIE reproduction - one-command local launcher (ASCII only: cmd reads
REM  this file in the console codepage, Chinese comments here break parsing).
REM  Human-readable output is produced by repro/scripts/progress.py instead.
REM
REM    run.cmd            start stage1 -> IWO stage2 (detached, auto-resume)
REM    run.cmd status     progress, s/epoch, hours left, expected finish time
REM    run.cmd watch      refresh progress every 30 s
REM    run.cmd stop       stop the chain, keep state_last.pt
REM    run.cmd eval       evaluate both checkpoints, then rebuild the report
REM    run.cmd verify     score the official pretrained checkpoint (no training)
REM    run.cmd report     rebuild results/report.md
REM    run.cmd dash       write + open results/dashboard.html (offline, inline SVG)
REM ===========================================================================
setlocal
set "ROOT=%~dp0"
set "REPRO=%ROOT%repro"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=start"

if not exist "%PY%" (
  echo [!] no virtualenv at "%PY%"
  echo     create it first:  powershell -ExecutionPolicy Bypass -File "%REPRO%\setup.ps1"
  exit /b 1
)

if /i "%ACTION%"=="start"  goto start
if /i "%ACTION%"=="resume" goto start
if /i "%ACTION%"=="status" goto status
if /i "%ACTION%"=="watch"  goto watch
if /i "%ACTION%"=="stop"   goto stop
if /i "%ACTION%"=="eval"   goto eval
if /i "%ACTION%"=="report" goto report
if /i "%ACTION%"=="verify" goto verify
if /i "%ACTION%"=="dash"   goto dash
if /i "%ACTION%"=="board"  goto board
goto help

:start
REM refuse a second concurrent chain: an accidental double launch would
REM silently restart a 18 h run from scratch
powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python\.exe|cmd\.exe)$' -and $_.CommandLine -match 'train\.py|run_two_stage' }) { exit 1 } else { exit 0 }"
if errorlevel 1 (
  echo [!] a training chain already looks alive - use run.cmd status, or run.cmd stop first
  exit /b 1
)
cd /d "%REPRO%"
echo [*] stage1 1000 ep + IWO stage2 1000 ep  =  about 17.8 h of GPU time on this laptop
echo [*] press Ctrl+C within 5 s to abort
timeout /t 5 /nobreak
if errorlevel 1 (
  echo [.] aborted, nothing was launched
  exit /b 1
)
echo [*] launching stage1 then IWO stage2, detached
powershell -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','\"%REPRO%\scripts\run_two_stage.cmd\"' -WindowStyle Hidden" 1>nul 2>nul
echo [*] launched; first status in 10 s...
timeout /t 10 /nobreak >nul
goto status

:status
cd /d "%REPRO%"
"%PY%" scripts\progress.py
exit /b 0

:watch
cd /d "%REPRO%"
"%PY%" scripts\progress.py --watch
exit /b 0

:stop
echo [*] killing the runner and trainer; state_last.pt is kept for --resume
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python\.exe|cmd\.exe)$' -and $_.ProcessId -ne $PID -and $_.CommandLine -match 'run_two_stage|train\.py' } | ForEach-Object { Write-Host ('  kill ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
goto status

:eval
cd /d "%REPRO%"
if exist "runs\lolv1_stage1\ckpt\model_best.pkl" (
  echo [*] evaluating stage1 - train graph, folded to slim and re-scored
  "%PY%" scripts\evaluate.py --config lolv1_stage1 --run-name lolv1_stage1 --tag stage1
) else (
  echo [ ] no runs\lolv1_stage1\ckpt\model_best.pkl - skipped
)
if exist "runs\lolv1_stage2_iwo\ckpt\model_best.pkl" (
  echo [*] evaluating stage2 IWO
  "%PY%" scripts\evaluate.py --config lolv1_stage2_iwo --run-name lolv1_stage2_iwo --tag stage2
) else (
  echo [ ] no runs\lolv1_stage2_iwo\ckpt\model_best.pkl - skipped
)
goto report

:verify
cd /d "%REPRO%"
echo [*] official checkpoint on eval15 vs paper Table 1
"%PY%" scripts\evaluate.py --config lolv1_pretrained --tag pretrained --save-images
exit /b 0

:report
cd /d "%REPRO%"
"%PY%" scripts\report.py 1>nul
echo [*] report: %REPRO%\results\report.md   curves: %REPRO%\results\curves.png
exit /b 0

:board
cd /d "%REPRO%"
start "" /min "%PY%" scripts\board.py --port 8765 --open
exit /b 0

:dash
cd /d "%REPRO%"
"%PY%" scripts\dashboard.py --open
exit /b 0

:help
echo.
echo   run.cmd            start two-stage training (detached, auto-retry resume)
echo   run.cmd status     progress / s per epoch / hours left / finish time
echo   run.cmd watch      refresh every 30 s
echo   run.cmd stop       stop, keep the checkpoint
echo   run.cmd eval       evaluate checkpoints and rebuild the report
echo   run.cmd verify     score the official pretrained checkpoint
echo   run.cmd report     rebuild results\report.md
echo   run.cmd dash       write + open results\dashboard.html (offline HTML, inline SVG)
echo   run.cmd board      start the live control board on http://127.0.0.1:8765 and open it
echo.
echo   docs: repro\*.md and cloud\README.md
exit /b 0
