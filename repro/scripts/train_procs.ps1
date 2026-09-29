# Print how many training processes are alive. Helper for queue_control.cmd.
# Matches the trainer and the chain wrapper only, never the board or this script.
# Prints ERR instead of a number when the query itself failed, so the caller can
# tell "GPU busy" apart from "I could not look".
$ErrorActionPreference = 'Stop'
try {
    $pat = 'train\.py|run_two_stage'
    $n = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and $_.CommandLine -match $pat
    }).Count
    Write-Output $n
} catch {
    Write-Output 'ERR'
}
