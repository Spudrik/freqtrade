param(
    [string]$Start = "2020-01-01",
    [string]$End = "2026-05-29",
    [int]$ChunkDays = 30,
    [int]$MaxWorkers = 4,
    [int]$ProgressEvery = 200,
    [int]$RetryDeferHours = 72,
    [int]$SleepSecondsOnFailure = 120,
    [switch]$ValidateExisting
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $repoRoot "user_data\research_news_data\gdelt\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("gkg_raw_download_loop_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-LoopLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Invoke-GkgRawDownload {
    param([string[]]$ExtraArgs)
    & $python -m user_data.Custom_Launcher.research.context_features.gkg_raw_download @ExtraArgs 2>&1
}

Write-LoopLog "Starting GKG raw download loop start=$Start end=$End chunk_days=$ChunkDays max_workers=$MaxWorkers retry_defer_hours=$RetryDeferHours validate_existing=$ValidateExisting"

while ($true) {
    $dryArgs = @(
        "--start", $Start,
        "--end", $End,
        "--next-missing-chunk-days", [string]$ChunkDays,
        "--defer-retry-failures-hours", [string]$RetryDeferHours,
        "--dry-run"
    )
    if ($ValidateExisting) {
        $dryArgs += "--validate-existing"
    }
    $dryOutput = Invoke-GkgRawDownload -ExtraArgs $dryArgs
    $dryOutput | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-LoopLog "Dry-run failed with exit code $LASTEXITCODE; sleeping $SleepSecondsOnFailure seconds."
        Start-Sleep -Seconds $SleepSecondsOnFailure
        continue
    }

    try {
        $drySummary = ($dryOutput -join "`n") | ConvertFrom-Json
    }
    catch {
        Write-LoopLog "Could not parse dry-run JSON; sleeping $SleepSecondsOnFailure seconds. Error: $($_.Exception.Message)"
        Start-Sleep -Seconds $SleepSecondsOnFailure
        continue
    }

    if ([int]$drySummary.files_planned -eq 0) {
        Write-LoopLog "GKG raw download loop complete for requested range."
        break
    }

    Write-LoopLog ("Running raw download chunk {0} to {1}; planned files={2}" -f $drySummary.start, $drySummary.end, $drySummary.files_planned)
    $runArgs = @(
        "--start", $Start,
        "--end", $End,
        "--next-missing-chunk-days", [string]$ChunkDays,
        "--max-workers", [string]$MaxWorkers,
        "--defer-retry-failures-hours", [string]$RetryDeferHours,
        "--progress-every", [string]$ProgressEvery
    )
    if ($ValidateExisting) {
        $runArgs += "--validate-existing"
    }
    $runOutput = Invoke-GkgRawDownload -ExtraArgs $runArgs
    $runOutput | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-LoopLog "Raw download chunk exited $LASTEXITCODE; completed rows were persisted, exit code 2 means retryable failures remain, sleeping $SleepSecondsOnFailure seconds."
        Start-Sleep -Seconds $SleepSecondsOnFailure
    }
}

Write-LoopLog "Finished."
