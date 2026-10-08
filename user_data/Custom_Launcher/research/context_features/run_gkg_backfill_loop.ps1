param(
    [string]$Start = "2020-01-01",
    [string]$End = "2026-05-29",
    [int]$ChunkDays = 14,
    [int]$MaxWorkers = 6,
    [int]$ProgressEvery = 200,
    [int]$SleepSecondsOnFailure = 120
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$logDir = Join-Path $repoRoot "user_data\research_news_data\gdelt\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logPath = Join-Path $logDir ("gkg_backfill_loop_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-LoopLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Invoke-GkgBackfill {
    param([string[]]$ExtraArgs)
    & $python -m user_data.Custom_Launcher.research.context_features.gkg_backfill @ExtraArgs 2>&1
}

Write-LoopLog "Starting GKG backfill loop start=$Start end=$End chunk_days=$ChunkDays max_workers=$MaxWorkers"

while ($true) {
    $dryArgs = @(
        "--start", $Start,
        "--end", $End,
        "--next-missing-chunk-days", [string]$ChunkDays,
        "--dry-run"
    )
    $dryOutput = Invoke-GkgBackfill -ExtraArgs $dryArgs
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
        Write-LoopLog "GKG backfill loop complete for requested range."
        break
    }

    Write-LoopLog ("Running chunk {0} to {1}; planned files={2}" -f $drySummary.start, $drySummary.end, $drySummary.files_planned)
    $runArgs = @(
        "--start", $Start,
        "--end", $End,
        "--next-missing-chunk-days", [string]$ChunkDays,
        "--max-workers", [string]$MaxWorkers,
        "--progress-every", [string]$ProgressEvery
    )
    $runOutput = Invoke-GkgBackfill -ExtraArgs $runArgs
    $runOutput | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Write-LoopLog "Chunk failed with exit code $LASTEXITCODE; sleeping $SleepSecondsOnFailure seconds."
        Start-Sleep -Seconds $SleepSecondsOnFailure
    }
}

Write-LoopLog "Finished."
