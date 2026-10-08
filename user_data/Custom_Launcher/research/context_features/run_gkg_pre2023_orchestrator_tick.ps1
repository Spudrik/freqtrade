param(
    [switch]$StatusOnly,
    [switch]$NoDelete
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if ($StatusOnly) {
    & $python -m user_data.Custom_Launcher.research.context_features.gkg_pre2023_orchestrator --status
    exit $LASTEXITCODE
}

$argsList = @("--tick")
if (-not $NoDelete) {
    $argsList += "--allow-delete"
}

& $python -m user_data.Custom_Launcher.research.context_features.gkg_pre2023_orchestrator @argsList
exit $LASTEXITCODE
