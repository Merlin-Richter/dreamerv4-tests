param(
    [string] $CheckpointPath,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PlayerArgs
)

$ErrorActionPreference = "Stop"
$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$GitCommonDir = (& git -C $TaskRoot rev-parse --path-format=absolute --git-common-dir).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not resolve the primary transformer checkout" }
$PrimaryRoot = Split-Path $GitCommonDir -Parent
$CodeRoot = Split-Path $PrimaryRoot -Parent
$PatchedDreamer4 = Join-Path (Split-Path $TaskRoot -Parent) "memmaze-community-dreamer4-upstream"
$CommunityRoot = Join-Path $CodeRoot "dreamer4"
$Python = Join-Path $PrimaryRoot "venv\Scripts\python.exe"
$Player = Join-Path $PSScriptRoot "play_dynamics.py"
$Checkpoint = if ($CheckpointPath) {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CheckpointPath)
} else {
    Join-Path $TaskRoot "checkpoints\memmaze-community-d4-mem2mem\memory-midrun-step-00221140-train-000126000.44s.pt"
}
$Tokenizer = Join-Path $CommunityRoot "checkpoints\memmaze-community-d4\tokenizer-final.pt"
$Frames = Join-Path $PrimaryRoot "data\memmaze9x9_val12.npy"
$Actions = Join-Path $PrimaryRoot "data\memmaze9x9_val12_actions.npy"

foreach ($Path in @($Python, $Player, $Checkpoint, $Tokenizer, $Frames, $Actions)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required player input: $Path" }
}
if (-not (Test-Path -LiteralPath (Join-Path $PatchedDreamer4 "dreamer4\model.py"))) {
    throw "Missing pinned memory-enabled Dreamer 4 checkout: $PatchedDreamer4"
}

& $Python -u $Player `
    --dreamer4 $PatchedDreamer4 `
    --checkpoint $Checkpoint `
    --tokenizer $Tokenizer `
    --frames $Frames `
    --actions $Actions `
    --model memory `
    --episode 0 `
    --start 0 `
    --seed 0 `
    @PlayerArgs

exit $LASTEXITCODE
