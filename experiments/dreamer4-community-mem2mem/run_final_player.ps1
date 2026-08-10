param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $PlayerArgs
)

$ErrorActionPreference = "Stop"
$TaskRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Runner = Join-Path $PSScriptRoot "run_midrun_player.ps1"
$Checkpoint = Join-Path $TaskRoot "checkpoints\memmaze-community-d4-mem2mem\memory-final.pt"

& $Runner -CheckpointPath $Checkpoint @PlayerArgs
exit $LASTEXITCODE
