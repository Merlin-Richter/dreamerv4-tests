$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$probe = Join-Path (Split-Path -Parent $PSScriptRoot) 'qualify_vast_instance.ps1'
. $probe -InternalNoRun

$tests = 0
function Assert-Equal {
    param($Actual, $Expected, [string]$Message)
    $script:tests++
    if ($Actual -ne $Expected) {
        throw "$Message (expected '$Expected', got '$Actual')"
    }
}

function Assert-True {
    param([bool]$Condition, [string]$Message)
    $script:tests++
    if (-not $Condition) { throw $Message }
}

Assert-Equal (Get-EndpointKind 'ssh7.vast.ai') 'Vast proxy' 'proxy endpoint classification'
Assert-Equal (Get-EndpointKind '154.64.230.67') 'direct IP' 'direct endpoint classification'
Assert-Equal (Get-EndpointKind 'gpu.example.net') 'direct/unknown hostname' 'hostname endpoint classification'
Assert-Equal (Get-OverallVerdict $true $true $true 'direct IP') 'PASS' 'healthy direct verdict'
Assert-Equal (Get-OverallVerdict $true $true $true 'Vast proxy') 'WARN' 'healthy proxy verdict'
Assert-Equal (Get-OverallVerdict $true $false $true 'direct IP') 'FAIL' 'threshold failure verdict'
Assert-Equal (Get-OverallVerdict $false $true $true 'direct IP') 'FAIL' 'core failure verdict'
Assert-Equal (Get-OverallVerdict $true $true $false 'direct IP') 'WARN' 'cleanup warning verdict'
Assert-Equal ([Math]::Round((Get-ProjectedCheckpointSeconds 1MB), 6)) 200.0 'checkpoint projection'
Assert-Equal (Get-TransferTimeoutSeconds (16MB)) 94 'fast transfer timeout'
$scpArguments = Get-ScpArguments -Source 'local.bin' -Destination 'root@example:/tmp/remote.bin'
Assert-True ($scpArguments -contains '-O') 'Vast SCP must force legacy protocol instead of SFTP'

$quoted = ConvertTo-NativeArgument 'path with spaces\key'
Assert-Equal $quoted '"path with spaces\key"' 'native argument quoting'
$quotedWithQuote = ConvertTo-NativeArgument 'a "quoted" value'
Assert-Equal $quotedWithQuote '"a \"quoted\" value"' 'embedded quote escaping'

$timeout = Invoke-BoundedProcess -FilePath ((Get-Command powershell.exe).Source) -Arguments @('-NoProfile', '-Command', 'Start-Sleep -Seconds 2') -TimeoutSeconds 1
Assert-True $timeout.TimedOut 'bounded process must time out'
Assert-True ($timeout.ElapsedSeconds -lt 4.0) 'bounded process timeout must return promptly'

$tempDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("vast-probe-unit-" + [Guid]::NewGuid().ToString('N'))
$tempFile = Join-Path $tempDirectory 'random.bin'
try {
    New-Item -ItemType Directory -Path $tempDirectory | Out-Null
    New-RandomFile -Path $tempFile -SizeBytes (2MB)
    Assert-Equal (Get-Item -LiteralPath $tempFile).Length (2MB) 'random file size'
    $bytes = [System.IO.File]::ReadAllBytes($tempFile)
    $allZero = -not ($bytes | Where-Object { $_ -ne 0 } | Select-Object -First 1)
    Assert-True (-not $allZero) 'test data must not be sparse/zero-filled'
}
finally {
    if (Test-Path -LiteralPath $tempDirectory) { Remove-Item -LiteralPath $tempDirectory -Recurse -Force }
}
Assert-True (-not (Test-Path -LiteralPath $tempDirectory)) 'unit-test temporary data cleanup'

# Replace the network primitive only for this final unit test. The retry helper
# must recover from OpenSSH's transport failure code without retrying forever.
$script:retryCalls = 0
function Invoke-ProbeSsh {
    param([string]$RemoteCommand, [int]$TimeoutSeconds)
    $script:retryCalls++
    if ($script:retryCalls -eq 1) {
        return [pscustomobject]@{ TimedOut = $false; ExitCode = 255; Stdout = ''; Stderr = 'transient' }
    }
    return [pscustomobject]@{ TimedOut = $false; ExitCode = 0; Stdout = 'ok'; Stderr = '' }
}
$retried = Invoke-IdempotentSshWithRetry -RemoteCommand 'true' -TimeoutSeconds 1 -Attempts 3
Assert-Equal $retried.ExitCode 0 'idempotent SSH retry result'
Assert-Equal $script:retryCalls 2 'idempotent SSH retry count'

Write-Output "PASS: $tests qualify_vast_instance tests"
