[CmdletBinding()]
param(
    [string]$SshHost,
    [ValidateRange(1, 65535)][int]$Port = 22,
    [string]$User = "root",
    [string]$IdentityFile,
    [ValidateRange(1, 1024)][int]$FastTransferMiB = 16,
    [ValidateRange(1.0, 1000000.0)][double]$MinRemoteDownloadMbps = 100.0,
    [ValidateRange(1.0, 1000000.0)][double]$MinRemoteUploadMbps = 20.0,
    [ValidateRange(1, 86400)][int]$MaxCheckpointPullSeconds = 300,
    [switch]$Confirm200MiB,
    [switch]$CheckWsl,
    [string]$WslIdentityFile,
    [switch]$InternalNoRun
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function ConvertTo-NativeArgument {
    param([AllowEmptyString()][string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }

    # Windows CreateProcess receives one command-line string. Apply the quoting
    # rules used by CommandLineToArgvW so paths and remote shell commands remain
    # single argv entries even when they contain spaces or quotes.
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (($backslashes * 2) + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-BoundedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][ValidateRange(1, 86400)][int]$TimeoutSeconds
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = (($Arguments | ForEach-Object { ConvertTo-NativeArgument ([string]$_) }) -join ' ')
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $process.Start()) {
            throw "failed to start $FilePath"
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $finished = $process.WaitForExit($TimeoutSeconds * 1000)
        if (-not $finished) {
            try { $process.Kill() } catch { }
            $process.WaitForExit()
        }
        $timer.Stop()
        return [pscustomobject]@{
            ExitCode = if ($finished) { $process.ExitCode } else { -1 }
            Stdout = $stdoutTask.Result.Trim()
            Stderr = $stderrTask.Result.Trim()
            TimedOut = -not $finished
            ElapsedSeconds = $timer.Elapsed.TotalSeconds
            Command = "$FilePath $($startInfo.Arguments)"
        }
    }
    finally {
        $timer.Stop()
        $process.Dispose()
    }
}

function Get-EndpointKind {
    param([Parameter(Mandatory = $true)][string]$Name)

    if ($Name -match '^(?i:ssh\d*\.vast\.ai)$') {
        return "Vast proxy"
    }
    $parsedAddress = $null
    if ([System.Net.IPAddress]::TryParse($Name, [ref]$parsedAddress)) {
        return "direct IP"
    }
    return "direct/unknown hostname"
}

function Get-ProjectedCheckpointSeconds {
    param([Parameter(Mandatory = $true)][double]$BytesPerSecond)
    if ($BytesPerSecond -le 0) { return [double]::PositiveInfinity }
    return (200.0 * 1024.0 * 1024.0) / $BytesPerSecond
}

function Format-Duration {
    param([double]$Seconds)
    if ([double]::IsInfinity($Seconds)) { return "unavailable" }
    if ($Seconds -lt 60) { return ("{0:0.0} s" -f $Seconds) }
    return ("{0:0.0} min" -f ($Seconds / 60.0))
}

function Get-OverallVerdict {
    param(
        [bool]$CorePassed,
        [bool]$ThresholdsPassed,
        [bool]$CleanupPassed,
        [string]$EndpointKind
    )
    if (-not $CorePassed -or -not $ThresholdsPassed) { return "FAIL" }
    if (-not $CleanupPassed -or $EndpointKind -ne "direct IP") { return "WARN" }
    return "PASS"
}

function New-RandomFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][long]$SizeBytes
    )

    $rng = New-Object System.Security.Cryptography.RNGCryptoServiceProvider
    $stream = New-Object System.IO.FileStream(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None,
        1048576,
        [System.IO.FileOptions]::SequentialScan
    )
    $buffer = New-Object byte[] 1048576
    $remaining = $SizeBytes
    try {
        while ($remaining -gt 0) {
            $count = [int][Math]::Min($buffer.Length, $remaining)
            $rng.GetBytes($buffer)
            $stream.Write($buffer, 0, $count)
            $remaining -= $count
        }
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
        $rng.Dispose()
    }
}

function ConvertTo-ShellSingleQuoted {
    param([Parameter(Mandatory = $true)][string]$Value)
    $embeddedSingleQuote = "'" + '"' + "'" + '"' + "'"
    return "'" + $Value.Replace("'", $embeddedSingleQuote) + "'"
}

function Get-BaseSshArguments {
    param([switch]$ForScp)

    $arguments = @()
    if ($ForScp) { $arguments += @('-P', [string]$Port) }
    else { $arguments += @('-p', [string]$Port, '-T') }
    $arguments += @(
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=10',
        '-o', 'ConnectionAttempts=1',
        '-o', 'ServerAliveInterval=5',
        '-o', 'ServerAliveCountMax=2',
        '-o', 'Compression=no',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'LogLevel=ERROR'
    )
    if ($IdentityFile) { $arguments += @('-i', $IdentityFile) }
    return $arguments
}

function Invoke-ProbeSsh {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteCommand,
        [ValidateRange(1, 86400)][int]$TimeoutSeconds = 30
    )
    $arguments = @(Get-BaseSshArguments) + @("${User}@${SshHost}", $RemoteCommand)
    return Invoke-BoundedProcess -FilePath $script:SshExecutable -Arguments $arguments -TimeoutSeconds $TimeoutSeconds
}

function Invoke-IdempotentSshWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteCommand,
        [ValidateRange(1, 86400)][int]$TimeoutSeconds = 30,
        [ValidateRange(1, 5)][int]$Attempts = 3
    )
    $result = $null
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $result = Invoke-ProbeSsh -RemoteCommand $RemoteCommand -TimeoutSeconds $TimeoutSeconds
        if (-not $result.TimedOut -and $result.ExitCode -eq 0) { return $result }
        # 255 is the OpenSSH transport/auth failure code. Do not retry a real
        # remote command failure; it is deterministic and should be diagnosed.
        if (-not $result.TimedOut -and $result.ExitCode -ne 255) { return $result }
        if ($attempt -lt $Attempts) { Start-Sleep -Seconds 1 }
    }
    return $result
}

function Get-ScpArguments {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    # Windows OpenSSH 9+ defaults scp to the SFTP subsystem. Vast SSH images do
    # not consistently expose SFTP, while their documented legacy SCP path does
    # work. All remote paths here are internally generated and contain no shell
    # metacharacters, so explicit -O is both compatible and safe.
    return @(Get-BaseSshArguments -ForScp) + @('-O', '-q', $Source, $Destination)
}

function Invoke-ProbeScp {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [ValidateRange(1, 86400)][int]$TimeoutSeconds
    )
    $arguments = Get-ScpArguments -Source $Source -Destination $Destination
    return Invoke-BoundedProcess -FilePath $script:ScpExecutable -Arguments $arguments -TimeoutSeconds $TimeoutSeconds
}

function Get-TransferTimeoutSeconds {
    param([long]$SizeBytes)
    # A failing link is allowed roughly 0.25 MiB/s plus connection overhead, but
    # never gets an unbounded transfer. The 16 MiB default therefore gets 94 s.
    return [int][Math]::Min(1800, [Math]::Max(45, 30 + ($SizeBytes / 262144.0)))
}

function Get-RemoteCurlMeasurement {
    param(
        [Parameter(Mandatory = $true)][string]$RemoteFile,
        [Parameter(Mandatory = $true)][string]$RemoteUploadFile
    )

    $downloadCommand = "LC_ALL=C curl --silent --show-error --fail --location --max-time 25 --output /dev/null --write-out 'PROBE_DOWNLOAD %{speed_download} %{time_total} %{http_code}' 'https://speed.cloudflare.com/__down?bytes=25000000'"
    $download = Invoke-ProbeSsh -RemoteCommand $downloadCommand -TimeoutSeconds 35
    if ($download.TimedOut -or $download.ExitCode -ne 0 -or $download.Stdout -notmatch 'PROBE_DOWNLOAD\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)') {
        return [pscustomobject]@{
            DownloadBytesPerSecond = 0.0
            DownloadSeconds = $download.ElapsedSeconds
            DownloadError = if ($download.TimedOut) { "timed out" } else { ($download.Stderr + ' ' + $download.Stdout).Trim() }
            UploadBytesPerSecond = 0.0
            UploadSeconds = 0.0
            UploadError = "skipped because download probe failed"
        }
    }
    $downloadBytesPerSecond = [double]::Parse($Matches[1], [System.Globalization.CultureInfo]::InvariantCulture)
    $downloadSeconds = [double]::Parse($Matches[2], [System.Globalization.CultureInfo]::InvariantCulture)

    $uploadCommand = "head -c 10000000 '$RemoteFile' > '$RemoteUploadFile' && LC_ALL=C curl --silent --show-error --fail --location --max-time 25 --output /dev/null --write-out 'PROBE_UPLOAD %{speed_upload} %{time_total} %{http_code}' --request POST --data-binary '@$RemoteUploadFile' 'https://speed.cloudflare.com/__up'"
    $upload = Invoke-ProbeSsh -RemoteCommand $uploadCommand -TimeoutSeconds 35
    if ($upload.TimedOut -or $upload.ExitCode -ne 0 -or $upload.Stdout -notmatch 'PROBE_UPLOAD\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)') {
        return [pscustomobject]@{
            DownloadBytesPerSecond = $downloadBytesPerSecond
            DownloadSeconds = $downloadSeconds
            DownloadError = ""
            UploadBytesPerSecond = 0.0
            UploadSeconds = $upload.ElapsedSeconds
            UploadError = if ($upload.TimedOut) { "timed out" } else { ($upload.Stderr + ' ' + $upload.Stdout).Trim() }
        }
    }
    return [pscustomobject]@{
        DownloadBytesPerSecond = $downloadBytesPerSecond
        DownloadSeconds = $downloadSeconds
        DownloadError = ""
        UploadBytesPerSecond = [double]::Parse($Matches[1], [System.Globalization.CultureInfo]::InvariantCulture)
        UploadSeconds = [double]::Parse($Matches[2], [System.Globalization.CultureInfo]::InvariantCulture)
        UploadError = ""
    }
}

function Invoke-WslHandoffCheck {
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) {
        return [pscustomobject]@{ Passed = $false; Detail = "wsl.exe is not installed"; ElapsedSeconds = 0.0 }
    }

    $parts = @(
        'ssh', '-p', [string]$Port, '-T',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=10',
        '-o', 'ConnectionAttempts=1',
        '-o', 'StrictHostKeyChecking=accept-new'
    )
    if ($WslIdentityFile) { $parts += @('-i', $WslIdentityFile) }
    $parts += @("${User}@${SshHost}", "printf WSL_PROBE_OK")
    $shellCommand = (($parts | ForEach-Object { ConvertTo-ShellSingleQuoted ([string]$_) }) -join ' ')
    $result = Invoke-BoundedProcess -FilePath $wsl.Source -Arguments @('-e', 'bash', '-lc', $shellCommand) -TimeoutSeconds 25
    $passed = (-not $result.TimedOut -and $result.ExitCode -eq 0 -and $result.Stdout -match 'WSL_PROBE_OK')
    $detail = if ($passed) { "authenticated independently" } elseif ($result.TimedOut) { "timed out" } else { ($result.Stderr + ' ' + $result.Stdout).Trim() }
    return [pscustomobject]@{ Passed = $passed; Detail = $detail; ElapsedSeconds = $result.ElapsedSeconds }
}

function Write-ProbeReport {
    param([hashtable]$State)

    Write-Output "=== Vast qualification: $($State.Verdict) ==="
    Write-Output "Transport: Windows-native $($State.EndpointKind)"
    Write-Output "Endpoint: ${User}@${SshHost}:$Port"
    if ($State.FailureReason) { Write-Output "Failure: $($State.FailureReason)" }
    if ($State.RemoteInternet) {
        $downloadStatus = if ($State.RemoteDownloadPassed) { "PASS" } else { "FAIL" }
        $uploadStatus = if ($State.RemoteUploadPassed) { "PASS" } else { "FAIL" }
        Write-Output ("Remote internet: {0} download {1:0.0} Mbps; {2} upload {3:0.0} Mbps" -f $downloadStatus, $State.RemoteDownloadMbps, $uploadStatus, $State.RemoteUploadMbps)
    }
    if ($State.PullBytesPerSecond -gt 0) {
        $pullStatus = if ($State.CheckpointPassed) { "PASS" } else { "FAIL" }
        Write-Output ("Checkpoint retrieval: {0} {1:0.0} Mbps; projected 200 MiB pull {2}" -f $pullStatus, $State.PullMbps, (Format-Duration $State.ProjectedCheckpointSeconds))
    }
    if ($State.IntegrityChecked) {
        Write-Output "Integrity: $(if ($State.IntegrityPassed) { 'PASS' } else { 'FAIL' }) (SHA-256 local, remote, and returned copy)"
    }
    Write-Output ("Cleanup: {0}; total duration {1}" -f $(if ($State.CleanupPassed) { 'PASS' } else { 'WARN' }), (Format-Duration $State.TotalSeconds))
    if ($State.EndpointKind -eq 'Vast proxy') {
        Write-Output "Diagnosis: this is a Vast proxy endpoint; its SSH rate may be proxy-limited. Prefer the direct endpoint."
    }
    if ($State.FullConfirmation) {
        Write-Output ("200 MiB confirmation: {0}; upload {1:0.0} Mbps, pull {2:0.0} Mbps" -f $State.FullConfirmation.Status, $State.FullConfirmation.UploadMbps, $State.FullConfirmation.PullMbps)
    }
    if ($State.WslCheck) {
        Write-Output "WSL handoff: $(if ($State.WslCheck.Passed) { 'PASS' } else { 'FAIL' }) ($($State.WslCheck.Detail))"
    }

    Write-Output ""
    Write-Output "--- Raw measurements ---"
    if ($State.SshLatencies.Count -gt 0) {
        Write-Output ("SSH command latency (3 fresh Windows connections): {0}" -f (($State.SshLatencies | ForEach-Object { "{0:0.000}s" -f $_ }) -join ', '))
    }
    if ($State.RemoteInternet) {
        Write-Output ("Remote download: {0:0.00} MiB/s ({1:0.0} Mbps), {2:0.00}s{3}" -f ($State.RemoteInternet.DownloadBytesPerSecond / 1MB), $State.RemoteDownloadMbps, $State.RemoteInternet.DownloadSeconds, $(if ($State.RemoteInternet.DownloadError) { "; error: $($State.RemoteInternet.DownloadError)" } else { '' }))
        Write-Output ("Remote upload:   {0:0.00} MiB/s ({1:0.0} Mbps), {2:0.00}s{3}" -f ($State.RemoteInternet.UploadBytesPerSecond / 1MB), $State.RemoteUploadMbps, $State.RemoteInternet.UploadSeconds, $(if ($State.RemoteInternet.UploadError) { "; error: $($State.RemoteInternet.UploadError)" } else { '' }))
    }
    if ($State.PushBytesPerSecond -gt 0) {
        Write-Output ("Windows -> Vast SCP: {0:0.00} MiB/s ({1:0.0} Mbps), {2:0.00}s for {3} MiB" -f ($State.PushBytesPerSecond / 1MB), $State.PushMbps, $State.PushSeconds, $FastTransferMiB)
    }
    if ($State.PullBytesPerSecond -gt 0) {
        Write-Output ("Vast -> Windows SCP: {0:0.00} MiB/s ({1:0.0} Mbps), {2:0.00}s for {3} MiB" -f ($State.PullBytesPerSecond / 1MB), $State.PullMbps, $State.PullSeconds, $FastTransferMiB)
    }
    if ($State.RemoteDiskMiBPerSecond -gt 0) {
        Write-Output ("Remote disk copy+sync sanity: {0:0.0} MiB/s; free /tmp space {1:0.0} GiB" -f $State.RemoteDiskMiBPerSecond, $State.RemoteFreeGiB)
    }
    Write-Output "Thresholds: remote download >= $MinRemoteDownloadMbps Mbps; remote upload >= $MinRemoteUploadMbps Mbps; 200 MiB pull <= $MaxCheckpointPullSeconds s"
    Write-Output "Internet test service: Cloudflare speed endpoints; SSH compression disabled."

    if ($State.Verdict -ne 'FAIL') {
        Write-Output ""
        Write-Output "--- Continue with this transport ---"
        $identityText = if ($IdentityFile) { " -i `"$IdentityFile`"" } else { "" }
        Write-Output "Windows SSH: ssh -p $Port$identityText ${User}@${SshHost}"
        Write-Output "Windows SCP: scp -P $Port$identityText <source> ${User}@${SshHost}:<destination>"
        Write-Output "WSL wrappers need a separate WSL-side key/config check; Windows success does not prove WSL success."
        Write-Output "VAST_SSH_HOST=$SshHost"
        Write-Output "VAST_SSH_PORT=$Port"
        Write-Output "VAST_SSH_USER=$User"
    }
}

function Invoke-FullConfirmation {
    param(
        [string]$LocalDirectory,
        [string]$RemoteDirectory
    )
    $sizeBytes = 200L * 1024L * 1024L
    $localFile = Join-Path $LocalDirectory 'checkpoint-200m.bin'
    $returnedFile = Join-Path $LocalDirectory 'checkpoint-200m.returned.bin'
    $remoteFile = "$RemoteDirectory/checkpoint-200m.bin"
    New-RandomFile -Path $localFile -SizeBytes $sizeBytes
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localFile).Hash.ToLowerInvariant()
    $timeout = Get-TransferTimeoutSeconds $sizeBytes
    $push = Invoke-ProbeScp -Source $localFile -Destination "${User}@${SshHost}:$remoteFile" -TimeoutSeconds $timeout
    if ($push.TimedOut -or $push.ExitCode -ne 0) {
        return [pscustomobject]@{ Status = "FAIL (upload)"; UploadMbps = 0.0; PullMbps = 0.0 }
    }
    $pull = Invoke-ProbeScp -Source "${User}@${SshHost}:$remoteFile" -Destination $returnedFile -TimeoutSeconds $timeout
    if ($pull.TimedOut -or $pull.ExitCode -ne 0) {
        return [pscustomobject]@{ Status = "FAIL (pull)"; UploadMbps = (($sizeBytes / $push.ElapsedSeconds) * 8 / 1000000); PullMbps = 0.0 }
    }
    $returnedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $returnedFile).Hash.ToLowerInvariant()
    $remoteHashResult = Invoke-ProbeSsh -RemoteCommand "sha256sum '$remoteFile' | cut -d ' ' -f 1" -TimeoutSeconds 20
    $integrity = ($returnedHash -eq $hash -and $remoteHashResult.ExitCode -eq 0 -and $remoteHashResult.Stdout.Trim().ToLowerInvariant() -eq $hash)
    return [pscustomobject]@{
        Status = if ($integrity) { "PASS" } else { "FAIL (integrity)" }
        UploadMbps = (($sizeBytes / $push.ElapsedSeconds) * 8 / 1000000)
        PullMbps = (($sizeBytes / $pull.ElapsedSeconds) * 8 / 1000000)
    }
}

function Invoke-VastQualification {
    if (-not $SshHost) { throw "-SshHost is required" }
    if ($IdentityFile -and -not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
        throw "identity file does not exist: $IdentityFile"
    }

    $script:SshExecutable = (Get-Command ssh.exe -ErrorAction Stop).Source
    $script:ScpExecutable = (Get-Command scp.exe -ErrorAction Stop).Source
    $endpointKind = Get-EndpointKind $SshHost
    $probeId = [Guid]::NewGuid().ToString('N')
    $localDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "vast-network-probe-$probeId"
    $remoteDirectory = "/tmp/vast-network-probe-$probeId"
    $localFile = Join-Path $localDirectory 'probe.bin'
    $returnedFile = Join-Path $localDirectory 'probe.returned.bin'
    $remoteFile = "$remoteDirectory/probe.bin"
    $remoteUploadFile = "$remoteDirectory/internet-upload.bin"
    $sizeBytes = [long]$FastTransferMiB * 1024L * 1024L
    $totalTimer = [System.Diagnostics.Stopwatch]::StartNew()
    $remoteCreated = $false
    $cleanupPassed = $true
    $failureReason = ""

    $state = @{
        Verdict = "FAIL"
        EndpointKind = $endpointKind
        FailureReason = ""
        SshLatencies = @()
        RemoteInternet = $null
        RemoteDownloadMbps = 0.0
        RemoteUploadMbps = 0.0
        RemoteDownloadPassed = $false
        RemoteUploadPassed = $false
        PushBytesPerSecond = 0.0
        PullBytesPerSecond = 0.0
        PushMbps = 0.0
        PullMbps = 0.0
        PushSeconds = 0.0
        PullSeconds = 0.0
        ProjectedCheckpointSeconds = [double]::PositiveInfinity
        CheckpointPassed = $false
        IntegrityChecked = $false
        IntegrityPassed = $false
        RemoteDiskMiBPerSecond = 0.0
        RemoteFreeGiB = 0.0
        CleanupPassed = $true
        TotalSeconds = 0.0
        FullConfirmation = $null
        WslCheck = $null
    }

    try {
        New-Item -ItemType Directory -Path $localDirectory -ErrorAction Stop | Out-Null

        $auth = Invoke-ProbeSsh -RemoteCommand "printf PROBE_AUTH_OK" -TimeoutSeconds 20
        if ($auth.TimedOut) { throw "Windows-native SSH authentication timed out" }
        if ($auth.ExitCode -ne 0 -or $auth.Stdout -notmatch 'PROBE_AUTH_OK') {
            throw "Windows-native SSH authentication failed: $(($auth.Stderr + ' ' + $auth.Stdout).Trim())"
        }

        $latencies = @()
        for ($index = 0; $index -lt 3; $index++) {
            $latency = Invoke-ProbeSsh -RemoteCommand "printf PROBE_LATENCY_OK" -TimeoutSeconds 20
            if ($latency.TimedOut -or $latency.ExitCode -ne 0) { throw "SSH latency trial $($index + 1) failed" }
            $latencies += $latency.ElapsedSeconds
        }
        $state.SshLatencies = $latencies

        $setupCommand = "set -eu; command -v curl >/dev/null; command -v sha256sum >/dev/null; command -v stat >/dev/null; command -v head >/dev/null; mkdir -p '$remoteDirectory'; chmod 700 '$remoteDirectory'; df -Pk '$remoteDirectory' | awk 'NR==2 {print `$4}'"
        # From this point onward cleanup is safe to attempt even if setup fails
        # after mkdir (for example, a broken df/stat tool on the image).
        $remoteCreated = $true
        $setup = Invoke-IdempotentSshWithRetry -RemoteCommand $setupCommand -TimeoutSeconds 20
        if ($setup.TimedOut -or $setup.ExitCode -ne 0) {
            throw "remote prerequisites/setup failed (exit $($setup.ExitCode), timeout $($setup.TimedOut)): $(($setup.Stderr + ' ' + $setup.Stdout).Trim())"
        }
        $freeKiB = 0.0
        if ([double]::TryParse(($setup.Stdout -split "`n")[-1].Trim(), [ref]$freeKiB)) {
            $state.RemoteFreeGiB = $freeKiB / 1024.0 / 1024.0
        }
        if (($freeKiB * 1024.0) -lt ($sizeBytes * 3.0)) { throw "remote /tmp has insufficient free space" }

        New-RandomFile -Path $localFile -SizeBytes $sizeBytes
        $localHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localFile).Hash.ToLowerInvariant()
        $transferTimeout = Get-TransferTimeoutSeconds $sizeBytes

        $push = Invoke-ProbeScp -Source $localFile -Destination "${User}@${SshHost}:$remoteFile" -TimeoutSeconds $transferTimeout
        if ($push.TimedOut) { throw "Windows-to-Vast SCP timed out after $transferTimeout seconds" }
        if ($push.ExitCode -ne 0) { throw "Windows-to-Vast SCP failed: $($push.Stderr)" }
        $state.PushSeconds = $push.ElapsedSeconds
        $state.PushBytesPerSecond = $sizeBytes / $push.ElapsedSeconds
        $state.PushMbps = $state.PushBytesPerSecond * 8.0 / 1000000.0

        $diskCommand = "set -eu; remote_hash=`$(sha256sum '$remoteFile' | cut -d ' ' -f 1); start=`$(date +%s%N); cp '$remoteFile' '$remoteDirectory/disk-copy.bin'; sync '$remoteDirectory/disk-copy.bin'; stop=`$(date +%s%N); rm -f '$remoteDirectory/disk-copy.bin'; echo PROBE_DISK `$remote_hash `$start `$stop"
        $disk = Invoke-ProbeSsh -RemoteCommand $diskCommand -TimeoutSeconds 30
        if ($disk.TimedOut -or $disk.ExitCode -ne 0 -or $disk.Stdout -notmatch 'PROBE_DISK\s+([a-fA-F0-9]{64})\s+(\d+)\s+(\d+)') {
            throw "remote integrity/disk sanity check failed: $(($disk.Stderr + ' ' + $disk.Stdout).Trim())"
        }
        $remoteHash = $Matches[1].ToLowerInvariant()
        $diskSeconds = ([double]$Matches[3] - [double]$Matches[2]) / 1000000000.0
        if ($diskSeconds -gt 0) { $state.RemoteDiskMiBPerSecond = ($sizeBytes / 1MB) / $diskSeconds }

        $pull = Invoke-ProbeScp -Source "${User}@${SshHost}:$remoteFile" -Destination $returnedFile -TimeoutSeconds $transferTimeout
        if ($pull.TimedOut) { throw "Vast-to-Windows SCP timed out after $transferTimeout seconds" }
        if ($pull.ExitCode -ne 0) { throw "Vast-to-Windows SCP failed: $($pull.Stderr)" }
        $state.PullSeconds = $pull.ElapsedSeconds
        $state.PullBytesPerSecond = $sizeBytes / $pull.ElapsedSeconds
        $state.PullMbps = $state.PullBytesPerSecond * 8.0 / 1000000.0
        $state.ProjectedCheckpointSeconds = Get-ProjectedCheckpointSeconds $state.PullBytesPerSecond
        $state.CheckpointPassed = $state.ProjectedCheckpointSeconds -le $MaxCheckpointPullSeconds

        $returnedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $returnedFile).Hash.ToLowerInvariant()
        $state.IntegrityChecked = $true
        $state.IntegrityPassed = ($localHash -eq $remoteHash -and $localHash -eq $returnedHash)
        if (-not $state.IntegrityPassed) { throw "SHA-256 mismatch across the SSH round trip" }

        $state.RemoteInternet = Get-RemoteCurlMeasurement -RemoteFile $remoteFile -RemoteUploadFile $remoteUploadFile
        $state.RemoteDownloadMbps = $state.RemoteInternet.DownloadBytesPerSecond * 8.0 / 1000000.0
        $state.RemoteUploadMbps = $state.RemoteInternet.UploadBytesPerSecond * 8.0 / 1000000.0
        $state.RemoteDownloadPassed = $state.RemoteDownloadMbps -ge $MinRemoteDownloadMbps
        $state.RemoteUploadPassed = $state.RemoteUploadMbps -ge $MinRemoteUploadMbps

        $thresholdsPassed = $state.RemoteDownloadPassed -and $state.RemoteUploadPassed -and $state.CheckpointPassed
        if ($Confirm200MiB -and $thresholdsPassed) {
            $state.FullConfirmation = Invoke-FullConfirmation -LocalDirectory $localDirectory -RemoteDirectory $remoteDirectory
            if ($state.FullConfirmation.Status -ne 'PASS') { throw "optional 200 MiB confirmation failed" }
        }
        elseif ($Confirm200MiB) {
            $state.FullConfirmation = [pscustomobject]@{ Status = "SKIPPED (short probe failed)"; UploadMbps = 0.0; PullMbps = 0.0 }
        }

        if ($CheckWsl) { $state.WslCheck = Invoke-WslHandoffCheck }
    }
    catch {
        $failureReason = $_.Exception.Message
    }
    finally {
        if ($remoteCreated) {
            try {
                $cleanup = Invoke-ProbeSsh -RemoteCommand "rm -rf -- '$remoteDirectory'" -TimeoutSeconds 15
                if ($cleanup.TimedOut -or $cleanup.ExitCode -ne 0) { $cleanupPassed = $false }
            }
            catch { $cleanupPassed = $false }
        }
        if (Test-Path -LiteralPath $localDirectory) {
            try { Remove-Item -LiteralPath $localDirectory -Recurse -Force -ErrorAction Stop }
            catch { $cleanupPassed = $false }
        }
        $totalTimer.Stop()
    }

    $corePassed = (-not $failureReason) -and $state.IntegrityPassed -and $state.PullBytesPerSecond -gt 0
    $thresholdsPassed = $state.RemoteDownloadPassed -and $state.RemoteUploadPassed -and $state.CheckpointPassed
    $state.CleanupPassed = $cleanupPassed
    $state.FailureReason = $failureReason
    $state.TotalSeconds = $totalTimer.Elapsed.TotalSeconds
    $state.Verdict = Get-OverallVerdict -CorePassed $corePassed -ThresholdsPassed $thresholdsPassed -CleanupPassed $cleanupPassed -EndpointKind $endpointKind
    Write-ProbeReport $state
    if ($state.Verdict -eq 'FAIL') { return 3 }
    return 0
}

if ($InternalNoRun) { return }

try {
    $probeOutput = @(Invoke-VastQualification)
    $exitCode = [int]$probeOutput[-1]
    if ($probeOutput.Count -gt 1) {
        $probeOutput[0..($probeOutput.Count - 2)] | ForEach-Object { Write-Output $_ }
    }
    exit $exitCode
}
catch {
    [Console]::Error.WriteLine("ERROR: BAD_CONFIG - $($_.Exception.Message)")
    exit 2
}
