# One-command Vast.ai connection and network qualification

## Goal
Make it cheap and reliable for a coding agent to qualify a newly rented Vast.ai instance before we invest
time moving datasets, environments, or checkpoints onto it. The agent will usually receive only the SSH
host or IP and port. From that, it should be able to establish the connection, run a fast network probe,
and report whether the machine is practical for training work.

This addresses prior rentals where network or file transfer was so slow or unreliable that substantial
time and money were lost before the problem became clear.

## User experience
- Provide one obvious command that accepts a host/IP and SSH port, with optional user and identity-key
  overrides.
- Require no remote repository clone, Vast API key, `cluster.env` edit, or pre-opened ControlMaster socket.
- Make direct SSH the bootstrap transport. Vast.ai is not subject to the ferranti/galvani wrapper-only
  security policy.
- Produce a concise pass/warn/fail summary suitable for a coding agent to paste back to the user, followed
  by the raw measurements and any actionable diagnosis.
- Finish the short probe in roughly two minutes on a healthy instance and use bounded timeouts so a broken
  or extremely slow machine fails quickly rather than hanging.

## Windows and WSL policy
- The primary entry point should run natively from Windows PowerShell using Windows OpenSSH. This is the
  lowest-friction path when a coding agent is handed only a host and port, and it uses the SSH key already
  stored in the Windows user profile.
- Do not require WSL or copy a private key into WSL merely to qualify a rental.
- Offer an optional WSL handoff check for instances that will subsequently use this repository's existing
  Vast wrapper workflow. That check should validate the WSL-side key and connection separately, because
  Windows and WSL have distinct SSH configuration, key paths, known-host state, and socket namespaces.
- Clearly label whether each result came from Windows-native direct SSH, WSL direct SSH, or a Vast proxy
  endpoint. Do not present measurements from one transport as proof that another transport is healthy.
- After a pass, report the connection/configuration information needed to continue either with direct
  Windows SSH/SCP or with the WSL-based repository wrappers.

## Probe coverage
- Verify non-interactive SSH authentication, basic command execution, and connection latency.
- Identify whether the supplied endpoint is a direct connection or a Vast proxy when this can be
  determined. Warn that proxy SSH may be the transfer bottleneck and prefer a direct endpoint when Vast
  exposes one.
- Measure the instance's outbound internet download and upload throughput with short, bounded tests.
- Measure file transfer in both directions between the local machine and the instance over the supplied
  SSH path.
- Use checkpoint-like, non-trivially-compressible data so SSH compression or sparse/zero-filled files
  cannot create misleading throughput numbers.
- Verify transferred data integrity and distinguish network throughput from obvious local or remote disk
  failures.
- Always remove temporary local and remote probe data, including after a timeout or failed test.

## Fast test and checkpoint projection
The default path should use a small bounded transfer to estimate real SSH file-transfer throughput. From
the measured Vast-to-local rate, report the projected time to retrieve a 200 MB checkpoint. Provide an
optional full 200 MB round-trip confirmation after the short probe passes; do not force a potentially long
full transfer on an already failing connection.

Report at least:
- Remote internet download and upload rates.
- Local-to-Vast and Vast-to-local SSH transfer rates.
- Projected duration for pulling a 200 MB checkpoint to the local machine.
- Whether integrity verification passed.
- Total probe duration and any timeout or endpoint limitation encountered.

## Default qualification thresholds
- **Remote download:** pass at 100 Mbps or above.
- **Remote internet upload:** pass at 20 Mbps or above.
- **Checkpoint retrieval:** pass when the measured Vast-to-local SSH rate projects a 200 MB checkpoint
  pull in five minutes or less.

Always show raw measurements and the projected checkpoint duration, even when a threshold fails. Allow
threshold overrides for unusual workloads, but keep these defaults so different coding agents reach the
same initial rental verdict.

Treat authentication failure, data corruption, an unbounded/hung transfer, or inability to complete the
Vast-to-local test as a failure. A proxy-limited result should be clearly diagnosed rather than silently
blamed on the instance's general internet connection.

## Documentation integration
- Add the probe to the Vast.ai section of the cluster guide as the first step after receiving a new
  instance's connection details.
- Explain the direct-versus-proxy distinction and the Windows-native-versus-WSL transport distinction.
- Keep the existing rule clear: ferranti and galvani remain wrapper-only, while Vast permits direct SSH,
  SCP, and rsync.
- Base connection guidance on Vast.ai's official SSH, Windows SSH, and data-movement documentation.

## Done means
- A coding agent given only a working Vast SSH host/IP and port can launch the qualification with one
  obvious Windows-native command.
- A healthy instance completes the default probe in roughly two minutes and receives a clear verdict.
- Slow, broken, proxy-limited, and authentication-failing cases terminate predictably with useful
  diagnostics.
- Both internet bandwidth and actual bidirectional local file-transfer performance are measured, with
  integrity verification and safe cleanup.
- The output includes a trustworthy projected time for retrieving a 200 MB checkpoint and supports an
  optional real 200 MB confirmation.
- The optional WSL handoff check makes it clear whether the existing Vast wrappers will work, without
  making WSL a prerequisite for initial qualification.
- The cluster documentation describes the new workflow and no longer conflates the academic-cluster
  security policy with Vast.ai access.
- The probe is validated against a live newly rented Vast instance before the task is closed.

## Progress (2026-07-13)

- Implemented `scripts/qualify_vast_instance.ps1`: Windows OpenSSH bootstrap, bounded auth/latency,
  Cloudflare-backed remote internet tests, 16 MiB random bidirectional SCP with SHA-256, remote disk
  sanity, 200 MiB projection/optional confirmation, cleanup, endpoint diagnosis, and optional separate
  WSL handoff.
- Added `scripts/tests/test_qualify_vast_instance.ps1` (17 deterministic checks; passing locally) and
  integrated the workflow into `HOWTO/cluster.md` / `scripts/README.md`.
- Live validation used a fresh Vast rental and found/fixed two real harness issues: report output was
  swallowed by PowerShell's success-output pipeline, and Windows OpenSSH 9's default SFTP-mode `scp` was
  closed by the Vast image (the probe now forces Vast's documented legacy SCP transport with `-O`).
- **Direct `154.64.230.67:23445`: FAIL** — remote download 35.2 Mbps; remote upload session forcibly
  closed; Windows→Vast 8.2 Mbps; Vast→Windows 8.6 Mbps; projected 200 MiB pull 3.2 min; SHA-256 PASS;
  remote disk 426.9 MiB/s; cleanup PASS.
- **Proxy `ssh7.vast.ai:39139`: FAIL** — remote download 37.3 Mbps; upload 20.7 Mbps;
  Windows→Vast 10.0 Mbps; Vast→Windows 11.6 Mbps; projected pull 2.4 min; SHA-256 PASS; remote disk
  666.5 MiB/s; cleanup PASS. The similar low remote-download result diagnoses the rental uplink, while
  the direct path is additionally less stable. The optional WSL handoff authenticated independently.
- Post-test remote inspection found no leftover probe directories. The full 200 MiB confirmation was
  correctly not run because the short probe failed its download threshold.

**Result:** Windows-native probe shipped with 20 passing deterministic tests and live direct/proxy/WSL
validation; it correctly rejected this slow/unreliable rental and cleaned up all local/remote test data.
