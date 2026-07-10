---
name: feedback-investigate-broken-claims
description: When the user says something is hanging/broken after a plausible-looking wait, investigate ground truth immediately rather than attributing it to known flakiness
metadata:
  type: feedback
---

When the user flags something as hanging or broken (e.g. "its been another 3 minutes, something is
broken") after I've already offered a flakiness-shaped explanation for slowness, stop and check
ground truth directly (actual remote/process state) rather than defaulting to "that's expected,
this environment is known to be slow/flaky."

**Why:** During the Vast.ai integration (2026-07-10), a polling script (`vast_wait.sh`) appeared to
hang. The environment genuinely IS flaky (every ssh call eats a reconnect tax), so it was tempting
to explain the delay away as more of the same. The user pushed back ("something is broken") instead
of accepting that framing — and it turned out to be a real logic bug (a login banner getting
captured via `2>&1` and contaminating a string comparison, `[ "$alive" = DEAD ]`, so it never
matched even though the job had finished minutes earlier). If I had kept attributing it to
flakiness, the bug would have shipped into infrastructure other sessions rely on.

**How to apply:** Known flakiness in an environment (retries, reconnects, variable latency) is a
real thing and a valid first hypothesis — but it's a hypothesis, not a license to stop looking. The
moment the user says a wait feels wrong, verify against ground truth (check the actual remote
process/file state directly, not just "it should be done by now" reasoning) before re-asserting the
flakiness explanation. This generalizes beyond this session: distinguish "this specific known
symptom, already characterized" from "an open-ended wait that's gone on longer than any established
baseline" — the latter deserves direct investigation, not a shrug.
