# Agent Client Protocol

Run `forge acp` for newline-delimited JSON-RPC on stdin and stdout. Protocol
messages are the only stdout output. The initialize response identifies the
application as `forge`, titled `Forge`.

| Method | Behavior |
|---|---|
| `initialize` | Negotiate protocol version and capabilities |
| `authenticate` | Acknowledge local credential use |
| `session/new` | Create a durable session bound to an absolute workspace |
| `session/load` | Restore a saved session and replay its transcript |
| `session/prompt` | Run a turn and stream session updates |
| `session/cancel` | Cancel the current turn; notification receives no response |

Saved sessions use the active Forge data root. Loading rejects an unavailable,
foreign or replaced workspace. The current Unix implementation binds the
workspace's canonical path, device and inode. Client-supplied stdio MCP rosters
are added to the configured roster for that session. The local Unix RPC,
code-mode and saved-session fork acceptance gate passes. Final-release
integration remains required.

## Session MCP setup

`session/new` and `session/load` validate the complete supplied roster before
launching a child. Each roster permits at most eight servers and 256 KiB of
arguments and environment entries. Server names must be unique, including
against configured global servers. Executables must be absolute files.
Unsupported HTTP and SSE transports receive an invalid-parameters response.
Malformed settings stop startup with a categorical error. A missing settings
file still uses defaults; invalid files cannot discard configured restrictions.

Supplied arguments and environment values are literal. Each child starts in
the session workspace. Saved session records do not serialize these configs.
Sessions have separate inventory caches and reuse existing global connections.

The connection permits eight additional stdio transports across its sessions.
Admission remains charged until owned transport cleanup completes. Setup has a
ten-second deadline and stops on disconnect. Empty-roster metadata requests
already accepted before input EOF still finish without starting a child.
Shutdown joins session transports and waits for admission to drain under a
three-second deadline. On Unix, owned child process groups are terminated and
confirmed gone before their admission is reclaimed. Windows process-tree
cleanup has not been verified.

Loading reserves the session ID before reading its record or starting MCP setup.
Concurrent loads cannot replace a pending owner. Failed and cancelled loads
release their reservation. Connected rosters retain setup cleanup ownership
through persistence, replay and registry activation. Notification refresh starts
before catalog readiness is awaited; cancelled refresh cannot keep a startup
child alive through a retained manager handle.

## Outbound request ownership

Each connection permits at most 64 outstanding outbound requests. Further
requests fail locally with an explicit limit error. Pending entries are
released after a response, request cancellation, serialization or write
failure, and reader shutdown. Reader ownership begins before the future is
first polled. Dropping an unstarted reader also releases its waiters.

Client EOF and reader failure reject later outbound requests before emission.
EOF interrupts a request blocked on output. An interrupted or failed JSON
frame makes subsequent output unavailable; another frame cannot reuse a
potentially partial line.

## Input and dispatch limits

Each JSON line is limited to 32 MiB, including its newline. Input is checked
before parsing for at most 100,000 structural units and 128 nesting levels.
Punctuation inside quoted strings does not count toward those limits.

Retained inbound messages share a 128 MiB accounting budget. The charge is
twice the wire bytes, 128 bytes per structural unit, and a 1 KiB floor. Queued
and active handlers keep their charge until completion. Replies waiting in
outbound-request channels keep their charge until consumption. One bounded
parser frame remains independent of that retained budget.

The input queue holds eight messages. At most eight request handlers run at
once, and completed handlers are reaped. Excess requests receive an error.
Cancellation notifications do not wait for a handler slot. Queue or budget
exhaustion closes the connection. Frame writes, including writer-lock wait,
have a ten-second deadline.

Output buffers grow in bounded steps rather than reallocating for every small
serializer write. Capacity remains within the frame limit.

On disconnect, an active prompt signals query cancellation and awaits its
cleanup before persisting the transcript. A prompt waiting for its session
gate returns cancelled without starting inference or appending its user turn.

## Verification and remaining work

The U013B scoped gate passes 62 ACP unit tests, 30 CLI unit tests and four
real executable smoke tests. One existing executable-upgrade helper test is
intentionally ignored and invoked by its parent test. Workspace check and
strict all-target Clippy pass at source fingerprint
`0b3af7123fc0d22a71e7f5e17d964400ba69f444a4839b7b18e34e44588f1f51`.
Evidence is recorded in
`.omx/state/forge/verification/u013-acp-bounded-transport/audit-r2.json`.

The executable smoke tests cover identity, session creation, restart/load
with workspace binding, and silent cancellation notifications. Eleven
initial connection regressions cover request ownership, disconnect, saturation
and partial-frame failures. The bounded transport tests additionally verify
frame boundaries, JSON complexity, budget ownership, reply routing at queue
capacity, handler limits, responsive cancellation and write deadlines.
Baseline failures are retained with the green gates.

Four additional real-executable cases pass on an immutable debug binary.
Disconnect cancels active inference and prevents a queued prompt from starting.
Cancellation remains responsive with all eight handlers occupied. Oversized
and dense input fails before provider traffic. The local synthetic provider
stays held while the application completes disconnect cleanup; no live-model
quality or latency claim follows from these fixtures.

Final validation used one Cargo job and one test thread. Peak sampled job RSS
was 742,162,432 bytes with at least 46% host headroom in the final gates.
Across the intermediate code gates, the largest sample was 1,448,165,376
bytes with 45% minimum headroom. No Terminal.app windows
or additional agents were used.

## Streaming and permission ownership

Each prompt owns an event channel with at most 128 retained events, an 8 MiB
accounting budget, and a 4 MiB maximum charge per event. The active write keeps
its charge. Admission counts string capacity, projected copies, JSON escaping
and parsed tool JSON structure. Events already ignored by ACP are filtered.
The event channel is passed explicitly to the query call; shared query
configuration does not carry it into child or scheduled work.

Event pressure, receiver loss and forwarding failure cancel the actual query.
The adapter persists its final transcript before returning a categorical
delivery error. It does not replay uncertain output. Normal channel EOF does
not cancel otherwise successful background work. Forwarding no longer retains
an unused map of active tool metadata.

A prompt permits 16 active permission forwards and 64 queued permissions.
Producers enforce queue admission under the store lock. Closing the store
denies retained decisions and rejects late or cancelled producers. The drainer
owns and joins every forwarding task, including when stopped before first poll.
The next serialized prompt reopens its store. Normal teardown cancels only the
drainer's child token.

U013C source verification passes 805 affected unit/smoke tests and two doctests,
workspace check and strict all-target Clippy at fingerprint
`892d0f4a6372bedbc9e91c5464c688881b3a99bd87f777921dee29c07c985edb`.
It includes controlled stalled/broken writers, structural amplification,
permission cancellation and producer admission. A deterministic regression
also covers completion between reading background cleanup state and waiting
for its notification. The waiter now registers before reading state.
Evidence is recorded in
`.omx/state/forge/verification/u013-acp-stream-ownership/foundation-r1.json`.
These source gates used serial guarded jobs. The final test run peaked at
1,267,302,400 sampled RSS bytes with at least 42% host headroom.

The immutable debug binary
`8f98a0f15fb869378cbc13b7c9e3897db6dbe27f7cb0afa38e643b073d192123`
passes eight real RPC fixtures: the four U013B transport cases and four
ownership cases. Cancelling a held approval denies its tool, ignores the late
reply and permits a fresh approval next turn. Oversized events and stalled
stdout return categorical delivery errors after a completed Write; another
prompt recovers without replaying the mutation. Broken stdout persists the
paired Write result while the local provider's stream tail remains held.

The previous U013B executable fails the approval-cancellation, event-size and
stalled-output fixtures. Its broken-output cleanup already passes and remains
a regression check. Each current fixture asserts exact provider call counts,
paired persisted results and terminal owned processes. The provider and
application state are isolated loopback fixtures. These results support no
live-model quality or release-speed claim. The build sampled at most
347,160,576 RSS bytes with 50% minimum host headroom.
Executable evidence is recorded in
`.omx/state/forge/verification/u013-acp-stream-ownership/runtime-audit-r1.json`.

At the U013C checkpoint, RPC and code-mode workflows, saved-session forks and
the executable matrix remained open. Their final U013E evidence follows below.
These accounting budgets are not OS reservations or a bound on the entire
server's memory use.

The U013D routing foundation passes 197 affected tests, workspace check and
strict all-target Clippy at source `5292973f`. The immutable debug binary
`f90387e7` passes five roster probes and the eight prior RPC regressions.
The roster probes cover new/load cwd and literal env, pre-launch rejection of
duplicate names and unsupported HTTP, and Unix descendant cleanup. Retained
failures show the descendant-cleanup repair and empty-roster EOF regression.
Evidence is in
`.omx/state/forge/verification/u013-acp-session-routing/foundation-r1.json`.

Matching gates used one serial guarded job at a time, no Terminal.app windows
and no new agents. Their maximum sampled job RSS was 866,631,680 bytes with at
least 73% host headroom. Short fixtures can finish between five-second samples;
their initial sample does not establish a runtime peak. This foundation does
not complete U01.3 or the remaining MCP acceptance matrix.

U013D session MCP acceptance now passes 198 affected tests, workspace check and
strict all-target Clippy at source `f98608bb`. The immutable debug binary
`249142d3` passes 29 actual cases: 18 roster/setup cases, three concurrent tool
workflows and the eight earlier RPC regressions. Its audit is
`.omx/state/forge/verification/u013-acp-session-routing/runtime-audit-r1.json`.

Two active sessions use the same private server and tool names with different
workspaces, literal environment values and image payloads. Each session calls
its own native connection; the configured global connection initializes once.
Selectors hide the denied tool and reject its attempted invocation. ACP approvals
belong to the correct session; rejecting one private call prevents that remote
call. Typed image results and EvidenceRead restoration preserve exact payloads.
A foreign session's evidence ID returns an error without its text or image.
Exact remote call counts also verify that restoring evidence does not replay MCP.

Startup cases cover duplicate load admission, failed-load retry, invalid/large
rosters, eight-connection exhaustion, malformed settings, persistence failure,
held initialization and catalog refresh, deadlines, disconnect and recovery.
Owned processes are terminal after each case. Matching final gates sampled at
most 797,851,648 RSS bytes with at least 73% host headroom; intermediate source
gates sampled at most 1,760,608,256 bytes with at least 74% headroom. All work
used serial guarded jobs, no Terminal.app windows and no new agents. The short
fixture sampling caveat and Unix platform limit remain applicable. This U013D
checkpoint left U01.3 and the full fork's final acceptance gate open.

## U01.3 workflow acceptance

The immutable debug executable `35d80ffa` at source `89bbf380` passes 47 actual
cases: seven saved-fork/resume cases, six code-mode cases, 21 session MCP cases,
eight transport/streaming cases and five print/TUI cases. Full fingerprints,
case results and retained failures are recorded in
`.omx/state/forge/verification/u013-modes-session-fork/runtime-audit-r2.json`
(SHA `b2aa213f`). U01.3 is verified for this local Unix scope. The final immutable
release and complete preserved-workflow matrix remain required by U01.7,
B01.9 and F01.1.

Persistent Python and JavaScript calls retain state and execute dependent
mutations in order. Different sessions remain isolated even in the same
workspace. Denied calls start no interpreter. Cancellation and disconnect reap
the interpreter and its delayed-writing descendant; a fresh turn recovers.

Saved forks retain the exact complete transcript prefix and parent lineage.
Restart, optional `--resume` and `--continue` select the saved branch without
replaying committed tools. Missing continuation fails before inference.
Malformed indices and boundaries with incomplete tool pairs fail before a
branch is saved. The TUI displays the fork error. Validation borrows the prefix
before copying it and does not silently repair history.

Final-source verification passes 737 core, commands, ACP and CLI tests, workspace
check and strict all-target Clippy. One existing helper test is ignored and
invoked by its parent. The full-workspace checkpoint passed 2,533 tests at
`1be2db7f`; the only later source change displays CLI command errors. The audit
reconstructs that checkpoint fingerprint by reversing the exact display block
in memory. It does not report the earlier full suite as a final-source run.

Jobs ran serially with one Cargo job and one test thread, no Terminal.app windows
and no new agents. Current-source gates sampled at most 1,237,024,768 RSS bytes
with at least 64% host headroom. The earlier full-workspace checkpoint sampled
at most 1,674,821,632 bytes with at least 65% headroom. Short fixture samples do
not establish runtime peaks. These local debug loopbacks establish no
live-model quality, release-speed or additional-platform claim.
