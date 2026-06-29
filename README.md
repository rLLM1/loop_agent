# Agent Review Loop

Interactive orchestration for two coding agents:

1. Claude Code implements a requested change and asks the user questions when
   input is required.
2. The orchestrator commits Claude's changes.
3. Codex reviews the cumulative implementation in read-only mode.
4. Claude assesses and fixes agreed findings.
5. The loop continues until Codex reports no issues or `--max-rounds` is
   reached.

## Requirements

- Python 3.9+
- Git
- Authenticated `claude` and `codex` CLIs on `PATH`

## Usage

```bash
cd /path/to/repository
python ~/Code/agent_loop/agent_review_loop.py \
  "Implement the requested feature and test it"
```

Add a document containing a list of known problems:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  "Fix the documented problems and test the result" \
  --problems-file /path/to/problems.md
```

Process a long plan as a checkbox task queue:

```markdown
- [ ] U0: scaffolding
  Implement only U0.
- [ ] U1: numerical path
  Implement only U1.
```

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  "Implement this plan one unit at a time" \
  --problems-file /path/to/work_queue.md \
  --work-queue \
  --run-key frozen-encoding
```

`--work-queue` selects the first unchecked Markdown checkbox item not already
done in the saved run state, then continues through the remaining unchecked
items in the same invocation. For each task it starts a fresh Claude session,
commits the task, runs Codex review on that task's commit range, and repeats
fresh Claude fix sessions until Codex reports clean. The plan file is read once
at run start and is not edited; the durable state stores compact parsed tasks,
not the full problem document. If the process is interrupted mid-task, the same
command resumes the saved run because `--run-key` points to the existing
incomplete state.

Every checkpoint is saved under `~/Code/agent_loop/run_logs/state/`. Resume a
specific run or the newest incomplete run:

```bash
python ~/Code/agent_loop/agent_review_loop.py --resume RUN_ID
python ~/Code/agent_loop/agent_review_loop.py --resume-last
```

Each program invocation writes one Markdown transcript to
`~/Code/agent_loop/run_logs/`. It records the request, problem-file path,
selected/current task text, agent questions/conclusions, user answers, review
findings, commits, and final status. It does not record agent thinking, shell
commands, raw tool output, or full problem documents in work-queue mode.

Claude/Codex subprocess stdout and stderr are streamed to raw logs under
`~/Code/agent_loop/run_logs/raw/`. Raw log filenames include the date/time and a
label derived from the problems-file name, falling back to the prompt when no
problems file is provided. The default heartbeat prints every 30 seconds with
the active raw log path, and each individual agent call times out after 60
minutes by default.

```bash
tail -f ~/Code/agent_loop/run_logs/raw/*.log
```

Useful options:

```text
--repo PATH
--problems-file PATH
--work-queue
--run-key KEY
--new-run
--break-lock
--resume RUN_ID_OR_STATE_PATH
--resume-last
--agent1-model MODEL
--agent2-model MODEL
--agent1-permission-mode MODE
--agent1-session-mode fresh|resume
--max-rounds N
--heartbeat-seconds N
--agent-timeout-minutes N
```

The target repository must have no tracked modifications when starting a new
run. Existing untracked files are not committed, even if Claude edits them.
The script creates commits but never pushes.

`--run-key` also adds a lock file under `run_logs/state/by_key/` so the same
key cannot run concurrently in two terminals. Use `--break-lock` only when the
previous process is definitely gone. If a keyed non-queue run has already
finished, rerunning the same key stops; use `--new-run` to intentionally start
over. In work-queue mode, rerunning a completed key reports the queue complete;
use `--new-run` to intentionally parse and run the queue again.

By default Agent 1 uses `--agent1-session-mode fresh`: every implement/fix call
starts a new Claude session and receives only the compact current task, state
summary, and latest review findings. This keeps token usage bounded. Use
`--agent1-session-mode resume` only when you explicitly want the legacy long
Claude conversation. Before every Claude call the orchestrator prints prompt
character count, estimated token count, and session mode.

## Concurrency and Resume Safety

The subprocess raw-log writer is thread-safe within one program process, but the
orchestrator is not a multi-process scheduler. Do not run two loop processes
against the same git worktree or resume the same saved run from two terminals.
Those cases can race on repo edits, commits, state JSON, and the Claude session.

Safe pattern:

```text
Terminal A -> repo/worktree A -> run A
Terminal B -> repo/worktree B -> run B
```

Avoid:

```text
Terminal A -> /path/to/repo
Terminal B -> /path/to/repo
```

Also avoid running this twice at the same time:

```bash
python ~/Code/agent_loop/agent_review_loop.py --resume-last
```

For parallel experiments, create separate git worktrees:

```bash
git worktree add ../repo-agent-2 feature/branch-name
```

Resume is checkpoint-based. The loop saves state before and after major
boundaries: Claude calls, user questions, commits, Codex reviews, completion,
errors, and interruption. If the process stops while waiting for user input,
after a review, or before/after a commit, resume should continue cleanly:

```bash
python ~/Code/agent_loop/agent_review_loop.py --resume-last
python ~/Code/agent_loop/agent_review_loop.py --resume RUN_ID
```

If interruption happens during an external Claude/Codex call, resume starts from
the last saved checkpoint. If the saved phase is about to call Claude but the
working tree already has tracked modifications, resume stops and asks you to
inspect the tree; that usually means Claude edited files but did not return
structured JSON. This avoids blindly running a fresh Claude session on top of
unknown partial edits.

## Test

```bash
cd ~/Code/agent_loop
pytest -q
```
