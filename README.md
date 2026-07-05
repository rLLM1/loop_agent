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

## Planned Mode Model

The next version should make the loop modes explicit instead of overloading one
prompt shape for planning, execution, review, and resume.

### Mode 1: Discussion

Discussion mode produces a task breakdown, issue list, or implementation plan.
It does not edit code. Agent A drafts the plan, Agent B reviews it, Agent A
revises or pushes back, and the loop continues until there is solid agreement or
the configured round limit is reached.

Discussion mode always writes a task-list artifact to disk. It may take only a
prompt, or it may take an optional `--context-file`. The context file can be one
issue, a list of issues, an unstructured plan, or a poorly formatted task list.
Regardless of input shape, the output is a normalized task list that can become
the checkpoint for execution mode.

Example intended command shape:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode discussion \
  --prompt "Review Rigel and produce a prioritized task list for code improvements." \
  --context-file /Users/pro/Code/Mds/loop_findings_to_improve_rigel.md \
  --task-list-file /Users/pro/Code/Mds/rigel_code_improvement_tasks.md
```

### Mode 2: Execution

Execution mode takes `--task-list-file`, validates that it contains executable
tasks, and iterates through them one by one. Each task becomes the checkpoint.
For each task, the orchestrator records `task_base_commit = HEAD`, starts a
fresh Agent A session, asks Agent A to implement only that task, commits the
work, then asks Agent B to review `task_base_commit..HEAD`. Agent A and Agent B
continue on that task until Agent B has no actionable feedback, the task is
skipped/deferred, or the loop blocks.

Agent A is allowed to push back on Agent B feedback. Agent A should classify
each finding as `agree`, `disagree`, `defer`, or `needs_user`, with a concise
reason and action. Blindly applying every review comment is not allowed.

Default behavior should be:

```text
--commit-policy squash-per-task
--no-push
--max-dispute-rounds 5
```

`squash-per-task` means every task ends as one clean focused commit. Intermediate
fix commits may exist during the task, but the final task state should be
squashed or amended before moving to the next task.

Example intended command shape:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode execute \
  --task-list-file /Users/pro/Code/Mds/rigel_code_improvement_tasks.md \
  --branch rigel_code_improvement_07_04
```

### Mode 3: Discussion Then Execution

Discussion-execution mode first runs discussion mode to create or refresh the
task list, then runs execution mode over that task list. `--auto-execute` should
default to enabled for this mode because the mode name already implies both
phases. A future `--approval-gate` flag can force a pause after task-list
generation when human approval is needed.

Example intended command shape:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode discuss-execute \
  --prompt "Turn this findings file into tasks and implement them one by one." \
  --context-file /Users/pro/Code/Mds/loop_findings_to_improve_rigel.md \
  --task-list-file /Users/pro/Code/Mds/rigel_code_improvement_tasks.md \
  --branch rigel_code_improvement_07_04
```

## Task List Contract

Execution mode should use task lists as the durable checkpoint format. A valid
task should eventually contain:

```text
id
title
problem
scope
acceptance criteria
test command
out of scope
risk/dependencies
status
```

Discussion mode is responsible for normalizing issue-shaped input into
task-shaped output. Execution mode should still validate the task list before
iteration. If an item is not executable, it should either normalize it locally,
skip it with a reason, or stop with `[USER TODO]` when a product/design decision
is required.

Recommended task statuses:

```text
todo
in_progress
done
skipped
deferred
blocked
needs_user
```

When the loop skips the current task, it should document the progress and
disagreement in the worklog using a visible marker:

```text
[USER TODO] T7 skipped: Agent A and Agent B disagree about whether this changes
the public precompute format. User decision required before implementation.
```

## Stop, Skip, And Dispute Rules

The loop should stop the whole run and ask the user when continuing could corrupt
the branch, implement the wrong design, or waste a large amount of work.

Must stop:

```text
Repo is dirty with unrelated user changes.
Branch creation or switching fails.
Merge conflict requires a product/design choice.
Task requires secrets, credentials, network, or external service access.
Task contradicts another task or project direction.
Agent A and Agent B disagree after --max-dispute-rounds in non-queue mode.
Tests fail in a way that may indicate a real regression and agents cannot isolate it.
Agent B cannot inspect the repo at all.
Agent A cannot produce valid structured output after retry.
Commit fails or git history state is unsafe.
Task requires broad refactor outside allowed scope.
```

Can skip to the next task:

```text
Task depends on a prior task that was skipped or failed.
Task is already fixed in current code.
Task duplicates an earlier task.
Task is no longer applicable after code inspection.
Task is too broad but can be converted into smaller subtasks.
Task needs an optional dependency unavailable locally and no code change is required.
Task is low-priority tuning/design, not a correctness bug.
Task is outside current branch scope.
Current work-queue task reaches --max-dispute-rounds; mark it skipped with
[USER TODO], drop that task's branch changes by resetting to task_base_commit,
and continue to the next task.
```

Can continue with a caveat:

```text
Full pytest was not run because of time.
GPU smoke test could not run on the local machine.
Optional dependency is missing for a non-targeted test.
Agent B found no code issue but noted unrun tests.
Formatting tool is unavailable and files are already style-consistent.
```

`--stop-on-dispute` should trigger when all of these are true:

```text
Agent B raises an actionable issue.
Agent A explicitly disagrees or refuses to change it.
Agent B reviews Agent A's disagreement and still says it is a real issue.
The disagreement survives --max-dispute-rounds rounds.
The issue is not merely style or preference.
```

Stop-worthy disputes include suspected label leakage, cache/schema versioning
breakage, public format changes, or task-scope violations. Non-blocking disputes
include variable names, optional docs, or full-suite test caveats when targeted
tests pass.

## Loop Improvement Plan

Implement the three-mode design in small changes:

1. Add explicit CLI modes: `--mode discussion`, `--mode execute`, and
   `--mode discuss-execute`. Keep old positional-prompt behavior as a
   compatibility alias.
2. Rename the broad document input to `--context-file` for discussion modes.
   Keep `--problems-file` as a deprecated alias.
3. Add `--task-list-file` for execution checkpoints. Execution mode should not
   consume arbitrary context files directly.
4. Add a task-list parser that accepts normalized task blocks and legacy
   checkbox tasks. Discussion mode should output normalized task blocks.
5. Add task-list validation before execution. Invalid tasks should be
   normalized, skipped with a reason, or marked `[USER TODO]`.
6. Make `--commit-policy squash-per-task` the default. Support `per-round` and
   `no-commit` as explicit alternatives.
7. Make `--no-push` the default and require an explicit future `--push` flag for
   any remote write.
8. Make `--auto-execute` default to enabled only for `--mode discuss-execute`.
9. Add `--max-dispute-rounds`, default `5`, and enforce it separately from
   generic review-round limits. In work-queue execution, reaching the dispute
   limit should skip the current task, write `[USER TODO]` to the worklog,
   reset back to that task's base commit so unresolved code does not contaminate
   later tasks, and continue to the next task. In non-queue mode, reaching the
   limit should stop the run.
10. Persist task checkpoints in state: current task id, task base commit, latest
    head, review rounds, Agent A decisions, Agent B findings, status, and
    worklog path.
11. Start fresh Agent A sessions per task and preferably per review-fix round,
    passing only the current task, compact state summary, and latest findings.
12. Make Agent B review range explicit: `task_base_commit..HEAD`.
13. Add structured Agent A decision output for each Agent B finding:
    `agree`, `disagree`, `defer`, or `needs_user`.
14. Add blocked-review detection only when Agent B cannot inspect the repo. A
    missing full pytest run is a caveat, not a blocker.
15. Write task progress, skipped tasks, deferred tasks, and unresolved disputes
    to the worklog with `[USER TODO]` where human action is required.

## Test

```bash
cd ~/Code/agent_loop
pytest -q
```
