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
  --task-list-file /path/to/work_queue.md \
  --work-queue \
  --run-key frozen-encoding
```

`--work-queue` selects the first unchecked Markdown checkbox item not already
done in the saved run state, then continues through the remaining unchecked
items in the same invocation. For each task it starts a fresh Claude session,
commits the task, runs Codex review on that task's commit range, and repeats
fresh Claude fix sessions until Codex reports clean. The plan file is read once
at run start and is not edited; the durable state stores compact parsed tasks,
not the full context document. If the process is interrupted mid-task, the same
command resumes the saved run because `--run-key` points to the existing
incomplete state.

The preferred explicit execution form is:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode execute \
  --prompt "Implement the task list one item at a time. Do not push." \
  --task-list-file /path/to/tasks.md \
  --run-key my-task-run
```

Generate a normalized task list without editing code:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode discussion \
  --prompt "Turn this findings file into bounded coding tasks." \
  --context-file /path/to/findings.md \
  --task-list-file /path/to/tasks.md
```

Generate the task list and then execute it:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode discuss-execute \
  --prompt "Turn this findings file into tasks and implement them one by one." \
  --context-file /path/to/findings.md \
  --run-key my-task-run
```

When `--task-list-file` is omitted, discussion and discussion-execution modes
write the checkpoint task list under:

```text
~/Code/agent_loop/run_logs/state/by_key/<run-key>.tasks.md
```

Every checkpoint is saved under `~/Code/agent_loop/run_logs/state/`. Resume a
specific run or the newest incomplete run:

```bash
python ~/Code/agent_loop/agent_review_loop.py --resume RUN_ID
python ~/Code/agent_loop/agent_review_loop.py --resume-last
```

Each program invocation writes one Markdown transcript to
`~/Code/agent_loop/run_logs/`. It records the request, context/task-list path,
selected/current task text, agent questions/conclusions, user answers, review
findings, commits, and final status. It does not record agent thinking, shell
commands, raw tool output, or full context documents in work-queue mode.

Claude/Codex subprocess stdout and stderr are streamed to raw logs under
`~/Code/agent_loop/run_logs/raw/`. Raw log filenames include the date/time and a
label derived from the context/task-list filename, falling back to the prompt
when no file is provided. The default heartbeat prints every 30 seconds with
the active raw log path, and each individual agent call times out after 60
minutes by default.

```bash
tail -f ~/Code/agent_loop/run_logs/raw/*.log
```

Useful options:

```text
--repo PATH
--mode legacy|discussion|execute|discuss-execute
--context-file PATH
--task-list-file PATH
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
--commit-policy squash-per-task|per-round|no-commit
--max-rounds N
--max-dispute-rounds N
--heartbeat-seconds N
--agent-timeout-minutes N
```

The target repository must have no tracked modifications when starting a new
execution run. Existing untracked files are not committed, even if Claude edits
them. The script creates commits but never pushes. `--no-push` is the default;
`--push` is intentionally rejected until remote-write behavior is implemented.

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

## Mode Model

The loop has explicit modes instead of overloading one prompt shape for
planning, execution, review, and resume.

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

Example command:

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

Example command:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode execute \
  --task-list-file /Users/pro/Code/Mds/rigel_code_improvement_tasks.md \
  --new-branch rigel_code_improvement_07_04
```

### Mode 3: Discussion Then Execution

Discussion-execution mode first runs discussion mode to create or refresh the
task list, then runs execution mode over that task list. `--auto-execute`
defaults to enabled for this mode because the mode name already implies both
phases. `--no-auto-execute` is currently rejected; use `--mode discussion` when
you want a planning-only run. If the discussion reaches `--max-rounds` but has
produced a task list, the loop marks tasks named in unresolved review findings
with `[USER TODO]`, records the review caveats, and continues executing other
ready tasks. If rerun with the same `--run-key`, the loop reuses the existing
task list, resumes executable tasks by status, and focuses discussion only when
no ready work remains but tasks still need user clarification.

Example command:

```bash
python ~/Code/agent_loop/agent_review_loop.py \
  --mode discuss-execute \
  --prompt "Turn this findings file into tasks and implement them one by one." \
  --context-file /Users/pro/Code/Mds/loop_findings_to_improve_rigel.md \
  --new-branch rigel_code_improvement_07_04
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

Task statuses:

```text
pending: created but not validated or started; executable.
ready: validated and safe to execute.
in_progress: currently running or interrupted mid-task; executable if state matches.
done: implemented, committed, and reviewed clean.
skipped: intentionally bypassed for this run; requires Reason/Context.
deferred: valid task postponed until a dependency or decision; requires Reason/Next action.
discussion_failed: task definition did not converge; requires [USER TODO].
discussion_failed_max_rounds: discussion hit --max-rounds; requires [USER TODO].
execution_failed: Agent A/B coding review did not converge; requires [USER TODO].
blocked: repo/environment prevents safe progress; stops the run.
needs_user: product/design choice required; requires [USER TODO], Context, Options.
invalid: malformed/non-executable task; requires Reason and Next action.
duplicate: covered by another task; requires duplicate-of context.
not_applicable: code inspection shows the issue no longer applies; requires evidence.
```

`[USER TODO]` is a human-action marker, not a status. It may appear on
`needs_user`, `discussion_failed`, `execution_failed`, or `deferred` tasks. It
does not mean the same thing as `skipped`.

Every non-success status must include enough context to decide what to do next:

```text
Reason: why this status happened
Context: relevant Agent A/B disagreement, code evidence, or blocker
Options: possible user choices when a user decision is needed
Recommended: optional recommended choice
Next action: what resumes or unblocks the task
```

When the loop skips, defers, or fails the current task, it should document the
progress and disagreement in the worklog using a visible marker:

```text
[USER TODO] T7 skipped: Agent A and Agent B disagree about whether this changes
the public precompute format. User decision required before implementation.
```

Execution follows a go-as-far-as-possible policy: ready/pending/in-progress
tasks are executed; `done`, `duplicate`, and `not_applicable` tasks are skipped;
`needs_user`, `discussion_failed`, and `[USER TODO]` tasks are left for focused
discussion or user input while other ready tasks continue. At program exit, the
transcript reports the task-list path so the user can inspect remaining work.

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

## Remaining Improvements

Implemented: explicit modes, `--context-file`, `--task-list-file`,
heading/checkbox task parsing, task-list discussion review, execution over task
lists, default `squash-per-task`, no-push default, blocked-review detection, and
work-queue dispute skipping with `[USER TODO]`.

Still worth improving:

1. Add stricter task-list validation before execution. Invalid tasks should be
   normalized, skipped with a reason, or marked `[USER TODO]`.
2. Persist richer task checkpoints in state: Agent A decisions, Agent B finding
   ids, status transitions, and worklog path.
3. Add structured Agent A decision output for each Agent B finding:
   `agree`, `disagree`, `defer`, or `needs_user`.
4. Make `per-round` and `no-commit` commit policies fully distinct. Today the
   useful default is `squash-per-task`; `per-round` mostly reflects the legacy
   behavior and `no-commit` is not useful for review ranges.
5. Add an optional approval gate between discussion and execution.

## Test

```bash
cd ~/Code/agent_loop
pytest -q
```
