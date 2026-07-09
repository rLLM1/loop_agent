from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "agent_review_loop.py"
SPEC = importlib.util.spec_from_file_location("agent_review_loop", SCRIPT)
assert SPEC and SPEC.loader
agent_loop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = agent_loop
SPEC.loader.exec_module(agent_loop)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("base\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    return tmp_path


def test_extract_json_object_supports_claude_result_envelope():
    raw = '{"type":"result","result":"{\\"status\\":\\"conclusion\\",'
    raw += '\\"message\\":\\"done\\",\\"commit_message\\":\\"Fix it\\"}"}'
    assert agent_loop._extract_json_object(raw)["message"] == "done"


def test_run_writes_raw_log_and_preserves_output(tmp_path: Path):
    raw_log = tmp_path / "raw" / "agent.log"

    result = agent_loop._run(
        [
            sys.executable,
            "-c",
            "import sys; print('visible stdout'); print('visible stderr', file=sys.stderr)",
        ],
        cwd=tmp_path,
        raw_log_path=raw_log,
        heartbeat_seconds=0,
        timeout_seconds=5,
        heartbeat_label="test agent",
    )

    assert result.returncode == 0
    assert "visible stdout" in result.stdout
    assert "visible stderr" in result.stderr
    text = raw_log.read_text()
    assert "label: test agent" in text
    assert "[stdout]" in text
    assert "visible stdout" in text
    assert "[stderr]" in text
    assert "visible stderr" in text


def test_run_times_out_and_records_raw_log(tmp_path: Path):
    raw_log = tmp_path / "raw" / "slow.log"

    with pytest.raises(agent_loop.OrchestrationError, match="timed out"):
        agent_loop._run(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            cwd=tmp_path,
            raw_log_path=raw_log,
            heartbeat_seconds=0,
            timeout_seconds=0.2,
            heartbeat_label="slow agent",
        )

    assert raw_log.exists()
    assert "label: slow agent" in raw_log.read_text()


def test_raw_log_label_uses_problem_file_stem(repo: Path, tmp_path: Path):
    state = agent_loop._new_state(
        repo=repo,
        prompt="Fix the plan",
        problems_file=None,
    )
    state.problems_file = "/tmp/Merge V2 Precompute To Cell.md"

    assert agent_loop._run_label(state) == "Merge-V2-Precompute-To-Cell"


def test_claude_raw_log_filename_includes_label(repo: Path, tmp_path: Path):
    fake_claude = tmp_path / "fake_claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({'status':'conclusion','message':'done','commit_message':'Do it'}))\n"
    )
    fake_claude.chmod(0o755)

    agent = agent_loop.ClaudeAgent(
        executable=str(fake_claude),
        repo=repo,
        model=None,
        permission_mode="acceptEdits",
        raw_log_dir=tmp_path / "raw",
        raw_log_label="Merge V2 Precompute To Cell.md",
    )
    reply = agent.ask(str(fake_claude))

    assert reply.message == "done"
    names = [path.name for path in (tmp_path / "raw").glob("*.log")]
    assert len(names) == 1
    assert "_Merge-V2-Precompute-To-Cell.md_" in names[0]


def test_claude_agent_retries_once_for_json_shape(repo: Path, tmp_path: Path):
    fake_claude = tmp_path / "fake_claude.py"
    counter = tmp_path / "count.txt"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib\n"
        f"counter = pathlib.Path({str(counter)!r})\n"
        "count = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(count + 1))\n"
        "if count == 0:\n"
        "    print(json.dumps({'type':'result','result':'plain prose, not JSON'}))\n"
        "else:\n"
        "    print(json.dumps({'status':'conclusion','message':'done','commit_message':'Do it'}))\n"
    )
    fake_claude.chmod(0o755)

    agent = agent_loop.ClaudeAgent(
        executable=str(fake_claude),
        repo=repo,
        model=None,
        permission_mode="acceptEdits",
        raw_log_dir=tmp_path / "raw",
    )

    assert agent.ask("do work").message == "done"
    assert counter.read_text() == "2"


def test_codex_reviewer_prompt_checks_big_picture_fit(repo: Path, tmp_path: Path):
    fake_codex = tmp_path / "fake_codex.py"
    captured_prompt = tmp_path / "prompt.txt"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(captured_prompt)!r}, 'w').write(sys.argv[-1])\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(json.dumps({'status':'clean','summary':'ok','issues':[]}))\n"
    )
    fake_codex.chmod(0o755)

    reviewer = agent_loop.CodexReviewer(
        executable=str(fake_codex),
        repo=repo,
        model=None,
    )
    review = reviewer.review(
        base_commit="base",
        commit="head",
        agent1_conclusion="Updated the plan.",
    )

    assert review.status == "clean"
    prompt = captured_prompt.read_text()
    assert "Big-picture fit" in prompt
    assert "Does this diff address the user's actual request" in prompt
    assert "For docs, plans, runbooks, and migrations" in prompt
    assert "Do not limit findings to lines changed" in prompt


def test_codex_reviewer_prompt_is_code_only(repo: Path, tmp_path: Path):
    fake_codex = tmp_path / "fake_codex.py"
    captured_prompt = tmp_path / "prompt.txt"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(captured_prompt)!r}, 'w').write(sys.argv[-1])\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(json.dumps({'status':'clean','summary':'ok','issues':[]}))\n"
    )
    fake_codex.chmod(0o755)

    reviewer = agent_loop.CodexReviewer(
        executable=str(fake_codex),
        repo=repo,
        model=None,
    )
    reviewer.review(
        base_commit="base",
        commit="head",
        agent1_conclusion="Updated the code.",
    )

    prompt = captured_prompt.read_text()
    # Normalize wrapping so the assertions guard wording, not source layout.
    normalized = " ".join(prompt.split())
    # The scope carve-out targets commit metadata only; it must not shrink the
    # review to "code only" and drop docs/plans/migration artifacts.
    assert (
        "Review repository changes only; do not comment on commit messages "
        "or add commit facts." in normalized
    )
    assert "Review code only" not in normalized
    # Broader review requirements survive the carve-out.
    assert "docs, plans, runbooks, and migrations" in normalized

    # Normal review stays free of commit-message fields.
    assert "commit" not in agent_loop.REVIEW_SCHEMA["properties"]
    assert "commit_message" not in agent_loop.REVIEW_SCHEMA["properties"]
    review_fields = {f.name for f in fields(agent_loop.ReviewReply)}
    assert review_fields == {"status", "summary", "issues"}


def test_blocked_review_detection():
    review = agent_loop.ReviewReply(
        "issues",
        "Unable to perform the requested repository review.",
        [{
            "severity": "critical",
            "title": "Review blocked by execution environment",
            "details": "No repository evidence was accessible because commands rejected under read-only sandbox approval.",
        }],
    )

    assert agent_loop._review_is_blocked(review)


def test_review_with_actionable_findings_and_readonly_test_caveat_is_not_blocked():
    review = agent_loop.ReviewReply(
        "issues",
        (
            "I found two remaining actionable regressions in the reviewed "
            "range. I did not run pytest because the session is read-only."
        ),
        [{
            "severity": "medium",
            "title": "Object columns containing numpy scalar values are dropped",
            "details": (
                "In dataset/schema_utils.py:111-115, _is_list_valued treats "
                "np.int64 values as list-like. Narrow this check and add a "
                "regression test."
            ),
        }],
    )

    assert not agent_loop._review_is_blocked(review)


def test_review_with_file_referenced_issues_is_not_blocked_by_sandbox_caveat():
    review = agent_loop.ReviewReply(
        "issues",
        "Found actionable issues. Could not run pytest in the sandbox.",
        [{
            "severity": "medium",
            "title": "Ragged distributed eval skips padding trim",
            "details": (
                "MetricTracker.from_snapshots skips max_samples trimming in "
                "train/metrics.py:119. Fix the ragged non-sampled path."
            ),
        }],
    )

    assert not agent_loop._review_is_blocked(review)


def test_commit_scopes_new_files_and_preserves_existing_untracked(repo: Path):
    existing = repo / "notes.txt"
    existing.write_text("do not commit\n")
    baseline = agent_loop._untracked_paths(repo)

    (repo / "tracked.txt").write_text("changed\n")
    (repo / "new.py").write_text("print('new')\n")
    commit = agent_loop._commit_agent_changes(
        repo,
        baseline_untracked=baseline,
        message="Add scoped changes",
    )

    assert commit == _git(repo, "rev-parse", "HEAD")
    assert set(_git(repo, "show", "--name-only", "--format=").splitlines()) == {
        "new.py",
        "tracked.txt",
    }
    assert existing.exists()
    assert "notes.txt" in agent_loop._untracked_paths(repo)


def test_run_loop_handles_question_review_fix_and_clean(repo: Path):
    existing = repo / "personal.txt"
    existing.write_text("keep me\n")

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def ask(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return agent_loop.Agent1Reply("question", "Which value?", "")
            if self.calls == 2:
                (repo / "tracked.txt").write_text("first\n")
                return agent_loop.Agent1Reply(
                    "conclusion", "Implemented first pass.", "Implement value"
                )
            (repo / "tracked.txt").write_text("fixed\n")
            return agent_loop.Agent1Reply(
                "conclusion", "Fixed reviewer finding.", "Fix review finding"
            )

    class FakeReviewer:
        def __init__(self):
            self.calls = 0
            self.ranges = []

        def review(self, **kwargs):
            self.calls += 1
            self.ranges.append((kwargs["base_commit"], kwargs["commit"]))
            if self.calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "One issue.",
                    [{
                        "severity": "medium",
                        "title": "Wrong value",
                        "details": "Use the corrected value.",
                    }],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

    answers = iter(["42"])
    reviewer = FakeReviewer()
    base_commit = _git(repo, "rev-parse", "HEAD")
    result = agent_loop.run_loop(
        repo=repo,
        initial_prompt="Do the work",
        agent1=FakeAgent(),
        reviewer=reviewer,
        max_rounds=3,
        input_fn=lambda _prompt: next(answers),
    )

    assert result == 0
    assert (repo / "tracked.txt").read_text() == "fixed\n"
    assert _git(repo, "rev-list", "--count", "HEAD") == "3"
    assert reviewer.ranges[0][0] == base_commit
    assert reviewer.ranges[1][0] == base_commit
    assert reviewer.ranges[0][1] != reviewer.ranges[1][1]
    assert existing.exists()


def test_agent1_review_base_overrides_starting_base(repo: Path):
    initial = _git(repo, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("intermediate\n")
    subprocess.run(["git", "commit", "-am", "intermediate", "-q"], cwd=repo, check=True)
    stale_start = _git(repo, "rev-parse", "HEAD")

    class FakeAgent:
        def ask(self, _prompt):
            (repo / "tracked.txt").write_text("class 1\n")
            return agent_loop.Agent1Reply(
                "conclusion",
                "Implemented class 1.",
                "Implement class 1",
                target_branch="target",
                work_branch="work",
                review_base=initial,
            )

    class CleanReviewer:
        def __init__(self):
            self.range = None

        def review(self, **kwargs):
            self.range = (kwargs["base_commit"], kwargs["commit"])
            return agent_loop.ReviewReply("clean", "No issues.", [])

    reviewer = CleanReviewer()
    result = agent_loop.run_loop(
        repo=repo,
        initial_prompt="Do class 1",
        agent1=FakeAgent(),
        reviewer=reviewer,
        max_rounds=1,
    )

    assert result == 0
    assert reviewer.range[0] == initial
    assert reviewer.range[0] != stale_start


def test_fake_full_sha_is_rejected(repo: Path):
    fake_sha = "67a007053a86fc70a5b7e2c62d0e5b3c4b9c9d1e"
    reply = agent_loop.Agent1Reply(
        "conclusion",
        "done",
        "Do it",
        review_base=fake_sha,
    )

    with pytest.raises(agent_loop.OrchestrationError, match="invalid review base"):
        agent_loop._resolve_review_base(repo, reply, "HEAD")


def test_work_queue_commit_range_ignores_agent_reported_range(
    repo: Path,
    tmp_path: Path,
):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] T01: fix one thing\n"
        "  Scope: tracked.txt only.\n"
    )
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="range-test",
        work_queue=True,
    )

    class CapturingStateStore:
        def __init__(self):
            self.saved = None

        def save(self, value):
            self.saved = value

        def load(self, _run_id):
            return self.saved

    class CapturingRunLog:
        def add(self, _title, _text):
            pass

    fake_sha = "67a007053a86fc70a5b7e2c62d0e5b3c4b9c9d1e"

    class FakeAgent:
        def ask(self, _prompt):
            (repo / "tracked.txt").write_text("fixed\n")
            return agent_loop.Agent1Reply(
                "conclusion",
                "Implemented task.",
                "Implement task",
                review_base=fake_sha,
                review_head=fake_sha,
            )

    class CleanReviewer:
        def __init__(self):
            self.range = None

        def review(self, **kwargs):
            self.range = (kwargs["base_commit"], kwargs["commit"])
            return agent_loop.ReviewReply("clean", "No issues.", [])

    store = CapturingStateStore()
    store.save(state)
    reviewer = CleanReviewer()
    base_commit = _git(repo, "rev-parse", "HEAD")

    result = agent_loop.run_state_loop(
        state=state,
        state_store=store,
        run_log=CapturingRunLog(),
        agent1=FakeAgent(),
        reviewer=reviewer,
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    assert reviewer.range is not None
    assert reviewer.range[0] == base_commit
    assert reviewer.range[1] == _git(repo, "rev-parse", "HEAD")
    assert reviewer.range[1] != fake_sha


def test_work_queue_resume_after_commit_uses_stored_commit_range(
    repo: Path,
    tmp_path: Path,
):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] T01: fix one thing\n"
        "  Scope: tracked.txt only.\n"
    )
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="resume-range-test",
        work_queue=True,
    )
    base_commit = _git(repo, "rev-parse", "HEAD")
    (repo / "tracked.txt").write_text("fixed before crash\n")
    subprocess.run(["git", "commit", "-am", "fixed before crash", "-q"], cwd=repo, check=True)
    committed_head = _git(repo, "rev-parse", "HEAD")
    fake_sha = "67a007053a86fc70a5b7e2c62d0e5b3c4b9c9d1e"
    state.phase = "commit"
    state.commit = committed_head
    state.reply = {
        "status": "conclusion",
        "message": "Implemented task.",
        "commit_message": "Implement task",
        "review_base": fake_sha,
        "review_head": fake_sha,
        "target_branch": "",
        "work_branch": "",
    }

    class CapturingStateStore:
        def __init__(self):
            self.saved = None

        def save(self, value):
            self.saved = value

        def load(self, _run_id):
            return self.saved

    class CapturingRunLog:
        def add(self, _title, _text):
            pass

    class UnusedAgent:
        def ask(self, _prompt):
            raise AssertionError("resume should not call Agent 1 again")

    class CleanReviewer:
        def __init__(self):
            self.range = None

        def review(self, **kwargs):
            self.range = (kwargs["base_commit"], kwargs["commit"])
            return agent_loop.ReviewReply("clean", "No issues.", [])

    store = CapturingStateStore()
    store.save(state)
    reviewer = CleanReviewer()

    result = agent_loop.run_state_loop(
        state=state,
        state_store=store,
        run_log=CapturingRunLog(),
        agent1=UnusedAgent(),
        reviewer=reviewer,
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    assert reviewer.range == (base_commit, committed_head)


def test_work_queue_task_memory_survives_into_later_fix_prompt(
    repo: Path,
    tmp_path: Path,
):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] T01: preserve decisions\n"
        "  Scope: tracked.txt only.\n"
        "  Out of scope: later tasks.\n"
        "  Tests: focused regression.\n"
    )
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="memory-test",
        work_queue=True,
    )

    class CapturingStateStore:
        def __init__(self):
            self.saved = None

        def save(self, value):
            self.saved = value

        def load(self, _run_id):
            return self.saved

    class CapturingRunLog:
        def add(self, _title, _text):
            pass

    class FakeAgent:
        def __init__(self):
            self.calls = 0
            self.prompts = []

        def ask(self, prompt):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                (repo / "tracked.txt").write_text("first\n")
                return agent_loop.Agent1Reply(
                    "conclusion",
                    "- Kept the public API unchanged.\n"
                    "Tests: focused regression passed.",
                    "First fix",
                )
            (repo / "tracked.txt").write_text("second\n")
            return agent_loop.Agent1Reply(
                "conclusion",
                "- Rejected broad refactor; fixed only the reviewed issue.\n"
                "Tests: focused regression passed again.",
                "Second fix",
            )

    class IssuesThenCleanReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "One actionable issue.",
                    [{
                        "severity": "high",
                        "title": "Missing propagation",
                        "details": "tracked.txt:1 does not propagate the value.",
                    }],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

    store = CapturingStateStore()
    store.save(state)
    agent = FakeAgent()
    result = agent_loop.run_state_loop(
        state=state,
        state_store=store,
        run_log=CapturingRunLog(),
        agent1=agent,
        reviewer=IssuesThenCleanReviewer(),
        max_rounds=3,
        max_dispute_rounds=3,
    )

    saved = store.load(state.run_id)
    task = saved.tasks[0]
    assert result == 0
    assert agent.calls == 2
    assert "Compact task memory from prior A/B rounds" in agent.prompts[1]
    assert "Kept the public API unchanged" in agent.prompts[1]
    assert "focused regression passed" in agent.prompts[1]
    assert "Missing propagation" in agent.prompts[1]
    assert task["status"] == "done"
    assert task["task_memory"]["open_review_findings"] == []
    assert any(
        "Agent 2 review round 2: clean" in item
        for item in task["task_memory"]["known_test_results"]
    )


def test_task_list_persists_compact_task_memory(repo: Path, tmp_path: Path):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] T01: persist memory\n"
        "  Scope: tracked.txt only.\n"
    )
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="memory-render-test",
        work_queue=True,
    )
    task = state.tasks[0]
    agent_loop._memory_add(task, "resolved_decisions", ["Committed abc123: keep API"])
    agent_loop._memory_replace(
        task,
        "open_review_findings",
        ["[high] Missing propagation: tracked.txt:1"],
    )

    agent_loop._persist_task_list_status(state)

    rendered = queue.read_text()
    assert "Task memory:" in rendered
    assert "Resolved decisions:" in rendered
    assert "Committed abc123: keep API" in rendered
    assert "Open review findings:" in rendered
    assert "Missing propagation" in rendered


def test_blocked_review_stops_before_agent1_fix(repo: Path):
    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def ask(self, _prompt):
            self.calls += 1
            (repo / "tracked.txt").write_text("changed\n")
            return agent_loop.Agent1Reply(
                "conclusion", "Implemented.", "Implement change"
            )

    class BlockedReviewer:
        def review(self, **_kwargs):
            return agent_loop.ReviewReply(
                "issues",
                "Unable to perform the requested repository review.",
                [{
                    "severity": "critical",
                    "title": "Review blocked by execution environment",
                    "details": "No repository evidence was accessible; commands rejected by sandbox approval.",
                }],
            )

    agent = FakeAgent()
    result = agent_loop.run_loop(
        repo=repo,
        initial_prompt="Do work",
        agent1=agent,
        reviewer=BlockedReviewer(),
        max_rounds=2,
    )

    assert result == 2
    assert agent.calls == 1


def test_run_loop_rejects_preexisting_tracked_changes(repo: Path):
    (repo / "tracked.txt").write_text("dirty\n")
    with pytest.raises(agent_loop.OrchestrationError, match="already modified"):
        agent_loop.run_loop(
            repo=repo,
            initial_prompt="task",
            agent1=object(),
            reviewer=object(),
            max_rounds=1,
        )


def test_combined_prompt_references_problem_document_without_inlining(tmp_path: Path):
    problems = tmp_path / "problems.md"
    problems.write_text("- Fix duplicate writes\n- Add validation\n")

    combined, text = agent_loop._combined_prompt("Implement fixes", problems)

    assert combined.startswith("Implement fixes")
    assert str(problems.resolve()) in combined
    assert "Fix duplicate writes" not in combined
    assert text == "- Fix duplicate writes\n- Add validation"


def test_parse_work_items_and_select_next():
    text = (
        "# Queue\n\n"
        "- [x] U0: done\n"
        "  Old work.\n"
        "- [ ] U1: next\n"
        "  Implement this.\n"
        "- [ ] U2: later\n"
    )

    items = agent_loop._parse_work_items(text)
    selected = agent_loop._select_next_work_item(
        problems_text=text,
        queue_progress={"done": {}},
    )
    selected_after_done = agent_loop._select_next_work_item(
        problems_text=text,
        queue_progress={"done": {selected.item_id: {}}},
    )

    assert [item.title for item in items] == ["U0: done", "U1: next", "U2: later"]
    assert items[0].checked
    assert selected.title == "U1: next"
    assert "Implement this." in selected.text
    assert selected_after_done.title == "U2: later"


def test_parse_task_queue_accepts_heading_tasks():
    text = (
        "# Tasks\n\n"
        "## T01: Fix first thing\n"
        "Status: todo\n"
        "Problem: first problem\n"
        "Acceptance: first acceptance\n\n"
        "## T02: Already done\n"
        "Status: done\n"
        "Problem: old problem\n"
    )

    tasks = agent_loop._parse_task_queue(text)

    assert [task["title"] for task in tasks] == [
        "T01: Fix first thing",
        "T02: Already done",
    ]
    assert [task["status"] for task in tasks] == ["pending", "done"]
    assert "first problem" in tasks[0]["text"]


def test_work_queue_state_prompt_limits_to_selected_item(repo: Path, tmp_path: Path):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] U0: scaffolding\n"
        "  Only this item.\n"
        "- [ ] U1: later\n"
        "  Not this item.\n"
    )

    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="frozen",
        work_queue=True,
    )

    assert state.run_key == "frozen"
    assert state.work_queue
    assert state.work_item_id == "U0-scaffolding"
    assert state.problems_text == ""
    assert state.tasks and len(state.tasks) == 2
    prompt = agent_loop._task_prompt(state)
    assert "Only this item." in prompt
    assert "Not this item." not in prompt


def test_work_queue_main_processes_all_items_in_one_invocation(repo: Path, tmp_path: Path):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] U0: scaffolding\n"
        "  First item.\n"
        "- [ ] U1: numerical\n"
        "  Second item.\n"
    )
    fake_claude = tmp_path / "fake_claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "prompt = sys.argv[-1]\n"
        "selected = 'U0' if 'U0: scaffolding' in prompt else 'U1'\n"
        "pathlib.Path('tracked.txt').write_text(selected + '\\n')\n"
        "print(json.dumps({\n"
        "  'status': 'conclusion',\n"
        "  'message': 'implemented ' + selected,\n"
        "  'commit_message': 'Implement ' + selected\n"
        "}))\n"
    )
    fake_claude.chmod(0o755)
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(json.dumps({'status':'clean','summary':'ok','issues':[]}))\n"
    )
    fake_codex.chmod(0o755)

    args = [
        "Implement queue",
        "--repo",
        str(repo),
        "--task-list-file",
        str(queue),
        "--work-queue",
        "--run-key",
        "frozen-test",
        "--run-log-dir",
        str(tmp_path / "logs"),
        "--agent1-command",
        str(fake_claude),
        "--agent2-command",
        str(fake_codex),
        "--heartbeat-seconds",
        "0",
    ]

    assert agent_loop.main(args) == 0
    assert (repo / "tracked.txt").read_text() == "U1\n"
    assert _git(repo, "log", "--oneline", "-2").count("Implement U") == 2
    state = json_load(tmp_path / "logs" / "state" / "by_key" / "frozen-test.json")
    assert [task["status"] for task in state["tasks"]] == ["done", "done"]
    assert state["phase"] == "complete"


def test_execute_mode_uses_task_list_file(repo: Path, tmp_path: Path):
    queue = tmp_path / "tasks.md"
    queue.write_text(
        "## T01: write value\n"
        "Problem: tracked.txt needs a value.\n"
        "Acceptance: tracked.txt says task-list.\n"
    )
    fake_claude = tmp_path / "fake_claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib\n"
        "pathlib.Path('tracked.txt').write_text('task-list\\n')\n"
        "print(json.dumps({\n"
        "  'status': 'conclusion',\n"
        "  'message': 'implemented task-list item',\n"
        "  'commit_message': 'Implement task-list item'\n"
        "}))\n"
    )
    fake_claude.chmod(0o755)
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(json.dumps({'status':'clean','summary':'ok','issues':[]}))\n"
    )
    fake_codex.chmod(0o755)

    result = agent_loop.main([
        "--mode",
        "execute",
        "--prompt",
        "Implement task list",
        "--repo",
        str(repo),
        "--task-list-file",
        str(queue),
        "--run-key",
        "execute-task-list-test",
        "--run-log-dir",
        str(tmp_path / "logs"),
        "--agent1-command",
        str(fake_claude),
        "--agent2-command",
        str(fake_codex),
        "--heartbeat-seconds",
        "0",
    ])

    assert result == 0
    assert (repo / "tracked.txt").read_text() == "task-list\n"
    assert _git(repo, "log", "-1", "--pretty=%s") == "Implement task-list item"


def test_task_queue_skips_needs_user_and_runs_ready(repo: Path, tmp_path: Path):
    queue = tmp_path / "tasks.md"
    queue.write_text(
        "- [ ] T01: Needs decision\n"
        "  Status: needs_user\n"
        "  [USER TODO]: choose a policy.\n"
        "- [ ] T02: Ready work\n"
        "  Status: ready\n"
        "  Problem: tracked.txt needs a value.\n"
    )
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="status-test",
        work_queue=True,
    )

    assert state.work_item_id == "T02-Ready-work"
    assert state.tasks[0]["status"] == "needs_user"
    assert state.tasks[1]["status"] == "running"


def test_work_queue_with_only_non_executable_tasks_exits_clean(repo: Path, tmp_path: Path):
    queue = tmp_path / "tasks.md"
    queue.write_text(
        "- [ ] T01: Needs decision\n"
        "  Status: needs_user\n"
        "  [USER TODO]: choose a policy.\n"
    )

    result = agent_loop.main([
        "--prompt",
        "Implement queue",
        "--repo",
        str(repo),
        "--task-list-file",
        str(queue),
        "--work-queue",
        "--run-key",
        "no-executable-test",
        "--run-log-dir",
        str(tmp_path / "logs"),
        "--agent1-command",
        sys.executable,
        "--agent2-command",
        sys.executable,
        "--heartbeat-seconds",
        "0",
    ])

    assert result == 0


def test_discussion_mode_writes_reviewed_task_list(repo: Path, tmp_path: Path):
    output = tmp_path / "tasks.md"
    context = tmp_path / "context.md"
    context.write_text("Finding: tracked.txt needs a bounded task.\n")
    fake_claude = tmp_path / "fake_claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "print(json.dumps({\n"
        "  'status': 'conclusion',\n"
        "  'message': '- [ ] T01: Fix tracked value\\\\n  Problem: tracked.txt needs a bounded task.\\\\n  Acceptance: test passes.',\n"
        "  'commit_message': 'Write task list'\n"
        "}))\n"
    )
    fake_claude.chmod(0o755)
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(json.dumps({'status':'clean','summary':'task list ok','issues':[]}))\n"
    )
    fake_codex.chmod(0o755)

    result = agent_loop.main([
        "--mode",
        "discussion",
        "--prompt",
        "Create tasks",
        "--repo",
        str(repo),
        "--context-file",
        str(context),
        "--task-list-file",
        str(output),
        "--run-log-dir",
        str(tmp_path / "logs"),
        "--agent1-command",
        str(fake_claude),
        "--agent2-command",
        str(fake_codex),
        "--heartbeat-seconds",
        "0",
    ])

    assert result == 0
    assert output.read_text().startswith("- [ ] T01: Fix tracked value")


def test_discuss_execute_defaults_task_list_under_run_key(repo: Path, tmp_path: Path):
    context = tmp_path / "context.md"
    context.write_text("Finding: tracked.txt needs a bounded task.\n")
    fake_claude = tmp_path / "fake_claude.py"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "prompt = sys.argv[-1]\n"
        "if 'DISCUSSION MODE' in prompt:\n"
        "    message = '- [ ] T01: Fix tracked value\\\\n  Problem: tracked.txt needs a value.\\\\n  Acceptance: tracked.txt says executed.'\n"
        "    commit_message = 'Write task list'\n"
        "else:\n"
        "    pathlib.Path('tracked.txt').write_text('executed\\n')\n"
        "    message = 'implemented generated task'\n"
        "    commit_message = 'Implement generated task'\n"
        "print(json.dumps({'status':'conclusion','message':message,'commit_message':commit_message}))\n"
    )
    fake_claude.chmod(0o755)
    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "open(out, 'w').write(json.dumps({'status':'clean','summary':'ok','issues':[]}))\n"
    )
    fake_codex.chmod(0o755)

    result = agent_loop.main([
        "--mode",
        "discuss-execute",
        "--prompt",
        "Create and run tasks",
        "--repo",
        str(repo),
        "--context-file",
        str(context),
        "--run-key",
        "generated-tasks",
        "--run-log-dir",
        str(tmp_path / "logs"),
        "--agent1-command",
        str(fake_claude),
        "--agent2-command",
        str(fake_codex),
        "--heartbeat-seconds",
        "0",
    ])

    generated = tmp_path / "logs" / "state" / "by_key" / "generated-tasks.tasks.md"
    assert result == 0
    assert generated.exists()
    assert "- [x] T01: Fix tracked value" in generated.read_text()
    assert "Status: done" in generated.read_text()
    assert (repo / "tracked.txt").read_text() == "executed\n"


def test_prepare_branch_creates_or_switches_without_breaking(repo: Path):
    initial_branch = _git(repo, "branch", "--show-current")
    agent_loop._prepare_branch(repo, branch="feature/test", branch_base=None)
    assert _git(repo, "branch", "--show-current") == "feature/test"
    subprocess.run(["git", "switch", initial_branch, "-q"], cwd=repo, check=True)

    agent_loop._prepare_branch(repo, branch="feature/test", branch_base=None)

    assert _git(repo, "branch", "--show-current") == "feature/test"


def test_parser_accepts_new_branch_and_legacy_branch_alias():
    parser = agent_loop._build_parser()

    parsed = parser.parse_args([
        "--mode",
        "execute",
        "--prompt",
        "Run tasks",
        "--task-list-file",
        "tasks.md",
        "--run-key",
        "tasks",
        "--new-branch",
        "feature/tasks",
    ])
    legacy = parser.parse_args([
        "--mode",
        "execute",
        "--prompt",
        "Run tasks",
        "--task-list-file",
        "tasks.md",
        "--run-key",
        "tasks",
        "--branch",
        "feature/legacy",
    ])

    assert parsed.new_branch == "feature/tasks"
    assert legacy.new_branch == "feature/legacy"


def test_agent1_reply_parses_and_round_trips_commit_update():
    reply = agent_loop._agent1_reply_from_value(
        {
            "status": "conclusion",
            "message": "done",
            "commit_message": "Fix bug",
            "commit_update": {
                "root_cause": ["null deref on empty input"],
                "changes": ["guard input", "add check"],
                "tests": ["unit: guard input"],
                "deferred": ["refactor later"],
                "unknown": ["should be dropped"],
            },
        }
    )
    expected = {
        "root_cause": ["null deref on empty input"],
        "changes": ["guard input", "add check"],
        "tests": ["unit: guard input"],
        "deferred": ["refactor later"],
    }
    assert reply.commit_update == expected
    # asdict round-trips the nested commit facts and rebuilds cleanly.
    dumped = agent_loop.asdict(reply)
    assert dumped["commit_update"] == expected
    assert agent_loop.Agent1Reply(**dumped).commit_update == expected


def test_agent1_reply_defaults_commit_update_when_absent():
    reply = agent_loop._agent1_reply_from_value(
        {
            "status": "conclusion",
            "message": "done",
            "commit_message": "Fix bug",
        }
    )
    assert reply.commit_update == {
        "root_cause": [],
        "changes": [],
        "tests": [],
        "deferred": [],
    }


def test_agent1_reply_tolerates_malformed_commit_update():
    empty = {
        "root_cause": [],
        "changes": [],
        "tests": [],
        "deferred": [],
    }

    partial = agent_loop._agent1_reply_from_value(
        {
            "status": "conclusion",
            "message": "done",
            "commit_message": "Fix bug",
            "commit_update": {
                "root_cause": "not a list",
                "changes": None,
                "tests": [1, 2],
            },
        }
    )
    assert partial.commit_update == {
        "root_cause": [],
        "changes": [],
        "tests": ["1", "2"],
        "deferred": [],
    }

    # A non-dict commit_update degrades to the four empty lists, never raises.
    non_dict = agent_loop._agent1_reply_from_value(
        {
            "status": "conclusion",
            "message": "done",
            "commit_message": "Fix bug",
            "commit_update": ["nope"],
        }
    )
    assert non_dict.commit_update == empty


def test_agent1_reply_commit_update_default_is_not_shared():
    first = agent_loop.Agent1Reply("conclusion", "a", "Fix a")
    second = agent_loop.Agent1Reply("conclusion", "b", "Fix b")

    assert first.commit_update is not second.commit_update
    first.commit_update["changes"].append("mutated")
    assert second.commit_update["changes"] == []


def _reply_with_facts(commit_message, **facts):
    return agent_loop.Agent1Reply(
        "conclusion",
        "done",
        commit_message,
        commit_update=agent_loop._normalize_commit_update(facts),
    )


def test_append_task_commit_log_numbers_rounds_by_log_length():
    task = {"task_commit_log": []}

    first = agent_loop._append_task_commit_log(
        task, _reply_with_facts("Implement thing", changes=["c1"])
    )
    # review_round is still 1 in the real flow when the first fix commits, so
    # round numbering must come from the log length, not review_round.
    second = agent_loop._append_task_commit_log(
        task, _reply_with_facts("Fix review finding", changes=["c2"])
    )

    assert first["round"] == 1
    assert second["round"] == 2
    log = task["task_commit_log"]
    assert [entry["round"] for entry in log] == [1, 2]
    assert log[0]["commit_message"] == "Implement thing"
    assert log[1]["commit_message"] == "Fix review finding"
    assert log[0]["changes"] == ["c1"]
    assert log[1]["changes"] == ["c2"]


def test_append_task_commit_log_initializes_missing_log():
    task = {}
    agent_loop._append_task_commit_log(task, _reply_with_facts("Subject"))
    assert task["task_commit_log"][0]["round"] == 1
    # A pre-existing non-list value is replaced with a fresh list.
    task2 = {"task_commit_log": "corrupt"}
    agent_loop._append_task_commit_log(task2, _reply_with_facts("Subject"))
    assert task2["task_commit_log"][0]["round"] == 1


def test_append_task_commit_log_preserves_duplicate_facts_verbatim():
    task = {"task_commit_log": []}
    facts = {"root_cause": ["same"], "changes": ["dup"], "tests": ["t"]}

    agent_loop._append_task_commit_log(task, _reply_with_facts("Round one", **facts))
    agent_loop._append_task_commit_log(task, _reply_with_facts("Round two", **facts))

    log = task["task_commit_log"]
    assert len(log) == 2
    # No dedup or merge: identical facts are stored on both rounds.
    assert log[0]["root_cause"] == log[1]["root_cause"] == ["same"]
    assert log[0]["changes"] == log[1]["changes"] == ["dup"]
    # Stored lists are independent copies, not shared references.
    log[0]["changes"].append("mutated")
    assert log[1]["changes"] == ["dup"]


def test_start_work_queue_task_initializes_empty_commit_log(repo: Path, tmp_path: Path):
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: do a thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="init-log",
        work_queue=True,
    )
    # Seed a stale log to prove the start resets it.
    state.tasks[0]["task_commit_log"] = [{"round": 9}]

    agent_loop._start_work_queue_task(state, repo, 0)

    assert state.tasks[0]["task_commit_log"] == []


def test_task_commit_log_survives_state_save_load_round_trip(repo: Path, tmp_path: Path):
    store = agent_loop.StateStore(tmp_path / "state")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=None,
    )
    state.work_queue = True
    state.current_task_index = 0
    task = {"id": "T01", "title": "t", "text": "t", "status": "running"}
    state.tasks = [task]
    agent_loop._append_task_commit_log(task, _reply_with_facts("Round one", changes=["c1"]))
    agent_loop._append_task_commit_log(task, _reply_with_facts("Round two", changes=["c1"]))
    expected = list(task["task_commit_log"])

    store.save(state)
    loaded = store.load(state.run_id)

    assert loaded.tasks[0]["task_commit_log"] == expected
    assert [entry["round"] for entry in loaded.tasks[0]["task_commit_log"]] == [1, 2]


def test_work_queue_commit_log_appends_one_entry_per_round(repo: Path, tmp_path: Path):
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="commit-log-rounds",
        work_queue=True,
    )

    class FakeAgent:
        def __init__(self):
            self.calls = 0

        def ask(self, _prompt):
            self.calls += 1
            if self.calls == 1:
                (repo / "tracked.txt").write_text("first\n")
                return _reply_with_facts("Implement task", root_cause=["rc"], changes=["c1"])
            (repo / "tracked.txt").write_text("fixed\n")
            # Repeats changes=["c1"] to prove duplicates survive across rounds.
            return _reply_with_facts("Fix finding", changes=["c1"], tests=["t2"])

    class FakeReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "One issue.",
                    [{"severity": "medium", "title": "x", "details": "fix it"}],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

    class CapturingStateStore:
        def save(self, _value):
            pass

    class CapturingRunLog:
        def add(self, _title, _text):
            pass

    result = agent_loop.run_state_loop(
        state=state,
        state_store=CapturingStateStore(),
        run_log=CapturingRunLog(),
        agent1=FakeAgent(),
        reviewer=FakeReviewer(),
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    log = state.tasks[0]["task_commit_log"]
    # Two commit rounds -> exactly two entries numbered 1 then 2, even though
    # review_round was still 1 when the fix commit appended (regression guard).
    assert [entry["round"] for entry in log] == [1, 2]
    assert log[0]["commit_message"] == "Implement task"
    assert log[1]["commit_message"] == "Fix finding"
    assert log[0]["changes"] == ["c1"]
    assert log[1]["changes"] == ["c1"]
    assert log[1]["tests"] == ["t2"]


class _NoopStore:
    def save(self, _value):
        pass


class _NoopLog:
    def add(self, _title, _text):
        pass


class _TwoCommitAgent:
    """Agent 1 that makes one commit initially and one on the review fix."""

    def __init__(self):
        self.calls = 0

    def ask(self, _prompt):
        self.calls += 1
        if self.calls == 1:
            (self.repo / "tracked.txt").write_text("first\n")
            return _reply_with_facts("Implement task", changes=["c1"])
        (self.repo / "tracked.txt").write_text("fixed\n")
        return _reply_with_facts("Fix finding", tests=["t2"])


class _OneCommitAgent:
    """Agent 1 that makes exactly one commit and needs no review fixes."""

    def __init__(self):
        self.calls = 0

    def ask(self, _prompt):
        self.calls += 1
        (self.repo / "tracked.txt").write_text("only\n")
        return _reply_with_facts("Implement task once", changes=["c1"])


def test_work_queue_multi_commit_converges_to_summarized_commit(
    repo: Path, tmp_path: Path
):
    """A converged multi-commit task ends as one commit with the Agent 2 summary."""
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="summary-converge",
        work_queue=True,
    )
    base = state.tasks[0]["task_base_commit"]

    agent = _TwoCommitAgent()
    agent.repo = repo
    captured = {}

    class FakeReviewer:
        def __init__(self):
            self.review_calls = 0
            self.summary_calls = 0

        def review(self, **_kwargs):
            self.review_calls += 1
            if self.review_calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "One issue.",
                    [{"severity": "medium", "title": "x", "details": "fix it"}],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

        def summarize_commit(
            self, *, base_commit, head_commit, task_commit_log, recorded_tests
        ):
            self.summary_calls += 1
            captured["base"] = base_commit
            captured["head"] = head_commit
            captured["log"] = list(task_commit_log)
            captured["tests"] = list(recorded_tests)
            return agent_loop.CommitSummary(
                "Summarize the whole task",
                "Root cause: c1\n\nValidation: t2",
            )

    reviewer = FakeReviewer()
    result = agent_loop.run_state_loop(
        state=state,
        state_store=_NoopStore(),
        run_log=_NoopLog(),
        agent1=agent,
        reviewer=reviewer,
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    head = _git(repo, "rev-parse", "HEAD")
    # Exactly one commit remains on top of the task base, carrying the summary.
    assert _git(repo, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert _git(repo, "log", "-1", "--format=%s", "HEAD") == "Summarize the whole task"
    body = _git(repo, "log", "-1", "--format=%b", "HEAD")
    assert "Root cause: c1" in body
    assert "Validation: t2" in body
    # The summary pass ran once against the pre-squash multi-commit tip and saw
    # the full append-only log, with the synthetic clean marker filtered out.
    assert reviewer.summary_calls == 1
    assert captured["base"] == base
    assert _git(repo, "rev-list", "--count", f"{base}..{captured['head']}") == "2"
    assert [entry["round"] for entry in captured["log"]] == [1, 2]
    assert "t2" in captured["tests"]
    assert not any("Agent 2 review round" in item for item in captured["tests"])
    # State and task now point at the squashed commit, consistent with HEAD.
    assert state.commit == head
    assert state.review_head == head
    assert state.tasks[0]["status"] == "done"
    assert state.tasks[0]["last_commit"] == head


def test_work_queue_multi_commit_summary_failure_uses_fallback(
    repo: Path, tmp_path: Path
):
    """A failed Agent 2 summary still squashes, using the deterministic fallback."""
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="summary-fallback",
        work_queue=True,
    )
    base = state.tasks[0]["task_base_commit"]

    agent = _TwoCommitAgent()
    agent.repo = repo

    class FakeReviewer:
        def __init__(self):
            self.review_calls = 0

        def review(self, **_kwargs):
            self.review_calls += 1
            if self.review_calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "One issue.",
                    [{"severity": "medium", "title": "x", "details": "fix it"}],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

        def summarize_commit(self, **_kwargs):
            raise agent_loop.OrchestrationError("codex summary crashed")

    result = agent_loop.run_state_loop(
        state=state,
        state_store=_NoopStore(),
        run_log=_NoopLog(),
        agent1=agent,
        reviewer=FakeReviewer(),
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    # Still squashed to one commit, but the message is the fallback built from
    # Agent 1 facts: subject = last non-empty round subject, body = labeled facts.
    assert _git(repo, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert _git(repo, "log", "-1", "--format=%s", "HEAD") == "Fix finding"
    body = _git(repo, "log", "-1", "--format=%b", "HEAD")
    assert "Implementation notes:" in body
    assert "- Round 1: c1" in body
    assert "Validation:" in body
    assert "- Round 2: t2" in body


def test_work_queue_single_commit_is_not_summarized_or_amended(
    repo: Path, tmp_path: Path
):
    """A single-commit task keeps Agent 1's message; the summary path never runs.

    This is the T08 decision: only multi-commit squash-per-task ranges are routed
    through the final Agent 2 commit-summary pass. A task that produced exactly
    one commit must be left byte-for-byte as Agent 1 committed it -- no amend, no
    re-created commit, and no ``summarize_commit`` call.
    """
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="single-commit-no-summary",
        work_queue=True,
    )
    base = state.tasks[0]["task_base_commit"]

    agent = _OneCommitAgent()
    agent.repo = repo

    class FakeReviewer:
        def __init__(self):
            self.review_calls = 0
            self.summary_calls = 0

        def review(self, **_kwargs):
            # Clean on the first (and only) review: the task ends as one commit.
            self.review_calls += 1
            return agent_loop.ReviewReply("clean", "No issues.", [])

        def summarize_commit(self, **_kwargs):
            self.summary_calls += 1
            raise AssertionError(
                "summarize_commit must not run for a single-commit task"
            )

    reviewer = FakeReviewer()
    result = agent_loop.run_state_loop(
        state=state,
        state_store=_NoopStore(),
        run_log=_NoopLog(),
        agent1=agent,
        reviewer=reviewer,
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    # The final Agent 2 summary pass never ran for a single-commit task.
    assert reviewer.summary_calls == 0
    # Exactly one commit on top of base, still carrying Agent 1's original
    # subject -- not the (absent) summary and not a rewritten commit.
    assert _git(repo, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert _git(repo, "log", "-1", "--format=%s", "HEAD") == "Implement task once"
    head = _git(repo, "rev-parse", "HEAD")
    # State/task point at the untouched single commit; the task still completes.
    assert state.commit == head
    assert state.tasks[0]["status"] == "done"
    assert state.tasks[0]["last_commit"] == head


class _SimulatedResumeCrash(Exception):
    """Marks the simulated process interruption in the end-to-end resume test."""


def test_end_to_end_squash_per_task_survives_resume(
    repo: Path, tmp_path: Path, tmp_path_factory
):
    """End-to-end work-queue squash-per-task run that crashes and then resumes.

    This is the T09 integration test. It drives the *whole* flow through the
    real ``StateStore`` (JSON persistence to disk):

        Agent 1 initial commit -> Agent 2 "issues" -> Agent 1 fix commit ->
        Agent 2 "clean" -> persist phase="commit_summary" -> [simulated crash]
        -> resume from disk -> Agent 2 commit summary -> squash.

    It asserts only the end-to-end concerns the T01-T06 unit tests do not own:
    that the accumulated ``task_commit_log`` survives the crash + reload intact,
    and that the resumed run finishes as one commit whose message is the Agent 2
    summary. Parsing (T01), schema/prompt text (T02), append round numbering
    (T03), the code-only review prompt (T04), the summary method internals
    (T05), and the fallback (T06) each have their own unit tests and are not
    re-asserted here.
    """
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="e2e-resume",
        work_queue=True,
    )

    agent = _TwoCommitAgent()
    agent.repo = repo

    class FlowReviewer:
        def __init__(self):
            self.review_calls = 0
            self.summary_calls = 0
            self.summary_log = None

        def review(self, **_kwargs):
            self.review_calls += 1
            if self.review_calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "One issue.",
                    [{"severity": "medium", "title": "x", "details": "fix it"}],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

        def summarize_commit(
            self, *, base_commit, head_commit, task_commit_log, recorded_tests
        ):
            self.summary_calls += 1
            # Snapshot what the resumed summary pass actually received so the test
            # can prove the preserved pre-crash log is *used*, not just persisted.
            self.summary_log = copy.deepcopy(list(task_commit_log))
            return agent_loop.CommitSummary(
                "Summarize the whole task",
                "Root cause: c1\n\nValidation: t2",
            )

    # The orchestrator's StateStore must live OUTSIDE the task repo. The `repo`
    # fixture returns `tmp_path`, so a `tmp_path / "state"` store would write its
    # JSON *inside* the git repo. Because the store is first written after
    # `_new_state` snapshots `baseline_untracked`, `_commit_agent_changes` would
    # treat `state/*.json` as agent-created files and sweep them into the
    # squash-per-task commits — making the end-to-end assertions pass against
    # orchestrator artifacts instead of the real production shape, where run logs
    # live outside the task diff. A separate temp dir keeps the store out of range.
    store_root = tmp_path_factory.mktemp("statestore")
    inner = agent_loop.StateStore(store_root)

    class CrashOnceStore:
        """Real StateStore that crashes the first time it persists the crash-safe
        ``commit_summary`` marker, modeling a process interruption between that
        marker and the history-rewriting squash. The state is written to disk
        before the crash, so a reload resumes from a consistent checkpoint."""

        def __init__(self):
            self.crashed = False

        def save(self, value):
            path = inner.save(value)
            if value.phase == "commit_summary" and not self.crashed:
                self.crashed = True
                raise _SimulatedResumeCrash("crash after persisting commit_summary")
            return path

        def load(self, run_id):
            return inner.load(run_id)

    store = CrashOnceStore()
    reviewer = FlowReviewer()

    # First invocation: full issues -> fix -> clean sequence, then a crash right
    # after phase="commit_summary" is persisted but before the squash lands.
    with pytest.raises(_SimulatedResumeCrash):
        agent_loop.run_state_loop(
            state=state,
            state_store=store,
            run_log=_NoopLog(),
            agent1=agent,
            reviewer=reviewer,
            max_rounds=3,
            max_dispute_rounds=3,
        )
    assert reviewer.summary_calls == 0  # crashed before the summary pass ran

    # Resume from persisted state: a fresh RunState deserialized from disk.
    resumed = store.load(state.run_id)
    assert resumed.phase == "commit_summary"
    # Deep-copy a real snapshot: run_state_loop mutates `resumed` in place, so an
    # alias of the live list would also mutate and mask an in-place clear/rewrite.
    log_before_resume = copy.deepcopy(resumed.tasks[0]["task_commit_log"])
    assert log_before_resume  # the pre-crash flow accumulated commit facts

    class UnusedAgent:
        def ask(self, _prompt):
            raise AssertionError("resume must not re-invoke Agent 1")

    result = agent_loop.run_state_loop(
        state=resumed,
        state_store=store,
        run_log=_NoopLog(),
        agent1=UnusedAgent(),
        reviewer=reviewer,
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    # The summary pass ran exactly once (on resume); review was not re-run.
    assert reviewer.summary_calls == 1
    assert reviewer.review_calls == 2
    # The resumed summary pass was fed the intact pre-crash log (used, not just
    # persisted) — compared against the immutable deep-copied snapshot.
    assert reviewer.summary_log == log_before_resume
    # The task converged to a single commit carrying the Agent 2 summary.
    base = resumed.tasks[0]["task_base_commit"]
    assert _git(repo, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert _git(repo, "log", "-1", "--format=%s", "HEAD") == "Summarize the whole task"
    body = _git(repo, "log", "-1", "--format=%b", "HEAD")
    assert "Root cause: c1" in body
    assert "Validation: t2" in body
    # The squashed task commit carries only the task's tracked change — never the
    # orchestrator's StateStore artifacts. This guards the store staying outside
    # the repo so the commit range reflects the real production shape (run logs
    # ignored/outside the task diff) rather than committed orchestrator state.
    committed = _git(repo, "diff", "--name-only", f"{base}..HEAD").splitlines()
    assert committed == ["tracked.txt"]
    # The commit log accumulated before the crash is preserved verbatim through
    # the resume boundary and into the finished task's persisted state (compared
    # against the immutable snapshot, not an alias of the mutated live list).
    final = store.load(state.run_id)
    assert final.tasks[0]["task_commit_log"] == log_before_resume


def test_commit_summary_resume_never_re_squashes(repo: Path, tmp_path: Path):
    """A crash between the squash ref-move and the state save must be idempotent.

    Simulates resuming at phase="commit_summary" when the squash already landed
    (HEAD is the single squashed commit) but the post-squash save never happened
    (state.review_head still points at the stale multi-commit tip). Resume must
    not re-run the summary pass or re-squash; it just repairs state to HEAD.
    """
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="summary-resume",
        work_queue=True,
    )
    base = state.tasks[0]["task_base_commit"]

    # Build a real multi-commit range on top of the task base.
    (repo / "tracked.txt").write_text("first\n")
    subprocess.run(["git", "commit", "-am", "c1", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("second\n")
    subprocess.run(["git", "commit", "-am", "c2", "-q"], cwd=repo, check=True)
    old_tip = _git(repo, "rev-parse", "HEAD")

    # Simulate the squash having already landed (crash was AFTER the ref move).
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    squashed = _git(
        repo,
        "commit-tree",
        tree,
        "-p",
        base,
        "-m",
        "Already squashed subject",
        "-m",
        "Already squashed body",
    )
    _git(repo, "reset", "--soft", squashed)
    head_before = _git(repo, "rev-parse", "HEAD")

    # Reconstruct the resumed state: phase="commit_summary", but review_head is
    # the stale pre-squash tip that the interrupted run never got to update.
    state.phase = "commit_summary"
    state.current_task_index = 0
    task = state.tasks[0]
    task["status"] = "running"
    task["task_commit_log"] = [
        {"round": 1, "commit_message": "c1", "root_cause": [], "changes": ["c1"], "tests": [], "deferred": []},
        {"round": 2, "commit_message": "c2", "root_cause": [], "changes": [], "tests": ["t2"], "deferred": []},
    ]
    state.review_head = old_tip
    state.commit = old_tip
    state.reply = agent_loop.asdict(_reply_with_facts("c2", tests=["t2"]))

    class GuardReviewer:
        def review(self, **_kwargs):
            raise AssertionError("review must not run when resuming at commit_summary")

        def summarize_commit(self, **_kwargs):
            raise AssertionError("summary must not re-run once the squash landed")

    result = agent_loop.run_state_loop(
        state=state,
        state_store=_NoopStore(),
        run_log=_NoopLog(),
        agent1=object(),
        reviewer=GuardReviewer(),
        max_rounds=3,
        max_dispute_rounds=3,
    )

    assert result == 0
    head = _git(repo, "rev-parse", "HEAD")
    # HEAD is unchanged (no second squash) and stays a single commit off base.
    assert head == head_before
    assert _git(repo, "rev-list", "--count", f"{base}..HEAD") == "1"
    assert _git(repo, "log", "-1", "--format=%s", "HEAD") == "Already squashed subject"
    # Stale state is repaired to the consistent HEAD and the task finishes.
    assert state.commit == head
    assert state.review_head == head
    assert state.tasks[0]["last_commit"] == head
    assert state.tasks[0]["status"] == "done"
    assert state.phase == "complete"


def test_validate_resume_tolerates_squashed_head_in_commit_summary(
    repo: Path, tmp_path: Path
):
    """Resume validation must accept a mid-squash HEAD advance.

    A crash between the squash ref move and the state save leaves HEAD on the
    squashed commit while state.commit still points at the pre-squash tip.
    _validate_resume must accept that (the phase reconciles it) but still reject
    a HEAD that is not built on the task base.
    """
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="validate-resume",
        work_queue=True,
    )
    base = state.tasks[0]["task_base_commit"]
    (repo / "tracked.txt").write_text("first\n")
    subprocess.run(["git", "commit", "-am", "c1", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("second\n")
    subprocess.run(["git", "commit", "-am", "c2", "-q"], cwd=repo, check=True)
    old_tip = _git(repo, "rev-parse", "HEAD")

    state.phase = "commit_summary"
    state.current_task_index = 0
    state.commit = old_tip

    # (1) Pre-squash: HEAD == state.commit -> accepted.
    agent_loop._validate_resume(repo, state)

    # (2) Post-squash crash: HEAD advanced to the squashed commit off base while
    #     state.commit still points at the stale tip -> accepted.
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    squashed = _git(repo, "commit-tree", tree, "-p", base, "-m", "squashed")
    _git(repo, "reset", "--soft", squashed)
    assert _git(repo, "rev-parse", "HEAD") != state.commit
    agent_loop._validate_resume(repo, state)

    # (3) A HEAD that is not built on the recorded task base is still rejected.
    state.tasks[0]["task_base_commit"] = old_tip
    state.commit = old_tip
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._validate_resume(repo, state)


def test_validate_resume_rejects_unrelated_descendant_in_commit_summary(
    repo: Path, tmp_path: Path
):
    """Resume must reject a *different* one-commit descendant of the task base.

    The commit-summary window only tolerates the saved pre-squash tip or the one
    squashed commit whose tree matches it. A sibling commit off the same base
    (HEAD advanced to an extra commit, or another branch built on the base) has a
    different tree, so accepting it would fold unrelated work into the task and
    mark it done. Regression for the over-broad is-ancestor acceptance.
    """
    queue = tmp_path / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="validate-resume-unrelated",
        work_queue=True,
    )
    base = state.tasks[0]["task_base_commit"]
    (repo / "tracked.txt").write_text("first\n")
    subprocess.run(["git", "commit", "-am", "c1", "-q"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("second\n")
    subprocess.run(["git", "commit", "-am", "c2", "-q"], cwd=repo, check=True)
    tip = _git(repo, "rev-parse", "HEAD")

    state.phase = "commit_summary"
    state.current_task_index = 0
    state.commit = tip

    # A one-commit descendant of the same base carrying a *different* tree.
    (repo / "tracked.txt").write_text("unrelated\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    other_tree = _git(repo, "write-tree")
    sibling = _git(repo, "commit-tree", other_tree, "-p", base, "-m", "unrelated")
    _git(repo, "reset", "--hard", sibling)
    assert _git(repo, "rev-parse", "HEAD") == sibling
    # One commit off base and base is its ancestor, but the tree differs from the
    # pre-squash tip, so it is not the task's squash -> rejected.
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._validate_resume(repo, state)

    # An *extra* commit stacked on the legitimate squash (count > 1) is rejected.
    squash_tree = _git(repo, "rev-parse", f"{tip}^{{tree}}")
    squashed = _git(repo, "commit-tree", squash_tree, "-p", base, "-m", "squashed")
    extra = _git(repo, "commit-tree", squash_tree, "-p", squashed, "-m", "extra")
    _git(repo, "reset", "--hard", extra)
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._validate_resume(repo, state)

    # The legitimate single squash of the recorded tip is still accepted.
    _git(repo, "reset", "--hard", squashed)
    agent_loop._validate_resume(repo, state)


def test_commit_log_append_is_gated_behind_review_range_validation(
    repo: Path, monkeypatch
):
    """Regression: the append must run *after* _validate_review_range.

    When validation raises during the commit phase, the top-level handler saves
    state at phase="commit" and a resume re-runs the whole commit block. The
    append is the only non-idempotent mutation, so if it ran before validation
    the round would be recorded twice on resume.
    """
    queue = repo / "queue.md"
    queue.write_text("- [ ] T01: fix one thing\n  Scope: tracked.txt only.\n")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="commit-log-validate-fail",
        work_queue=True,
    )
    agent_loop._start_work_queue_task(state, repo, 0)
    # Enter the commit phase directly, as if Agent 1 just returned a reply.
    state.phase = "commit"
    state.agent_mode = "initial"
    state.reply = agent_loop.asdict(
        _reply_with_facts("Implement task", changes=["c1"])
    )

    validate_calls = {"n": 0}
    real_validate = agent_loop._validate_review_range

    def flaky_validate(repo_arg, base, head):
        validate_calls["n"] += 1
        if validate_calls["n"] == 1:
            raise agent_loop.OrchestrationError("review base is not an ancestor")
        return real_validate(repo_arg, base, head)

    monkeypatch.setattr(agent_loop, "_validate_review_range", flaky_validate)

    class NoopStore:
        def save(self, _value):
            pass

    class NoopLog:
        def add(self, _title, _text):
            pass

    class StopReviewer:
        # Halt the loop right after the commit phase so no second round runs.
        def review(self, **_kwargs):
            raise agent_loop.OrchestrationError("stop after commit")

    def run_once():
        return agent_loop.run_state_loop(
            state=state,
            state_store=NoopStore(),
            run_log=NoopLog(),
            agent1=object(),
            reviewer=StopReviewer(),
            max_rounds=3,
            max_dispute_rounds=3,
        )

    # First pass: validation raises before the append, so nothing is logged and
    # the phase stays at "commit" (what a resume would pick up).
    with pytest.raises(agent_loop.OrchestrationError):
        run_once()
    assert state.tasks[0]["task_commit_log"] == []
    assert state.phase == "commit"

    # Resume: validation passes, the append runs exactly once, and the reviewer
    # stop-guard halts before any second commit round -> exactly one entry.
    with pytest.raises(agent_loop.OrchestrationError):
        run_once()
    log = state.tasks[0]["task_commit_log"]
    assert [entry["round"] for entry in log] == [1]
    assert log[0]["commit_message"] == "Implement task"
    assert log[0]["changes"] == ["c1"]


def test_agent1_schema_commit_update_shape():
    props = agent_loop.AGENT1_SCHEMA["properties"]
    # commit_update is an optional (non-required) top-level field.
    assert "commit_update" in props
    assert "commit_update" not in agent_loop.AGENT1_SCHEMA["required"]

    commit_update = props["commit_update"]
    assert commit_update["type"] == "object"
    assert commit_update["additionalProperties"] is False
    assert set(commit_update["properties"]) == {
        "root_cause",
        "changes",
        "tests",
        "deferred",
    }
    for field_schema in commit_update["properties"].values():
        assert field_schema == {"type": "array", "items": {"type": "string"}}


def test_agent1_schema_validates_replies_with_and_without_commit_update():
    jsonschema = pytest.importorskip("jsonschema")

    without = {
        "status": "conclusion",
        "message": "done",
        "commit_message": "Fix bug",
    }
    with_facts = {
        **without,
        "commit_update": {
            "root_cause": ["null deref"],
            "changes": ["guard input"],
            "tests": ["unit: guard input"],
            "deferred": [],
        },
    }

    # Both shapes validate against the schema.
    jsonschema.validate(without, agent_loop.AGENT1_SCHEMA)
    jsonschema.validate(with_facts, agent_loop.AGENT1_SCHEMA)

    # additionalProperties False still rejects unknown commit_update keys.
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {**without, "commit_update": {"bogus": ["x"]}},
            agent_loop.AGENT1_SCHEMA,
        )


def test_agent1_system_prompt_contains_commit_update_guidance():
    system = agent_loop.AGENT1_SYSTEM
    for expected in (
        "commit_update",
        "imperative subject line",
        "72 characters",
        "append-only",
        "root_cause, changes, tests, and deferred",
        "no remove, replace, or dedup",
        "no diffs",
        "at most 10 bullets",
    ):
        assert expected in system, expected


def test_work_queue_skips_task_after_dispute_limit_and_continues(
    repo: Path,
    tmp_path: Path,
):
    queue = tmp_path / "queue.md"
    queue.write_text(
        "- [ ] U0: disputed\n"
        "  This item will not converge.\n"
        "- [ ] U1: clean\n"
        "  This item should still run.\n"
    )
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement queue",
        problems_file=queue,
        run_key="skip-test",
        work_queue=True,
    )
    class CapturingStateStore:
        def __init__(self):
            self.saved = None

        def save(self, value):
            self.saved = value

        def load(self, _run_id):
            return self.saved

    store = CapturingStateStore()
    store.save(state)

    class CapturingRunLog:
        def __init__(self):
            self.entries = []

        def add(self, title, text):
            self.entries.append((title, text))

    class FakeAgent:
        def ask(self, prompt):
            if "U0: disputed" in prompt:
                (repo / "tracked.txt").write_text("bad-u0\n")
                return agent_loop.Agent1Reply(
                    "conclusion",
                    "Implemented U0 but review will not converge.",
                    "Implement U0",
                )
            (repo / "tracked.txt").write_text("good-u1\n")
            return agent_loop.Agent1Reply(
                "conclusion",
                "Implemented U1.",
                "Implement U1",
            )

    class DisputedThenCleanReviewer:
        def __init__(self):
            self.calls = 0

        def review(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return agent_loop.ReviewReply(
                    "issues",
                    "Still disputed.",
                    [{
                        "severity": "high",
                        "title": "Unresolved correctness issue",
                        "details": "tracked.txt:1 still contains the bad U0 value.",
                    }],
                )
            return agent_loop.ReviewReply("clean", "No issues.", [])

    run_log = CapturingRunLog()
    reviewer = DisputedThenCleanReviewer()
    result = agent_loop.run_state_loop(
        state=state,
        state_store=store,
        run_log=run_log,
        agent1=FakeAgent(),
        reviewer=reviewer,
        max_rounds=10,
        max_dispute_rounds=1,
    )

    saved = store.load(state.run_id)
    assert result == 0
    assert [task["status"] for task in saved.tasks] == ["skipped", "done"]
    assert "[USER TODO]" in saved.state_summary
    assert "U0-disputed skipped" in saved.state_summary
    assert (repo / "tracked.txt").read_text() == "good-u1\n"
    assert _git(repo, "log", "--oneline", "--all").count("Implement U0") == 0
    assert _git(repo, "log", "-1", "--pretty=%s") == "Implement U1"
    assert any(
        title == "Task Skipped" and "[USER TODO]" in text
        for title, text in run_log.entries
    )


def json_load(path: Path):
    import json

    return json.loads(path.read_text())


def test_state_store_round_trip_and_latest_incomplete(repo: Path, tmp_path: Path):
    store = agent_loop.StateStore(tmp_path / "state")
    state = agent_loop._new_state(
        repo=repo,
        prompt="Do work",
        problems_file=None,
    )
    store.save(state)

    loaded = store.load(state.run_id)

    assert loaded == state
    assert store.load_latest_incomplete() == state
    state.phase = "complete"
    store.save(state)
    with pytest.raises(agent_loop.OrchestrationError, match="no incomplete"):
        store.load_latest_incomplete()


def test_run_log_records_useful_messages_only(repo: Path, tmp_path: Path):
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement the feature",
        problems_file=None,
    )
    run_log = agent_loop.RunLog(
        root=tmp_path / "logs",
        state=state,
        resumed=False,
    )
    run_log.add("Agent 1", "Implemented the feature and tests pass.")
    run_log.finish("Nothing to do.")

    text = run_log.path.read_text()
    assert "Implement the feature" in text
    assert "Implemented the feature and tests pass." in text
    assert "Nothing to do." in text
    assert "bash command" not in text
    assert "thinking process" not in text


def test_resumed_log_includes_pending_question(repo: Path, tmp_path: Path):
    state = agent_loop._new_state(
        repo=repo,
        prompt="Implement the feature",
        problems_file=None,
    )
    state.phase = "awaiting_user"
    state.reply = {
        "status": "question",
        "message": "Which behavior should be used?",
        "commit_message": "",
    }

    run_log = agent_loop.RunLog(
        root=tmp_path / "logs",
        state=state,
        resumed=True,
    )

    assert "Pending Agent 1 Question" in run_log.path.read_text()
    assert "Which behavior should be used?" in run_log.path.read_text()


def test_interrupted_question_can_resume(repo: Path, tmp_path: Path):
    state = agent_loop._new_state(
        repo=repo,
        prompt="Set the requested value",
        problems_file=None,
    )
    store = agent_loop.StateStore(tmp_path / "state")
    store.save(state)

    class QuestionAgent:
        def ask(self, _prompt):
            return agent_loop.Agent1Reply("question", "Which value?", "")

    with pytest.raises(KeyboardInterrupt):
        agent_loop.run_state_loop(
            state=state,
            state_store=store,
            run_log=agent_loop.NullRunLog(),
            agent1=QuestionAgent(),
            reviewer=object(),
            max_rounds=2,
            input_fn=lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
        )

    saved = store.load(state.run_id)
    assert saved.phase == "awaiting_user"
    assert saved.reply["message"] == "Which value?"

    class FinishingAgent:
        def ask(self, prompt):
            assert "42" in prompt
            (repo / "tracked.txt").write_text("42\n")
            return agent_loop.Agent1Reply(
                "conclusion",
                "Set the value to 42.",
                "Set requested value",
            )

    class CleanReviewer:
        def review(self, **_kwargs):
            return agent_loop.ReviewReply("clean", "No issues.", [])

    answers = iter(["42"])
    result = agent_loop.run_state_loop(
        state=saved,
        state_store=store,
        run_log=agent_loop.NullRunLog(),
        agent1=FinishingAgent(),
        reviewer=CleanReviewer(),
        max_rounds=2,
        input_fn=lambda _prompt: next(answers),
    )

    assert result == 0
    assert store.load(state.run_id).phase == "complete"
    assert (repo / "tracked.txt").read_text() == "42\n"
    assert _git(repo, "log", "-1", "--pretty=%s") == "Set requested value"


def test_summarize_commit_prompt_includes_all_rounds(repo: Path, tmp_path: Path):
    # Two commits so base..head is a real range for git diff --stat.
    (repo / "tracked.txt").write_text("changed\n")
    subprocess.run(["git", "commit", "-aqm", "second"], cwd=repo, check=True)
    base = _git(repo, "rev-parse", "HEAD~1")
    head = _git(repo, "rev-parse", "HEAD")

    fake_codex = tmp_path / "fake_codex.py"
    captured_prompt = tmp_path / "prompt.txt"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(captured_prompt)!r}, 'w').write(sys.argv[-1])\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "payload = {"
        "'status': 'commit_summary',"
        "'final_commit_subject': 'Summarize task',"
        "'final_commit_body': 'Body text'}\n"
        "open(out, 'w').write(json.dumps(payload))\n"
    )
    fake_codex.chmod(0o755)

    reviewer = agent_loop.CodexReviewer(
        executable=str(fake_codex),
        repo=repo,
        model=None,
    )
    task_commit_log = [
        {
            "round": 1,
            "commit_message": "Implement feature",
            "root_cause": ["rc1"],
            "changes": ["c1"],
            "tests": ["t1"],
            "deferred": [],
        },
        {
            "round": 2,
            "commit_message": "Fix first finding",
            "root_cause": [],
            "changes": ["c2"],
            "tests": ["t2"],
            "deferred": ["d2"],
        },
        {
            "round": 3,
            "commit_message": "Fix second finding",
            "root_cause": [],
            "changes": ["c3"],
            "tests": [],
            "deferred": [],
        },
    ]

    summary = reviewer.summarize_commit(
        base_commit=base,
        head_commit=head,
        task_commit_log=task_commit_log,
        recorded_tests=["t1", "t2"],
    )

    assert isinstance(summary, agent_loop.CommitSummary)
    assert summary.final_commit_subject == "Summarize task"
    assert summary.final_commit_body == "Body text"

    prompt = captured_prompt.read_text()
    # Every recorded round appears in the prompt.
    assert "Round 1:" in prompt
    assert "Round 2:" in prompt
    assert "Round 3:" in prompt
    assert "Implement feature" in prompt
    assert "Fix first finding" in prompt
    assert "Fix second finding" in prompt
    # The diff stat range and changed file are supplied.
    assert f"Diff stat for {base}..{head}:" in prompt
    assert "tracked.txt" in prompt
    # Recorded tests are fed in.
    assert "- t1" in prompt
    assert "- t2" in prompt
    # Read-only summarizing instructions are present.
    assert "read-only summarizing pass" in prompt
    assert "Remove duplicate and" in prompt
    assert "Do not invent tests" in prompt


def test_commit_summary_or_fallback_blank_subject_uses_fallback(
    repo: Path, tmp_path: Path
):
    """A schema-valid but whitespace-only Codex subject degrades to the fallback."""
    (repo / "tracked.txt").write_text("changed\n")
    subprocess.run(["git", "commit", "-aqm", "second"], cwd=repo, check=True)
    base = _git(repo, "rev-parse", "HEAD~1")
    head = _git(repo, "rev-parse", "HEAD")

    fake_codex = tmp_path / "fake_codex.py"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output-last-message') + 1]\n"
        "payload = {"
        "'status': 'commit_summary',"
        "'final_commit_subject': '   ',"
        "'final_commit_body': 'Body text'}\n"
        "open(out, 'w').write(json.dumps(payload))\n"
    )
    fake_codex.chmod(0o755)

    reviewer = agent_loop.CodexReviewer(
        executable=str(fake_codex),
        repo=repo,
        model=None,
    )
    task = {
        "task_commit_log": [
            {
                "round": 1,
                "commit_message": "Implement feature",
                "root_cause": ["rc1"],
                "changes": ["c1"],
                "tests": ["t1"],
                "deferred": [],
            },
        ],
    }

    summary = agent_loop._commit_summary_or_fallback(
        reviewer=reviewer,
        repo=repo,
        task=task,
        base=base,
        head=head,
        run_log=_NoopLog(),
    )

    # The blank reviewer subject is rejected, so the fallback (last non-empty
    # round subject + labeled facts) is used instead of committing whitespace.
    assert summary.final_commit_subject == "Implement feature"
    assert "Implementation notes:" in summary.final_commit_body
    assert "- Round 1: c1" in summary.final_commit_body


def test_commit_summary_from_value_parses_and_validates():
    summary = agent_loop._commit_summary_from_value(
        {
            "status": "commit_summary",
            "final_commit_subject": "Do the thing",
            "final_commit_body": "Details.",
        }
    )
    assert summary.final_commit_subject == "Do the thing"
    assert summary.final_commit_body == "Details."

    # Surrounding whitespace is stripped from the subject.
    stripped = agent_loop._commit_summary_from_value(
        {
            "status": "commit_summary",
            "final_commit_subject": "  Do the thing  ",
            "final_commit_body": "Details.",
        }
    )
    assert stripped.final_commit_subject == "Do the thing"

    # A whitespace-only subject is treated as a summary failure so the caller
    # degrades to the deterministic fallback instead of committing a blank
    # subject.
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._commit_summary_from_value(
            {
                "status": "commit_summary",
                "final_commit_subject": "   ",
                "final_commit_body": "Details.",
            }
        )
    # An empty subject is likewise rejected.
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._commit_summary_from_value(
            {
                "status": "commit_summary",
                "final_commit_subject": "",
                "final_commit_body": "Details.",
            }
        )

    # A missing required field raises OrchestrationError.
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._commit_summary_from_value(
            {
                "status": "commit_summary",
                "final_commit_subject": "Do the thing",
            }
        )
    # A non-string body raises OrchestrationError.
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._commit_summary_from_value(
            {
                "status": "commit_summary",
                "final_commit_subject": "Do the thing",
                "final_commit_body": ["not", "a", "string"],
            }
        )
    # The wrong status raises OrchestrationError.
    with pytest.raises(agent_loop.OrchestrationError):
        agent_loop._commit_summary_from_value(
            {
                "status": "clean",
                "final_commit_subject": "s",
                "final_commit_body": "b",
            }
        )


def test_recorded_tests_for_summary_excludes_clean_marker():
    task = {
        "task_commit_log": [
            {
                "round": 1,
                "commit_message": "m",
                "root_cause": [],
                "changes": [],
                "tests": ["pytest tests/test_x.py -k a"],
                "deferred": [],
            },
            {
                "round": 2,
                "commit_message": "m",
                "root_cause": [],
                "changes": [],
                "tests": ["pytest tests/test_x.py -k b"],
                "deferred": [],
            },
        ],
        "task_memory": {
            "known_test_results": [
                "manual smoke test passed",
                "Agent 2 review round 1: clean",
                "Agent 2 review round 2: clean",
            ],
        },
    }

    tests = agent_loop._recorded_tests_for_summary(task)

    # Real recorded tests from the log and known_test_results are kept.
    assert "pytest tests/test_x.py -k a" in tests
    assert "pytest tests/test_x.py -k b" in tests
    assert "manual smoke test passed" in tests
    # The synthetic clean-review marker is never fed into the summary pass.
    assert all(not agent_loop._is_clean_review_marker(t) for t in tests)
    assert not any(t.startswith("Agent 2 review round") for t in tests)


def test_is_clean_review_marker_matches_only_synthetic_marker():
    assert agent_loop._is_clean_review_marker("Agent 2 review round 1: clean")
    assert agent_loop._is_clean_review_marker("Agent 2 review round 12: clean")
    assert not agent_loop._is_clean_review_marker("Agent 2 review round 1: issues")
    assert not agent_loop._is_clean_review_marker("pytest passed")


def _commit_log_entry(round_no, commit_message="", **facts):
    entry = {"round": round_no, "commit_message": commit_message}
    for field_name in agent_loop.COMMIT_UPDATE_FIELDS:
        entry[field_name] = list(facts.get(field_name, []))
    return entry


def test_fallback_commit_summary_subject_is_last_nonempty():
    # A later empty subject must not override an earlier non-empty one; the
    # subject is the last non-empty commit_message across all rounds.
    log = [
        _commit_log_entry(1, "Add parser"),
        _commit_log_entry(2, "Refine parser edge cases"),
        _commit_log_entry(3, ""),
    ]

    summary = agent_loop._fallback_commit_summary(log)

    assert isinstance(summary, agent_loop.CommitSummary)
    assert summary.final_commit_subject == "Refine parser edge cases"


def test_fallback_commit_summary_body_labels_each_round_fact():
    log = [
        _commit_log_entry(
            1,
            "Add parser",
            root_cause=["missing null guard"],
            changes=["add guard in parse()"],
            tests=["pytest -k parse"],
        ),
        _commit_log_entry(
            2,
            "Handle empty input",
            changes=["return default on empty"],
            deferred=["revisit unicode later"],
        ),
    ]

    summary = agent_loop._fallback_commit_summary(log)
    body = summary.final_commit_body

    # Facts appear under their section labels, tagged with the source round,
    # verbatim and without dedup or invention.
    assert "Root cause:" in body
    assert "- Round 1: missing null guard" in body
    assert "Implementation notes:" in body
    assert "- Round 1: add guard in parse()" in body
    assert "- Round 2: return default on empty" in body
    assert "Validation:" in body
    assert "- Round 1: pytest -k parse" in body
    assert "Deferred:" in body
    assert "- Round 2: revisit unicode later" in body
    # Sections with no facts are omitted entirely.
    assert body.count("Root cause:") == 1


def test_fallback_commit_summary_empty_log_has_safe_subject():
    summary = agent_loop._fallback_commit_summary([])

    assert summary.final_commit_subject
    assert summary.final_commit_subject == agent_loop._FALLBACK_COMMIT_SUBJECT
    assert summary.final_commit_body == ""

    # A log whose only subjects are empty still yields the safe fallback subject.
    only_empty = agent_loop._fallback_commit_summary(
        [_commit_log_entry(1, ""), _commit_log_entry(2, "   ")]
    )
    assert only_empty.final_commit_subject == agent_loop._FALLBACK_COMMIT_SUBJECT
