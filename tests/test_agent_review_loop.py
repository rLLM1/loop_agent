from __future__ import annotations

import importlib.util
import subprocess
import sys
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
        "--problems-file",
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
