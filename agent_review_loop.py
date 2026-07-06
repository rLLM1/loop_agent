#!/usr/bin/env python3
"""Run a resumable implement-review-fix loop with Claude Code and Codex."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set


APP_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_LOG_DIR = APP_DIR / "run_logs"
STATE_VERSION = 1

AGENT1_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["question", "conclusion"]},
        "message": {"type": "string"},
        "commit_message": {"type": "string"},
        "target_branch": {"type": "string"},
        "work_branch": {"type": "string"},
        "review_base": {"type": "string"},
        "review_head": {"type": "string"},
    },
    "required": ["status", "message", "commit_message"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["issues", "clean"]},
        "summary": {"type": "string"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                    "title": {"type": "string"},
                    "details": {"type": "string"},
                },
                "required": ["severity", "title", "details"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "summary", "issues"],
    "additionalProperties": False,
}

AGENT1_SYSTEM = """\
You are agent 1, the implementing coding agent. Work directly in the current
repository. Investigate, edit, and test as needed. Do not commit or push; the
orchestrator owns commits.

Return exactly the requested JSON object:
- status="question" only when user input is genuinely required before work can
  continue. Put the question and useful context in message.
- status="conclusion" when the current work is complete. Put a concise summary
  and test results in message, and provide a meaningful imperative git commit
  subject in commit_message.
- If you create or switch branches, include target_branch, work_branch,
  review_base, and review_head when known. review_base must be the commit Agent
  2 should diff from, not a stale branch from a previous task.

When responding to reviewer feedback, assess every issue independently. Fix all
issues you agree with and explain any disagreements in message. Do not make
unrelated changes.

Keep message compact: at most 10 bullets, no full diffs, no long logs, no file
dumps. Include changed files, key decisions, tests run, and blockers only.
"""


class OrchestrationError(RuntimeError):
    """Raised for an invalid response or unsafe repository state."""


@dataclass
class Agent1Reply:
    status: str
    message: str
    commit_message: str
    target_branch: str = ""
    work_branch: str = ""
    review_base: str = ""
    review_head: str = ""


@dataclass
class ReviewReply:
    status: str
    summary: str
    issues: List[Dict[str, str]]


@dataclass
class WorkItem:
    item_id: str
    title: str
    text: str
    checked: bool
    status: str = "pending"
    task_base_commit: str = ""
    last_commit: str = ""
    review_round: int = 1
    summary: str = ""


EXECUTABLE_TASK_STATUSES = {
    "pending",
    "ready",
    "in_progress",
    "execution_failed",
}

NON_EXECUTABLE_TASK_STATUSES = {
    "done",
    "skipped",
    "deferred",
    "discussion_failed",
    "discussion_failed_max_rounds",
    "blocked",
    "needs_user",
    "invalid",
    "duplicate",
    "not_applicable",
}

TASK_MEMORY_FIELDS = (
    "resolved_decisions",
    "constraints",
    "rejected_approaches",
    "known_test_results",
    "open_review_findings",
)

TASK_MEMORY_LIMITS = {
    "resolved_decisions": 5,
    "constraints": 5,
    "rejected_approaches": 3,
    "known_test_results": 5,
    "open_review_findings": 3,
}


@dataclass
class RunState:
    version: int
    run_id: str
    repo: str
    prompt: str
    problems_file: Optional[str]
    problems_text: str
    base_commit: str
    baseline_untracked: List[str]
    agent1_session_id: str
    agent1_started: bool = False
    phase: str = "agent_initial"
    agent_mode: str = "initial"
    pending_prompt: str = ""
    reply: Optional[Dict[str, str]] = None
    review: Optional[Dict[str, Any]] = None
    commit: Optional[str] = None
    review_base: Optional[str] = None
    review_head: Optional[str] = None
    target_branch: Optional[str] = None
    work_branch: Optional[str] = None
    prior_context: str = ""
    review_round: int = 1
    status_message: str = ""
    run_key: Optional[str] = None
    work_queue: bool = False
    work_item_id: Optional[str] = None
    work_item_title: Optional[str] = None
    work_item_text: Optional[str] = None
    tasks: List[Dict[str, Any]] = None
    current_task_index: int = -1
    queue_base_commit: Optional[str] = None
    final_review: bool = False
    agent1_session_mode: str = "fresh"
    state_summary: str = ""

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RunState":
        if value.get("version") != STATE_VERSION:
            raise OrchestrationError(
                f"unsupported state version {value.get('version')!r}"
            )
        try:
            return cls(**value)
        except TypeError as exc:
            raise OrchestrationError(f"invalid run state: {exc}") from exc


class StateStore:
    def __init__(
        self,
        root: Path = DEFAULT_RUN_LOG_DIR / "state",
        *,
        run_key: Optional[str] = None,
    ) -> None:
        self.root = root
        self.run_key = run_key

    def path_for(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def key_path_for(self, run_key: str) -> Path:
        return self.root / "by_key" / f"{_slugify(run_key)}.json"

    def save(self, state: RunState) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(state.run_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        key = state.run_key or self.run_key
        if key:
            key_path = self.key_path_for(key)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_tmp = key_path.with_suffix(".json.tmp")
            key_tmp.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            key_tmp.replace(key_path)
        return path

    def load(self, identifier: str) -> RunState:
        candidate = Path(identifier).expanduser()
        path = candidate if candidate.exists() else self.path_for(identifier)
        if not path.exists():
            raise OrchestrationError(f"run state not found: {identifier}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError(f"could not read run state {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise OrchestrationError(f"run state must be a JSON object: {path}")
        return RunState.from_dict(value)

    def load_latest_incomplete(self) -> RunState:
        if not self.root.exists():
            raise OrchestrationError("no saved runs found")
        paths = sorted(
            self.root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in paths:
            state = self.load(str(path))
            if state.phase not in {"complete", "max_rounds", "review_blocked"}:
                return state
        raise OrchestrationError("no incomplete saved runs found")

    def load_by_key(self, run_key: str) -> RunState:
        return self.load(str(self.key_path_for(run_key)))

    def has_key(self, run_key: str) -> bool:
        return self.key_path_for(run_key).exists()


class MemoryStateStore:
    """State-store-compatible test helper that performs no filesystem writes."""

    def save(self, state: RunState) -> None:
        return None


class RunLog:
    """Write one concise Markdown transcript for each program invocation."""

    def __init__(
        self,
        *,
        root: Path,
        state: RunState,
        resumed: bool,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = root / f"{timestamp}_{state.run_id[:8]}.md"
        started = datetime.now(timezone.utc).isoformat()
        header = "# Agent Review Loop\n\n" f"- Run ID: `{state.run_id}`\n"
        if state.run_key:
            header += f"- Run key: `{state.run_key}`\n"
        header += (
            f"- Started: `{started}`\n"
            f"- Repository: `{state.repo}`\n"
            f"- Invocation: `{'resume' if resumed else 'new'}`\n\n"
            "## User Request\n\n"
            f"{state.prompt}\n"
        )
        self.path.write_text(header, encoding="utf-8")
        if state.work_queue and state.work_item_text:
            self.add(
                "Selected Work Item",
                f"ID: `{state.work_item_id}`\n\n{state.work_item_text}",
            )
        if state.problems_text and not state.work_queue:
            source = state.problems_file or "provided document"
            self.add("Context Document", f"Source: `{source}`\n\n{state.problems_text}")
        elif state.problems_file:
            self.add("Context / Task-List File", f"Source: `{state.problems_file}`")
        if resumed and state.phase == "awaiting_user" and state.reply:
            self.add("Pending Agent 1 Question", state.reply.get("message", ""))

    def add(self, title: str, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n## {title}\n\n{text.strip() or '(empty)'}\n")

    def finish(self, text: str) -> None:
        self.add("Exit", text)


class NullRunLog:
    path = Path("")

    def add(self, title: str, text: str) -> None:
        return None

    def finish(self, text: str) -> None:
        return None


def _slugify(value: str, *, default: str = "run", max_length: int = 48) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not slug:
        slug = default
    return slug[:max_length].strip("-._") or default


def _run_label(state: RunState) -> str:
    if state.run_key:
        return _slugify(state.run_key, default="run-key")
    if state.problems_file:
        return _slugify(Path(state.problems_file).stem, default="problems")
    return _slugify(state.prompt, default="prompt")


def _terminal_phase(phase: str) -> bool:
    return phase in {"complete", "max_rounds", "review_blocked"}


def _prompt_stats(prompt: str) -> Dict[str, int]:
    return {"chars": len(prompt), "est_tokens": (len(prompt) + 3) // 4}


def _parse_work_items(markdown: str) -> List[WorkItem]:
    items: List[WorkItem] = []
    current: Optional[Dict[str, Any]] = None
    checkbox_re = re.compile(r"^\s*[-*]\s+\[([ xX])\]\s+(.+?)\s*$")

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        body = "\n".join(current["lines"]).rstrip()
        text = current["title"] if not body else f"{current['title']}\n{body}"
        item_id = _slugify(current["title"], max_length=80)
        items.append(
            WorkItem(
                item_id=item_id,
                title=current["title"],
                text=text,
                checked=current["checked"],
            )
        )
        current = None

    for line in markdown.splitlines():
        match = checkbox_re.match(line)
        if match:
            flush()
            current = {
                "checked": match.group(1).lower() == "x",
                "title": match.group(2).strip(),
                "lines": [],
            }
            continue
        if current is not None:
            current["lines"].append(line)
    flush()
    return items


def _parse_heading_work_items(markdown: str) -> List[WorkItem]:
    items: List[WorkItem] = []
    current: Optional[Dict[str, Any]] = None
    heading_re = re.compile(r"^\s*#{2,6}\s+(.+?)\s*$")

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        body = "\n".join(current["lines"]).strip()
        title = current["title"].strip()
        text = title if not body else f"{title}\n{body}"
        lowered = f"{title}\n{body}".lower()
        checked = bool(
            re.search(r"^\s*status\s*:\s*(done|complete|completed)\s*$", lowered, re.M)
            or "[x]" in title.lower()
        )
        items.append(
            WorkItem(
                item_id=_slugify(title, max_length=80),
                title=re.sub(r"^\[[ xX]\]\s*", "", title),
                text=text,
                checked=checked,
            )
        )
        current = None

    for line in markdown.splitlines():
        match = heading_re.match(line)
        if match:
            flush()
            current = {"title": match.group(1).strip(), "lines": []}
            continue
        if current is not None:
            current["lines"].append(line)
    flush()
    return items


class QueueProgressStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, run_key: str) -> Path:
        return self.root / "queues" / f"{_slugify(run_key)}.json"

    def load(self, run_key: str) -> Dict[str, Any]:
        path = self.path_for(run_key)
        if not path.exists():
            return {"done": {}}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OrchestrationError(f"could not read queue state {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise OrchestrationError(f"queue state must be a JSON object: {path}")
        value.setdefault("done", {})
        if not isinstance(value["done"], dict):
            raise OrchestrationError(f"queue state field 'done' must be an object: {path}")
        return value

    def mark_done(self, run_key: str, state: RunState) -> Path:
        if not state.work_item_id:
            raise OrchestrationError("cannot mark queue item done: state has no item id")
        value = self.load(run_key)
        value["done"][state.work_item_id] = {
            "title": state.work_item_title or state.work_item_id,
            "commit": state.commit,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.path_for(run_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path


def _select_next_work_item(
    *,
    problems_text: str,
    queue_progress: Dict[str, Any],
) -> Optional[WorkItem]:
    done = set(queue_progress.get("done", {}).keys())
    for item in _parse_work_items(problems_text):
        if not item.checked and item.item_id not in done:
            return item
    return None


def _work_item_to_task(item: WorkItem) -> Dict[str, Any]:
    status = _extract_task_status(item.text)
    if item.checked:
        status = "done"
    return {
        "id": item.item_id,
        "title": item.title,
        "text": item.text,
        "checked": item.checked,
        "status": status or "pending",
        "task_base_commit": "",
        "last_commit": "",
        "review_round": 1,
        "summary": "",
        "task_memory": _extract_task_memory_from_text(item.text),
    }


def _task_to_work_item(task: Dict[str, Any]) -> WorkItem:
    return WorkItem(
        item_id=str(task.get("id") or _slugify(str(task.get("title", "task")))),
        title=str(task.get("title") or task.get("id") or "task"),
        text=str(task.get("text") or task.get("title") or ""),
        checked=bool(task.get("checked") or task.get("status") == "done"),
        status=str(task.get("status") or "pending"),
        task_base_commit=str(task.get("task_base_commit") or ""),
        last_commit=str(task.get("last_commit") or ""),
        review_round=int(task.get("review_round") or 1),
        summary=str(task.get("summary") or ""),
    )


def _parse_task_queue(problems_text: str) -> List[Dict[str, Any]]:
    work_items = _parse_work_items(problems_text)
    if not work_items:
        work_items = _parse_heading_work_items(problems_text)
    tasks = [_work_item_to_task(item) for item in work_items]
    if not tasks:
        raise OrchestrationError(
            "task list has no executable tasks; use Markdown checkboxes or "
            "task headings"
        )
    return tasks


def _extract_task_status(text: str) -> str:
    match = re.search(r"^\s*Status\s*:\s*([A-Za-z0-9_-]+)\s*$", text, re.M)
    if not match:
        return ""
    status = match.group(1).strip().lower()
    aliases = {
        "todo": "pending",
        "complete": "done",
        "completed": "done",
    }
    return aliases.get(status, status)


def _task_needs_user(task: Dict[str, Any]) -> bool:
    text = f"{task.get('title', '')}\n{task.get('text', '')}"
    return "[USER TODO]" in text


def _task_is_executable(task: Dict[str, Any]) -> bool:
    status = str(task.get("status") or "pending").lower()
    if _task_needs_user(task):
        return False
    if status in NON_EXECUTABLE_TASK_STATUSES:
        return False
    return status in EXECUTABLE_TASK_STATUSES or not status


def _task_requires_discussion(task: Dict[str, Any]) -> bool:
    status = str(task.get("status") or "pending").lower()
    return _task_needs_user(task) or status in {
        "discussion_failed",
        "discussion_failed_max_rounds",
        "needs_user",
        "invalid",
    }


def _current_task(state: RunState) -> Optional[Dict[str, Any]]:
    if not state.tasks or state.current_task_index < 0:
        return None
    if state.current_task_index >= len(state.tasks):
        return None
    return state.tasks[state.current_task_index]


def _select_next_task_index(state: RunState) -> Optional[int]:
    for index, task in enumerate(state.tasks or []):
        if _task_is_executable(task):
            return index
    return None


def _task_queue_summary(tasks: Sequence[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    if not counts:
        return "No tasks."
    return ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))


def _upsert_status_line(body: str, status: str) -> str:
    if re.search(r"^\s*Status\s*:", body, re.M):
        return re.sub(r"^\s*Status\s*:.*$", f"  Status: {status}", body, count=1, flags=re.M)
    lines = body.splitlines()
    if not lines:
        return f"  Status: {status}"
    return "\n".join([lines[0], f"  Status: {status}", *lines[1:]])


def _empty_task_memory() -> Dict[str, List[str]]:
    return {field: [] for field in TASK_MEMORY_FIELDS}


def _clean_memory_item(text: str, *, max_len: int = 260) -> str:
    value = re.sub(r"\s+", " ", text.strip().lstrip("-*0123456789. ")).strip()
    if len(value) > max_len:
        value = value[: max_len - 3].rstrip() + "..."
    return value


def _normalize_task_memory(value: Any) -> Dict[str, List[str]]:
    memory = _empty_task_memory()
    if not isinstance(value, dict):
        return memory
    for field in TASK_MEMORY_FIELDS:
        raw_items = value.get(field, [])
        if isinstance(raw_items, str):
            raw_items = [raw_items]
        if not isinstance(raw_items, list):
            continue
        seen = set()
        items: List[str] = []
        for raw in raw_items:
            item = _clean_memory_item(str(raw))
            if not item or item in seen:
                continue
            seen.add(item)
            items.append(item)
        memory[field] = items[-TASK_MEMORY_LIMITS[field]:]
    return memory


def _memory_add(task: Dict[str, Any], field: str, items: Sequence[str]) -> None:
    if field not in TASK_MEMORY_FIELDS:
        return
    memory = _normalize_task_memory(task.get("task_memory"))
    existing = list(memory[field])
    seen = set(existing)
    for raw in items:
        item = _clean_memory_item(raw)
        if not item or item in seen:
            continue
        existing.append(item)
        seen.add(item)
    memory[field] = existing[-TASK_MEMORY_LIMITS[field]:]
    task["task_memory"] = memory


def _memory_replace(task: Dict[str, Any], field: str, items: Sequence[str]) -> None:
    if field not in TASK_MEMORY_FIELDS:
        return
    memory = _normalize_task_memory(task.get("task_memory"))
    cleaned: List[str] = []
    seen = set()
    for raw in items:
        item = _clean_memory_item(raw)
        if not item or item in seen:
            continue
        cleaned.append(item)
        seen.add(item)
    memory[field] = cleaned[-TASK_MEMORY_LIMITS[field]:]
    task["task_memory"] = memory


def _extract_task_memory_from_text(text: str) -> Dict[str, List[str]]:
    memory = _empty_task_memory()
    mapping = {
        "out of scope": "constraints",
        "dependencies/risk": "constraints",
        "acceptance": "constraints",
        "tests": "known_test_results",
    }
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z /_-]+)\s*:\s*(.+)$", line)
        if not match:
            continue
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        field = mapping.get(key)
        if field:
            memory[field].append(_clean_memory_item(f"{match.group(1).strip()}: {value}"))
    return _normalize_task_memory(memory)


def _extract_agent1_memory(reply: Agent1Reply, commit: Optional[str]) -> Dict[str, List[str]]:
    memory = _empty_task_memory()
    if commit:
        memory["resolved_decisions"].append(f"Committed {commit[:12]}: {reply.commit_message}")
    for line in reply.message.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("test", "tests", "pytest")) or " passed" in lower:
            memory["known_test_results"].append(stripped)
        elif "out of scope" in lower or "constraint" in lower or "kept" in lower:
            memory["constraints"].append(stripped)
        elif "rejected" in lower or "did not" in lower or "no change" in lower:
            memory["rejected_approaches"].append(stripped)
        elif stripped.startswith(("-", "*")):
            memory["resolved_decisions"].append(stripped)
    return _normalize_task_memory(memory)


def _review_issue_memory(review: ReviewReply) -> List[str]:
    items = []
    for issue in review.issues[: TASK_MEMORY_LIMITS["open_review_findings"]]:
        severity = str(issue.get("severity") or "issue")
        title = str(issue.get("title") or "").strip()
        details = str(issue.get("details") or "").strip()
        text = f"[{severity}] {title}: {details}" if title else f"[{severity}] {details}"
        items.append(text)
    return items


def _format_task_memory(task: Dict[str, Any]) -> str:
    memory = _normalize_task_memory(task.get("task_memory"))
    if not any(memory.values()):
        return "(none)"
    labels = {
        "resolved_decisions": "Resolved decisions",
        "constraints": "Constraints",
        "rejected_approaches": "Rejected approaches",
        "known_test_results": "Known test results",
        "open_review_findings": "Open review findings",
    }
    lines: List[str] = []
    for field in TASK_MEMORY_FIELDS:
        items = memory[field]
        if not items:
            continue
        lines.append(f"{labels[field]}:")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _render_task_memory_for_file(task: Dict[str, Any]) -> List[str]:
    memory_text = _format_task_memory(task)
    if memory_text == "(none)":
        return []
    return ["  Task memory:", *[f"    {line}" for line in memory_text.splitlines()]]


def _persist_task_list_status(state: RunState) -> None:
    if not state.work_queue or not state.problems_file or not state.tasks:
        return
    path = Path(state.problems_file)
    lines = [
        "# Agent Loop Task Queue",
        "",
        f"Run key: {state.run_key or ''}",
        f"Execution status: {state.phase}",
        f"Current task: {state.work_item_id or ''}",
        f"Last updated: {datetime.now(timezone.utc).isoformat()}",
        f"Summary: {_task_queue_summary(state.tasks)}",
        "",
    ]
    for task in state.tasks:
        title = str(task.get("title") or task.get("id") or "task")
        status = str(task.get("status") or "pending")
        marker = "x" if status == "done" else " "
        raw_text = str(task.get("text") or title)
        body_lines = raw_text.splitlines()
        body = "\n".join(body_lines[1:]).strip() if body_lines else ""
        body = _upsert_status_line(body or f"  Status: {status}", status)
        lines.append(f"- [{marker}] {title}")
        if body:
            lines.append(body)
        if task.get("last_commit"):
            lines.append(f"  Last commit: {task.get('last_commit')}")
        if task.get("summary"):
            summary = " ".join(str(task.get("summary")).split())
            lines.append(f"  Last summary: {summary[:500]}")
        lines.extend(_render_task_memory_for_file(task))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _start_work_queue_task(state: RunState, repo: Path, index: int) -> None:
    state.current_task_index = index
    task = state.tasks[index]
    task["status"] = "running"
    task["task_base_commit"] = _git(repo, "rev-parse", "HEAD")
    task["review_round"] = 1
    task["last_commit"] = ""
    task["task_memory"] = _normalize_task_memory(task.get("task_memory"))
    state.work_item_id = str(task.get("id"))
    state.work_item_title = str(task.get("title"))
    state.work_item_text = str(task.get("text"))
    state.agent_mode = "initial"
    state.phase = "agent_initial"
    state.reply = None
    state.review = None
    state.prior_context = ""
    state.review_round = 1
    state.review_base = None
    state.review_head = None
    state.pending_prompt = _task_prompt(state)


def _skip_current_work_queue_task(
    *,
    state: RunState,
    repo: Path,
    run_log: Any,
    reason: str,
) -> None:
    task = _current_task(state)
    if task is None:
        raise OrchestrationError("cannot skip work-queue task: no current task")
    if _tracked_dirty_paths(repo):
        raise OrchestrationError(
            "cannot skip current task because tracked files are modified; "
            "inspect the working tree before continuing"
        )
    task_base = str(task.get("task_base_commit") or "")
    if not task_base:
        raise OrchestrationError(
            "cannot skip current task because task_base_commit is missing"
        )
    _git(repo, "reset", "--hard", task_base)
    task["status"] = "skipped"
    task["review_round"] = state.review_round
    task["summary"] = reason
    _memory_add(task, "constraints", [f"[USER TODO] skipped: {reason}"])
    if state.review:
        _memory_replace(task, "open_review_findings", _review_issue_memory(_review_from_state(state)))
    task_id = str(task.get("id") or task.get("title") or "current task")
    message = f"[USER TODO] {task_id} skipped: {reason}"
    state.state_summary = (
        f"{state.state_summary}\n{message}".strip()
        if state.state_summary
        else message
    )
    state.commit = _git(repo, "rev-parse", "HEAD")
    run_log.add("Task Skipped", message)
    _persist_task_list_status(state)


def _task_block(task: Dict[str, Any]) -> str:
    return (
        f"- id: {task.get('id')}\n"
        f"- title: {task.get('title')}\n\n"
        f"{task.get('text', '').strip()}"
    ).strip()


def _compact_state_summary(state: RunState) -> str:
    done = [
        str(t.get("id") or t.get("title"))
        for t in (state.tasks or [])
        if t.get("status") == "done"
    ]
    current = _current_task(state)
    parts = []
    if done:
        parts.append("Completed tasks: " + ", ".join(done[-8:]))
    if current:
        parts.append(
            f"Current task: {current.get('id')} ({current.get('status', 'pending')})"
        )
        if current.get("last_commit"):
            parts.append(f"Current task last commit: {current.get('last_commit')}")
    if state.state_summary:
        parts.append(state.state_summary)
    return "\n".join(parts) if parts else "(no prior task summary)"


def _task_prompt(state: RunState, *, review_text: str = "") -> str:
    task = _current_task(state)
    if task is None:
        raise OrchestrationError("no current task selected")
    source = state.problems_file or "provided task-list file"
    mode = "fix the Codex review findings" if review_text else "implement the task"
    extra = (
        "\n\nCodex review findings to address:\n"
        f"{review_text}"
        if review_text
        else ""
    )
    return (
        f"{state.prompt}\n\n"
        f"WORK QUEUE MODE: {mode}.\n"
        "Process exactly the current task. Do not start later tasks. Keep the "
        "diff limited to this task.\n"
        "Do not commit or push; the orchestrator owns commits.\n"
        "Return compact JSON only. message <= 10 bullets; no full diffs or long logs.\n\n"
        f"Task-list/context file path: {source}\n\n"
        "Current task:\n"
        f"{_task_block(task)}\n\n"
        "Compact task memory from prior A/B rounds:\n"
        f"{_format_task_memory(task)}\n\n"
        "Compact state summary:\n"
        f"{_compact_state_summary(state)}"
        f"{extra}"
    )


def _work_queue_prompt(prompt: str, item: WorkItem, problems_file: Optional[Path]) -> str:
    source = str(problems_file.expanduser().resolve()) if problems_file else "the task-list file"
    return (
        f"{prompt}\n\n"
        "WORK QUEUE MODE\n"
        "Process exactly one work-queue item in this program run.\n"
        "Do not start later items. Keep the diff limited to this item.\n"
        "Commit exactly this item as one focused commit; the orchestrator owns the commit.\n"
        "After Agent 2 review is clean, stop. If this item has an unresolved "
        "precondition, return a question or blocked conclusion instead of guessing.\n\n"
        f"Task-list file: {source}\n\n"
        f"Selected item id: {item.item_id}\n"
        f"Selected item:\n{item.text}"
    )


class RunLock:
    def __init__(self, path: Path, *, break_lock: bool = False) -> None:
        self.path = path
        self.break_lock = break_lock
        self.acquired = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.break_lock and self.path.exists():
            self.path.unlink()
        payload = (
            f"pid={os.getpid()}\n"
            f"started={datetime.now(timezone.utc).isoformat()}\n"
        )
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = ""
            try:
                existing = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            raise OrchestrationError(
                f"run lock already exists: {self.path}"
                + (f" ({existing})" if existing else "")
                + ". Use --break-lock only if the previous process is gone."
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        self.acquired = True
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    input_text: Optional[str] = None,
    raw_log_path: Optional[Path] = None,
    heartbeat_seconds: float = 0,
    timeout_seconds: Optional[float] = None,
    heartbeat_label: str = "command",
) -> subprocess.CompletedProcess[str]:
    if raw_log_path is None and heartbeat_seconds <= 0 and timeout_seconds is None:
        try:
            return subprocess.run(
                list(command),
                cwd=cwd,
                input=input_text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise OrchestrationError(
                f"could not execute {command[0]!r}: {exc}"
            ) from exc

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []
    raw_handle = None
    raw_lock = threading.Lock()
    if raw_log_path is not None:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        raw_handle = raw_log_path.open("a", encoding="utf-8")
        raw_handle.write(
            f"# Raw subprocess log\n"
            f"started: {datetime.now(timezone.utc).isoformat()}\n"
            f"label: {heartbeat_label}\n"
            f"cwd: {cwd}\n\n"
        )
        raw_handle.flush()

    def read_stream(stream: Any, sink: List[str], stream_name: str) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                sink.append(chunk)
                if raw_handle is not None:
                    with raw_lock:
                        raw_handle.write(f"\n[{stream_name}]\n{chunk}")
                        raw_handle.flush()
        finally:
            stream.close()

    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            text=True,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        if raw_handle is not None:
            raw_handle.close()
        raise OrchestrationError(
            f"could not execute {command[0]!r}: {exc}"
        ) from exc

    if input_text is not None and process.stdin is not None:
        try:
            process.stdin.write(input_text)
            process.stdin.close()
        except BrokenPipeError:
            pass

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_parts, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_parts, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    started = time.monotonic()
    next_heartbeat = started + heartbeat_seconds if heartbeat_seconds > 0 else None
    try:
        while process.poll() is None:
            now = time.monotonic()
            elapsed = now - started
            if timeout_seconds is not None and elapsed >= timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise OrchestrationError(
                    f"{heartbeat_label} timed out after {timeout_seconds / 60:.1f} "
                    "minutes"
                )
            if next_heartbeat is not None and now >= next_heartbeat:
                raw_hint = f" Raw log: {raw_log_path}" if raw_log_path else ""
                print(
                    f"[orchestrator] Waiting for {heartbeat_label}... "
                    f"{elapsed / 60:.1f} minutes elapsed.{raw_hint}",
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds
            time.sleep(0.2)
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        if raw_handle is not None:
            with raw_lock:
                raw_handle.write(
                    f"\nfinished: {datetime.now(timezone.utc).isoformat()}\n"
                    f"exit_code: {process.returncode}\n"
                )
            raw_handle.close()

    return subprocess.CompletedProcess(
        list(command),
        process.returncode,
        "".join(stdout_parts),
        "".join(stderr_parts),
    )


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        raise OrchestrationError("agent returned an empty response")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OrchestrationError(
            f"agent did not return valid JSON: {text[:500]}"
        ) from exc

    if isinstance(value, dict):
        for key in ("structured_output", "output"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
        result = value.get("result")
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                nested = json.loads(result)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, dict):
                return nested
        return value
    raise OrchestrationError("agent JSON response must be an object")


def _require_fields(
    value: Dict[str, Any],
    *,
    status_values: Set[str],
    text_fields: Iterable[str],
) -> None:
    if value.get("status") not in status_values:
        raise OrchestrationError(
            f"invalid agent status {value.get('status')!r}; "
            f"expected one of {sorted(status_values)}"
        )
    for field_name in text_fields:
        if not isinstance(value.get(field_name), str):
            raise OrchestrationError(
                f"agent field {field_name!r} must be a string"
            )


def _agent1_reply_from_value(value: Dict[str, Any]) -> Agent1Reply:
    _require_fields(
        value,
        status_values={"question", "conclusion"},
        text_fields={"message", "commit_message"},
    )
    return Agent1Reply(
        status=value["status"],
        message=value["message"],
        commit_message=value["commit_message"],
        target_branch=value.get("target_branch") or "",
        work_branch=value.get("work_branch") or "",
        review_base=value.get("review_base") or "",
        review_head=value.get("review_head") or "",
    )


class ClaudeAgent:
    def __init__(
        self,
        *,
        executable: str,
        repo: Path,
        model: Optional[str],
        permission_mode: str,
        raw_log_dir: Optional[Path] = None,
        raw_log_label: str = "run",
        heartbeat_seconds: float = 0,
        timeout_seconds: Optional[float] = None,
        session_id: Optional[str] = None,
        started: bool = False,
        session_mode: str = "fresh",
    ) -> None:
        self.executable = executable
        self.repo = repo
        self.model = model
        self.permission_mode = permission_mode
        self.raw_log_dir = raw_log_dir
        self.raw_log_label = _slugify(raw_log_label)
        self.heartbeat_seconds = heartbeat_seconds
        self.timeout_seconds = timeout_seconds
        self.session_id = session_id or str(uuid.uuid4())
        self.started = started
        self.session_mode = session_mode
        self.call_count = 0

    def ask(self, prompt: str) -> Agent1Reply:
        self.call_count += 1
        if self.session_mode == "fresh":
            self.session_id = str(uuid.uuid4())
            self.started = False
        stats = _prompt_stats(prompt)
        print(
            "[orchestrator] Agent1 prompt: "
            f"chars={stats['chars']} est_tokens={stats['est_tokens']} "
            f"session={self.session_mode}",
            flush=True,
        )
        schema = json.dumps(AGENT1_SCHEMA, separators=(",", ":"))
        try:
            return self._ask_once(prompt, schema, retry=False)
        except OrchestrationError as first_error:
            first_error_text = str(first_error)
            if (
                first_error_text.startswith("Claude Code failed")
                or first_error_text.startswith("could not execute")
                or " timed out after " in first_error_text
            ):
                raise
            correction = (
                "Your previous response was not valid for the orchestrator. "
                "Return only one JSON object matching the required schema. "
                "Do not include Markdown or prose outside JSON. Required fields: "
                'status ("question" or "conclusion"), message, commit_message. '
                "Optional fields: target_branch, work_branch, review_base, "
                "review_head.\n\n"
                f"Original parser error: {first_error}"
            )
            try:
                return self._ask_once(correction, schema, retry=True)
            except OrchestrationError as second_error:
                raise OrchestrationError(
                    "Agent 1 did not return valid JSON after one retry. "
                    f"First error: {first_error}; retry error: {second_error}"
                ) from second_error

    def _ask_once(self, prompt: str, schema: str, *, retry: bool) -> Agent1Reply:
        command = [
            self.executable,
            "--print",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--permission-mode",
            self.permission_mode,
        ]
        if self.model:
            command.extend(["--model", self.model])
        if self.started or retry:
            command.extend(["--resume", self.session_id])
        else:
            command.extend([
                "--session-id",
                self.session_id,
                "--append-system-prompt",
                AGENT1_SYSTEM,
            ])
        command.append(prompt)

        raw_log_path = None
        if self.raw_log_dir is not None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            suffix = "retry" if retry else f"{self.call_count:03d}"
            raw_log_path = (
                self.raw_log_dir
                / (
                    f"{stamp}_{self.raw_log_label}_{self.session_id[:8]}"
                    f"_agent1_{suffix}.log"
                )
            )
        result = _run(
            command,
            cwd=self.repo,
            raw_log_path=raw_log_path,
            heartbeat_seconds=self.heartbeat_seconds,
            timeout_seconds=self.timeout_seconds,
            heartbeat_label=(
                f"agent 1 call {self.call_count}"
                + (" JSON retry" if retry else "")
            ),
        )
        if result.returncode != 0:
            raise OrchestrationError(
                "Claude Code failed "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        self.started = True
        return _agent1_reply_from_value(_extract_json_object(result.stdout))


class CodexReviewer:
    def __init__(
        self,
        *,
        executable: str,
        repo: Path,
        model: Optional[str],
        raw_log_dir: Optional[Path] = None,
        raw_log_label: str = "run",
        heartbeat_seconds: float = 0,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executable = executable
        self.repo = repo
        self.model = model
        self.raw_log_dir = raw_log_dir
        self.raw_log_label = _slugify(raw_log_label)
        self.heartbeat_seconds = heartbeat_seconds
        self.timeout_seconds = timeout_seconds
        self.call_count = 0

    def review(
        self,
        *,
        base_commit: str,
        commit: str,
        agent1_conclusion: str,
        prior_context: str = "",
    ) -> ReviewReply:
        self.call_count += 1
        prompt = f"""\
Review the cumulative code changes in {base_commit}..{commit} in the current
repository. Recheck the full range on every round, not only the latest fix
commit. Review only: do not edit files, commit, or run destructive commands.

Prior implementing-agent conclusion:
{agent1_conclusion}

Prior review/disagreement context, if any:
{prior_context or "(none)"}

Look for correctness bugs, regressions, unsafe behavior, missing validation,
compatibility problems, and important missing tests.

Review on two levels:

1. Big-picture fit:
   - Does this diff address the user's actual request and task/context document?
   - Is this diff necessary and useful for the stated goal?
   - Did agent 1 solve the right problem, or only make a locally plausible edit?
   - For docs, plans, runbooks, and migrations, is the result concrete enough
     to execute, or still too high-level?
   - Did agent 1 inspect the relevant code, docs, data, or generated artifacts
     needed to justify the change?

2. Diff correctness:
   - Look for correctness bugs, regressions, unsafe behavior, missing
     validation, compatibility problems, stale docs, contradictory
     instructions, unsupported claims, missing tests, and execution hazards.
   - For documentation or planning changes, check for vague rows, missing
     file-level detail, contradictions with nearby sections, commands that
     would fail, and claims not supported by available repo evidence.

Report only actionable findings. Do not limit findings to lines changed in the
latest diff if the user's request requires reviewing the whole affected
artifact. If no issue remains, return status="clean" with an empty issues array.
"""
        with tempfile.TemporaryDirectory(prefix="agent-review-") as tmp:
            tmp_path = Path(tmp)
            schema_path = tmp_path / "review_schema.json"
            output_path = tmp_path / "review.json"
            schema_path.write_text(json.dumps(REVIEW_SCHEMA), encoding="utf-8")
            command = [
                self.executable,
                "exec",
                "--sandbox",
                "read-only",
                "--cd",
                str(self.repo),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append(prompt)
            raw_log_path = None
            if self.raw_log_dir is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_log_path = (
                    self.raw_log_dir
                    / (
                        f"{stamp}_{self.raw_log_label}"
                        f"_agent2_review_{self.call_count:03d}.log"
                    )
                )
            result = _run(
                command,
                cwd=self.repo,
                raw_log_path=raw_log_path,
                heartbeat_seconds=self.heartbeat_seconds,
                timeout_seconds=self.timeout_seconds,
                heartbeat_label=f"agent 2 review {self.call_count}",
            )
            if result.returncode != 0:
                raise OrchestrationError(
                    f"Codex review failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            raw = output_path.read_text() if output_path.exists() else result.stdout

        value = _extract_json_object(raw)
        _require_fields(
            value,
            status_values={"issues", "clean"},
            text_fields={"summary"},
        )
        issues = value.get("issues")
        if not isinstance(issues, list):
            raise OrchestrationError("review field 'issues' must be a list")
        if value["status"] == "clean" and issues:
            raise OrchestrationError("clean review must have an empty issues list")
        return ReviewReply(
            status=value["status"],
            summary=value["summary"],
            issues=issues,
        )


class CodexPlanReviewer:
    def __init__(
        self,
        *,
        executable: str,
        repo: Path,
        model: Optional[str],
        raw_log_dir: Optional[Path] = None,
        raw_log_label: str = "run",
        heartbeat_seconds: float = 0,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.executable = executable
        self.repo = repo
        self.model = model
        self.raw_log_dir = raw_log_dir
        self.raw_log_label = _slugify(raw_log_label)
        self.heartbeat_seconds = heartbeat_seconds
        self.timeout_seconds = timeout_seconds
        self.call_count = 0

    def review_task_list(
        self,
        *,
        prompt: str,
        task_list: str,
        context_path: Optional[str] = None,
    ) -> ReviewReply:
        self.call_count += 1
        review_prompt = f"""\
Review this proposed task list. Review only: do not edit files, commit, or run
destructive commands.

Original user prompt:
{prompt}

Context file path, if any:
{context_path or "(none)"}

Proposed task list:
{task_list}

Check:
- Is each item executable as one bounded coding task?
- Does each task have enough scope, acceptance criteria, tests, and out-of-scope
  boundaries for an implementation agent?
- Are issues/tasks deduplicated without dropping distinct correctness concerns?
- Are broad design disputes marked as needing user input instead of hidden?
- Is the list suitable for checkpointed execution one item at a time?

Report only actionable issues with the task list. If no issue remains, return
status="clean" with an empty issues array.
"""
        with tempfile.TemporaryDirectory(prefix="agent-plan-review-") as tmp:
            tmp_path = Path(tmp)
            schema_path = tmp_path / "review_schema.json"
            output_path = tmp_path / "review.json"
            schema_path.write_text(json.dumps(REVIEW_SCHEMA), encoding="utf-8")
            command = [
                self.executable,
                "exec",
                "--sandbox",
                "read-only",
                "--cd",
                str(self.repo),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if self.model:
                command.extend(["--model", self.model])
            command.append(review_prompt)
            raw_log_path = None
            if self.raw_log_dir is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_log_path = (
                    self.raw_log_dir
                    / (
                        f"{stamp}_{self.raw_log_label}"
                        f"_agent2_plan_review_{self.call_count:03d}.log"
                    )
                )
            result = _run(
                command,
                cwd=self.repo,
                raw_log_path=raw_log_path,
                heartbeat_seconds=self.heartbeat_seconds,
                timeout_seconds=self.timeout_seconds,
                heartbeat_label=f"agent 2 plan review {self.call_count}",
            )
            if result.returncode != 0:
                raise OrchestrationError(
                    f"Codex plan review failed (exit {result.returncode}): "
                    f"{result.stderr.strip()}"
                )
            raw = output_path.read_text() if output_path.exists() else result.stdout

        value = _extract_json_object(raw)
        _require_fields(
            value,
            status_values={"issues", "clean"},
            text_fields={"summary"},
        )
        issues = value.get("issues")
        if not isinstance(issues, list):
            raise OrchestrationError("review field 'issues' must be a list")
        if value["status"] == "clean" and issues:
            raise OrchestrationError("clean review must have an empty issues list")
        return ReviewReply(
            status=value["status"],
            summary=value["summary"],
            issues=issues,
        )


def _git(repo: Path, *args: str) -> str:
    result = _run(["git", *args], cwd=repo)
    if result.returncode != 0:
        raise OrchestrationError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _tracked_dirty_paths(repo: Path) -> Set[str]:
    paths = set(_git(repo, "diff", "--name-only").splitlines())
    paths.update(_git(repo, "diff", "--cached", "--name-only").splitlines())
    return {path for path in paths if path}


def _untracked_paths(repo: Path) -> Set[str]:
    output = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    return {path for path in output.split("\0") if path}


def _changed_by_agent(repo: Path, baseline_untracked: Set[str]) -> Set[str]:
    paths = _tracked_dirty_paths(repo)
    paths.update(_untracked_paths(repo) - baseline_untracked)
    return paths


def _commit_agent_changes(
    repo: Path,
    *,
    baseline_untracked: Set[str],
    message: str,
) -> Optional[str]:
    paths = sorted(_changed_by_agent(repo, baseline_untracked))
    if not paths:
        return None
    subject = " ".join(message.strip().split()) or "Implement agent changes"
    _git(repo, "add", "--", *paths)
    _git(repo, "commit", "-m", subject)
    return _git(repo, "rev-parse", "HEAD")


def _prepare_branch(repo: Path, *, branch: Optional[str], branch_base: Optional[str]) -> None:
    if not branch:
        return
    if _tracked_dirty_paths(repo):
        raise OrchestrationError(
            "tracked files are already modified; commit or stash them before "
            "switching branches"
        )
    base = branch_base or "HEAD"
    exists = _run(["git", "rev-parse", "--verify", branch], cwd=repo)
    if exists.returncode == 0:
        _git(repo, "switch", branch)
        return
    _git(repo, "switch", "-c", branch, base)


def _default_task_list_file(log_root: Path, *, run_key: Optional[str]) -> Path:
    if run_key:
        return log_root / "state" / "by_key" / f"{_slugify(run_key)}.tasks.md"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return log_root / "state" / "task_lists" / f"{timestamp}_tasks.md"


def _squash_current_task_commit(repo: Path, task: Dict[str, Any]) -> Optional[str]:
    base = str(task.get("task_base_commit") or "")
    if not base:
        return None
    head = _git(repo, "rev-parse", "HEAD")
    if head == base:
        return head
    count_text = _git(repo, "rev-list", "--count", f"{base}..HEAD")
    try:
        count = int(count_text)
    except ValueError:
        count = 1
    if count <= 1:
        return head
    subjects = _git(repo, "log", "--reverse", "--format=%s", f"{base}..HEAD").splitlines()
    subject = subjects[0] if subjects else str(task.get("title") or "Implement task")
    _git(repo, "reset", "--soft", base)
    _git(repo, "commit", "-m", subject)
    return _git(repo, "rev-parse", "HEAD")


def _read_context_excerpt(context_file: Optional[Path], *, max_chars: int = 60000) -> tuple[Optional[str], str]:
    if context_file is None:
        return None, ""
    resolved = context_file.expanduser().resolve()
    try:
        text = resolved.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OrchestrationError(f"could not read context file {context_file}: {exc}") from exc
    if not text:
        raise OrchestrationError("context file is empty")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[TRUNCATED: context file exceeded prompt budget]"
    return str(resolved), text


def _discussion_prompt(
    *,
    prompt: str,
    context_path: Optional[str],
    context_text: str,
    review_text: str = "",
) -> str:
    review_block = (
        "\n\nAgent B review feedback to address before finalizing the task list:\n"
        f"{review_text}\n"
        if review_text
        else ""
    )
    return (
        f"{prompt}\n\n"
        "DISCUSSION MODE. Do not edit repository files and do not commit. "
        "Produce a normalized Markdown task list suitable for execution mode.\n\n"
        "Each task must be a Markdown checkbox item using this shape:\n"
        "- [ ] T01: Short imperative title\n"
        "  Status: ready\n"
        "  Problem: ...\n"
        "  Scope: ...\n"
        "  Acceptance: ...\n"
        "  Tests: ...\n"
        "  Out of scope: ...\n"
        "  Dependencies/Risk: ...\n"
        "  Reason: required for any status other than ready/done\n"
        "  Context: required for any skipped/deferred/failed/needs_user task\n"
        "  Options: required when user input is needed\n"
        "  Next action: required for any non-success status\n\n"
        "Rules: make tasks bounded, deduplicate duplicates, preserve distinct "
        "correctness concerns, mark user-decision items with [USER TODO], and "
        "set user-decision task Status to needs_user. "
        "Return only the task list in message. Set commit_message to "
        "'Write task list'.\n\n"
        f"Context file path: {context_path or '(none)'}\n\n"
        f"Context excerpt:\n{context_text or '(none)'}"
        f"{review_block}"
    )


def _mark_discussion_todo_tasks(task_list: str, review_text: str) -> str:
    task_ids = set(re.findall(r"\bT\d+\b", review_text))
    if not task_ids:
        return task_list
    lines = task_list.splitlines()
    output: List[str] = []
    for line in lines:
        output.append(line)
        match = re.match(r"^(\s*[-*]\s+\[[ xX]\]\s+)(T\d+)\b", line)
        if match and match.group(2) in task_ids:
            output.append("  Status: needs_user")
            output.append(
                "  [USER TODO]: Discussion did not converge for this task; "
                "see the Pre-execution Review Caveats section."
            )
    return "\n".join(output)


def run_discussion_loop(
    *,
    prompt: str,
    context_file: Optional[Path],
    task_list_file: Path,
    agent1: Any,
    plan_reviewer: Any,
    run_log: Any,
    max_rounds: int,
    allow_partial: bool = False,
) -> int:
    context_path, context_text = _read_context_excerpt(context_file)
    review_text = ""
    latest_task_list = ""
    last_review_text = ""
    for round_index in range(1, max_rounds + 1):
        reply = agent1.ask(
            _discussion_prompt(
                prompt=prompt,
                context_path=context_path,
                context_text=context_text,
                review_text=review_text,
            )
        )
        if reply.status == "question":
            raise OrchestrationError(
                "discussion mode cannot continue without user input: "
                f"{reply.message}"
            )
        latest_task_list = reply.message.strip()
        if not latest_task_list:
            raise OrchestrationError("Agent 1 returned an empty task list")
        task_list_file.parent.mkdir(parents=True, exist_ok=True)
        task_list_file.write_text(latest_task_list + "\n", encoding="utf-8")
        run_log.add(f"Agent 1 Task List {round_index}", latest_task_list)
        review = plan_reviewer.review_task_list(
            prompt=prompt,
            task_list=latest_task_list,
            context_path=context_path,
        )
        formatted = _format_review(review)
        last_review_text = formatted
        run_log.add(f"Agent 2 Plan Review {round_index}", formatted)
        print(f"\n[agent 2 plan review {round_index}]\n{formatted}", flush=True)
        if review.status == "clean":
            run_log.add("Result", f"Task list written to `{task_list_file}`.")
            return 0
        review_text = formatted
    if allow_partial and latest_task_list:
        latest_task_list = _mark_discussion_todo_tasks(latest_task_list, last_review_text)
        caveats = (
            "\n\n## Pre-execution Review Caveats\n\n"
            "Status: discussion_failed_max_rounds\n\n"
            "[USER TODO]: Discussion reached the maximum review rounds. Tasks "
            "named below need user clarification; execution may continue on "
            "other ready tasks.\n\n"
            f"{last_review_text.strip()}\n"
        )
        task_list_file.write_text(latest_task_list.rstrip() + caveats, encoding="utf-8")
        run_log.add(
            "Result",
            f"Discussion maxed out after {max_rounds} rounds; partial task "
            f"list with [USER TODO] caveats written to `{task_list_file}`.",
        )
        return 0
    raise OrchestrationError(
        f"discussion did not converge after {max_rounds} review rounds; "
        f"latest task list was written to {task_list_file}"
    )


def _format_review(review: ReviewReply) -> str:
    lines = [review.summary]
    for index, issue in enumerate(review.issues, 1):
        lines.append(
            f"{index}. [{issue.get('severity', 'unknown')}] "
            f"{issue.get('title', '')}: {issue.get('details', '')}"
        )
    return "\n".join(lines)


_BLOCKED_REVIEW_PATTERNS = (
    "unable to perform the requested repository review",
    "could not inspect the repository",
    "cannot inspect the repository",
    "can't inspect the repository",
    "could not inspect the repo",
    "cannot inspect the repo",
    "can't inspect the repo",
    "no repository evidence",
    "no repo evidence",
    "all commands rejected",
    "all available shell commands are rejected",
    "could not run git diff",
    "cannot run git diff",
    "can't run git diff",
)

_ACTIONABLE_REVIEW_HINTS = (
    ".py:",
    ".md:",
    ".toml:",
    ".yaml:",
    ".yml:",
    ".json:",
    ".rs:",
    ".go:",
    ".ts:",
    ".tsx:",
    ".js:",
    ".jsx:",
    "add a regression test",
    "add tests",
    "fix ",
    "narrow this check",
    "either ",
)


def _review_is_blocked(review: ReviewReply) -> bool:
    if review.status != "issues" or not review.issues:
        return False
    text = " ".join(
        [review.summary]
        + [
            f"{issue.get('title', '')} {issue.get('details', '')}"
            for issue in review.issues
        ]
    ).lower()
    # A reviewer may include a test caveat such as "I could not run pytest
    # because the session is read-only" while still giving concrete repo
    # findings. That is not a blocked review; only stop when the review itself
    # says repo inspection failed.
    if any(hint in text for hint in _ACTIONABLE_REVIEW_HINTS):
        return False
    return any(pattern in text for pattern in _BLOCKED_REVIEW_PATTERNS)


def _resolve_review_base(repo: Path, reply: Agent1Reply, fallback: str) -> str:
    candidate = reply.review_base or fallback
    try:
        return _git(repo, "rev-parse", "--verify", f"{candidate}^{{commit}}")
    except OrchestrationError as exc:
        raise OrchestrationError(
            f"Agent 1 provided invalid review base {candidate!r}: {exc}"
        ) from exc


def _resolve_review_head(repo: Path, reply: Agent1Reply, fallback: str) -> str:
    candidate = reply.review_head or fallback
    try:
        return _git(repo, "rev-parse", "--verify", f"{candidate}^{{commit}}")
    except OrchestrationError as exc:
        raise OrchestrationError(
            f"Agent 1 provided invalid review head {candidate!r}: {exc}"
        ) from exc


def _resolve_commit(repo: Path, candidate: str, label: str) -> str:
    try:
        return _git(repo, "rev-parse", "--verify", f"{candidate}^{{commit}}")
    except OrchestrationError as exc:
        raise OrchestrationError(f"invalid {label} {candidate!r}: {exc}") from exc


def _validate_review_range(repo: Path, base: str, head: str) -> None:
    try:
        _git(repo, "merge-base", "--is-ancestor", base, head)
    except OrchestrationError as exc:
        raise OrchestrationError(
            f"review_base {base[:12]} is not an ancestor of "
            f"review_head {head[:12]}"
        ) from exc


def _combined_prompt(prompt: str, problems_file: Optional[Path]) -> tuple[str, str]:
    if problems_file is None:
        return prompt, ""
    resolved = problems_file.expanduser().resolve()
    try:
        problems_text = resolved.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OrchestrationError(
            f"could not read task/context document {problems_file}: {exc}"
        ) from exc
    if not problems_text:
        raise OrchestrationError("task/context document is empty")
    combined = (
        f"{prompt}\n\n"
        "Use the supplied context document as task context, but do not copy it "
        "into your response. Read only the relevant sections needed for the "
        "current request.\n\n"
        f"Context file: {resolved}"
    )
    return combined, problems_text


def _new_state(
    *,
    repo: Path,
    prompt: str,
    problems_file: Optional[Path],
    run_key: Optional[str] = None,
    work_queue: bool = False,
    work_item: Optional[WorkItem] = None,
) -> RunState:
    if _tracked_dirty_paths(repo):
        raise OrchestrationError(
            "tracked files are already modified; commit or stash them before "
            "running the agent loop"
        )
    combined, problems_text = _combined_prompt(prompt, problems_file)
    tasks: List[Dict[str, Any]] = []
    current_task_index = -1
    queue_base_commit = None
    if work_queue:
        if not problems_text:
            raise OrchestrationError("work queue mode needs a task-list document")
        tasks = _parse_task_queue(problems_text)
        if work_item is not None:
            wanted = work_item.item_id
            current_task_index = next(
                (i for i, task in enumerate(tasks) if task.get("id") == wanted),
                -1,
            )
        if current_task_index < 0:
            current_task_index = next(
                (i for i, task in enumerate(tasks) if _task_is_executable(task)),
                -1,
            )
        if current_task_index < 0:
            raise OrchestrationError("work queue has no executable unfinished tasks")
        tasks[current_task_index]["status"] = "running"
        tasks[current_task_index]["task_base_commit"] = _git(repo, "rev-parse", "HEAD")
        work_item = _task_to_work_item(tasks[current_task_index])
        queue_base_commit = _git(repo, "rev-parse", "HEAD")
        # In queue mode the state and prompts carry only the selected task, not
        # the full task/context document. The parsed compact task list is the durable
        # queue state.
        problems_text = ""
        combined = ""
    return RunState(
        version=STATE_VERSION,
        run_id=str(uuid.uuid4()),
        repo=str(repo),
        prompt=prompt,
        problems_file=(
            str(problems_file.expanduser().resolve()) if problems_file else None
        ),
        problems_text=problems_text,
        base_commit=_git(repo, "rev-parse", "HEAD"),
        baseline_untracked=sorted(_untracked_paths(repo)),
        agent1_session_id=str(uuid.uuid4()),
        pending_prompt=combined,
        run_key=run_key,
        work_queue=work_queue,
        work_item_id=work_item.item_id if work_item else None,
        work_item_title=work_item.title if work_item else None,
        work_item_text=work_item.text if work_item else None,
        tasks=tasks,
        current_task_index=current_task_index,
        queue_base_commit=queue_base_commit,
    )


def _reply_from_state(state: RunState) -> Agent1Reply:
    if state.reply is None:
        raise OrchestrationError("saved state is missing agent 1's conclusion")
    return Agent1Reply(**state.reply)


def _review_from_state(state: RunState) -> ReviewReply:
    if state.review is None:
        raise OrchestrationError("saved state is missing agent 2's review")
    return ReviewReply(**state.review)


def _validate_resume(repo: Path, state: RunState) -> None:
    if repo != Path(state.repo):
        raise OrchestrationError(
            f"saved run belongs to {state.repo}, not {repo}"
        )
    expected = state.commit or state.base_commit
    head = _git(repo, "rev-parse", "HEAD")
    if head != expected:
        raise OrchestrationError(
            "repository HEAD changed since the saved checkpoint: "
            f"expected {expected}, found {head}"
        )
    if state.phase in {"agent_initial", "agent_fix"} and _tracked_dirty_paths(repo):
        raise OrchestrationError(
            "resume stopped: tracked files are modified while the saved phase "
            "is waiting to call Agent 1. This usually means the previous Claude "
            "call edited files but did not return structured JSON. Inspect the "
            "working tree before resuming."
        )


def run_state_loop(
    *,
    state: RunState,
    state_store: Any,
    run_log: Any,
    agent1: Any,
    reviewer: Any,
    max_rounds: int,
    max_dispute_rounds: Optional[int] = None,
    commit_policy: str = "squash-per-task",
    input_fn: Callable[[str], str] = input,
) -> int:
    repo = Path(state.repo)
    baseline_untracked = set(state.baseline_untracked)
    dispute_limit = max_dispute_rounds or max_rounds

    while True:
        if state.phase in {"complete", "max_rounds", "review_blocked"}:
            message = state.status_message or "Nothing to do; run already finished."
            print(f"\n[orchestrator] {message}", flush=True)
            return 0 if state.phase == "complete" else 2

        if state.phase in {"agent_initial", "agent_fix"}:
            if state.work_queue:
                if _current_task(state) is None:
                    next_index = _select_next_task_index(state)
                    if next_index is None:
                        state.phase = "complete"
                        state.status_message = (
                            "All work-queue tasks are complete."
                            + (
                                f" Task list: {state.problems_file}"
                                if state.problems_file
                                else ""
                            )
                        )
                        _persist_task_list_status(state)
                        state_store.save(state)
                        return 0
                    _start_work_queue_task(state, repo, next_index)
                    _persist_task_list_status(state)
                if not state.pending_prompt:
                    state.pending_prompt = _task_prompt(state)
                _persist_task_list_status(state)
            # Persist resume intent before the external call. If interrupted,
            # the next invocation resumes the same Claude session and prompt.
            was_started = state.agent1_started
            state.agent1_started = True
            state_store.save(state)
            try:
                reply = agent1.ask(state.pending_prompt)
            except OrchestrationError:
                # A normal CLI error means no usable response was created.
                # Preserve resume intent only for process interruption.
                state.agent1_started = getattr(agent1, "started", was_started)
                state_store.save(state)
                raise
            state.reply = asdict(reply)
            run_log.add("Agent 1", reply.message)
            print("\n[agent 1]\n" + reply.message, flush=True)
            if reply.status == "question":
                state.phase = "awaiting_user"
            else:
                state.phase = "commit"
            state_store.save(state)
            continue

        if state.phase == "awaiting_user":
            question = _reply_from_state(state).message
            answer = input_fn("\nYour response: ")
            run_log.add("User Response", answer)
            state.pending_prompt = (
                "The user answered your question:\n\n"
                f"{answer}\n\nContinue the task."
            )
            state.phase = "agent_initial" if state.agent_mode == "initial" else "agent_fix"
            state_store.save(state)
            continue

        if state.phase == "commit":
            reply = _reply_from_state(state)
            new_commit = _commit_agent_changes(
                repo,
                baseline_untracked=baseline_untracked,
                message=reply.commit_message,
            )
            if new_commit is None:
                if state.commit is None:
                    state.commit = _git(repo, "rev-parse", "HEAD")
                message = "Agent 1 made no file changes."
            else:
                state.commit = new_commit
                message = f"Committed `{new_commit[:12]}`: {reply.commit_message}"
            task = _current_task(state) if state.work_queue else None
            if task is not None and state.commit:
                task["last_commit"] = state.commit
                task["summary"] = reply.message
                agent1_memory = _extract_agent1_memory(reply, state.commit)
                for field, items in agent1_memory.items():
                    _memory_add(task, field, items)
            task_review_base = ""
            if task is not None:
                task_review_base = str(task.get("task_base_commit") or state.base_commit)
            if task is not None and state.commit:
                review_base = _resolve_commit(
                    repo,
                    task_review_base,
                    "task review base",
                )
                review_head = _resolve_commit(
                    repo,
                    state.commit,
                    "orchestrator-managed review head",
                )
            else:
                review_base = _resolve_review_base(
                    repo,
                    reply,
                    (task.get("task_base_commit") if task else None)
                    or state.review_base
                    or state.base_commit,
                )
                review_head = _resolve_review_head(
                    repo,
                    reply,
                    state.commit or state.base_commit,
                )
            _validate_review_range(repo, review_base, review_head)
            state.review_base = review_base
            state.review_head = review_head
            state.target_branch = reply.target_branch or state.target_branch
            state.work_branch = reply.work_branch or state.work_branch
            print(f"\n[orchestrator] {message}", flush=True)
            run_log.add("Commit", message)

            if state.agent_mode == "fix":
                review_text = _format_review(_review_from_state(state))
                state.prior_context = (
                    f"Agent 2 previously reported:\n{review_text}\n\n"
                    f"Agent 1 responded:\n{reply.message}"
                )
                state.review_round += 1
            state.phase = "review"
            state_store.save(state)
            continue

        if state.phase == "review":
            round_limit = dispute_limit if state.work_queue else max_rounds
            if state.review_round > round_limit:
                if state.work_queue:
                    task = _current_task(state)
                    reason = (
                        f"Agent A and Agent B still have unresolved findings "
                        f"after {round_limit} review/dispute rounds. Dropped "
                        "this task's branch changes and moving to the next task."
                    )
                    _skip_current_work_queue_task(
                        state=state,
                        repo=repo,
                        run_log=run_log,
                        reason=reason,
                    )
                    next_index = _select_next_task_index(state)
                    if next_index is not None:
                        skipped_id = task.get("id") if task else ""
                        _start_work_queue_task(state, repo, next_index)
                        state.status_message = (
                            f"Task {skipped_id} skipped after dispute limit; "
                            f"starting {state.tasks[next_index].get('id')}."
                        )
                        state_store.save(state)
                        run_log.add("Result", state.status_message)
                        continue
                    state.phase = "complete"
                    state.status_message = (
                        "All work-queue tasks are complete; at least one task "
                        "was skipped with [USER TODO] follow-up."
                        + (
                            f" Task list: {state.problems_file}"
                            if state.problems_file
                            else ""
                        )
                    )
                    _persist_task_list_status(state)
                    state_store.save(state)
                    run_log.add("Result", state.status_message)
                    return 0
                state.phase = "max_rounds"
                state.status_message = (
                    f"Stopped after {max_rounds} review rounds with unresolved "
                    "findings."
                )
                state_store.save(state)
                run_log.add("Result", state.status_message)
                return 2

            reply = _reply_from_state(state)
            review = reviewer.review(
                base_commit=state.review_base or state.base_commit,
                commit=state.review_head or state.commit or state.base_commit,
                agent1_conclusion=reply.message,
                prior_context=state.prior_context,
            )
            state.review = asdict(review)
            review_text = _format_review(review)
            title = f"Agent 2 Review {state.review_round}"
            run_log.add(title, review_text)
            print(f"\n[{title.lower()}]\n{review_text}", flush=True)
            if review.status == "clean":
                if state.work_queue:
                    task = _current_task(state)
                    if task is not None:
                        _memory_replace(task, "open_review_findings", [])
                        _memory_add(
                            task,
                            "known_test_results",
                            [f"Agent 2 review round {state.review_round}: clean"],
                        )
                        if commit_policy == "squash-per-task":
                            squashed = _squash_current_task_commit(repo, task)
                            if squashed:
                                state.commit = squashed
                                state.review_head = squashed
                                task["last_commit"] = squashed
                        task["status"] = "done"
                        task["review_round"] = state.review_round
                        task["summary"] = _reply_from_state(state).message
                        _persist_task_list_status(state)
                    next_index = _select_next_task_index(state)
                    if next_index is not None:
                        _start_work_queue_task(state, repo, next_index)
                        _persist_task_list_status(state)
                        state.status_message = (
                            f"Task {task.get('id') if task else ''} clean; "
                            f"starting {state.tasks[next_index].get('id')}."
                        )
                        state_store.save(state)
                        run_log.add("Result", state.status_message)
                        continue
                    state.phase = "complete"
                    state.status_message = (
                        "All work-queue tasks are complete."
                        + (
                            f" Task list: {state.problems_file}"
                            if state.problems_file
                            else ""
                        )
                    )
                    _persist_task_list_status(state)
                else:
                    state.phase = "complete"
                    state.status_message = "Agent 2 found no remaining issues."
                state_store.save(state)
                run_log.add("Result", state.status_message)
                return 0
            if _review_is_blocked(review):
                state.phase = "review_blocked"
                state.status_message = (
                    "Agent 2 review blocked by its execution environment; "
                    "not sending the blocker back to Agent 1 as a code issue."
                )
                state_store.save(state)
                run_log.add("Result", state.status_message)
                return 2

            if state.work_queue:
                task = _current_task(state)
                if task is not None:
                    _memory_replace(task, "open_review_findings", _review_issue_memory(review))
            state.agent_mode = "fix"
            state.pending_prompt = (
                _task_prompt(state, review_text=review_text)
                if state.work_queue
                else (
                    "Agent 2 reviewed your latest committed work. Assess every "
                    "finding, fix all findings you agree with, test the result, "
                    "and explain any disagreements. Do not commit.\n\n"
                    f"{review_text}"
                )
            )
            state.phase = "agent_fix"
            state_store.save(state)
            continue

        raise OrchestrationError(f"unknown saved phase: {state.phase}")


def run_loop(
    *,
    repo: Path,
    initial_prompt: str,
    agent1: Any,
    reviewer: Any,
    max_rounds: int,
    max_dispute_rounds: Optional[int] = None,
    commit_policy: str = "squash-per-task",
    input_fn: Callable[[str], str] = input,
) -> int:
    """Compatibility wrapper used by callers that do not need persistence."""
    state = _new_state(repo=repo, prompt=initial_prompt, problems_file=None)
    return run_state_loop(
        state=state,
        state_store=MemoryStateStore(),
        run_log=NullRunLog(),
        agent1=agent1,
        reviewer=reviewer,
        max_rounds=max_rounds,
        max_dispute_rounds=max_dispute_rounds,
        commit_policy=commit_policy,
        input_fn=input_fn,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Initial request for agent 1")
    parser.add_argument(
        "--prompt",
        dest="prompt_option",
        help="Initial request for agent 1; alternative to the positional prompt",
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "discussion", "execute", "discuss-execute"],
        default="legacy",
        help="Run the legacy loop, task-list discussion, execution, or both",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="Optional context document for discussion modes",
    )
    parser.add_argument(
        "--task-list-file",
        type=Path,
        help="Normalized Markdown task-list checkpoint file",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="Git repository to modify (default: current directory)",
    )
    parser.add_argument(
        "--work-queue",
        action="store_true",
        help="Process task-list items one at a time",
    )
    parser.add_argument(
        "--new-branch",
        dest="new_branch",
        help=(
            "Create this branch before starting a new execution run, or switch "
            "to it if it already exists"
        ),
    )
    parser.add_argument(
        "--branch",
        dest="new_branch",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--branch-base",
        help="Base commit/ref used when creating --new-branch (default: HEAD)",
    )
    parser.add_argument(
        "--run-key",
        help="Stable key for same-command resume and work-queue progress",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a fresh keyed run instead of resuming an existing run",
    )
    parser.add_argument(
        "--break-lock",
        action="store_true",
        help="Break a stale --run-key lock before starting",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        metavar="RUN_ID_OR_STATE_PATH",
        help="Resume a saved run by ID or state JSON path",
    )
    resume.add_argument(
        "--resume-last",
        action="store_true",
        help="Resume the most recently saved incomplete run",
    )
    parser.add_argument("--agent1-command", default="claude")
    parser.add_argument("--agent2-command", default="codex")
    parser.add_argument("--agent1-model")
    parser.add_argument("--agent2-model")
    parser.add_argument(
        "--agent1-permission-mode",
        choices=["acceptEdits", "auto", "bypassPermissions", "dontAsk"],
        default="acceptEdits",
        help="Claude Code permission mode (default: acceptEdits)",
    )
    parser.add_argument(
        "--agent1-session-mode",
        choices=["fresh", "resume"],
        default="fresh",
        help="Use fresh Claude sessions per call or legacy resumed sessions",
    )
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--commit-policy",
        choices=["squash-per-task", "per-round", "no-commit"],
        default="squash-per-task",
        help="Commit cleanup policy; default squashes each queue task to one commit",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=False,
        help="Reserved for future use; current script never pushes",
    )
    parser.add_argument(
        "--no-push",
        action="store_false",
        dest="push",
        help="Do not push changes (default)",
    )
    parser.add_argument(
        "--auto-execute",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For discuss-execute mode, continue into execution after writing tasks",
    )
    parser.add_argument(
        "--max-dispute-rounds",
        type=int,
        default=5,
        help=(
            "In work-queue mode, skip the current task after this many "
            "unresolved Agent A/B review rounds and continue to the next task"
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30,
        help="Print a waiting heartbeat this often for Claude/Codex calls",
    )
    parser.add_argument(
        "--agent-timeout-minutes",
        type=float,
        default=60,
        help="Abort an individual Claude/Codex call after this many minutes",
    )
    parser.add_argument(
        "--run-log-dir",
        type=Path,
        default=DEFAULT_RUN_LOG_DIR,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.prompt and args.prompt_option:
        raise SystemExit("provide either positional prompt or --prompt, not both")
    prompt_arg = args.prompt_option or args.prompt
    if args.push:
        raise SystemExit("--push is not implemented; the agent loop is no-push only")
    if args.commit_policy == "no-commit":
        raise SystemExit(
            "--commit-policy no-commit is not implemented because Agent 2 "
            "reviews committed ranges"
        )
    log_root = args.run_log_dir.expanduser().resolve()
    raw_log_dir = log_root / "raw"
    mode = args.mode
    context_file = args.context_file
    task_list_file = args.task_list_file
    queue_file: Optional[Path] = None
    if mode == "legacy":
        if args.work_queue:
            queue_file = task_list_file
        task_list_file = None
    elif mode == "execute":
        args.work_queue = True
        if task_list_file is None:
            raise SystemExit("--mode execute requires --task-list-file")
        queue_file = task_list_file
    elif mode == "discussion":
        if task_list_file is None:
            task_list_file = _default_task_list_file(log_root, run_key=args.run_key)
        queue_file = None
    elif mode == "discuss-execute":
        if args.auto_execute is False:
            raise SystemExit(
                "--no-auto-execute is not supported yet for discuss-execute; "
                "use --mode discussion instead"
            )
        if task_list_file is None:
            task_list_file = _default_task_list_file(log_root, run_key=args.run_key)
        args.work_queue = True
        queue_file = task_list_file
    else:
        queue_file = None
    if args.max_rounds <= 0:
        raise SystemExit("--max-rounds must be positive")
    if args.max_dispute_rounds <= 0:
        raise SystemExit("--max-dispute-rounds must be positive")
    if args.heartbeat_seconds < 0:
        raise SystemExit("--heartbeat-seconds must be non-negative")
    if args.agent_timeout_minutes <= 0:
        raise SystemExit("--agent-timeout-minutes must be positive")
    if args.work_queue and not queue_file:
        raise SystemExit("--work-queue requires --task-list-file")
    if args.work_queue and not args.run_key:
        raise SystemExit("--work-queue requires --run-key")
    if args.new_run and not args.run_key:
        raise SystemExit("--new-run requires --run-key")
    if args.break_lock and not args.run_key:
        raise SystemExit("--break-lock requires --run-key")

    state_store = StateStore(log_root / "state", run_key=args.run_key)
    resumed = bool(args.resume or args.resume_last)
    lock = (
        RunLock(
            log_root / "state" / "by_key" / f"{_slugify(args.run_key)}.lock",
            break_lock=args.break_lock,
        )
        if args.run_key
        else None
    )
    try:
        if lock is not None:
            lock.__enter__()
        if mode == "discussion":
            if args.resume or args.resume_last:
                raise OrchestrationError("discussion mode resume is not implemented")
            prompt = prompt_arg
            if not prompt:
                prompt = input("Task for agent 1: ").strip()
            if not prompt:
                raise OrchestrationError("a non-empty prompt is required")
            repo = (args.repo or Path.cwd()).expanduser().resolve()
            if not (repo / ".git").exists():
                raise OrchestrationError(f"not a git repository: {repo}")
            for executable in (args.agent1_command, args.agent2_command):
                if shutil.which(executable) is None:
                    raise OrchestrationError(
                        f"required executable not found: {executable}"
                    )
            raw_log_dir.mkdir(parents=True, exist_ok=True)
            dummy_state = RunState(
                version=STATE_VERSION,
                run_id=str(uuid.uuid4()),
                repo=str(repo),
                prompt=prompt,
                problems_file=str(context_file.expanduser().resolve()) if context_file else None,
                problems_text="",
                base_commit=_git(repo, "rev-parse", "HEAD"),
                baseline_untracked=sorted(_untracked_paths(repo)),
                agent1_session_id=str(uuid.uuid4()),
                run_key=args.run_key,
            )
            run_log = RunLog(root=log_root, state=dummy_state, resumed=False)
            print(f"[orchestrator] Run log: {run_log.path}", flush=True)
            print(f"[orchestrator] Raw logs: {raw_log_dir}", flush=True)
            raw_log_label = _run_label(dummy_state)
            agent1 = ClaudeAgent(
                executable=args.agent1_command,
                repo=repo,
                model=args.agent1_model,
                permission_mode=args.agent1_permission_mode,
                raw_log_dir=raw_log_dir,
                raw_log_label=raw_log_label,
                heartbeat_seconds=args.heartbeat_seconds,
                timeout_seconds=args.agent_timeout_minutes * 60,
                session_mode=args.agent1_session_mode,
            )
            plan_reviewer = CodexPlanReviewer(
                executable=args.agent2_command,
                repo=repo,
                model=args.agent2_model,
                raw_log_dir=raw_log_dir,
                raw_log_label=raw_log_label,
                heartbeat_seconds=args.heartbeat_seconds,
                timeout_seconds=args.agent_timeout_minutes * 60,
            )
            result = run_discussion_loop(
                prompt=prompt,
                context_file=context_file,
                task_list_file=task_list_file.expanduser().resolve(),
                agent1=agent1,
                plan_reviewer=plan_reviewer,
                run_log=run_log,
                max_rounds=args.max_rounds,
                allow_partial=False,
            )
            run_log.finish(f"Discussion complete. Task list: `{task_list_file}`")
            return result
        if mode == "discuss-execute":
            if args.resume or args.resume_last:
                raise OrchestrationError("discuss-execute mode resume is not implemented")
            prompt = prompt_arg
            if not prompt:
                prompt = input("Task for agent 1: ").strip()
            if not prompt:
                raise OrchestrationError("a non-empty prompt is required")
            repo = (args.repo or Path.cwd()).expanduser().resolve()
            if not (repo / ".git").exists():
                raise OrchestrationError(f"not a git repository: {repo}")
            existing_tasks: List[Dict[str, Any]] = []
            resolved_task_list = task_list_file.expanduser().resolve()
            if resolved_task_list.exists():
                try:
                    existing_tasks = _parse_task_queue(
                        resolved_task_list.read_text(encoding="utf-8")
                    )
                except OrchestrationError:
                    existing_tasks = []
            has_executable = any(_task_is_executable(task) for task in existing_tasks)
            has_discussion_tasks = any(
                _task_requires_discussion(task) for task in existing_tasks
            )
            should_run_discussion = (
                not existing_tasks or (not has_executable and has_discussion_tasks)
            )
            for executable in (args.agent1_command, args.agent2_command):
                if shutil.which(executable) is None:
                    raise OrchestrationError(
                        f"required executable not found: {executable}"
                    )
            if should_run_discussion:
                raw_log_dir.mkdir(parents=True, exist_ok=True)
                discussion_context = (
                    resolved_task_list
                    if existing_tasks and resolved_task_list.exists()
                    else context_file
                )
                discussion_prompt = prompt
                if existing_tasks:
                    discussion_prompt = (
                        f"{prompt}\n\nResume discussion only for tasks marked "
                        "[USER TODO], needs_user, discussion_failed, or "
                        "discussion_failed_max_rounds. Do not rewrite clean or "
                        "ready tasks."
                    )
                dummy_state = RunState(
                    version=STATE_VERSION,
                    run_id=str(uuid.uuid4()),
                    repo=str(repo),
                    prompt=discussion_prompt,
                    problems_file=str(discussion_context.expanduser().resolve()) if discussion_context else None,
                    problems_text="",
                    base_commit=_git(repo, "rev-parse", "HEAD"),
                    baseline_untracked=sorted(_untracked_paths(repo)),
                    agent1_session_id=str(uuid.uuid4()),
                    run_key=args.run_key,
                )
                discussion_log = RunLog(root=log_root, state=dummy_state, resumed=False)
                print(f"[orchestrator] Discussion run log: {discussion_log.path}", flush=True)
                print(f"[orchestrator] Raw logs: {raw_log_dir}", flush=True)
                raw_log_label = _run_label(dummy_state)
                agent1 = ClaudeAgent(
                    executable=args.agent1_command,
                    repo=repo,
                    model=args.agent1_model,
                    permission_mode=args.agent1_permission_mode,
                    raw_log_dir=raw_log_dir,
                    raw_log_label=raw_log_label,
                    heartbeat_seconds=args.heartbeat_seconds,
                    timeout_seconds=args.agent_timeout_minutes * 60,
                    session_mode=args.agent1_session_mode,
                )
                plan_reviewer = CodexPlanReviewer(
                    executable=args.agent2_command,
                    repo=repo,
                    model=args.agent2_model,
                    raw_log_dir=raw_log_dir,
                    raw_log_label=raw_log_label,
                    heartbeat_seconds=args.heartbeat_seconds,
                    timeout_seconds=args.agent_timeout_minutes * 60,
                )
                result = run_discussion_loop(
                    prompt=discussion_prompt,
                    context_file=discussion_context,
                    task_list_file=resolved_task_list,
                    agent1=agent1,
                    plan_reviewer=plan_reviewer,
                    run_log=discussion_log,
                    max_rounds=args.max_rounds,
                    allow_partial=True,
                )
                discussion_log.finish(f"Discussion complete. Task list: `{task_list_file}`")
                if result != 0:
                    return result
            else:
                print(
                    f"[orchestrator] Existing task list has ready work; "
                    f"skipping discussion: {resolved_task_list}",
                    flush=True,
                )
            # Fall through into execution using the generated task list.
            queue_file = resolved_task_list
        if args.resume:
            if prompt_arg or queue_file:
                raise OrchestrationError(
                    "a resumed run cannot accept a new prompt or task-list file"
                )
            state = state_store.load(args.resume)
        elif args.resume_last:
            if prompt_arg or queue_file:
                raise OrchestrationError(
                    "a resumed run cannot accept a new prompt or task-list file"
                )
            state = state_store.load_latest_incomplete()
        else:
            prompt = prompt_arg
            if not prompt:
                prompt = input("Task for agent 1: ").strip()
            if not prompt:
                raise OrchestrationError("a non-empty prompt is required")
            repo = (args.repo or Path.cwd()).expanduser().resolve()
            if not (repo / ".git").exists():
                raise OrchestrationError(f"not a git repository: {repo}")
            if mode in {"execute", "discuss-execute"}:
                _prepare_branch(
                    repo,
                    branch=args.new_branch,
                    branch_base=args.branch_base,
                )
            problems_path = (
                queue_file.expanduser().resolve()
                if queue_file
                else None
            )
            if args.work_queue and problems_path and problems_path.exists():
                parsed_tasks = _parse_task_queue(problems_path.read_text(encoding="utf-8"))
                if not any(_task_is_executable(task) for task in parsed_tasks):
                    print(
                        "[orchestrator] Task list has no executable unfinished "
                        f"tasks. Summary: {_task_queue_summary(parsed_tasks)}. "
                        f"Task list: {problems_path}",
                        flush=True,
                    )
                    return 0
            if args.run_key and state_store.has_key(args.run_key) and not args.new_run:
                existing = state_store.load_by_key(args.run_key)
                if not _terminal_phase(existing.phase):
                    if existing.prompt != prompt:
                        raise OrchestrationError(
                            "existing incomplete keyed run uses a different prompt; "
                            "use the same command, --resume, --new-run, or a new --run-key"
                        )
                    if existing.problems_file != (str(problems_path) if problems_path else None):
                        raise OrchestrationError(
                            "existing incomplete keyed run uses a different task-list file; "
                            "use the same command, --resume, --new-run, or a new --run-key"
                        )
                    state = existing
                    resumed = True
                elif args.work_queue:
                    print(
                        f"[orchestrator] Work queue `{args.run_key}` is complete. "
                        "Use --new-run to start over.",
                        flush=True,
                    )
                    return 0
                else:
                    raise OrchestrationError(
                        f"run key {args.run_key!r} already finished; use "
                        "--new-run to start over"
                    )
            else:
                state = _new_state(
                    repo=repo,
                    prompt=prompt,
                    problems_file=problems_path,
                    run_key=args.run_key,
                    work_queue=args.work_queue,
                )
                state_store.save(state)

        repo = Path(state.repo)
        if args.repo and args.repo.expanduser().resolve() != repo:
            raise OrchestrationError(
                f"--repo does not match saved run repository {repo}"
            )
        if resumed:
            _validate_resume(repo, state)
        for executable in (args.agent1_command, args.agent2_command):
            if shutil.which(executable) is None:
                raise OrchestrationError(
                    f"required executable not found: {executable}"
                )

        run_log = RunLog(root=log_root, state=state, resumed=resumed)
        print(f"[orchestrator] Run log: {run_log.path}", flush=True)
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[orchestrator] Raw logs: {raw_log_dir}", flush=True)
        raw_log_label = _run_label(state)
        run_log.add(
            "Raw Logs",
            f"Raw Claude/Codex subprocess logs are written to `{raw_log_dir}`.\n"
            f"Raw log filename label: `{raw_log_label}`.\n"
            "Use `tail -f` on the current file for live progress.",
        )
        agent1 = ClaudeAgent(
            executable=args.agent1_command,
            repo=repo,
            model=args.agent1_model,
            permission_mode=args.agent1_permission_mode,
            raw_log_dir=raw_log_dir,
            raw_log_label=raw_log_label,
            heartbeat_seconds=args.heartbeat_seconds,
            timeout_seconds=args.agent_timeout_minutes * 60,
            session_id=state.agent1_session_id,
            started=state.agent1_started,
            session_mode=args.agent1_session_mode,
        )
        reviewer = CodexReviewer(
            executable=args.agent2_command,
            repo=repo,
            model=args.agent2_model,
            raw_log_dir=raw_log_dir,
            raw_log_label=raw_log_label,
            heartbeat_seconds=args.heartbeat_seconds,
            timeout_seconds=args.agent_timeout_minutes * 60,
        )
        try:
            result = run_state_loop(
                state=state,
                state_store=state_store,
                run_log=run_log,
                agent1=agent1,
                reviewer=reviewer,
                max_rounds=args.max_rounds,
                max_dispute_rounds=args.max_dispute_rounds,
                commit_policy=args.commit_policy,
            )
        except KeyboardInterrupt:
            state_store.save(state)
            run_log.finish("Interrupted. The saved run can be resumed safely.")
            print("\nInterrupted; run state saved.", file=sys.stderr)
            return 130
        except OrchestrationError as exc:
            state_store.save(state)
            run_log.finish(f"Error: {exc}")
            print(f"error: {exc}", file=sys.stderr)
            return 1
        run_log.finish(state.status_message or "Nothing to do.")
        return result
    except OrchestrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
