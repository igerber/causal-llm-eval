"""Three-layer telemetry capture for agent runs.

Layer 1 - Stream-JSON event log from Claude Code:
    Parsed from `claude --print --output-format stream-json` output. Contains
    every user/assistant turn, every tool call (Bash, Read, Edit, Write, Grep)
    with arguments and results, and file reads with paths.

Layer 2 - In-process Python instrumentation (the discoverability ground truth):
    A `sitecustomize.py` installed in the per-run venv hooks the target library
    and writes per-event JSON records. Catches access that stream-JSON misses
    (e.g., `python -c "from diff_diff import get_llm_guide"` reads the file via
    Python, not Claude's Read tool). See `harness/sitecustomize_template.py`.

Layer 3 - Subprocess stderr capture:
    Captures CLI-level errors emitted by `claude --bare` and any other stderr
    the agent's process writes. Python-level warnings from the agent's
    diff_diff calls are captured by layer 2 via the `warnings.warn` wrapper
    (stack-inspecting, stacklevel-safe), not here.

The merger (`merge_layers`) parses layers 1+2 and emits a per-run
`TelemetryRecord` with arm-aware sentinel semantics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TelemetryRecord:
    """Merged per-run record assembled from three layers.

    Discoverability fields use a three-state encoding:
        - ``True``  - the agent accessed this surface
        - ``False`` - the agent did NOT access this surface (and could have)
        - ``None``  - this surface is "not applicable" to the arm; absence is
          structural, not behavioral. Used for arm-2 (statsmodels) on the
          `opened_llms_*` and `called_get_llm_guide` fields, since statsmodels
          ships no LLM-targeted guides.

    Comparator-fairness analysis distinguishes "could-have-but-didn't" (False)
    from "couldn't-have-because-no-such-feature" (None). Collapsing these
    biases the comparison in favor of the arm with no guidance surface.

    The ``arm`` field is required so ``merge_layers()`` and downstream graders
    can validate that the sentinel pattern matches the arm's contract (e.g.,
    ``opened_llms_txt is None`` iff ``arm == "statsmodels"``).
    """

    arm: str  # "diff_diff" or "statsmodels"; drives sentinel semantics below
    stream_json_path: Path
    in_process_events_path: Path
    stderr_path: Path
    # Discoverability flags - tri-state per docstring above
    opened_llms_txt: bool | None = None
    opened_llms_practitioner: bool | None = None
    opened_llms_autonomous: bool | None = None
    opened_llms_full: bool | None = None
    called_get_llm_guide: bool | None = None
    get_llm_guide_variants: tuple[str, ...] = ()
    # The remaining flags ARE applicable to both arms (warnings, diagnostics,
    # estimator instantiation), so they remain plain bool / tuple.
    saw_fit_time_warning: bool = False
    diagnostic_methods_invoked: tuple[str, ...] = ()
    estimator_classes_instantiated: tuple[str, ...] = ()

    # Set of fields whose tri-state encoding depends on arm.
    _ARM_SENTINEL_FIELDS = (
        "opened_llms_txt",
        "opened_llms_practitioner",
        "opened_llms_autonomous",
        "opened_llms_full",
        "called_get_llm_guide",
    )
    _VALID_ARMS = ("diff_diff", "statsmodels")

    def __post_init__(self) -> None:
        """Enforce arm-specific contracts on construction.

        - Reject unknown arms.
        - For arm == "diff_diff", every sentinel-bearing field MUST be a bool
          (not None). The diff-diff arm has guide surfaces; the merger is
          obligated to fill them in.
        - For arm == "statsmodels", every sentinel-bearing field MUST be None
          (not bool). The statsmodels arm has no guide surfaces; encoding
          False would conflate "not applicable" with "not discovered".

        Catching this at construction prevents downstream graders/analysis from
        silently consuming corrupted records.
        """
        if self.arm not in self._VALID_ARMS:
            raise ValueError(f"TelemetryRecord.arm={self.arm!r} is not one of {self._VALID_ARMS}")
        for field_name in self._ARM_SENTINEL_FIELDS:
            value = getattr(self, field_name)
            if self.arm == "diff_diff":
                if not isinstance(value, bool):
                    raise ValueError(
                        f"TelemetryRecord(arm='diff_diff').{field_name} must be "
                        f"bool (True/False), got {value!r}. The diff-diff arm "
                        f"has guide surfaces; the merger must record discovery "
                        f"outcome, not leave the field as None."
                    )
            elif self.arm == "statsmodels":
                if value is not None:
                    raise ValueError(
                        f"TelemetryRecord(arm='statsmodels').{field_name} must "
                        f"be None (not applicable), got {value!r}. statsmodels "
                        f"has no guide surfaces; encoding True/False would "
                        f"conflate 'not applicable' with 'not discovered'."
                    )


class TelemetryMergeError(RuntimeError):
    """Raised when ``merge_layers`` cannot produce a valid TelemetryRecord.

    Distinct cases:
    - In-process event log file missing entirely.
    - Malformed line in the event log (non-JSON).
    - Runner-written ``telemetry_missing`` sentinel present (post-exec the
      agent's event log disappeared; the runner already wrote the sentinel
      and downgraded exit_code to -2).
    - Cross-layer inconsistency: agent's transcript shows python invocations
      but the in-process layer has no ``session_start`` event (shim never
      loaded; cold-start eval is invalid).

    All four cases are fail-closed; the merger does not silently downgrade
    to a "no agent activity" record.
    """


# diff_diff bundled guide files mapped to the variant string `get_llm_guide`
# accepts. Verified against `diff_diff/_guides_api.py:7-12`. Key ordering
# follows the source for diff-friendliness.
_VARIANT_TO_FILENAME: dict[str, str] = {
    "concise": "llms.txt",
    "full": "llms-full.txt",
    "practitioner": "llms-practitioner.txt",
    "autonomous": "llms-autonomous.txt",
}


# Word-boundary regex for python-invocation detection in the layer-1
# cross-check. Matches `python`, `python3`, `python3.11` at a token boundary.
# Boundary set includes `/` so absolute interpreter paths
# (`/usr/bin/python3 script.py`, `/opt/venv/bin/python -c "..."`) are caught,
# and `;`/`&`/`|`/`(` so compound shell commands
# (`pip install foo && python script.py`) are caught. The trailing `\s|$`
# requirement guards against false positives like `/opt/python/` (the `/`
# after `python` is neither whitespace nor end-of-string).
_PYTHON_INVOCATION_RE = re.compile(r"(?:^|[\s;&|()/])python(?:3(?:\.\d+)?)?(?:[\s<>]|$)")


# Shell patterns that mutate or override the resolved interpreter location.
# Any of these in a command containing a python invocation means the
# resolved interpreter may NOT be the per-arm-venv python — fail-closed.
#
# Forms detected:
# - Inline / pre-invocation / exported PATH mutation
# - venv activation: `source X`, `. X`, `conda activate`, `pyenv shell`
# - env-driven resolution: `env python`, `env -u VAR python`
# - Local binary: `./python`
#
# Detection is conservative: ANY of these patterns in a command that ALSO
# contains a python invocation triggers fail-closed. This over-catches
# benign cases (e.g. `echo PATH=/foo && python`), but agents rarely emit
# such patterns for non-bypass reasons; failing-closed is correct.
_PATH_ASSIGNMENT_RE = re.compile(r"(?:^|[\s;&|])(?:export\s+)?PATH=\S+")
# PYTHONPATH / PYTHONHOME / PYTHONSTARTUP / PYTHONUSERBASE all alter
# Python's import resolution or run startup code in the agent's process;
# if either is set, the interpreter resolves modules differently from a
# clean per-arm-venv start, and the shim's discoverability claims are
# at risk.
_PYTHON_ENV_VAR_RE = re.compile(
    r"(?:^|[\s;&|])(?:export\s+)?PYTHON(?:PATH|HOME|STARTUP|USERBASE)=\S+"
)
_SHELL_ACTIVATION_RE = re.compile(
    r"(?:^|[\s;&|])(?:source\s+\S+|\.\s+\S+|conda\s+activate|pyenv\s+shell)"
)
# Wrapper commands that launch Python through a tool-managed environment.
# The launched Python may not have sitecustomize installed (uv/poetry use
# their own venv; conda run / pyenv exec use the named env).
_PYTHON_WRAPPER_RE = re.compile(r"(?:^|[\s;&|])(?:uv\s+run|poetry\s+run|conda\s+run|pyenv\s+exec)")
# env wrapper: bare ``env``, ``/usr/bin/env``, ``/bin/env``, ``/usr/local/bin/env``,
# all followed by optional flags (e.g. ``env -S python``, ``env -u VAR python``)
# and then the python token. The path-prefix variants resolve to the same
# binary as bare ``env`` and bypass PATH-based interpreter resolution the
# same way.
_ENV_BEFORE_PYTHON_RE = re.compile(
    r"(?:^|[\s;&|()])(?:/(?:usr/(?:local/)?)?bin/)?env" r"(?:\s+-\S+(?:\s+\S+)?)*\s+python"
)
_DOT_PYTHON_RE = re.compile(r"(?:^|[\s;&|])\./python")
# Command-substitution that resolves to a python interpreter path:
# ``$(which python)``, ``$(command -v python)``, ``$(type -p python)``, and
# backtick equivalents. The substitution resolves at shell-eval time to a
# path we cannot predict, so any python invocation through this form is
# fail-closed (no session_start argv can be matched against an opaque
# substitution).
_PYTHON_COMMAND_SUBSTITUTION_RE = re.compile(
    r"(?:\$\(|`)\s*(?:which|command\s+-v|type\s+-p)\s+python(?:3(?:\.\d+)?)?\s*(?:\)|`)"
)
# Shell wrappers that take a quoted code payload (``bash -c``, ``sh -c``,
# ``zsh -c``, ``bash -lc``, etc. plus ``eval`` and ``exec``). When such a
# wrapper appears AND the command also contains a ``python``-shaped token
# anywhere (including inside the quoted payload), the inner python invocation
# is invisible to the regex-based attribution path; fail-closed.
_SHELL_WRAPPER_RE = re.compile(
    # Shell name, then zero or more option tokens (lazy), then -c (or -lc /
    # -ic / -Sc etc). The lazy `(?:\s+\S+)*?` allows option-token forms like
    # `bash --noprofile -c "..."` or `bash -o pipefail -c "..."`.
    r"(?:^|[\s;&|()])(?:bash|sh|zsh|dash|ksh)\b(?:\s+\S+)*?\s+-[a-zA-Z]*c(?:\s|$)"
)
_EVAL_WRAPPER_RE = re.compile(r"(?:^|[\s;&|()])(?:eval|exec)(?:\s|$)")
# Command modifier prefixes that take a command as their argument:
# `command python ...`, `time python ...`, `nohup python ...`, `nice python ...`,
# `timeout 30 python ...`, `xargs python ...`. The python token is not in
# command position by `_is_in_command_position`, so the extractor skips it.
# When a command pairs one of these with a python literal AND a `-S`-style
# bypass flag literal, fail closed.
_COMMAND_MODIFIER_BEFORE_PYTHON_RE = re.compile(
    r"(?:^|[\s;&|()])(?:command|time|nohup|nice|timeout|xargs)"
    r"(?:\s+-?\d+\S*)*"  # optional numeric/short-flag args (e.g. `timeout 30`)
    r"\s+(?:[A-Z_][A-Za-z0-9_]*=\S+\s+)*"  # optional KEY=value env-prefixes
    r"python(?:3(?:\.\d+)?)?\b"
)
# Word-boundary `-S` short-flag pattern (also catches compact forms like
# `-Sc`, `-IS`). Used together with the modifier pattern above to detect
# command-prefix bypass without enumerating every modifier+flag combination.
_BARE_BYPASS_FLAG_RE = re.compile(r"(?:^|\s)-[A-Za-z]*S[A-Za-z]*(?:[\s=]|$)")
# Word-boundary ``python`` match that ignores ``pythonic`` / ``/opt/python_libs/``
# but DOES find ``python`` inside quoted strings (because string content is
# inspected raw here, not via shlex). Used together with the wrapper regexes
# to detect inner-payload bypasses.
_PYTHON_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_])python(?:3(?:\.\d+)?)?(?![A-Za-z0-9_])")


# Recursive shell unwrapper: strip known wrapper forms (modifiers, shell -c
# wrappers, brace groups, control prefixes, command substitutions) so the
# inner python invocation is exposed to the existing primitive detectors.
# The architectural insight: PR #5's per-arm-venv install makes wrappers
# THEMSELVES harmless (any python that PATH-resolves to the venv binary
# auto-loads sitecustomize). The remaining bypass primitives are ``-S``,
# absolute non-venv interpreter paths, and ``PATH=`` overrides; what the
# unwrapper protects is letting THOSE be hidden inside arbitrary wrappers.
#
# Each extractor is intentionally narrow: it pattern-matches one recognized
# wrapper form and yields the inner content. The orchestrator iterates to
# fixed point so e.g. ``time bash -c "python -S ..."`` unwraps to the
# inner ``python -S ...`` in two passes.
_BASH_DASH_C_RE = re.compile(
    r"(?:^|[\s;&|()])(?:bash|sh|zsh|dash|ksh)\b(?:\s+\S+)*?\s+-[a-zA-Z]*c\s+(['\"])((?:(?!\1).)*)\1"
)
_EVAL_PAYLOAD_RE = re.compile(r"(?:^|[\s;&|()])(?:eval|exec)\s+(['\"])((?:(?!\1).)*)\1")
_DOLLAR_PAREN_RE = re.compile(r"\$\(([^()]+)\)")
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_BRACE_GROUP_RE = re.compile(r"\{\s+(.+?);\s*\}")
_PAREN_SUBSHELL_RE = re.compile(r"(?:^|\s)\(\s*([^()]+?)\s*\)")
_IF_PREFIX_RE = re.compile(r"^if\s+(.+?);\s*then\b", re.DOTALL)
_WHILE_PREFIX_RE = re.compile(r"^(?:while|until)\s+(.+?);\s*do\b", re.DOTALL)
_NEGATION_PREFIX_RE = re.compile(r"^!\s+(.+)$")
# Shell control BODIES (test command vs body command): the prefix patterns
# above extract the test command; these extract the body. ``then BODY (else
# BODY)? fi``, ``do BODY done``, and ``case ... in PATTERN) BODY ;;`` arms.
_THEN_BODY_RE = re.compile(r"\bthen\s+(.+?)(?:\s*;\s*(?:else|elif|fi)\b|$)", re.DOTALL)
_ELSE_BODY_RE = re.compile(r"\belse\s+(.+?)(?:\s*;\s*fi\b|$)", re.DOTALL)
_DO_BODY_RE = re.compile(r"\bdo\s+(.+?)(?:\s*;\s*done\b|$)", re.DOTALL)
_CASE_ARM_RE = re.compile(r"\)\s+([^;)]+?)\s*;;", re.DOTALL)

# Recognized command modifiers that take a command (and possibly options /
# numeric args) before that command.
_COMMAND_MODIFIER_NAMES = (
    "command",
    "time",
    "nohup",
    "nice",
    "timeout",
    "env",
    "xargs",
    "exec",
    "stdbuf",
    "ionice",
    "chrt",
)


def _strip_command_modifier_prefix(command: str) -> list[str]:
    """If ``command`` starts with a recognized modifier word (``command``,
    ``time``, ``nohup``, ``nice``, ``timeout``, ``env``, ``xargs``,
    ``exec``, etc.), generate progressive-strip variants and return them.

    Generates multiple variants by progressively stripping the modifier
    word, then the modifier + 1 leading token, then + 2 tokens, etc.,
    stopping at the first variant whose remainder begins with a python
    invocation. This catches modifier-with-args forms like ``nice -n 10
    python ...`` and ``timeout --signal=KILL 30 python ...`` without
    requiring per-modifier knowledge of which options take values: the
    extractor sees the python-prefixed variant and pulls the
    invocation, while the in-between variants are no-ops.

    Returns ``[]`` if no modifier is present.
    """
    s = command.lstrip()
    for mod in _COMMAND_MODIFIER_NAMES:
        if s.startswith(mod + " ") or s.startswith(mod + "\t"):
            after_mod = s[len(mod) :].lstrip()
            if not after_mod:
                return []
            results = [after_mod]
            tokens = after_mod.split()
            for i in range(1, len(tokens)):
                rest = " ".join(tokens[i:])
                if not rest:
                    continue
                results.append(rest)
                # Stop once we've reached a python-prefixed remainder; further
                # stripping would just re-traverse what extraction already covers.
                if re.match(r"python(?:3(?:\.\d+)?)?(?:\s|$)", rest):
                    break
            return results
    return []


def _strip_shell_control_prefix(command: str) -> list[str]:
    """Extract test commands AND body commands from shell-control structures.

    Test commands (return what the structure tests):
    - ``! CMD`` -> ``CMD``
    - ``if CMD; then ...`` -> ``CMD``
    - ``while CMD; do ...`` / ``until CMD; do ...`` -> ``CMD``

    Body commands (return what the structure executes):
    - ``then BODY ; (else|elif|fi)`` -> ``BODY``
    - ``else BODY ; fi`` -> ``BODY``
    - ``do BODY ; done`` -> ``BODY`` (covers for/while/until loops)
    - ``case ... in PATTERN) BODY ;;`` -> each ``BODY`` arm

    Body extraction matters because a one-line ``for f in x; do python "$f";
    done`` or ``if true; then python -S; fi`` would otherwise hide the
    python invocation from extraction (the python token isn't in command
    position by ``_is_in_command_position`` because ``do`` / ``then`` are
    not in the separator set).
    """
    out: list[str] = []
    s = command.lstrip()
    m = _NEGATION_PREFIX_RE.match(s)
    if m:
        out.append(m.group(1).strip())
    m = _IF_PREFIX_RE.match(s)
    if m:
        out.append(m.group(1).strip())
    m = _WHILE_PREFIX_RE.match(s)
    if m:
        out.append(m.group(1).strip())
    # Body extraction: search anywhere in the command (not just at start).
    for m in _THEN_BODY_RE.finditer(s):
        out.append(m.group(1).strip())
    for m in _ELSE_BODY_RE.finditer(s):
        out.append(m.group(1).strip())
    for m in _DO_BODY_RE.finditer(s):
        out.append(m.group(1).strip())
    for m in _CASE_ARM_RE.finditer(s):
        out.append(m.group(1).strip())
    return out


def _unwrap_command_for_inspection(command: str, max_depth: int = 10) -> list[str]:
    """Return a list of inner-command strings extracted by recursively
    unwrapping known shell wrapper forms.

    The original command is always the first element. Each recognized
    wrapper contributes one or more entries: the bare inner command after
    the wrapper is stripped. Recursion is bounded by ``max_depth`` and a
    visited set so cyclic-looking constructs cannot loop.

    Wrappers handled:

    - **Quoted-payload shell wrappers**: ``bash [opts] -c "CODE"``,
      ``sh -c 'CODE'``, ``eval "CODE"``, ``exec "CODE"`` (and zsh/dash/ksh).
    - **Command modifiers**: ``command``, ``time``, ``nohup``, ``nice
      [-n N]``, ``timeout [opts] N``, ``env [-u VAR | VAR=val]*``,
      ``xargs [opts]``, ``exec``, ``stdbuf``, ``ionice``, ``chrt``.
    - **Shell control**: ``! CMD``, ``if CMD; then ...``, ``while CMD; do
      ...``, ``until CMD; do ...``, ``{ CMD; }`` brace groups,
      ``( CMD )`` subshells.
    - **Substitution**: ``$(CMD)``, ``\\`CMD\\```.

    The bypass detector iterates over the returned list and runs primitive
    bypass checks (``-S`` flag literal, absolute non-venv path, ``PATH=``)
    on each variant. A primitive hidden inside any recognized wrapper is
    surfaced.
    """
    seen: set[str] = {command}
    out: list[str] = [command]
    queue: list[tuple[str, int]] = [(command, 0)]
    extractors = (
        lambda c: [m.group(2) for m in _BASH_DASH_C_RE.finditer(c)],
        lambda c: [m.group(2) for m in _EVAL_PAYLOAD_RE.finditer(c)],
        lambda c: [m.group(1).strip() for m in _DOLLAR_PAREN_RE.finditer(c)],
        lambda c: [m.group(1).strip() for m in _BACKTICK_RE.finditer(c)],
        lambda c: [m.group(1).strip() for m in _BRACE_GROUP_RE.finditer(c)],
        lambda c: [m.group(1).strip() for m in _PAREN_SUBSHELL_RE.finditer(c)],
        _strip_command_modifier_prefix,
        _strip_shell_control_prefix,
    )
    while queue:
        cmd, depth = queue.pop()
        if depth >= max_depth:
            continue
        for extractor in extractors:
            for inner in extractor(cmd):
                if inner and inner not in seen:
                    seen.add(inner)
                    out.append(inner)
                    queue.append((inner, depth + 1))
    return out


# Python interpreter flag that bypasses `sitecustomize.py`: `-S` disables
# `site.py` (which is what imports sitecustomize). Note `-I` (isolated mode)
# implies `-E`, `-P`, and `-s` (lowercase) — NOT `-S`. Isolated mode does
# not skip sitecustomize when the shim is installed in the venv's
# site-packages, so it is not in the bypass list.
#
# Compact short-flag forms are matched too: `python -Sc 'code'` is
# equivalent to `python -S -c 'code'` (the `-S` is the bypass; `c` is the
# `-c` short flag for inline code). Lowercase `-s` (skip user site) does
# NOT bypass sitecustomize and is correctly ignored (regex requires
# uppercase `S` specifically).
def _argv_contains_bypass_flag(args_tokens: list[str]) -> bool:
    """Return True if any pre-script Python interpreter short-flag token
    contains uppercase ``S`` (the sitecustomize-disabling flag).

    Walks shlex-tokenized argv left-to-right. A token starting with a
    single ``-`` and at least one alphabetic character is a Python
    short-flag combination (``-S``, ``-Sc``, ``-IS``, etc.); scan it for
    ``S``. A token starting with ``--`` is a long flag and ignored. The
    first non-flag entry (the script path, the ``-`` stdin marker, or any
    other argv element) ends the interpreter-flag region. A ``-S`` after
    that point is ``sys.argv`` for the script, not an interpreter flag,
    and does not disable sitecustomize.

    Operating on tokens (not raw text) makes the detector quote-aware:
    a ``-S`` substring inside a quoted ``-c`` code argument (e.g.
    ``python -c "print(' -S ')"``) lives inside the single token
    ``print(' -S ')``, which does not start with ``-``, so the walk
    terminates without flagging.
    """
    for tok in args_tokens:
        if not tok:
            continue
        if tok == "-":  # stdin marker, not a flag
            return False
        if tok.startswith("--"):
            continue  # long flag; cannot disable sitecustomize
        if tok.startswith("-"):
            if "S" in tok[1:]:
                return True
            continue
        # First non-flag entry; interpreter flag region ends here.
        return False
    return False


def _parse_jsonl_strict(path: Path, label: str) -> list[dict]:
    """Parse a JSONL file strictly; raise TelemetryMergeError on malformed lines.

    Shared parser used by both layer-2 event-log reads and layer-1 transcript
    reads. Empty lines are skipped; non-JSON content raises; non-object
    (scalar/list) entries also raise — the runner's contract is that every
    line is a JSON OBJECT. The ``label`` string identifies the layer/file
    in the error message.

    An empty (0-byte) file is permitted and returns ``[]``. The caller is
    responsible for whether emptiness is acceptable in that layer's
    semantics.
    """
    events: list[dict] = []
    with open(path) as f:
        for line_num, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise TelemetryMergeError(
                    f"malformed JSON in {label} {path} at line {line_num}: {e}"
                ) from e
            if not isinstance(parsed, dict):
                raise TelemetryMergeError(
                    f"non-object JSON in {label} {path} at line {line_num}: "
                    f"expected a JSON object, got {type(parsed).__name__}"
                )
            events.append(parsed)
    return events


# Substring marker the shim's `_write_event` prints to stderr when a write
# fails mid-run. Used by the merger to fail-closed when telemetry events were
# dropped after the hook deliberately continued executing the wrapped call.
_SHIM_WRITE_FAILURE_MARKER = "[pyruntime] cannot write event"


def _scan_stderr_for_shim_failures(stderr_path: Path) -> bool:
    """Return True if outer Claude CLI stderr contains the shim's event-write
    failure marker.

    The shim's hook wrappers catch transient OSError on event writes so the
    agent's diff_diff call still completes (avoids aborting on telemetry
    hiccups). When that happens, `_write_event` first prints
    `[pyruntime] cannot write event to <path>: <err>` to stderr. The merger
    looks for that marker post-hoc: presence means at least one event was
    dropped, so the per-run record may be silently incomplete.

    NOTE: this only covers stderr from the outer ``claude --bare`` process.
    When the shim runs inside an agent-invoked python subprocess (the
    common case), its stderr is captured into the corresponding Bash
    ``tool_result`` block in the stream-JSON transcript, not into
    ``cli_stderr.log``. Use ``_scan_tool_results_for_shim_failures`` for
    that path; both are checked in ``merge_layers``.
    """
    try:
        content = stderr_path.read_text(errors="replace")
    except OSError:
        # Existence was already validated; a read error here is itself a
        # capture problem worth surfacing.
        return True
    return _SHIM_WRITE_FAILURE_MARKER in content


def _scan_tool_results_for_shim_failures(transcript_entries: list[dict]) -> bool:
    """Return True if any Bash ``tool_result`` content contains the shim's
    event-write failure marker.

    Most shim event-write failures occur inside an agent-spawned python
    subprocess (Claude's Bash tool runs the python child whose stderr is
    captured by Claude into the matching ``tool_result`` block). The
    outer ``cli_stderr.log`` only carries stderr from the ``claude
    --bare`` process itself, which rarely sees inner subprocess output,
    so a marker emitted by the shim is invisible to the layer-3 scan.

    Iterating ``tool_result`` blocks closes that gap. Stringified content
    (Claude sometimes returns a single string) and list-shaped content
    (one or more text blocks) are both supported. A match is enough to
    fail closed; per-block granularity is not needed because the
    per-run record cannot be trusted once a single event is known to
    have been dropped.
    """
    for entry in transcript_entries:
        for block in _iter_tool_result_blocks(entry):
            content = block.get("content")
            if isinstance(content, str):
                if _SHIM_WRITE_FAILURE_MARKER in content:
                    return True
            elif isinstance(content, list):
                for sub in content:
                    if isinstance(sub, dict):
                        text = sub.get("text") or sub.get("content") or ""
                        if isinstance(text, str) and _SHIM_WRITE_FAILURE_MARKER in text:
                            return True
                    elif isinstance(sub, str) and _SHIM_WRITE_FAILURE_MARKER in sub:
                        return True
    return False


def _validate_layer_artifacts(stream_json_path: Path, stderr_path: Path) -> list[dict]:
    """Fail-closed preflight: layer-1 and layer-3 capture files must exist,
    and the transcript must be a non-empty sequence of JSON objects.
    Returns the parsed transcript entries so the caller doesn't reparse.

    The three-layer telemetry contract requires the stream-JSON transcript
    (layer 1) and the CLI stderr capture (layer 3) to be present for every
    run.

    - Missing files mean the runner did not finish capturing or the output
      directory was tampered with.
    - An EMPTY stream-JSON transcript means stdout capture failed or was
      truncated; without at least one entry, the merger cannot distinguish
      a Read-tool guide access from no agent activity, so emitting
      definitive ``opened_llms_*=False`` would be silent layer-1 loss.
    - A non-object JSON line (scalar / list) violates the runner's contract
      that every line is a single JSON object.

    Empty stderr capture remains valid (a trivial agent response can produce
    an empty stderr log; layer-3 only carries CLI-level errors).
    """
    if not stream_json_path.exists():
        raise TelemetryMergeError(
            f"stream-JSON transcript missing at {stream_json_path}; "
            f"layer-1 capture incomplete, per-run record cannot be validated"
        )
    if not stderr_path.exists():
        raise TelemetryMergeError(
            f"CLI stderr capture missing at {stderr_path}; "
            f"layer-3 capture incomplete, per-run record cannot be validated"
        )
    transcript_entries = _parse_jsonl_strict(stream_json_path, "stream-JSON transcript")
    if not transcript_entries:
        raise TelemetryMergeError(
            f"stream-JSON transcript at {stream_json_path} is empty; "
            f"mergeable runs require at least one transcript entry "
            f"(empty transcript likely indicates stdout-capture failure "
            f"and would silently zero out layer-1 evidence)"
        )
    # Truncation check: a complete Claude stream-json transcript ends with a
    # successful `result` entry. A transcript whose final entry is anything
    # else (assistant message, tool_use, tool_result, or a `result` carrying
    # `is_error=true`) indicates capture was cut short before the run
    # finished, and per-run evidence (Bash invocations, Read tool_results,
    # later guide accesses) may be silently missing.
    last = transcript_entries[-1]
    if not (isinstance(last, dict) and last.get("type") == "result" and not last.get("is_error")):
        raise TelemetryMergeError(
            f"stream-JSON transcript at {stream_json_path} does not end "
            f"with a successful `type=result` entry; capture is truncated "
            f"or the run did not complete cleanly, and per-run telemetry "
            f"cannot be treated as complete"
        )
    return transcript_entries


def _read_events(path: Path) -> list[dict]:
    """Parse the in-process event log into a list of dicts.

    Raises ``TelemetryMergeError`` if the file is missing or contains a
    non-JSON line. An empty (0-byte) file is permitted and returns ``[]``;
    the validity check is the caller's responsibility.
    """
    if not path.exists():
        raise TelemetryMergeError(
            f"in-process event log not found at {path}; "
            f"the runner should have written a telemetry_missing sentinel"
        )
    events = _parse_jsonl_strict(path, "event log")
    _validate_event_schemas(events, path)
    return events


# Required field schemas for known event types. The shim writes these under
# strict producer-side control (see ``harness/sitecustomize_template.py``);
# missing required fields means the event is malformed and would silently
# zero out telemetry fields if accepted as-is. Reject at parse time.
_EVENT_SCHEMA: dict[str, tuple[str, ...]] = {
    "session_start": ("argv",),
    "module_import": ("module",),
    "guide_file_read": (
        "via",
    ),  # via=='get_llm_guide' needs variant; via=='open' needs filename - checked below
    "estimator_init": ("class",),
    "estimator_fit": ("class",),
    "diagnostic_call": ("name",),
    "warning_emitted": ("filename",),
    # telemetry_missing sentinel has no required fields; the merger raises on
    # its presence regardless of payload.
    "telemetry_missing": (),
}


def _validate_event_schemas(events: list[dict], path: Path) -> None:
    """Reject malformed known-event records.

    Validates two layers per known event type:

    1. **Required field presence**: missing required fields would silently
       zero out telemetry under downstream ``event.get("field", "")``
       reads.
    2. **Required field type and enum**: a present-but-wrong-typed value
       (e.g. ``argv="string"`` instead of ``list``) or an unknown enum
       member (e.g. ``variant="not-real"``, ``filename="not-a-guide.txt"``)
       would also silently zero out telemetry because the unknown value
       doesn't match any bundled-guide check.

    Unknown event types are ignored (forward-compatibility: a future shim
    may emit new event types, and we don't want old mergers to reject
    them outright). Only KNOWN event types are schema-checked.
    """
    valid_filenames = set(_VARIANT_TO_FILENAME.values())
    valid_variants = set(_VARIANT_TO_FILENAME.keys())
    for i, event in enumerate(events):
        event_type = event.get("event")
        if not isinstance(event_type, str):
            continue  # not a typed event; ignore
        required = _EVENT_SCHEMA.get(event_type)
        if required is None:
            continue

        def _bail(msg: str) -> None:
            raise TelemetryMergeError(f"event log {path} entry {i} (event={event_type!r}): {msg}")

        for field in required:
            if field not in event:
                _bail(
                    f"missing required field {field!r}; the shim wrote a "
                    f"malformed record and downstream telemetry would "
                    f"silently omit this evidence"
                )

        # Per-event-type type and enum checks.
        if event_type == "session_start":
            argv = event["argv"]
            if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                _bail(f"argv must be list[str], got {type(argv).__name__}={argv!r}")
        elif event_type == "module_import":
            module = event["module"]
            if not isinstance(module, str):
                _bail(f"module must be str, got {type(module).__name__}={module!r}")
        elif event_type == "guide_file_read":
            via = event.get("via")
            if via not in ("get_llm_guide", "open"):
                _bail(f"unknown via={via!r}; recognized values: " f"'get_llm_guide', 'open'")
            if via == "get_llm_guide":
                if "variant" not in event:
                    _bail("missing required 'variant' field for via='get_llm_guide'")
                variant = event["variant"]
                if not isinstance(variant, str):
                    _bail(f"variant must be str, got " f"{type(variant).__name__}={variant!r}")
                if variant not in valid_variants:
                    _bail(
                        f"unknown variant={variant!r}; recognized values: "
                        f"{sorted(valid_variants)!r}"
                    )
            elif via == "open":
                if "filename" not in event:
                    _bail("missing required 'filename' field for via='open'")
                filename = event["filename"]
                if not isinstance(filename, str):
                    _bail(f"filename must be str, got " f"{type(filename).__name__}={filename!r}")
                if filename not in valid_filenames:
                    _bail(
                        f"unknown filename={filename!r}; recognized values: "
                        f"{sorted(valid_filenames)!r}"
                    )
        elif event_type in ("estimator_init", "estimator_fit"):
            class_name = event["class"]
            if not isinstance(class_name, str):
                _bail(f"class must be str, got " f"{type(class_name).__name__}={class_name!r}")
        elif event_type == "diagnostic_call":
            name = event["name"]
            if not isinstance(name, str):
                _bail(f"name must be str, got {type(name).__name__}={name!r}")
        elif event_type == "warning_emitted":
            filename = event["filename"]
            if not isinstance(filename, str):
                _bail(f"filename must be str, got " f"{type(filename).__name__}={filename!r}")


def _iter_tool_result_blocks(entry):
    """Yield tool_result blocks from a stream-JSON entry.

    tool_result entries live on user messages (the role="user" turn that
    carries tool outputs back to the assistant). The block contains
    `tool_use_id` matching the originating Read/Bash request, plus
    `is_error` and `content` indicating success/failure.
    """
    if not isinstance(entry, dict):
        return
    msg = entry.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    yield block
    if entry.get("type") == "tool_result":
        yield entry


def _validate_bash_tool_results_complete(entries: list[dict]) -> None:
    """Raise ``TelemetryMergeError`` if any Bash ``tool_use`` in the
    transcript lacks a matching ``tool_result``.

    Claude's tool-call flow guarantees every ``tool_use`` is followed by a
    ``tool_result`` in the same conversation (success or error). A
    missing match indicates the transcript was truncated between the
    request and the response, which silently drops the subprocess stderr
    for that command. If that stderr would have contained the shim's
    ``[pyruntime] cannot write event`` marker, a dropped layer-2 event
    becomes silently invisible and the merger emits an all-False
    telemetry record. Reciprocal of the Read tool_result fail-closed
    check (``_scan_read_tool_guide_accesses_in_entries``); same
    invariant, different tool surface.

    Tool uses without an ``id`` field are also rejected; without an id
    the match cannot be established and the result may be missing
    silently.
    """
    bash_uses: dict[str, dict] = {}
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_use_id = block.get("id")
            if not isinstance(tool_use_id, str) or not tool_use_id:
                raise TelemetryMergeError(
                    "Bash tool_use is missing or has empty 'id' field; the "
                    "transcript cannot match its tool_result, so a silently "
                    "dropped stderr (including a possible shim event-write "
                    "failure marker) cannot be ruled out"
                )
            bash_uses[tool_use_id] = block

    if not bash_uses:
        return

    matched_ids: set[str] = set()
    for entry in entries:
        for result in _iter_tool_result_blocks(entry):
            tool_use_id = result.get("tool_use_id")
            if isinstance(tool_use_id, str) and tool_use_id in bash_uses:
                matched_ids.add(tool_use_id)

    unmatched = set(bash_uses) - matched_ids
    if unmatched:
        raise TelemetryMergeError(
            f"Bash tool_use(s) {sorted(unmatched)!r} have no matching "
            f"tool_result in the transcript. The transcript is truncated "
            f"between request and response, so the subprocess stderr "
            f"(which may contain the shim event-write failure marker) "
            f"is silently missing. Per-run record cannot be trusted."
        )


def _scan_read_tool_guide_accesses_in_entries(
    entries: list[dict],
) -> dict[str, bool]:
    """Return ``{filename: True}`` for each bundled guide file SUCCESSFULLY
    accessed via Claude's Read tool in the transcript.

    Catches the case where an agent reads `llms.txt` (or any other bundled
    guide) via the Read tool without invoking Python; the in-process shim
    sees nothing in that path because no `open()` or `get_llm_guide` call
    runs in the agent's subprocess. Without this layer-1 check the merger
    would emit a definitive `opened_llms_txt=False` for what was actually
    a guide discovery.

    Path-matching uses adjacent path parts (``diff_diff`` / ``guides`` /
    ``<filename>``). Substring match would accept paths like
    ``/tmp/notdiff_diff/guides/llms.txt`` or
    ``/some/path/diff_diff/guides_extra/llms.txt``; the part-adjacency
    check requires the read to be in the real bundled location.

    A Read tool_use is only counted as evidence if the matching
    tool_result is non-error (``is_error`` falsy or absent). A failed,
    denied, or hallucinated Read does not flip `opened_llms_*`. If a
    guide-file Read request has no matching tool_result, the merger
    fails closed: an incomplete transcript that would silently emit
    `opened_llms_*=False` for a Read that may have succeeded is treated
    the same as the terminal-result truncation check - the per-run
    record cannot be trusted.

    Only Read-tool evidence is recognized here. Bash-level guide reads
    (e.g. ``cat llms.txt``) are still part of the broader layer-1 parsing
    deferred to a future PR.
    """
    known_filenames = set(_VARIANT_TO_FILENAME.values())

    # First pass: collect candidate Read requests that target a guide path.
    pending: dict[str, str] = {}  # tool_use_id -> filename
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Read":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            file_path = tool_input.get("file_path", "")
            if not isinstance(file_path, str) or not file_path:
                continue
            p = Path(file_path)
            if not (
                p.name in known_filenames
                and p.parent.name == "guides"
                and p.parent.parent.name == "diff_diff"
            ):
                continue
            tool_use_id = block.get("id")
            if isinstance(tool_use_id, str):
                pending[tool_use_id] = p.name

    if not pending:
        return {}

    # Second pass: find matching tool_results.
    opened: dict[str, bool] = {}
    matched_ids: set[str] = set()
    for entry in entries:
        for result in _iter_tool_result_blocks(entry):
            tool_use_id = result.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                continue
            filename = pending.get(tool_use_id)
            if filename is None:
                continue
            matched_ids.add(tool_use_id)
            if not result.get("is_error", False):
                opened[filename] = True

    unmatched_ids = set(pending) - matched_ids
    if unmatched_ids:
        unmatched_filenames = sorted({pending[i] for i in unmatched_ids})
        raise TelemetryMergeError(
            f"Read tool_use for bundled guide file(s) {unmatched_filenames!r} "
            f"has no matching tool_result in the transcript (tool_use_ids: "
            f"{sorted(unmatched_ids)!r}). Transcript is incomplete; the "
            f"per-run record cannot be trusted to reflect guide-discovery state."
        )
    return opened


def _iter_tool_use_blocks(entry):
    """Yield tool_use blocks from a stream-JSON entry, handling both
    nested (`message.content[*]`) and shallow (top-level type=tool_use) shapes.
    """
    if not isinstance(entry, dict):
        return
    msg = entry.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block
    if entry.get("type") == "tool_use":
        yield entry


_HEREDOC_OPEN_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")


def _strip_heredoc_bodies(command: str) -> str:
    """Remove heredoc bodies (text between ``<<TAG`` and a closing ``TAG``
    line) from a Bash command string.

    Heredoc bodies are stdin content for the receiving command, not part
    of any argv, so they must not be scanned for python invocations or
    bypass flags. For example, ``cat <<'EOF'\\npython x.py\\nEOF`` is a
    cat-creates-file pattern; the inner ``python x.py`` is data, not an
    invocation.

    Supports the ``<<TAG`` / ``<<-TAG`` / ``<<'TAG'`` / ``<<"TAG"`` forms.
    The closing tag matches a line consisting of the tag with optional
    leading whitespace (mandatory for the ``<<-`` form, optional in our
    detection because Bash strips leading tabs only).

    Multiple heredocs in a single command (rare, e.g.
    ``cat <<A; cat <<B``) are handled by scanning iteratively.
    Malformed heredocs (no closing tag) are treated as "body extends to
    end-of-command" and stripped entirely from the opener onward.
    """
    out_parts: list[str] = []
    pos = 0
    while pos < len(command):
        m = _HEREDOC_OPEN_RE.search(command, pos)
        if not m:
            out_parts.append(command[pos:])
            break
        # Keep everything up to and including the heredoc opener token; the
        # opener itself is fine to scan (it's just a redirection operator).
        out_parts.append(command[pos : m.end()])
        tag = m.group(2)
        # Body starts after the next newline (if any).
        body_start_match = re.search(r"\n", command[m.end() :])
        if body_start_match is None:
            # No newline after opener: malformed; drop the rest.
            break
        body_start = m.end() + body_start_match.end()
        # Look for the closing tag on its own line.
        close_re = re.compile(rf"\n[ \t]*{re.escape(tag)}(?:\n|$)")
        close_match = close_re.search(command, body_start)
        if close_match is None:
            # No closing tag: drop body to end.
            break
        # Skip the body and the closing tag; resume after. Insert a synthetic
        # ``\n`` separator so any post-heredoc command (``cat <<EOF\n...\nEOF\n
        # python script.py``) is not glued onto the heredoc opener and lost
        # to the python-invocation regex. The synthetic newline is a real
        # shell command separator and is honored by ``_PYTHON_INVOCATION_RE``,
        # ``_is_in_command_position``, and ``_find_unquoted_separator``.
        out_parts.append("\n")
        pos = close_match.end()
    return "".join(out_parts)


def _find_unquoted_separator(s: str) -> int | None:
    """Return the position of the first unquoted shell separator (``;``,
    ``&``, ``|``, ``)``, ``\\n``) in ``s``, or ``None`` if every such char
    is inside a single- or double-quoted region.

    Includes ``)`` so that ``(python script.py)`` truncates correctly
    without the closing paren attaching to the last argv token, and
    includes ``\\n`` because newline is a shell command separator (a
    multiline Bash command's ``cmd1\\ncmd2`` is two commands).

    Tracks shell quoting state so that the inline ``;`` in
    ``python -c 'import os; print(1)'`` does not falsely terminate the
    python invocation's argument region. Single-quoted regions are
    literal; double-quoted regions honor backslash escapes for ``"`` and
    ``\\`` only (no command substitution / variable expansion handling,
    which is fine here because we only need to identify quote boundaries).
    """
    i = 0
    quote_char: str | None = None
    while i < len(s):
        c = s[i]
        if quote_char is None:
            if c in ("'", '"'):
                quote_char = c
            elif c in (";", "&", "|", ")", "\n"):
                return i
        else:
            if quote_char == '"' and c == "\\" and i + 1 < len(s):
                # Backslash inside double quotes escapes the next char.
                i += 2
                continue
            if c == quote_char:
                quote_char = None
        i += 1
    return None


def _is_in_command_position(command: str, interp_start: int) -> bool:
    """Return True iff the python token at ``command[interp_start:]`` is in
    shell command position (start of a command segment), not a positional
    argument to another command.

    Walks left from ``interp_start``, skipping:

    - Whitespace.
    - Posix env-var assignment prefixes (``KEY=value`` or ``KEY=``); these
      precede the command in shell syntax (``MPLBACKEND=Agg python ...``)
      and do not change command position.

    Then checks the previous char: a separator (``;``, ``&``, ``|``,
    ``(``, ``\\n``) or start-of-command means command position; anything
    else (e.g. ``p`` of ``grep``, ``o`` of ``echo``) means the token is
    an argument and should NOT be treated as a python invocation.
    """
    i = interp_start - 1
    while i >= 0 and command[i] in (" ", "\t"):
        i -= 1
    if i < 0:
        return True
    while i >= 0:
        # Try to parse a trailing KEY=VALUE (or KEY=) preceding this
        # position. Walk back to find a `=` whose left side is `[A-Za-z_][A-Za-z0-9_]*`.
        j = i
        while j >= 0 and command[j] not in (" ", "\t", ";", "&", "|", "(", "\n"):
            j -= 1
        # command[j+1 : i+1] is a single shell word (no whitespace/separators).
        word = command[j + 1 : i + 1]
        if "=" in word:
            key = word.split("=", 1)[0]
            if (
                key
                and (key[0].isalpha() or key[0] == "_")
                and all(c.isalnum() or c == "_" for c in key)
            ):
                # Skip past this assignment and any preceding whitespace.
                i = j
                while i >= 0 and command[i] in (" ", "\t"):
                    i -= 1
                if i < 0:
                    return True
                continue
        break
    return command[i] in (";", "&", "|", "(", "\n")


def _is_redirection_token(tok: str) -> bool:
    """Return True if tok is a shell I/O redirection / pipe operator that
    terminates a python program's argv list.

    Catches:
    - Pure redirections / heredocs: ``<``, ``>``, ``<<``, ``>>``, ``<<<``,
      ``>out.txt`` (no space before filename), ``<infile`` etc.
    - fd-prefixed: ``2>``, ``1>``, ``2>>``, ``2>&1``, ``&>`` etc.
    - Pipes / sequencers: ``|``, ``&``, ``&&``, ``||``, ``;`` (defense in
      depth; the outer regex usually splits on these first).

    Tokens that contain ``<`` / ``>`` INSIDE quotes (e.g. ``"print(1>0)"``)
    are kept by ``shlex.split`` as a single token starting with ``"`` or
    ``p`` (an alphanumeric), so they are not flagged here.
    """
    if not tok:
        return False
    if tok in ("|", "&", "&&", "||", ";"):
        return True
    if tok[0] in ("<", ">"):
        return True
    # fd-prefixed: leading digit(s) then `<` or `>` (e.g. `2>`, `2>&1`, `1>>`).
    i = 0
    while i < len(tok) and tok[i].isdigit():
        i += 1
    if 0 < i < len(tok) and tok[i] in ("<", ">"):
        return True
    return False


def _truncate_at_redirection(tokens: list[str]) -> list[str]:
    """Drop everything from the first I/O redirection / pipe token onward.

    Shell-managed I/O (heredoc body, redirection target file, fd dup) is
    NOT part of the Python program's ``sys.orig_argv``; including those
    tokens in the visible argv would prevent attribution against a
    legitimate session_start.
    """
    result: list[str] = []
    for tok in tokens:
        if _is_redirection_token(tok):
            break
        result.append(tok)
    return result


def _extract_python_invocations_from_segment(command: str) -> list[list[str]]:
    """Return argv-shaped token lists for each python invocation in a single
    pre-stripped, pre-unwrapped command segment. Used by
    ``_extract_python_invocations_from_command`` per shell variant.
    """
    invocations: list[list[str]] = []
    for m in _PYTHON_INVOCATION_RE.finditer(command):
        match_start = m.start()
        boundary_char = m.group(0)[0] if m.group(0) else ""
        # interp_end is the position right after the python token (exclusive).
        # The trailing group is ``(?:[\s<>]|$)`` - when a real boundary char
        # (space, `<`, `>`) was consumed, m.end() points one past it and the
        # python token ends at m.end() - 1. When the match ended via ``$``
        # (zero-width, no boundary consumed), m.end() IS the end of the
        # python token and subtracting 1 would truncate the last char.
        if m.group(0) and m.group(0)[-1] in (" ", "\t", "\n", "<", ">"):
            interp_end = m.end() - 1
        else:
            interp_end = m.end()
        if boundary_char == "/":
            # Absolute path: walk left to recover the rest of the path
            # (e.g. `/usr/bin/python3`). Stop at any shell separator,
            # whitespace, redirection, or `=` (which marks an inline
            # ``KEY=value`` env-var assignment ending and is not a valid
            # interpreter path char).
            i = match_start
            while i > 0 and command[i - 1] not in (
                " ",
                "\t",
                "\n",
                ";",
                "&",
                "|",
                "(",
                ")",
                "<",
                ">",
                "=",
            ):
                i -= 1
            interp_start = i
        elif boundary_char in (" ", "\t", "\n", ";", "&", "|", "(", ")"):
            # Boundary char is a separator; interpreter starts after it.
            # ``\n`` is included so post-heredoc / multiline-Bash python
            # invocations on a fresh line are extracted.
            interp_start = match_start + 1
        else:
            # ``^`` start-of-command boundary (zero-width); m.start() is
            # already the first char of the python token.
            interp_start = match_start
        # Command-position check: skip ``python`` tokens that are arguments
        # to another command (``grep python script.py``, ``echo python``).
        # ``python`` must appear at the start of a shell command segment
        # (after ``;``, ``&``, ``|``, ``(``, ``\\n``, or start-of-command,
        # optionally preceded by ``KEY=value`` env-var assignments).
        if not _is_in_command_position(command, interp_start):
            continue
        interpreter = command[interp_start:interp_end]
        # Slice the args region: from end of match to next UNQUOTED shell
        # separator. A raw-regex scan would falsely terminate at a
        # in-quoted ``;`` in commands like
        # ``python -c 'import os; print(1)'``.
        # Args region starts at the trailing boundary char (so e.g. ``python -c``
        # gives args ``  -c`` with leading whitespace that shlex tolerates).
        # When the match ended via ``$`` (no boundary char consumed), there is
        # no trailing boundary to include; rest is empty.
        if m.group(0) and m.group(0)[-1] in (" ", "\t", "\n", "<", ">"):
            rest_start = m.end() - 1
        else:
            rest_start = m.end()
        rest = command[rest_start:]
        sep_pos = _find_unquoted_separator(rest)
        args_region = rest[:sep_pos] if sep_pos is not None else rest
        # Tokenize args with shlex; fall back to whitespace split if shlex
        # raises (unbalanced quotes, etc.).
        try:
            import shlex

            args_tokens = shlex.split(args_region, comments=False, posix=True)
        except ValueError:
            args_tokens = args_region.split()
        args_tokens = _truncate_at_redirection(args_tokens)
        invocations.append([interpreter, *args_tokens])
    return invocations


def _extract_python_invocations_from_command(command: str) -> list[list[str]]:
    """Return argv-shaped token lists for every python invocation visible in
    ``command``, INCLUDING those hidden inside shell wrappers.

    Pipeline:

    1. Strip heredoc bodies (so script content inside ``<<EOF ... EOF`` is
       not falsely matched as an invocation).
    2. Recursively unwrap known shell wrappers (modifiers, ``bash -c``,
       brace groups, ``if/while/!``, ``$()``, backticks) via
       ``_unwrap_command_for_inspection``.
    3. Run the per-segment regex extractor on each variant.
    4. Deduplicate argv lists (different unwrap paths often surface the
       same inner invocation).

    Each entry is the full argv slice ``[interpreter, *args]`` matching
    what the Python process would see in ``sys.orig_argv``. Args run up
    to the next unquoted shell separator (``;``, ``&``, ``|``, ``)``,
    ``\\n``, end-of-string), with quoting handled by ``shlex.split``,
    and any I/O redirection tokens (``>``, ``<``, ``<<``, ``2>&1``,
    etc.) are stripped because shell redirection is not part of the
    Python program's argv.
    """
    command = _strip_heredoc_bodies(command)
    seen_argvs: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for variant in _unwrap_command_for_inspection(command):
        for argv in _extract_python_invocations_from_segment(variant):
            key = tuple(argv)
            if key in seen_argvs:
                continue
            seen_argvs.add(key)
            out.append(argv)
    return out


def _session_argv_matches_invocation(session_argv: list[str], visible_argv: list[str]) -> bool:
    """Return True iff a session_start's argv matches a transcript-visible
    python invocation's argv.

    Args after the interpreter must match exactly (shell-tokenized argv from
    the visible Bash command compared against ``sys.orig_argv[1:]``).

    Interpreter argv[0] matching is asymmetric:

    - If the VISIBLE argv[0] is absolute (``startswith("/")``), require
      exact equality with the session's argv[0]. Basename-only fallback
      would silently attribute ``/usr/bin/python3 script.py`` (off-venv,
      no sitecustomize) to a session whose argv[0] is
      ``/per-arm-venv/bin/python3`` and whose args also match. The visible
      absolute path is unambiguous and must identify the exact
      interpreter.
    - If the VISIBLE argv[0] is relative (bare ``python`` / ``python3``
      / ``python3.11``), allow path-vs-basename fallback. The shell may
      have PATH-resolved the bare name to an absolute interpreter, in
      which case ``sys.orig_argv[0]`` carries the resolved path even
      though the visible argv[0] is just the bare token.
    """
    if not session_argv or not visible_argv:
        return False
    if session_argv[1:] != visible_argv[1:]:
        return False
    if session_argv[0] == visible_argv[0]:
        return True
    # Asymmetric fallback: only when visible is relative.
    if visible_argv[0].startswith("/"):
        return False
    return Path(session_argv[0]).name == Path(visible_argv[0]).name


def _attribute_python_invocations(transcript_entries: list[dict], events: list[dict]) -> None:
    """Per-invocation cross-check by argv matching.

    Every transcript-visible python invocation must match an unused
    ``session_start`` by argv (interpreter basename + args). Surplus
    session_starts that have no visible counterpart are allowed; pip
    console-scripts and child processes spawned by an attributed run
    record session_start without surfacing as a Bash ``python`` argv[0]
    in the transcript.

    Argv matching closes the masking class where the transcript shows
    e.g. ``pip --version && python script.py`` while the event log
    contains a session_start for pip (matching the visible ``python``
    token by name) but none for the actual script. With basename-only
    pooling, pip's session would be popped and the script's missing
    session would go undetected. Argv matching binds pip's session to
    pip's argv shape (e.g. ``[".../bin/pip", "--version"]``), leaving the
    visible ``[python, script.py]`` invocation unattributed and forcing
    the merger to fail closed.
    """
    sessions: list[dict] = [e for e in events if e.get("event") == "session_start"]
    available_idx: list[int] = list(range(len(sessions)))

    visible_invocations: list[tuple[str, list[str]]] = []
    for entry in transcript_entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command", "")
            if not isinstance(command, str):
                continue
            for argv in _extract_python_invocations_from_command(command):
                if argv:
                    visible_invocations.append((command, argv))

    for command, visible_argv in visible_invocations:
        matched_idx: int | None = None
        for idx in available_idx:
            session_argv = sessions[idx].get("argv")
            if not isinstance(session_argv, list):
                continue
            if _session_argv_matches_invocation(session_argv, visible_argv):
                matched_idx = idx
                break
        if matched_idx is None:
            remaining = [sessions[i].get("argv") for i in available_idx]
            raise TelemetryMergeError(
                f"python invocation argv={visible_argv!r} in command "
                f"{command[:160]!r} has no matching session_start "
                f"(remaining session argvs: {remaining!r}). The interpreter "
                f"either ran without sitecustomize, or its session was "
                f"masked by an unrelated entry point (e.g. a pip "
                f"console-script). Cold-start eval is invalid."
            )
        available_idx.remove(matched_idx)


def _find_python_bypass_invocations_in_entries(entries: list[dict]) -> list[str]:
    """Return Bash commands containing a `-S` (or compact `-Sc`) flag,
    inline `PATH=...` interpreter override, `env python`, or `./python`.

    All of these forms can cause a Python interpreter to run without the
    shim loading: `-S` disables `site.py`; `PATH=/usr/bin python3` resolves
    to a non-per-arm-venv interpreter; `env python` resolves via PATH;
    `./python` invokes a local binary that almost certainly isn't the per-
    arm-venv interpreter.

    Per-invocation attribution cannot recover from these because the
    bypassed interpreter never writes `session_start`. Fail closed.

    Note: `-I` (isolated mode) is NOT a bypass; it implies `-E`, `-P`, and
    lowercase `-s` — none of which disable sitecustomize.
    """
    bypass_commands: list[str] = []
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            raw_command = tool_input.get("command", "")
            if not isinstance(raw_command, str):
                continue
            # Strip heredoc bodies first: text inside ``<<EOF ... EOF`` is
            # data on the receiving command's stdin, not a python invocation,
            # PATH override, or `-S` bypass even if the literal substrings
            # appear in the body.
            command = _strip_heredoc_bodies(raw_command)
            # PATH mutation / activation script / env / ./python: any of
            # these in a command containing a python invocation triggers
            # fail-closed because the resolved interpreter may not be the
            # per-arm-venv python.
            has_python = bool(_PYTHON_INVOCATION_RE.search(command))
            has_path_mutation = bool(_PATH_ASSIGNMENT_RE.search(command))
            has_python_env_var = bool(_PYTHON_ENV_VAR_RE.search(command))
            has_activation = bool(_SHELL_ACTIVATION_RE.search(command))
            has_wrapper = bool(_PYTHON_WRAPPER_RE.search(command))
            # Shell-wrapper bypass: `bash -c "python -S ..."`, `eval 'python -S ...'`,
            # etc. The inner python token lives inside a quoted payload that
            # _PYTHON_INVOCATION_RE does not visit. Only flag as bypass
            # when there's actually a primitive bypass vector (-S, PATH=,
            # source/activation) inside the wrapper - otherwise the
            # unwrap+attribution path handles benign wrappers correctly.
            has_shell_wrapper = bool(_SHELL_WRAPPER_RE.search(command))
            has_eval_wrapper = bool(_EVAL_WRAPPER_RE.search(command))
            has_python_literal = bool(_PYTHON_LITERAL_RE.search(command))
            if (has_shell_wrapper or has_eval_wrapper) and has_python_literal:
                # Require a primitive bypass marker (-S flag literal OR
                # PATH= manipulation) in either the outer command or any
                # unwrapped variant - a primitive hidden inside a `bash -c`
                # quoted payload IS a bypass even if the outer regex
                # doesn't see it. Without any primitive in any variant,
                # fall through to unwrap+attribution.
                variants = _unwrap_command_for_inspection(command)
                if any(
                    _BARE_BYPASS_FLAG_RE.search(v) or _PATH_ASSIGNMENT_RE.search(v)
                    for v in variants
                ):
                    bypass_commands.append(command)
                    continue
            # Command modifier prefix bypass: `command python -S ...`,
            # `time python -S ...`, `nohup python -S ...`, `timeout 30
            # python -S ...`. The python token is not in command position
            # (it's an arg to the modifier), so the extractor skips it; the
            # `-S` then runs without producing any visible invocation.
            # When the modifier-prefixed-python pattern is paired with a
            # word-bounded `-S` flag literal, fail closed.
            if _COMMAND_MODIFIER_BEFORE_PYTHON_RE.search(command) and _BARE_BYPASS_FLAG_RE.search(
                command
            ):
                bypass_commands.append(command)
                continue
            if (
                (has_path_mutation and has_python)
                or (has_python_env_var and has_python)
                or (has_activation and has_python)
                or has_wrapper
                or _ENV_BEFORE_PYTHON_RE.search(command)
                or _DOT_PYTHON_RE.search(command)
                or _PYTHON_COMMAND_SUBSTITUTION_RE.search(command)
            ):
                bypass_commands.append(command)
                continue
            # -S / -Sc / etc. as a python interpreter flag (NOT inside
            # quoted code, NOT after the script argument). The argv-walking
            # check operates on shlex-tokenized argv so a -S substring
            # inside a quoted -c code argument cannot false-positive.
            argv_walk_hit = False
            for argv in _extract_python_invocations_from_command(command):
                if _argv_contains_bypass_flag(argv[1:]):
                    bypass_commands.append(command)
                    argv_walk_hit = True
                    break  # one bypass per command is enough to record
            if argv_walk_hit:
                continue
            # Wrapper-aware unwrap: extract inner-command variants from
            # known shell wrappers (modifiers, bash -c "...", brace groups,
            # if/while/!, $(...), `...`) and check each variant for the
            # python+`-S` primitive. Catches forms the regexes above missed
            # because the wrapper hid the python invocation from outer
            # scanning. The original command is element [0]; the regex
            # checks above already handled that, so skip it here.
            for variant in _unwrap_command_for_inspection(command)[1:]:
                if _PYTHON_LITERAL_RE.search(variant) and _BARE_BYPASS_FLAG_RE.search(variant):
                    bypass_commands.append(command)
                    break
    return bypass_commands


def _count_python_invocations(stream_json_path: Path) -> int:
    """Return the total number of python interpreter invocations across all
    Bash tool commands in the transcript.

    Counts EVERY occurrence within each command, not just one per Bash call,
    so compound commands like
    ``python a.py && python -S b.py`` register as 2 invocations. The
    cross-check in ``_validate_shim_loaded`` requires this count to match
    the number of ``session_start`` events; an undercount here would let
    partial instrumentation silently pass.

    Parses each line of the stream-JSON transcript strictly via the shared
    ``_parse_jsonl_strict`` helper. Existence of the file is the caller's
    responsibility (``_validate_layer_artifacts`` runs first in
    ``merge_layers``).

    The word-boundary regex in ``_PYTHON_INVOCATION_RE`` and the structural
    tool_use/Bash/command match guard against false positives like
    ``/opt/python/`` (directory) or ``pythonic`` (substring).
    """
    entries = _parse_jsonl_strict(stream_json_path, "stream-JSON transcript")
    count = 0
    for entry in entries:
        for block in _iter_tool_use_blocks(entry):
            if block.get("name") != "Bash":
                continue
            tool_input = block.get("input") or block.get("tool_input")
            if not isinstance(tool_input, dict):
                continue
            command = tool_input.get("command", "")
            if isinstance(command, str):
                count += len(_PYTHON_INVOCATION_RE.findall(command))
    return count


def _validate_shim_loaded(events: list[dict], transcript_entries: list[dict]) -> None:
    """Cross-layer fail-closed check before building the record.

    Cases:
    1. ``telemetry_missing`` sentinel present (written by the runner when
       the event log disappeared post-exec): raise.
    2. ANY visible python invocation uses ``-S`` (sitecustomize bypass) or
       a compact form like ``-Sc`` / ``-Sm``: raise. Per-invocation
       attribution cannot help here because the bypassed process WOULD
       have a sys.executable but never fires `session_start`.
    3. Per-invocation attribution by argv: every visible python invocation
       in the transcript MUST claim a `session_start` whose ``argv``
       matches the shell-tokenized invocation (interpreter token by
       basename, args[1:] exact). Unmatchable invocations raise. This
       replaces the previous sys_executable-only pooling, which could be
       masked by unrelated instrumented Python processes (e.g. `pip
       --version` supplying a session_start that satisfied a later
       relative `python` token without actually corresponding to it).
    4. Else accept.
    """
    for event in events:
        if event.get("event") == "telemetry_missing":
            raise TelemetryMergeError(
                "telemetry_missing sentinel present in event log; "
                "the agent's event log disappeared post-exec and the run "
                "is invalid for evaluation"
            )
    # A genuine shim-produced event log always writes `session_start`
    # before any hook events. Nonempty event log + zero session_starts
    # indicates truncation, deletion-then-recreation, or fabrication —
    # fail closed.
    has_session_start = any(e.get("event") == "session_start" for e in events)
    has_hook_events = any(
        e.get("event")
        in {
            "module_import",
            "guide_file_read",
            "estimator_init",
            "estimator_fit",
            "diagnostic_call",
            "warning_emitted",
        }
        for e in events
    )
    if has_hook_events and not has_session_start:
        raise TelemetryMergeError(
            "in-process event log has hook events but no session_start; "
            "log was truncated, deleted-and-recreated, or fabricated. "
            "Cold-start eval is invalid."
        )
    bypass_commands = _find_python_bypass_invocations_in_entries(transcript_entries)
    if bypass_commands:
        first = bypass_commands[0]
        if len(first) > 200:
            first = first[:200] + "..."
        raise TelemetryMergeError(
            f"python interpreter bypass flag (-S, including compact forms "
            f"like -Sc) detected in {len(bypass_commands)} transcript "
            f"command(s); first: {first!r}. This flag skips sitecustomize.py "
            f"and bypasses the in-process shim, leaving layer-2 silently "
            f"incomplete."
        )
    _attribute_python_invocations(transcript_entries, events)


def _build_diff_diff_record(
    events: list[dict],
    transcript_entries: list[dict],
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Construct a TelemetryRecord for arm='diff_diff' from parsed events.

    Discoverability fields default to ``False`` (the diff-diff arm requires
    explicit bool encoding per `TelemetryRecord.__post_init__`). Any event
    flips its field to ``True``.
    """
    guide_reads = [e for e in events if e.get("event") == "guide_file_read"]
    warnings_emitted = [e for e in events if e.get("event") == "warning_emitted"]
    estimator_inits = [e for e in events if e.get("event") == "estimator_init"]
    estimator_fits = [e for e in events if e.get("event") == "estimator_fit"]
    diagnostic_calls = [e for e in events if e.get("event") == "diagnostic_call"]

    opened = {filename: False for filename in _VARIANT_TO_FILENAME.values()}
    variants_seen: set[str] = set()
    for r in guide_reads:
        via = r.get("via")
        if via == "get_llm_guide":
            variant = r.get("variant", "")
            if variant in _VARIANT_TO_FILENAME:
                variants_seen.add(variant)
                opened[_VARIANT_TO_FILENAME[variant]] = True
        elif via == "open":
            # The shim writes basename-only filenames after its own
            # `_path_is_diff_diff_guide` check confirmed the path was under
            # `diff_diff/guides/`. Match exactly here; defense in depth.
            filename = r.get("filename", "")
            if filename in opened:
                opened[filename] = True

    # Layer-1 evidence: Claude's Read tool accessing a bundled guide file
    # without invoking Python. Layer-2 hooks see nothing in that path, so
    # without this OR-merge the merger would silently emit
    # opened_llms_*=False for an actual discovery.
    read_tool_opens = _scan_read_tool_guide_accesses_in_entries(transcript_entries)
    for filename, was_opened in read_tool_opens.items():
        if was_opened:
            opened[filename] = True

    return TelemetryRecord(
        arm="diff_diff",
        stream_json_path=stream_json_path,
        in_process_events_path=in_process_events_path,
        stderr_path=stderr_path,
        opened_llms_txt=opened["llms.txt"],
        opened_llms_practitioner=opened["llms-practitioner.txt"],
        opened_llms_autonomous=opened["llms-autonomous.txt"],
        opened_llms_full=opened["llms-full.txt"],
        called_get_llm_guide=any(r.get("via") == "get_llm_guide" for r in guide_reads),
        get_llm_guide_variants=tuple(sorted(variants_seen)),
        saw_fit_time_warning=bool(warnings_emitted),
        diagnostic_methods_invoked=tuple(
            sorted({e["name"] for e in diagnostic_calls if "name" in e})
        ),
        estimator_classes_instantiated=tuple(
            sorted({e["class"] for e in (estimator_inits + estimator_fits) if "class" in e})
        ),
    )


def _build_statsmodels_record(
    events: list[dict],
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Construct a TelemetryRecord for arm='statsmodels' from parsed events.

    Guide-related fields are ``None`` (sentinel: not applicable). Bool/tuple
    fields are populated from events targeting `statsmodels` — the shim
    in PR #4 has no statsmodels-specific hooks, so these will be False/()
    until the statsmodels arm instrumentation lands.
    """
    statsmodels_warnings = [
        e
        for e in events
        if e.get("event") == "warning_emitted" and "statsmodels" in str(e.get("filename", ""))
    ]
    return TelemetryRecord(
        arm="statsmodels",
        stream_json_path=stream_json_path,
        in_process_events_path=in_process_events_path,
        stderr_path=stderr_path,
        opened_llms_txt=None,
        opened_llms_practitioner=None,
        opened_llms_autonomous=None,
        opened_llms_full=None,
        called_get_llm_guide=None,
        get_llm_guide_variants=(),
        saw_fit_time_warning=bool(statsmodels_warnings),
        diagnostic_methods_invoked=(),
        estimator_classes_instantiated=(),
    )


def merge_layers(
    arm: str,
    stream_json_path: Path,
    in_process_events_path: Path,
    stderr_path: Path,
) -> TelemetryRecord:
    """Merge the three telemetry layers into a single record.

    `arm` drives the sentinel semantics on guide-discovery fields (see
    TelemetryRecord docstring): for arm == "statsmodels", `opened_llms_*`
    and `called_get_llm_guide` are encoded as None ("not applicable").

    Raises ``TelemetryMergeError`` on fail-closed conditions:
    - In-process event log missing or malformed.
    - Runner-written ``telemetry_missing`` sentinel present.
    - Empty or non-object transcript.
    - Shim event-write failure marker in outer cli_stderr.log OR in any
      Bash tool_result content (the agent's python subprocess stderr is
      captured into tool_result.content, not into cli_stderr.log).
    - Python bypass flag (`-S`) in any transcript command.
    - Stream-JSON transcript missing a terminal successful `result` entry
      (truncated capture).
    - Any visible python invocation that cannot be attributed to a
      session_start by ``argv`` (interpreter basename + args).
    - A guide-file Read tool_use with no matching tool_result
      (incomplete transcript / cannot determine success).
    - A Bash tool_use with no matching tool_result (subprocess stderr
      silently missing; could contain the shim event-write marker).

    The transcript is parsed exactly once (per-invocation attribution,
    guide-read scan, and bypass detection all consume the parsed entries).
    """
    transcript_entries = _validate_layer_artifacts(stream_json_path, stderr_path)
    _validate_bash_tool_results_complete(transcript_entries)
    if _scan_stderr_for_shim_failures(stderr_path):
        raise TelemetryMergeError(
            f"shim event-write failure marker present in {stderr_path}; "
            f"at least one layer-2 event was dropped mid-run and the "
            f"per-run record may be silently incomplete"
        )
    if _scan_tool_results_for_shim_failures(transcript_entries):
        raise TelemetryMergeError(
            "shim event-write failure marker present in a Bash tool_result "
            "(agent python subprocess stderr); at least one layer-2 event "
            "was dropped mid-run and the per-run record may be silently "
            "incomplete"
        )
    events = _read_events(in_process_events_path)
    _validate_shim_loaded(events, transcript_entries)
    if arm == "diff_diff":
        return _build_diff_diff_record(
            events,
            transcript_entries,
            stream_json_path,
            in_process_events_path,
            stderr_path,
        )
    elif arm == "statsmodels":
        return _build_statsmodels_record(
            events, stream_json_path, in_process_events_path, stderr_path
        )
    else:
        raise ValueError(f"unknown arm: {arm!r}")
