"""Bash command parsing for layer-1 telemetry attestation.

Walks the bashlex AST of an agent-visible Bash command and enumerates
Python invocations. Replaces the previous regex-based extractor whose
wrapper-form coverage was enumeration-based - one regex per recognized
shape - and consistently lagged real shell semantics. The AST walker
treats wrapper forms as language structure rather than as a list of
patterns to recognize, so quoted command words, path-qualified wrappers,
modifiers after shell separators, command modifiers behind env-prefix
assignments, and the like all surface naturally from the same walker
rather than each requiring its own regex.

Failure-mode contract: when static parsing cannot prove whether a
command word IS or IS NOT a Python launch, raise rather than guess. The
neutral exception name (``RunValidityError``) does not telegraph the
specific check.
"""

from __future__ import annotations

import os.path
import re
from typing import Iterator, Optional

import bashlex
import bashlex.errors


class RunValidityError(Exception):
    """Base class for any layer-1 / layer-2 / layer-3 validity failure
    raised by the merger. Neutral name to avoid telegraphing the
    specific check that fired."""


class ShellCommandIndeterminate(RunValidityError):
    """A Bash command contained a Python invocation whose command-word
    or argv-word could not be statically resolved (variable expansion,
    command substitution, parameter expansion). Static parsing cannot
    prove the resolved value is not load-bearing for layer-1 attestation."""


class ShellCommandParseError(RunValidityError):
    """bashlex failed to parse a Bash command, or recursive eval/sh-c
    payload re-parsing exceeded the recursion bound."""


_PYTHON_WRAPPER_BASENAMES = frozenset(
    {
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
        "sudo",
        "doas",
    }
)

_TIME_KEYWORD_RE = re.compile(r"(?:^|(?<=[\s;&|()\n{}]))time(?=\s)")

# bashlex requires unquoted heredoc delimiters. `<<'EOF'`/`<<"EOF"` produce
# a parse error because bashlex looks for the LITERAL `'EOF'`/`"EOF"` as
# the terminator on its own line. Real bash strips the quotes; pre-process
# the delimiter to match real bash semantics.
_HEREDOC_QUOTED_DELIM_RE = re.compile(r"<<(-?)\s*(['\"])(\w+)\2")

_MAX_EVAL_RECURSION = 10

_INLINE_PARSER_NAMES = frozenset({"eval"})
_DASH_C_PARSER_NAMES = frozenset({"sh", "bash", "zsh", "dash", "ksh", "ash"})

_PYTHON_BASENAME_RE = re.compile(r"^python(?:3(?:\.\d+)?)?$")
# PR #5: any ``*-real`` basename matching a python interpreter alias
# (``python-real``, ``python3-real``, ``python3.11-real``) is treated as a
# python invocation AND a bypass primitive. The layer-1.5 wrapper renames
# the venv's real interpreter to ``python-real``; any direct invocation
# of that name (e.g., ``${venv}/bin/python-real script.py``) skips the
# wrapper, so the merger has no layer-1.5 attestation for the run. Adding
# it to bypass detection makes the merger fail closed.
_PYTHON_REAL_BASENAME_RE = re.compile(r"^python(?:3(?:\.\d+)?)?-real$")


def _preprocess(command: str) -> str:
    """Pre-parse fixups for bashlex limitations.

    1. Replace bareword ``time`` (the bash timespec keyword) with the
       builtin ``command``. bashlex raises ``NotImplementedError`` on
       the keyword; both words are in ``_PYTHON_WRAPPER_BASENAMES`` so
       the walker strips them identically.
    2. Append a trailing newline if missing. bashlex requires heredoc
       delimiters to be terminated by a newline; ``cat <<'EOF'\\n...\\nEOF``
       (no trailing newline) raises an "end-of-file" parse error.
    """
    preprocessed = _TIME_KEYWORD_RE.sub("command", command)
    # Strip quotes around heredoc delimiters: <<'EOF' / <<"EOF" / <<-'EOF'
    # are real bash but bashlex looks for the literal quoted form as the
    # terminator. Real bash strips the quotes when matching.
    preprocessed = _HEREDOC_QUOTED_DELIM_RE.sub(r"<<\1\3", preprocessed)
    if not preprocessed.endswith("\n"):
        preprocessed += "\n"
    return preprocessed


def _parse(command: str) -> list:
    """Parse ``command`` via bashlex; raise ``ShellCommandParseError``
    on any failure (including ``NotImplementedError`` for Bash forms
    bashlex does not model, e.g. ``case`` patterns or coproc)."""
    preprocessed = _preprocess(command)
    try:
        return bashlex.parse(preprocessed)
    except bashlex.errors.ParsingError as e:
        raise ShellCommandParseError(f"could not parse Bash command: {e}") from e
    except NotImplementedError as e:
        raise ShellCommandParseError(f"unsupported Bash form in command: {e}") from e


def _is_python_basename(name: str) -> bool:
    """Match ``python``, ``python3``, ``python3.x``; reject ``pythonw``,
    ``python2``, ``python-config``."""
    return bool(_PYTHON_BASENAME_RE.match(name))


def _is_python_real_basename(name: str) -> bool:
    """Match ``python-real``, ``python3-real``, ``python3.x-real``.

    PR #5: the layer-1.5 wrapper renames the venv's real interpreter to
    ``python-real`` so the wrapper at ``${venv}/bin/python`` can find it
    via ``$(dirname "$0")/python-real``. Direct invocation of the real
    interpreter (e.g., ``${venv}/bin/python-real script.py``) skips the
    wrapper entirely - no exec_python event fires - which would let an
    agent bypass layer-1.5 attestation. The parser treats this as both
    a python invocation (so the AST walker yields it) AND a bypass
    primitive (so ``find_python_bypass_invocations`` flags the run).
    """
    return bool(_PYTHON_REAL_BASENAME_RE.match(name))


def _is_literal_word(node) -> bool:
    """True iff a WordNode contains no non-literal sub-parts. bashlex
    attaches ``parameter`` / ``commandsubstitution`` / ``arithmetic``
    children to non-literal words; literal words have ``parts == []``."""
    return not getattr(node, "parts", None)


def _assert_literal(node, position: str) -> None:
    """Raise ``ShellCommandIndeterminate`` if ``node`` is non-literal."""
    if not _is_literal_word(node):
        raise ShellCommandIndeterminate(
            f"Bash {position} word contains a non-literal expansion "
            f"(variable, command substitution, or parameter expansion): "
            f"{node.word!r}. Run cannot be validated."
        )


def _walk_commands(nodes, depth: int = 0) -> Iterator:  # noqa: C901
    """Recursively yield every CommandNode reachable from ``nodes``.

    bashlex node attributes:
    - Most non-leaf nodes (list, pipeline, if, while, for, function)
      hold children in ``.parts``.
    - The ``compound`` node (wrapping ``{...}``, ``(...)``, and the
      body of if/while/for) holds its children in ``.list``.
    - WordNodes whose value contains command substitution (``$(...)``,
      ``\\`...\\```) have a ``commandsubstitution`` sub-part with a
      ``.command`` attribute pointing at the inner CommandNode.
    - ``eval``/``sh -c`` payloads are recursively re-parsed via
      ``_maybe_recurse_eval``.

    Walking all of these surfaces every Python invocation regardless of
    whether it appears at top level, inside a control-flow body, inside
    a quoted shell payload, or inside a command substitution. ``depth``
    bounds eval/sh-c re-parse recursion (NOT generic AST descent,
    which is bounded by construction).
    """
    items = nodes if isinstance(nodes, list) else [nodes]
    for n in items:
        # bashlex can return string fragments for some malformed inputs
        # (e.g. lone comments). Skip non-node items defensively.
        if not hasattr(n, "kind"):
            continue
        kind = getattr(n, "kind", None)
        if kind == "command":
            yield from _maybe_recurse_eval(n, depth)
            yield n
            # Also walk into any command-substitutions in the word args.
            yield from _walk_command_substitutions_in_words(n.parts, depth)
            continue
        if kind in (
            "list",
            "pipeline",
            "compound",
            "if",
            "while",
            "until",
            "for",
            "case",
            "function",
        ):
            children = getattr(n, "parts", None) or getattr(n, "list", None) or []
            yield from _walk_commands(children, depth)
            continue
        # reservedword / operator / pipe / redirect / word / assignment are leaves; skip.


def _walk_command_substitutions_in_words(parts, depth: int = 0) -> Iterator:
    """Walk every embedded substitution under ``parts`` and yield the
    CommandNodes inside.

    Recursively traverses each part looking for ``commandsubstitution``
    (``$(...)``, backticks) and ``processsubstitution`` (``<(...)``,
    ``>(...)``) nodes. The substitution can be hidden anywhere a Bash
    command exposes word-like or wordlist-like content:

    - WordNode args: ``echo $(python ...)``
    - AssignmentNode values: ``OUT=$(python ...)``
    - RedirectNode targets: ``echo ok > "$(python ...)"`` (the redirect's
      .output / .input / .heredoc fields are WordNodes that can carry
      substitutions)
    - ProcessSubstitution: ``cat <(python ...)`` (its own node kind
      mirroring commandsubstitution)

    Without this coverage an agent could execute Python via a substitution
    in any of these surfaces and the AST walker would miss the
    invocation, allowing layer-2 loss to merge as a clean record.
    """
    for p in parts:
        yield from _yield_embedded_commands(p, depth)


def _yield_embedded_commands(node, depth: int = 0) -> Iterator:
    """Recursively visit ``node`` looking for command/process substitution
    sub-parts and yield the CommandNodes inside. Generic over node kind
    so the walker doesn't need to enumerate every container shape."""
    if node is None:
        return
    kind = getattr(node, "kind", None)
    if kind in ("commandsubstitution", "processsubstitution"):
        inner = getattr(node, "command", None)
        if inner is not None:
            yield from _walk_commands(inner, depth)
        return
    # Recurse through every attribute that can hold sub-nodes.
    for attr in ("parts", "list", "command", "input", "output", "heredoc"):
        sub = getattr(node, attr, None)
        if sub is None:
            continue
        if isinstance(sub, list):
            for s in sub:
                yield from _yield_embedded_commands(s, depth)
        else:
            yield from _yield_embedded_commands(sub, depth)


def _maybe_recurse_eval(command_node, depth: int) -> Iterator:
    """If ``command_node`` is ``eval CMD`` or ``bash -c CMD`` (or
    sh/zsh/dash/ksh -c), bashlex-parse ``CMD`` and yield its
    CommandNodes. Bounded by ``_MAX_EVAL_RECURSION`` levels of nesting."""
    word_parts = [p for p in command_node.parts if getattr(p, "kind", None) == "word"]
    if not word_parts:
        return
    # Gate: only recurse if the FIRST word is a recognized wrapper /
    # shell / eval / env. Otherwise the shell-payload tokens later in
    # the word list are arguments to a non-executing command (e.g.
    # ``echo bash -c "python script.py"`` - bash here is an arg to
    # echo, not an executed shell wrapper). R32 P2: pre-gate the scan
    # to avoid false-positive recursion that would extract a Python
    # invocation from non-executed text.
    first = word_parts[0]
    if not _is_literal_word(first):
        return
    first_basename = os.path.basename(first.word)
    if (
        first_basename not in _PYTHON_WRAPPER_BASENAMES
        and first_basename not in _INLINE_PARSER_NAMES
        and first_basename not in _DASH_C_PARSER_NAMES
        and first_basename != "env"
    ):
        return
    # Scan word positions for shell-payload wrappers. R28 P0: forms
    # like `timeout 30 bash -c "python ..."`, `sudo bash -c "..."`,
    # `nice sh -c "..."`, and `env -S "..."` hide a shell payload
    # behind a recognized wrapper; "only first word" missed those, so
    # we scan beyond position 0 (but the gate above ensures we're in
    # an executable-chain context).
    i = 0
    while i < len(word_parts):
        w = word_parts[i]
        if not _is_literal_word(w):
            i += 1
            continue
        basename = os.path.basename(w.word)

        if basename in _INLINE_PARSER_NAMES:
            if i + 1 < len(word_parts):
                payload = word_parts[i + 1]
                yield from _recurse_payload(payload, depth, "eval")
            i += 2
            continue
        if basename in _DASH_C_PARSER_NAMES:
            # Look for -c (or compact -lc/-ic) flag anywhere after this
            # shell-wrapper word. The lazy `-o pipefail -c "..."`
            # shape is supported because the inner scan skips non-c
            # short-flag bundles.
            consumed = 1
            for j in range(i + 1, len(word_parts)):
                w2 = word_parts[j]
                if not _is_literal_word(w2):
                    continue
                if w2.word.startswith("-") and not w2.word.startswith("--") and "c" in w2.word[1:]:
                    if j + 1 < len(word_parts):
                        payload = word_parts[j + 1]
                        yield from _recurse_payload(payload, depth, f"{basename} -c")
                    consumed = j - i + 2
                    break
            i += consumed
            continue
        if basename == "env":
            # `env -S PAYLOAD` and `env --split-string=PAYLOAD` carry an
            # inline shell command string. `env --split-string PAYLOAD`
            # (separate arg) also valid. Recurse the same way as `sh -c`.
            consumed = 1
            for j in range(i + 1, len(word_parts)):
                w2 = word_parts[j]
                if not _is_literal_word(w2):
                    continue
                if w2.word in ("-S", "--split-string"):
                    if j + 1 < len(word_parts):
                        payload = word_parts[j + 1]
                        yield from _recurse_payload(payload, depth, "env -S")
                    consumed = j - i + 2
                    break
                if w2.word.startswith("--split-string="):
                    payload_text = w2.word[len("--split-string=") :]
                    yield from _recurse_payload_str(payload_text, depth, "env --split-string=")
                    consumed = j - i + 1
                    break
                # Compact short-option attached form: `env -S<payload>`
                # with no space. POSIX short options allow this; bashlex
                # tokenizes the whole `-S<payload>` as a single word.
                # Excludes the bare `-S` (handled above) and `--split-`
                # long forms (handled above).
                if w2.word.startswith("-S") and len(w2.word) > 2 and not w2.word.startswith("--"):
                    payload_text = w2.word[2:]
                    yield from _recurse_payload_str(payload_text, depth, "env -S")
                    consumed = j - i + 1
                    break
            i += consumed
            continue
        i += 1


def _recurse_payload(payload_node, depth: int, label: str) -> Iterator:
    """Recursively bashlex-parse a literal WordNode payload and yield
    its CommandNodes. Raises if the payload is non-literal or recursion
    depth would exceed the bound."""
    if not _is_literal_word(payload_node):
        raise ShellCommandIndeterminate(
            f"{label} payload is non-literal: " f"{payload_node.word!r}. Run cannot be validated."
        )
    yield from _recurse_payload_str(payload_node.word, depth, label)


def _recurse_payload_str(payload_text: str, depth: int, label: str) -> Iterator:
    """Recursively bashlex-parse a literal payload string and yield its
    CommandNodes."""
    if depth >= _MAX_EVAL_RECURSION:
        raise ShellCommandParseError(
            f"{label} recursion depth exceeded {_MAX_EVAL_RECURSION}: "
            f"nested wrapper chain too deep"
        )
    inner = _parse(payload_text)
    yield from _walk_commands(inner, depth + 1)


def _extract_python_argv(command_node) -> Optional[list[str]]:
    """If ``command_node`` is a Python invocation, return its argv as
    ``[interpreter, *args]``. Else return ``None``.

    Walks the CommandNode's WordNodes (after stripping AssignmentNode
    prefixes). The first word is the command-word: if it is NOT python
    AND NOT a known wrapper, this is not a Python invocation - return
    ``None``. If it IS a known wrapper, scan forward through the
    remaining words looking for a Python token. Once found, the
    interpreter token is the literal command-word and the remaining
    literal words are args.

    This wrapper-then-scan logic preserves the existing parser's
    behavior for wrapper-with-args forms (``nice -n 10 python ...``,
    ``timeout --signal=KILL 30 python ...``, ``sudo -u alice python
    ...``) where the wrapper's argv structure varies per modifier. We
    don't model individual modifier arg shapes; instead, we scan past
    wrapper args until python appears. This has a small false-positive
    risk (e.g. ``nice grep python file.py`` would extract python from
    the grep argv) identical to what the previous regex parser
    accepted; in practice agents don't search for the literal string
    'python' under a wrapper.

    Raises ``ShellCommandIndeterminate`` if the command word OR any
    argv-position word is non-literal.
    """
    words = [p for p in command_node.parts if getattr(p, "kind", None) == "word"]
    if not words:
        return None
    i = 0
    after_wrapper = False
    while i < len(words):
        w = words[i]
        _assert_literal(w, "command-word" if not after_wrapper else "argv")
        basename = os.path.basename(w.word)
        if _is_python_basename(basename) or _is_python_real_basename(basename):
            args: list[str] = []
            for p in words[i + 1 :]:
                _assert_literal(p, "argv")
                args.append(p.word)
            return [w.word, *args]
        if basename in _PYTHON_WRAPPER_BASENAMES:
            after_wrapper = True
            i += 1
            continue
        if not after_wrapper:
            # First word is neither python nor a wrapper - this
            # CommandNode is not a Python invocation. Return without
            # scanning the rest of the words (they are positional args
            # to whatever the command-word is).
            return None
        # After a wrapper; the current word is a wrapper-arg or
        # wrapper-arg-value. Advance and keep scanning for python.
        i += 1
    return None


def _prefix_assignments(command_node) -> list[tuple[str, str]]:
    """Return ``[(key, value), ...]`` for AssignmentNodes prefixing a
    CommandNode. ``PATH=/usr/bin VAR=1 python ...`` -> ``[('PATH',
    '/usr/bin'), ('VAR', '1')]``."""
    out: list[tuple[str, str]] = []
    for p in command_node.parts:
        if getattr(p, "kind", None) != "assignment":
            break
        word = getattr(p, "word", "")
        if "=" in word:
            k, _, v = word.partition("=")
            out.append((k, v))
    return out


def _first_word_node(command_node):
    """Return the first WordNode of a CommandNode (after assignments).
    Used to detect activation scripts (``.`` / ``source``)."""
    for p in command_node.parts:
        if getattr(p, "kind", None) == "word":
            return p
    return None


def parse_python_invocations(command: str) -> list[list[str]]:
    """Return argv lists for every Python invocation visible in
    ``command``.

    Each argv is ``[interpreter, *args]`` matching ``sys.orig_argv``
    shape: the interpreter is the literal first command-word (after
    env-prefix and wrapper stripping), and args are the remaining
    literal argv-position words.

    Raises:
      ShellCommandIndeterminate: a Python command-word or argv-word
        contains non-literal expansion (variable / command-substitution
        / parameter expansion). Static parsing cannot prove the resolved
        value is not load-bearing.
      ShellCommandParseError: bashlex could not parse the Bash command
        (including bash forms bashlex does not model, e.g. ``case``).
    """
    nodes = _parse(command)
    out: list[list[str]] = []
    for cn in _walk_commands(nodes):
        argv = _extract_python_argv(cn)
        if argv is not None:
            out.append(argv)
    return out


_REAL_INTERPRETER_BYPASS_RE = re.compile(
    r"\.pyruntime-real\b|\bpython(?:3(?:\.\d+)?)?-real\b|\.actual-python\b"
)


def find_python_bypass_invocations(command: str) -> list[str]:
    """Return ``[command]`` if it contains any Python bypass primitive,
    else ``[]``.

    Bypass primitives:
      - ``-S`` (or compact ``-Sc`` / ``-IS``) in a Python interpreter's
        pre-script argv: disables ``site.py`` import.
      - ``PATH=...`` or ``PYTHON*`` env-prefix on a Python CommandNode:
        env-prefix assignment changes the interpreter or import path.
      - ``export PATH=...`` / ``export PYTHON*=...`` in any CommandNode
        earlier in the same outer command: env mutation persists for
        subsequent commands in the same shell.
      - ``env -u VAR`` wrapper: unsets an env var (specifically
        ``_PYRUNTIME_EVENT_LOG``) before exec'ing the wrapped Python.
      - ``.``, ``source``, ``conda activate``, ``pyenv shell`` activation
        command appearing in the same outer command before a Python
        CommandNode: shell activation mutates env (often PATH).
      - **Any occurrence of ``.pyruntime-real`` or ``python-real`` /
        ``python3-real`` / ``python3.X-real`` in the visible Bash command**:
        these tokens reference the layer-1.5 wrapper's hidden real
        interpreter location. Direct command-word invocation is caught by
        the AST walker; delegated forms like
        ``find venv -name python-real -exec {} -S script.py \\;`` keep
        the token inside a non-CommandNode argv but still trigger this
        substring check. The merger fails closed regardless of how the
        agent threaded the reference. (PR #5 R2 P0 #1.)

    Raises ShellCommandIndeterminate / ShellCommandParseError per the
    parser's normal contract.
    """
    # Cheap pre-pass: any reference to the real-interpreter location or
    # its basename in the visible command is a bypass primitive, even if
    # the token sits inside a delegated execution argv (find -exec,
    # xargs, fd -x, parallel) that the AST walker cannot model.
    if _REAL_INTERPRETER_BYPASS_RE.search(command):
        return [command]
    nodes = _parse(command)
    has_activation = False
    has_path_mutation = False
    out: list[str] = []
    for cn in _walk_commands(nodes):
        first_word = _first_word_node(cn)
        if first_word is not None and _is_literal_word(first_word):
            fw = first_word.word
            if fw in (".", "source"):
                has_activation = True
            elif fw == "conda" and _has_arg(cn, "activate"):
                has_activation = True
            elif fw == "pyenv" and _has_arg(cn, "shell"):
                has_activation = True
            elif fw == "export" and _has_path_or_python_assignment_arg(cn):
                has_path_mutation = True
        # Standalone assignment-only commands like ``PATH=/usr/bin``
        # (no command-word) also mutate the shell's env for following
        # commands. The CommandNode has only AssignmentNode parts and
        # no WordNodes.
        if _is_assignment_only_path_or_python(cn):
            has_path_mutation = True
        # `env -u VAR python` removes VAR before exec. Detect before
        # walking the wrapper-prefix stripper (which would otherwise
        # treat env as a benign wrapper).
        if _env_wrapper_bypasses_python(cn):
            out.append(command)
            break
        argv = _extract_python_argv(cn)
        if argv is None:
            continue
        # PR #5: any direct invocation of ``python-real`` /
        # ``python3-real`` / ``python3.X-real`` is a bypass primitive
        # (skips the layer-1.5 wrapper and the wrapper's exec_python
        # event emission). The argv[0] is the path the agent invoked;
        # the basename check catches both ``python-real`` and
        # ``${venv}/bin/python-real``.
        if _is_python_real_basename(os.path.basename(argv[0])):
            out.append(command)
            break
        if argv_contains_bypass_flag(argv[1:]):
            out.append(command)
            break
        bypass_via_assignment = False
        for k, _value in _prefix_assignments(cn):
            if k == "PATH" or k.startswith("PYTHON") or k == "_PYRUNTIME_EVENT_LOG":
                bypass_via_assignment = True
                break
        if bypass_via_assignment:
            out.append(command)
            break
        if has_activation or has_path_mutation:
            out.append(command)
            break
    return out


def _has_arg(command_node, target: str) -> bool:
    """Return True if any literal WordNode arg of `command_node` equals
    `target`."""
    for p in command_node.parts:
        if getattr(p, "kind", None) == "word" and _is_literal_word(p):
            if p.word == target:
                return True
    return False


def _has_path_or_python_assignment_arg(command_node) -> bool:
    """For an `export` CommandNode: True if any arg word's ``KEY=`` prefix
    matches PATH, PYTHON*, or ``_PYRUNTIME_EVENT_LOG``. The value may be
    literal or non-literal (``export PATH=/usr/bin:$PATH``); only the
    key matters for detection. PR #5 R6 P0: ``_PYRUNTIME_EVENT_LOG``
    retargeting (``export _PYRUNTIME_EVENT_LOG=/tmp/fake``) sends shim
    events to an attacker-chosen file while the runner-owned log
    appears empty."""
    for p in command_node.parts:
        if getattr(p, "kind", None) != "word":
            continue
        w = getattr(p, "word", "")
        if "=" in w:
            k, _, _ = w.partition("=")
            if k == "PATH" or k.startswith("PYTHON") or k == "_PYRUNTIME_EVENT_LOG":
                return True
    return False


def _is_assignment_only_path_or_python(command_node) -> bool:
    """True if ``command_node`` has only AssignmentNode parts (no command
    word) AND any assignment's key is PATH or PYTHON*. Bash parses
    ``PATH=/usr/bin`` as a CommandNode whose first part is an
    AssignmentNode and which has no WordNodes - this assigns the env var
    for the remainder of the shell."""
    has_word = False
    has_target_assignment = False
    for p in command_node.parts:
        kind = getattr(p, "kind", None)
        if kind == "word":
            has_word = True
        elif kind == "assignment":
            word = getattr(p, "word", "")
            if "=" in word:
                k, _, _ = word.partition("=")
                if k == "PATH" or k.startswith("PYTHON") or k == "_PYRUNTIME_EVENT_LOG":
                    has_target_assignment = True
    return has_target_assignment and not has_word


def _env_wrapper_bypasses_python(command_node) -> bool:
    """For a CommandNode whose command-word is ``env``: True if it
    carries any env-mutation primitive followed by a Python invocation.

    Detected env primitives (all mirror the prefix/export bypass forms):

    - ``-u VAR`` / ``-uVAR`` / ``--unset=VAR`` / ``--unset VAR``: remove
      an env var (the agent can target ``_PYRUNTIME_EVENT_LOG`` directly).
    - ``-i`` / ``--ignore-environment``: clear the entire env before
      exec, dropping every shim env var.
    - ``PATH=...`` arg: redirect interpreter resolution.
    - ``PYTHON*=...`` arg (PYTHONHOME, PYTHONPATH, PYTHONSTARTUP, etc.):
      change import resolution or run startup code.

    Plain ``env python script.py`` (no primitive) is NOT a bypass; env
    just inherits the existing environment and resolves python via PATH.
    """
    words = [p for p in command_node.parts if getattr(p, "kind", None) == "word"]
    if not words:
        return False
    first = words[0]
    if not _is_literal_word(first):
        return False
    if os.path.basename(first.word) != "env":
        return False
    has_primitive = False
    has_python = False
    for w in words[1:]:
        if not _is_literal_word(w):
            continue
        tok = w.word
        if tok in ("-i", "--ignore-environment"):
            has_primitive = True
            continue
        if tok == "-u" or tok.startswith("-u") or tok.startswith("--unset"):
            has_primitive = True
            continue
        if "=" in tok and not tok.startswith("-"):
            k = tok.partition("=")[0]
            if k == "PATH" or k.startswith("PYTHON") or k == "_PYRUNTIME_EVENT_LOG":
                has_primitive = True
            continue
        bn = os.path.basename(tok)
        if _is_python_basename(bn) or _is_python_real_basename(bn):
            has_python = True
            break
    return has_primitive and has_python


def argv_contains_bypass_flag(args_tokens: list[str]) -> bool:
    """Return True if any pre-script Python interpreter short-flag
    token contains uppercase ``S`` (sitecustomize-disabling flag).

    Walks tokens left-to-right. A token starting with single ``-`` and
    at least one alpha char is a short-flag combination (``-S``,
    ``-Sc``, ``-IS``); scan for uppercase ``S``. ``--`` starts a long
    flag (ignored). The first non-flag token ends the interpreter-flag
    region; ``-S`` after that point is an arg to the script, not an
    interpreter flag.

    Tokens are already shlex-tokenized by bashlex, so a ``-S``
    substring inside a quoted ``-c`` code argument lives in a single
    token whose first char is not ``-`` and does not flag.
    """
    for tok in args_tokens:
        if not tok:
            continue
        if tok == "-":
            return False
        if tok.startswith("--"):
            continue
        if tok.startswith("-"):
            if "S" in tok[1:]:
                return True
            continue
        return False
    return False
