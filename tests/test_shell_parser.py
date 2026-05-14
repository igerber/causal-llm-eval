"""Direct unit tests for ``harness.shell_parser``.

Most parser behavior is exercised end-to-end via test_telemetry_merger.py
calling ``merge_layers``. This file targets the parser API directly to
pin behavior for: every wrapper shape the CI reviewer enumerated across
R20-R26, indeterminate command-words, bashlex parse failures, recursive
eval/sh-c payload parsing, and command-substitution recursion.
"""

from __future__ import annotations

import pytest

from harness.shell_parser import (
    RunValidityError,
    ShellCommandIndeterminate,
    ShellCommandParseError,
    argv_contains_bypass_flag,
    find_python_bypass_invocations,
    parse_python_invocations,
)

# ---------------------------------------------------------------------------
# Positive: every wrapper shape extracts the inner python invocation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        # Direct
        ("python script.py", [["python", "script.py"]]),
        ("python3 script.py", [["python3", "script.py"]]),
        ("python3.11 -c 'pass'", [["python3.11", "-c", "pass"]]),
        # Compound (R25)
        ("cd /tmp && python script.py", [["python", "script.py"]]),
        ("cd /tmp; python script.py", [["python", "script.py"]]),
        # `time` keyword (bashlex doesn't handle; preprocessed)
        ("time python script.py", [["python", "script.py"]]),
        ("cd /tmp && time python script.py", [["python", "script.py"]]),
        # Path-qualified wrappers (R26)
        ("/usr/bin/time python script.py", [["python", "script.py"]]),
        (
            "/usr/bin/time /usr/bin/python script.py",
            [["/usr/bin/python", "script.py"]],
        ),
        ("/usr/bin/timeout 30 python script.py", [["python", "script.py"]]),
        # Modifiers with args (R25 / R26)
        ("nice -n 10 python script.py", [["python", "script.py"]]),
        (
            "timeout --signal=KILL 30 python script.py",
            [["python", "script.py"]],
        ),
        ("sudo -u alice python script.py", [["python", "script.py"]]),
        # Sudo (R26)
        ("sudo python script.py", [["python", "script.py"]]),
        # Quoted command words (R26)
        ('"python" script.py', [["python", "script.py"]]),
        ("'python' script.py", [["python", "script.py"]]),
        ('"/usr/bin/python" script.py', [["/usr/bin/python", "script.py"]]),
        # env-prefix (R20)
        ("PATH=/usr/bin python script.py", [["python", "script.py"]]),
        ("MPLBACKEND=Agg python script.py", [["python", "script.py"]]),
        # Recursive eval/sh -c
        ('bash -c "python script.py"', [["python", "script.py"]]),
        ("eval 'python script.py'", [["python", "script.py"]]),
        # -lc / -ic / -Sc variants (recursive parse)
        ('bash -lc "python script.py"', [["python", "script.py"]]),
        # Control flow bodies
        ("if true; then python script.py; fi", [["python", "script.py"]]),
        (
            "for f in a; do python f.py; done",
            [["python", "f.py"]],
        ),
        ("while true; do python s.py; break; done", [["python", "s.py"]]),
        # Pipes / background / redirection
        ("python script.py | grep result", [["python", "script.py"]]),
        ("python script.py &", [["python", "script.py"]]),
        ("python script.py > out.txt", [["python", "script.py"]]),
        # Command-substitution containing python (recursed via word parts)
        (
            "echo $(python -S script.py)",
            [["python", "-S", "script.py"]],
        ),
        (
            "echo `python -S script.py`",
            [["python", "-S", "script.py"]],
        ),
        # Multiple invocations
        (
            "python a.py && python b.py",
            [["python", "a.py"], ["python", "b.py"]],
        ),
    ],
)
def test_parse_python_invocations_positive(command, expected):
    assert parse_python_invocations(command) == expected


# ---------------------------------------------------------------------------
# Negative: non-python commands return empty list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # python as an argument to another command
        "grep python script.py",
        "echo python",
        "cat python.txt",
        "ls /opt/python/",
        # Other languages
        "node script.js",
        "ruby script.rb",
        "perl script.pl",
        # Empty / whitespace
        "true",
        "false",
        ":",
    ],
)
def test_parse_python_invocations_no_python(command):
    assert parse_python_invocations(command) == []


# ---------------------------------------------------------------------------
# Indeterminate command-word / argv: fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # Command-word variable expansion
        "${PY} script.py",
        "${PYTHON:-python} script.py",
        "$PYBIN script.py",
        # Command-word command substitution
        "$(which python) script.py",
        "`which python` script.py",
        "$(command -v python) script.py",
        "`type -p python3` script.py",
        # Argv-position non-literal
        "python ${SCRIPT}",
        'python -c "$CODE"',
        "python $(generate_args.sh)",
        # eval / sh -c with non-literal payload
        'bash -c "$VAR"',
        "eval $CMD",
    ],
)
def test_parse_python_invocations_indeterminate_raises(command):
    with pytest.raises(ShellCommandIndeterminate):
        parse_python_invocations(command)


# ---------------------------------------------------------------------------
# Parse-failure: bashlex cannot model the form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # bashlex does not implement case-pattern parsing
        "case x in foo) python script.py;; esac",
        "case $x in y) python s.py;; esac",
    ],
)
def test_parse_python_invocations_parse_error(command):
    with pytest.raises(ShellCommandParseError):
        parse_python_invocations(command)


# ---------------------------------------------------------------------------
# Bypass detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        # -S flag (R13/R14)
        "python -S script.py",
        "python -Sc 'import os'",
        "cd /tmp && python -S script.py",
        # PATH= env-prefix on the python command
        "PATH=/usr/bin python script.py",
        # PYTHON* env-prefix
        "PYTHONHOME=/foo python script.py",
        "PYTHONPATH=/bar python script.py",
        # Standalone PATH= assignment then python
        "PATH=/usr/bin && python script.py",
        # export PATH=
        "export PATH=/usr/bin && python script.py",
        "export PATH=/usr/bin:$PATH && python script.py",
        # Activation
        ". venv/activate && python script.py",
        "source venv/activate && python script.py",
        "conda activate myenv && python script.py",
        "pyenv shell myenv && python script.py",
        # env -u
        "env -u _PYRUNTIME_EVENT_LOG python script.py",
    ],
)
def test_find_python_bypass_invocations_positive(command):
    assert find_python_bypass_invocations(command) == [command]


@pytest.mark.parametrize(
    "command",
    [
        # No python: nothing to bypass
        "ls -la",
        "echo hello",
        "cat file.txt",
        # Benign python without bypass primitives
        "python script.py",
        "/usr/bin/python script.py",
        "time python script.py",
        "nice python script.py",
        # -s lowercase (NOT a bypass)
        "python -s script.py",
        # PATH= but no python
        "PATH=/usr/bin ls",
    ],
)
def test_find_python_bypass_invocations_negative(command):
    assert find_python_bypass_invocations(command) == []


# ---------------------------------------------------------------------------
# argv bypass-flag walker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,expected",
    [
        # Positive
        (["-S", "script.py"], True),
        (["-Sc", "code"], True),
        (["-IS", "script.py"], True),
        (["-c", "import os; print(1)", "-S"], False),  # -S after script
        # Negative
        ([], False),
        (["script.py"], False),
        (["-s", "script.py"], False),  # lowercase
        (["-c", "code"], False),
        (["-", "stdin-mode"], False),
        # Long flags don't end the interpreter-flag region; -S after a
        # long flag IS detected (consistent with the original walker).
        (["--config=x", "-S"], True),
        (["--", "-S"], True),
        # Plain script arg ends the region; -S after script arg is the
        # script's own arg, not an interpreter flag.
        (["script.py", "-S"], False),
    ],
)
def test_argv_contains_bypass_flag(argv, expected):
    assert argv_contains_bypass_flag(argv) is expected


# ---------------------------------------------------------------------------
# Recursive eval/sh-c depth bound
# ---------------------------------------------------------------------------


def test_eval_sh_c_recursion_single_level():
    """Single-level ``bash -c "python ..."`` extracts the inner Python
    invocation."""
    result = parse_python_invocations('bash -c "python script.py"')
    assert result == [["python", "script.py"]]


def test_eval_sh_c_recursion_two_levels():
    """Two-level nesting with alternating quote types extracts the
    innermost Python invocation. (Bash single-quotes do not nest, so we
    alternate single/double around each level.)"""
    cmd = "bash -c 'bash -c \"python script.py\"'"
    result = parse_python_invocations(cmd)
    assert result == [["python", "script.py"]]


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_exception_hierarchy():
    """``RunValidityError`` is the neutral parent of the parser's
    specific subclasses. Callers catching the parent catch both."""
    assert issubclass(ShellCommandIndeterminate, RunValidityError)
    assert issubclass(ShellCommandParseError, RunValidityError)


def test_run_validity_error_catches_both_subclasses():
    """A handler matching ``RunValidityError`` catches both parser
    exception subclasses (used by the merger to unify fail-closed
    handling)."""
    with pytest.raises(RunValidityError):
        parse_python_invocations("${PY} script.py")
    with pytest.raises(RunValidityError):
        parse_python_invocations("case x in y) python;; esac")
