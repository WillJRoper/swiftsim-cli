"""The main module containing the swift-cli command-line tool."""

import argparse
import os
import subprocess
import sys
import warnings
from typing import NoReturn

from swiftsim_cli.cosmology import clear_cosmology_cache
from swiftsim_cli.modes import load_mode
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs
from swiftsim_cli.params import clear_parameter_cache, load_parameters
from swiftsim_cli.profile import move_legacy_dir, set_profile_override

# Errors caused by user input or the environment rather than a bug. These
# are reported as a one-line message (ProfileError is a ValueError, and
# FileNotFoundError, PermissionError, ... are OSErrors). EOFError is a
# prompt reading from a closed stdin.
USER_ERRORS = (ValueError, OSError, EOFError, argparse.ArgumentTypeError)

DEBUG_HINT = " (run with --debug for the full traceback)"


def _fail(message: object, code: int, hint: bool = False) -> NoReturn:
    """Print an error message to stderr and exit.

    Args:
        message: The error (or message) to print.
        code: The exit code.
        hint: Whether to add a hint about --debug.
    """
    text = str(message) or type(message).__name__
    if isinstance(message, EOFError):
        text = "no input (stdin is closed)"
    print(f"error: {text}{DEBUG_HINT if hint else ''}", file=sys.stderr)
    sys.exit(code)


def main(argv: list[str] | None = None) -> None:
    """Run the swiftsim-cli command-line interface.

    User-facing errors are printed as ``error: <message>`` and exit with
    code 1 (a failed subprocess exits with its own code, an interrupt with
    130, a usage error with 2). Pass ``--debug`` or set
    ``SWIFT_CLI_DEBUG=1`` to get the full traceback instead.

    Args:
        argv: The command-line arguments to parse. If None, uses sys.argv.
    """
    debug = os.environ.get("SWIFT_CLI_DEBUG") == "1"
    # Expected and harmless: an analysis with nothing labelled to legend
    warnings.filterwarnings(
        "ignore", message="No artists with labels found to put in legend"
    )
    try:
        # Parse the multi-mode command-line arguments
        multi_args = MultiModeCLIArgs(argv)
        debug = debug or multi_args.global_args.debug

        # Older versions kept their files in ~/.swiftsim-utils
        move_legacy_dir()

        # Run every mode against --profile NAME instead of the current one
        set_profile_override(multi_args.global_args.profile)

        # Execute each mode in sequence
        modes = multi_args.modes
        for i, (mode_name, args) in enumerate(modes):
            # Let a mode see what follows it (update skips its build when
            # a later config/make in the chain will build anyway)
            args.later_modes = [name for name, _ in modes[i + 1 :]]

            # Load the parameters if they exist (for modes that use them)
            clear_parameter_cache()
            clear_cosmology_cache()
            _ = load_parameters(getattr(args, "params", None))

            load_mode(mode_name).run(args)
    except KeyboardInterrupt:
        if debug:
            raise
        _fail("interrupted", 130)
    except subprocess.CalledProcessError as exc:
        if debug:
            raise
        # A negative return code means the child died from that signal;
        # report it the way a shell would (128 + signal number).
        code = exc.returncode
        _fail(exc, 128 - code if code < 0 else code)
    except USER_ERRORS as exc:
        if debug:
            raise
        # A failed build command (CommandError) exits with its own code
        _fail(exc, getattr(exc, "returncode", 1), hint=True)
