"""The main module containing the swift-cli command-line tool."""

import argparse
import os
import subprocess
import sys
from typing import NoReturn

from swiftsim_cli.cosmology import clear_cosmology_cache
from swiftsim_cli.modes import MODE_MODULES
from swiftsim_cli.multi_mode_args import MultiModeCLIArgs
from swiftsim_cli.params import clear_parameter_cache, load_parameters

# Errors caused by user input or the environment rather than a bug. These
# are reported as a one-line message (ProfileError is a ValueError, and
# FileNotFoundError, PermissionError, ... are OSErrors).
USER_ERRORS = (ValueError, OSError, argparse.ArgumentTypeError)


def _fail(message: object, code: int) -> NoReturn:
    """Print an error message to stderr and exit.

    Args:
        message: The error (or message) to print.
        code: The exit code.
    """
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def main(argv: list[str] | None = None) -> None:
    """Run the SWIFT-utils command-line interface.

    User-facing errors are printed as ``error: <message>`` and exit with
    code 1 (a failed subprocess exits with its own code, an interrupt with
    130). Pass ``--debug`` or set ``SWIFT_CLI_DEBUG=1`` to get the full
    traceback instead.

    Args:
        argv: The command-line arguments to parse. If None, uses sys.argv.
    """
    debug = os.environ.get("SWIFT_CLI_DEBUG") == "1"
    try:
        # Parse the multi-mode command-line arguments
        multi_args = MultiModeCLIArgs(argv)
        debug = debug or multi_args.global_args.debug

        # Execute each mode in sequence
        for mode_name, args in multi_args.modes:
            # Load the parameters if they exist (for modes that use them)
            clear_parameter_cache()
            clear_cosmology_cache()
            _ = load_parameters(getattr(args, "params", None))

            # Get the mode module and execute it
            mode_module = MODE_MODULES[mode_name]
            mode_module.run(args)
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
        _fail(exc, 1)
