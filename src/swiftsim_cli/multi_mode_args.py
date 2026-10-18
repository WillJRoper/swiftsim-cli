"""Multi-mode argument parser for swift-cli.

A command line is a list of global options followed by one or more modes,
each with its own arguments::

    swift-cli [--profile NAME] [--debug] <mode> [args] [<mode> [args]] ...

A token naming a mode starts a new mode section unless the current mode's
parser expects it as a value (e.g. ``new --path make`` or ``switch show``).
"""

import argparse
import re
import sys
from typing import List, Sequence, Tuple, cast

from swiftsim_cli import __version__
from swiftsim_cli.modes import (
    AVAILABLE_MODES,
    MODE_DESCRIPTIONS,
    MODE_MODULES,
    Mode,
)
from swiftsim_cli.profile import set_profile_override

# Variable-length nargs values (argparse.OPTIONAL, ZERO_OR_MORE, ...)
_VARIABLE_NARGS = ("?", "*", "+")

# Tokens like "-1" or "-2.5e3" are values, not options
_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+(e[-+]?\d+)?$", re.I)


def _nargs(action: argparse.Action) -> int | str:
    """Return the number of values an action consumes.

    Args:
        action: The argparse action.

    Returns:
        The fixed count, or one of "?", "*", "+" for variable counts.
    """
    if action.nargs is None:
        return 1
    if isinstance(action.nargs, int):
        return action.nargs
    if action.nargs in _VARIABLE_NARGS:
        return action.nargs
    return "*"  # argparse.REMAINDER and friends


class _ModeScanner:
    """Track which tokens a mode's parser would consume as values.

    Feed each token after the mode name to `belongs`; it returns False
    for a mode name the parser does not expect, which starts a new mode.
    Expected values are read from the parser's own actions: the nargs of
    the last option, the remaining positionals and any subcommands.
    """

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        """Start scanning the arguments of a parser.

        Args:
            parser: The mode's argument parser.
        """
        self._enter(parser)

    def _enter(self, parser: argparse.ArgumentParser) -> None:
        """Make a (sub)parser the one whose arguments are being scanned."""
        self.parser = parser
        self.positionals = [a for a in parser._actions if not a.option_strings]
        self.owed = 0  # values still owed to a fixed-nargs argument
        self.open: int | str | None = None  # an open variable-nargs argument
        self.seen = 0  # values given to the open argument so far

    def _option_nargs(self, token: str) -> int | str:
        """Return how many values follow an option token."""
        if "=" in token and token.startswith("--"):
            return 0  # --opt=value carries its value
        actions = self.parser._option_string_actions
        action = actions.get(token)
        if action is None and token.startswith("--"):
            # argparse accepts unambiguous abbreviations of long options
            matches = {a for s, a in actions.items() if s.startswith(token)}
            if len(matches) == 1:
                action = matches.pop()
        # Unknown options (e.g. configure flags) and attached short values
        # (-j4) take nothing from the following tokens
        return 0 if action is None else _nargs(action)

    def belongs(self, token: str) -> bool:
        """Consume a token, returning False if it starts a new mode.

        Args:
            token: The next command-line token.

        Returns:
            Whether the token belongs to the current mode.
        """
        is_mode = token in AVAILABLE_MODES

        if self.owed:
            self.owed -= 1
            return True

        if (
            token.startswith("-")
            and len(token) > 1
            and not _NEGATIVE_NUMBER.match(token)
        ):
            nargs = self._option_nargs(token)
            if isinstance(nargs, int):
                self.open, self.owed = None, nargs
            else:
                self.open, self.seen = nargs, 0
            return True

        if self.open is not None:
            # A mode name ends an optional value list, but is taken as the
            # first value of a "+" argument (which needs at least one)
            if is_mode and (self.open != "+" or self.seen):
                return False
            self.seen += 1
            if self.open == "?":
                self.open = None
            return True

        if not self.positionals:
            return not is_mode

        action = self.positionals.pop(0)
        if isinstance(action, argparse._SubParsersAction):
            if token in action.choices:
                self._enter(action.choices[token])
                return True
            return not is_mode

        nargs = _nargs(action)
        if isinstance(nargs, int):
            self.owed = nargs - 1
            return True
        if is_mode and nargs != "+":
            return False
        self.open, self.seen = (None if nargs == "?" else nargs), 1
        return True


class MultiModeCLIArgs:
    """Multi-mode command-line argument parser for swift-cli.

    Allows executing multiple modes in sequence, e.g.:
    swift-cli config --enable-debug make -j 32
    """

    def __init__(self, argv: Sequence[str] | None = None) -> None:
        """Initialize and parse multi-mode arguments.

        Args:
            argv: Command-line arguments to parse. If None, uses sys.argv.
        """
        if argv is None:
            argv = sys.argv[1:]

        # Parse the global options that precede the first mode
        self.global_args, remaining_argv = self._parse_global_args(list(argv))

        # Split remaining arguments into mode sections and parse each one
        self.modes: List[Tuple[Mode, argparse.Namespace]] = [
            (mode_name, self._parse_mode_args(mode_name, parser, mode_argv))
            for mode_name, parser, mode_argv in self._split_by_modes(
                remaining_argv
            )
        ]

    @staticmethod
    def _global_parser() -> argparse.ArgumentParser:
        """Build the parser for the global options."""
        parser = argparse.ArgumentParser(
            prog="swift-cli",
            usage="swift-cli [global options] <mode> [mode args] "
            "[<mode> [mode args]] ...",
            add_help=False,
            allow_abbrev=False,
        )
        parser.add_argument("-h", "--help", action="store_true")
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {__version__}"
        )
        parser.add_argument("--profile", metavar="NAME", default=None)
        parser.add_argument("--debug", action="store_true")
        return parser

    def _parse_global_args(
        self, argv: List[str]
    ) -> Tuple[argparse.Namespace, List[str]]:
        """Parse the global options that precede the first mode.

        Prints the help and exits for -h/--help or when no mode is given,
        and exits with a usage error for anything else before the first
        mode.

        Args:
            argv: Full argument list

        Returns:
            Tuple of (global_args, remaining_argv)
        """
        parser = self._global_parser()

        # The global section ends at the first mode name not taken as an
        # option value (e.g. "--profile make" names a profile)
        scanner = _ModeScanner(parser)
        split = next(
            (i for i, arg in enumerate(argv) if not scanner.belongs(arg)),
            len(argv),
        )
        global_args, unknown = parser.parse_known_args(argv[:split])
        if global_args.help:
            self._print_help()
            sys.exit(0)
        if unknown:
            parser.error(
                f"unknown mode or option '{unknown[0]}' (choose from: "
                f"{', '.join(AVAILABLE_MODES)})"
            )
        if split == len(argv):
            self._print_help()
            sys.exit(1)

        # Run every mode against --profile NAME instead of the current one
        set_profile_override(global_args.profile)
        return global_args, argv[split:]

    def _split_by_modes(
        self, argv: List[str]
    ) -> List[Tuple[Mode, argparse.ArgumentParser, List[str]]]:
        """Split argument list into mode sections.

        Args:
            argv: Arguments starting with a mode name.

        Returns:
            List of (mode_name, mode_parser, mode_args) tuples.
        """
        sections: List[Tuple[Mode, argparse.ArgumentParser, List[str]]] = []
        scanner = None
        for arg in argv:
            if scanner is None or not scanner.belongs(arg):
                mode_name = cast(Mode, arg)
                parser = self._mode_parser(mode_name)
                sections.append((mode_name, parser, []))
                scanner = _ModeScanner(parser)
            else:
                sections[-1][2].append(arg)
        return sections

    @staticmethod
    def _mode_parser(mode_name: Mode) -> argparse.ArgumentParser:
        """Build the argument parser for a mode.

        Args:
            mode_name: Name of the mode

        Returns:
            The mode's parser.
        """
        parser = argparse.ArgumentParser(
            prog=f"swift-cli {mode_name}",
            description=MODE_DESCRIPTIONS[mode_name],
        )
        MODE_MODULES[mode_name].add_arguments(parser)
        return parser

    @staticmethod
    def _parse_mode_args(
        mode_name: Mode,
        parser: argparse.ArgumentParser,
        mode_argv: List[str],
    ) -> argparse.Namespace:
        """Parse arguments for a specific mode.

        Args:
            mode_name: Name of the mode
            parser: The mode's parser.
            mode_argv: Arguments for this mode

        Returns:
            Parsed arguments namespace
        """
        # Config passes unknown --<option> flags through to ./configure
        if mode_name == "config":
            args, unknown = parser.parse_known_args(mode_argv)
            args.options = unknown
            return args
        return parser.parse_args(mode_argv)

    def _print_help(self) -> None:
        """Print comprehensive help for all modes."""
        print("swift-cli: Utilities for SWIFT development workflows")
        print()
        print("Usage:")
        print(
            "  swift-cli [global options] <mode> [mode args] "
            "[<mode> [mode args]] ..."
        )
        print()
        print("Global options (must come before the first mode):")
        print("  -h, --help      Show this help and exit.")
        print("  --version       Show the version and exit.")
        print("  --profile NAME  Run the modes against profile NAME.")
        print("  --debug         Show full tracebacks for errors (or set")
        print("                  SWIFT_CLI_DEBUG=1).")
        print()
        print("Available modes:")

        for mode_name in AVAILABLE_MODES:
            print(f"  {mode_name:<15} {MODE_DESCRIPTIONS[mode_name]}")

        print()
        print("Examples:")
        print("  swift-cli config --enable-debug")
        print("  swift-cli make -j 8")
        print("  swift-cli config --enable-debug make -j 32")
        print("  swift-cli update config --enable-debug make -j 8")
        print("  swift-cli --profile zoom make -j 16")
        print()
        print("For mode-specific help: swift-cli <mode> --help")
