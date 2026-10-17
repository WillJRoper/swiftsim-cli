"""Multi-mode argument parser for swift-cli."""

import argparse
import sys
from typing import List, Sequence, Tuple, cast

from swiftsim_cli.modes import AVAILABLE_MODES, MODE_MODULES, Mode
from swiftsim_cli.profile import set_profile_override


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

        # Parse global arguments first
        self.global_args, remaining_argv = self._parse_global_args(list(argv))

        # Split remaining arguments by mode keywords
        self.modes: List[Tuple[Mode, argparse.Namespace]] = []
        mode_sections = self._split_by_modes(remaining_argv)

        # Parse each mode section
        for mode_name, mode_argv in mode_sections:
            parsed_args = self._parse_mode_args(mode_name, mode_argv)
            # Add global args to each mode's args
            self._merge_global_args(parsed_args)
            self.modes.append((mode_name, parsed_args))

    def _parse_global_args(
        self, argv: List[str]
    ) -> Tuple[argparse.Namespace, List[str]]:
        """Parse global arguments that apply to all modes.

        Args:
            argv: Full argument list

        Returns:
            Tuple of (global_args, remaining_argv)
        """
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-v", "--verbose", action="store_true")
        parser.add_argument("--profile", default=None)

        # Parse known global args and return the rest
        global_args, remaining = parser.parse_known_args(argv)

        # Run every mode against --profile NAME instead of the current one
        set_profile_override(global_args.profile)
        return global_args, remaining

    def _split_by_modes(self, argv: List[str]) -> List[Tuple[Mode, List[str]]]:
        """Split argument list into mode sections.

        Args:
            argv: Arguments to split by mode keywords

        Returns:
            List of (mode_name, mode_args) tuples
        """
        if not argv:
            return []

        mode_sections = []
        current_mode = None
        current_args: list[str] = []

        for arg in argv:
            if arg in AVAILABLE_MODES:
                # Found a new mode
                if current_mode is not None:
                    mode_sections.append(
                        (cast(Mode, current_mode), current_args)
                    )
                current_mode = arg
                current_args = []
            elif arg == "--help" or arg == "-h":
                # Handle help specially
                if current_mode is None:
                    # Global help
                    self._print_help()
                    sys.exit(0)
                else:
                    # Mode-specific help
                    current_args.append(arg)
            else:
                if current_mode is None:
                    raise argparse.ArgumentTypeError(
                        f"Unknown argument '{arg}'. Expected one "
                        f"of: {', '.join(AVAILABLE_MODES)}"
                    )
                current_args.append(arg)

        # Add the last mode section
        if current_mode is not None:
            mode_sections.append((cast(Mode, current_mode), current_args))

        if not mode_sections:
            self._print_help()
            sys.exit(1)

        return mode_sections

    def _parse_mode_args(
        self, mode_name: Mode, mode_argv: List[str]
    ) -> argparse.Namespace:
        """Parse arguments for a specific mode.

        Args:
            mode_name: Name of the mode
            mode_argv: Arguments for this mode

        Returns:
            Parsed arguments namespace
        """
        # Create parser for this mode
        parser = argparse.ArgumentParser(
            prog=f"swift-cli {mode_name}",
            add_help=True,
        )

        # Get the mode module and add its arguments
        mode_module = MODE_MODULES[mode_name]
        mode_module.add_arguments(parser)

        # Special handling for config mode - capture unknown arguments
        if mode_name == "config":
            args, unknown = parser.parse_known_args(mode_argv)
            args.options = unknown
            return args
        else:
            # Standard parsing for other modes
            return parser.parse_args(mode_argv)

    def _merge_global_args(self, mode_args: argparse.Namespace) -> None:
        """Merge global arguments into mode-specific arguments.

        Args:
            mode_args: Mode-specific arguments to update with globals
        """
        # Add global args to mode args if they're not already set
        for attr_name, value in vars(self.global_args).items():
            if (
                not hasattr(mode_args, attr_name)
                or getattr(mode_args, attr_name) is None
            ):
                setattr(mode_args, attr_name, value)

    def _print_help(self) -> None:
        """Print comprehensive help for all modes."""
        print("swift-cli: Utilities for Swift development workflows")
        print()
        print("Usage:")
        print("  swift-cli <mode1> [mode1_args] [<mode2> [mode2_args]] ...")
        print()
        print("Available modes:")

        for mode_name in AVAILABLE_MODES:
            mode_module = MODE_MODULES[mode_name]

            # Create a parser to extract help text
            parser = argparse.ArgumentParser(add_help=False)
            mode_module.add_arguments(parser)

            # Get the first line of the module docstring as description
            desc = (
                mode_module.__doc__.split(".")[0]
                if mode_module.__doc__
                else f"{mode_name.title()} mode"
            )
            print(f"  {mode_name:<15} {desc}")

        print()
        print("Examples:")
        print("  swift-cli config --enable-debug")
        print("  swift-cli make -j 8")
        print("  swift-cli config --enable-debug make -j 32")
        print("  swift-cli update config --enable-debug make -j 8")
        print()
        print("For mode-specific help: swift-cli <mode> --help")


def parse_multimode_args(
    argv: Sequence[str] | None = None,
) -> MultiModeCLIArgs:
    """Parse multi-mode command-line arguments.

    Args:
        argv: Command-line arguments to parse. If None, uses sys.argv.

    Returns:
        MultiModeCLIArgs instance with parsed arguments.
    """
    return MultiModeCLIArgs(argv)
