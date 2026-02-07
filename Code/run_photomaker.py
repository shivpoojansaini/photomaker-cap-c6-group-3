"""
Simple launcher for the PhotoMaker V2 CLI.

This script imports the `main()` function from `PhotoMaker_Extensions.cli`
and executes it when run directly. It provides a clean entrypoint for
command-line usage or integration with external tools.
"""

from PhotoMaker_Extensions.cli import main

if __name__ == "__main__":
    main()
