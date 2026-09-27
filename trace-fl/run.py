#!/usr/bin/env python3
"""Build and run TRACE-FL in Docker.

    python run.py                 interactive menu
    python run.py sim             run the simulation
    python run.py test [ARGS]     run pytest (ARGS passed through)
    python run.py shell           open a shell in the container
    python run.py rebuild         rebuild the image from scratch

Standard library only, so any Python 3 on the host can run it.
"""
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "config.yaml"
SERVICE = "trace-fl"

EDITABLE = [
    ("num_clients", int),
    ("num_rounds", int),
    ("epochs", int),
    ("batch_size", int),
    ("learning_rate", float),
    ("seed", int),
]


def compose(*args):
    sys.stdout.flush()
    return subprocess.call(["docker", "compose", *args], cwd=str(ROOT))


def check_docker():
    if shutil.which("docker") is None:
        sys.exit("Docker is not installed. Install Docker Desktop (macOS/Windows) or Docker Engine (Linux).")
    if subprocess.call(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        sys.exit("Docker is installed but not running. Start Docker Desktop and try again.")


def run_sim(config_text=None):
    """Run with config_text (default: config/config.yaml) without modifying the tracked file."""
    if config_text is None:
        config_text = CONFIG.read_text()

    run_dir = ROOT / "experiments" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_dir / "config.yaml"
    run_config.write_text(config_text)
    print("Run config saved to {}".format(run_config.relative_to(ROOT)))

    return compose(
        "run", "--rm", "--build",
        "-v", "{}:/app/config/config.yaml:ro".format(run_config),
        SERVICE,
    )


def run_tests(pytest_args=()):
    return compose("run", "--rm", "--build", SERVICE, "pytest", *pytest_args)


def open_shell():
    return compose("run", "--rm", "--build", SERVICE, "bash")


def rebuild():
    return compose("build", "--no-cache")


def ask(prompt, default=""):
    answer = input(prompt).strip()
    return answer or default


def ask_yes_no(prompt, default=False):
    suffix = " [Y/n] " if default else " [y/N] "
    answer = ask(prompt + suffix).lower()
    return default if not answer else answer.startswith("y")


def _key_pattern(key):
    return re.compile(r"^(\s*" + re.escape(key) + r":\s*)(\S+)", re.MULTILINE)


def read_settings(text):
    settings = {}
    for key, _ in EDITABLE:
        match = _key_pattern(key).search(text)
        if match:
            settings[key] = match.group(2)
    return settings


def edit_config():
    """Prompt for settings; return the config text to run with, or None if unchanged."""
    text = CONFIG.read_text()
    current = read_settings(text)

    print("\nDefault settings (config/config.yaml):")
    for key, value in current.items():
        print("  {:<14} {}".format(key, value))

    if not ask_yes_no("\nChange any settings for this run?"):
        return None

    print("Press Enter to keep a value.")
    changed = False
    for key, cast in EDITABLE:
        if key not in current:
            continue
        while True:
            value = ask("  {} [{}]: ".format(key, current[key]), current[key])
            try:
                if cast(value) <= 0 and key != "seed":
                    raise ValueError
                break
            except ValueError:
                print("    Enter a positive {}.".format(cast.__name__))
        if value != current[key]:
            text = _key_pattern(key).sub(lambda m: m.group(1) + value, text, count=1)
            changed = True

    return text if changed else None


def interactive_sim():
    override = edit_config()
    code = run_sim(override)
    if override is not None and ask_yes_no("\nSave these settings as the new defaults in config/config.yaml?"):
        CONFIG.write_text(override)
        print("Saved config/config.yaml.")
    return code


def menu():
    options = {
        "1": ("Run simulation", interactive_sim),
        "2": ("Run tests", lambda: run_tests(ask("Test file or pytest args (blank = all): ").split())),
        "3": ("Open a shell in the container", open_shell),
        "4": ("Rebuild image from scratch (only if something seems broken)", rebuild),
    }
    print("TRACE-FL (Docker)")
    for key, (label, _) in options.items():
        print("  {}) {}".format(key, label))
    print("  q) Quit")

    choice = ask("\nChoose: ").lower()
    if choice not in options:
        return 0
    return options[choice][1]()


def main():
    commands = {
        "sim": run_sim,
        "test": lambda: run_tests(sys.argv[2:]),
        "shell": open_shell,
        "rebuild": rebuild,
    }
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command is not None and command not in commands:
        sys.exit(__doc__)

    check_docker()
    return commands[command]() if command else menu()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(130)
