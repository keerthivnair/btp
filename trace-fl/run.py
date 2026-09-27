#!/usr/bin/env python3
"""Build and run TRACE-FL in Docker.

    python run.py                 interactive menu
    python run.py sim             run the simulation
    python run.py deploy          run with server and clients as separate containers
    python run.py test [ARGS]     run pytest (ARGS passed through)
    python run.py shell           open a shell in the container
    python run.py rebuild         rebuild the image from scratch

Standard library only, so any Python 3 on the host can run it.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "config.yaml"
SERVICE = "trace-fl"
IMAGE = "trace-fl:latest"

DEPLOY_PROJECT = "trace-fl-deploy"
# Rough estimates: each client container loads its own PyTorch + MNIST.
DEPLOY_BASE_GB = 1.5
DEPLOY_GB_PER_CLIENT = 0.6

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


def run_sim(config_text=None, deploy=False):
    """Run with config_text (default: config/config.yaml) without modifying the tracked file."""
    if config_text is None:
        config_text = CONFIG.read_text()

    run_dir = ROOT / "experiments" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_dir / "config.yaml"
    run_config.write_text(config_text)
    print("Run config saved to {}".format(run_config.relative_to(ROOT)))

    if deploy:
        num_clients = int(read_settings(config_text)["num_clients"])
        return run_deployment(run_dir, run_config, num_clients)

    return compose(
        "run", "--rm", "--build",
        "-v", "{}:/app/config/config.yaml:ro".format(run_config),
        SERVICE,
    )


def docker_memory_gb():
    out = subprocess.run(
        ["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True, text=True
    ).stdout.strip()
    return int(out) / 1e9 if out.isdigit() else None


def deployment_fits(num_clients):
    needed = DEPLOY_BASE_GB + DEPLOY_GB_PER_CLIENT * num_clients
    available = docker_memory_gb()
    if available is None:
        return True
    print("Deployment mode: {} client containers need ~{:.1f} GB; Docker has {:.1f} GB.".format(
        num_clients, needed, available))
    if needed <= available * 0.9:
        return True
    print("This may run out of memory. Lower num_clients or give Docker more memory.")
    return not sys.stdin.isatty() or ask_yes_no("Continue anyway?")


def deployment_spec(run_config, flwr_config, num_clients):
    services = {
        "superlink": {
            "image": IMAGE,
            "command": ["flower-superlink", "--insecure", "--host", "0.0.0.0"],
        },
        # Submits the run (`flwr run`) and polls node status; stays idle otherwise.
        "cli": {
            "image": IMAGE,
            "command": ["sleep", "infinity"],
            "init": True,
            "volumes": [
                {"type": "bind", "source": str(run_config),
                 "target": "/app/config/config.yaml", "read_only": True},
                {"type": "bind", "source": str(flwr_config),
                 "target": "/root/.flwr/config.toml"},
            ],
            "depends_on": ["superlink"],
        },
    }
    for i in range(num_clients):
        services["supernode-{}".format(i)] = {
            "image": IMAGE,
            "command": [
                "flower-supernode", "--insecure", "--superlink", "superlink:9092",
                "--node-config", "partition-id={} num-partitions={}".format(i, num_clients),
            ],
            "depends_on": ["superlink"],
        }
    return {"name": DEPLOY_PROJECT, "services": services}


def wait_for_supernodes(compose_file, num_clients, timeout=180):
    cmd = ["docker", "compose", "-f", str(compose_file), "exec", "-T", "cli",
           "flwr", "supernode", "ls", "deploy", "--format", "json"]
    deadline = time.monotonic() + timeout
    online = 0
    while time.monotonic() < deadline:
        out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True).stdout
        online = len(re.findall(r'"status":\s*"online"', out))
        print("\rWaiting for clients: {}/{} online".format(online, num_clients), end="", flush=True)
        if online >= num_clients:
            print()
            return True
        time.sleep(2)
    print("\nOnly {}/{} clients came online within {}s.".format(online, num_clients, timeout))
    return False


def run_deployment(run_dir, run_config, num_clients):
    if not deployment_fits(num_clients):
        return 1

    flwr_config = run_dir / "flwr-config.toml"
    flwr_config.write_text(
        '[superlink]\ndefault = "deploy"\n\n'
        '[superlink.deploy]\naddress = "superlink:8000"\ninsecure = true\n'
    )
    compose_file = run_dir / "docker-compose.deploy.json"
    compose_file.write_text(json.dumps(deployment_spec(run_config, flwr_config, num_clients), indent=2))

    def deploy_compose(*args):
        return compose("-f", str(compose_file), *args)

    if compose("build") != 0:
        return 1
    try:
        if deploy_compose("up", "-d") != 0:
            return 1
        if not wait_for_supernodes(compose_file, num_clients):
            return 1
        return deploy_compose("exec", "cli", "flwr", "run", ".", "deploy", "--stream")
    finally:
        with open(str(run_dir / "containers.log"), "w") as log:
            subprocess.call(["docker", "compose", "-f", str(compose_file), "logs", "--no-color"],
                            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        print("Container logs saved to {}".format((run_dir / "containers.log").relative_to(ROOT)))
        deploy_compose("down", "--remove-orphans")


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
    deploy = ask_yes_no(
        "\nRun server and clients as separate containers (deployment mode, more memory)?")
    code = run_sim(override, deploy=deploy)
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
        "deploy": lambda: run_sim(deploy=True),
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
