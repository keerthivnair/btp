from pathlib import Path

import yaml

# Resolved from the code location, not the CWD: Flower loads the app from an installed
# bundle (FAB) whose working directory differs between simulation and deployment.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
