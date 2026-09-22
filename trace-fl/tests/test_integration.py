import os
import subprocess

def test_integration_environment():
    """Ensure check_environment script runs without errors."""
    result = subprocess.run(["python", "scripts/check_environment.py"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Environment is configured correctly." in result.stdout
