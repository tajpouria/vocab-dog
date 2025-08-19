import os
import sys
import subprocess


def test_main_exits_when_env_vars_missing():
    """Test that main.py exits with code 1 when any required environment variable is missing"""
    required_vars = ["TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"]

    for missing_var in required_vars:
        env = os.environ.copy()
        for var in required_vars:
            env[var] = "test_value"
        env.pop(missing_var, None)

        result = subprocess.run(
            [sys.executable, "main.py"], env=env, capture_output=True
        )
        assert result.returncode == 1
