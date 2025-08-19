import os
import sys
import subprocess


def test_main_exits_when_no_token():
    """Test that main.py exits with code 1 when TELEGRAM_BOT_TOKEN is not set"""
    env = os.environ.copy()
    env.pop('TELEGRAM_BOT_TOKEN', None)
    
    result = subprocess.run(
        [sys.executable, 'main.py'],
        env=env,
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 1
    assert "Error: TELEGRAM_BOT_TOKEN environment variable is not set" in result.stderr
