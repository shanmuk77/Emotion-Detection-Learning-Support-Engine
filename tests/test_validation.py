import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.validation import validate_challenge

def test_validate_empty_input():
    is_valid, msg = validate_challenge("")
    assert not is_valid
    assert msg == "Please enter a learning challenge."

def test_validate_whitespace_only():
    is_valid, msg = validate_challenge("    \n\t  ")
    assert not is_valid
    assert msg == "Please enter a learning challenge."

def test_validate_too_short():
    is_valid, msg = validate_challenge("math")
    assert not is_valid
    assert "at least 5 characters" in msg

def test_validate_too_long():
    is_valid, msg = validate_challenge("a" * 1001)
    assert not is_valid
    assert "too long" in msg

def test_validate_valid_input():
    is_valid, msg = validate_challenge("I am confused about recursion.")
    assert is_valid
    assert msg == ""
