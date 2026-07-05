import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.preprocessing import clean_text

def test_lowercase():
    assert clean_text("HELLO") == "hello"

def test_trim_spaces():
    assert clean_text("  hello world  ") == "hello world"

def test_remove_urls():
    assert clean_text("Check this out http://example.com/page") == "check this out"
    assert clean_text("Visit www.google.com for info") == "visit for info"

def test_remove_punctuation():
    assert clean_text("Ohh! This seems fascinating, but now... I am tired.") == "ohh this seems fascinating but now i am tired"

def test_whitespace_normalization():
    assert clean_text("hello    world\n\tthis is test") == "hello world this is test"
