import pytest
from main import _words_to_paragraphs

def test_words_to_paragraphs_simple():
    words = [
        {"text": "Hello", "top": 10, "x0": 10},
        {"text": "World", "top": 10, "x0": 60},
    ]
    paras = _words_to_paragraphs(words)
    assert len(paras) == 1
    assert paras[0] == "Hello World"

def test_words_to_paragraphs_multi_line():
    words = [
        {"text": "Line1", "top": 10, "x0": 10},
        {"text": "Line2", "top": 20, "x0": 10}, # Gap < 14
    ]
    paras = _words_to_paragraphs(words)
    assert len(paras) == 1
    assert "Line1\nLine2" in paras[0]

def test_words_to_paragraphs_split_para():
    words = [
        {"text": "Para1", "top": 10, "x0": 10},
        {"text": "Para2", "top": 40, "x0": 10}, # Gap > 14
    ]
    paras = _words_to_paragraphs(words)
    assert len(paras) == 2
    assert paras[0] == "Para1"
    assert paras[1] == "Para2"
