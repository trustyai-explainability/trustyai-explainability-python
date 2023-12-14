# pylint: disable=import-error, wrong-import-position, wrong-import-order, duplicate-code, unused-import
"""Language metrics test suite"""

from common import *
from trustyai.metrics.language import word_error_rate
import math

tolerance = 1e-4

REFERENCES = [
    "This is the test reference, to which I will compare alignment against.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur condimentum velit id velit posuere dictum. Fusce euismod tortor massa, nec euismod sapien laoreet non. Donec vulputate mi velit, eu ultricies nibh iaculis vel. Aenean posuere urna nec sapien consectetur, vitae porttitor sapien finibus. Duis nec libero convallis lectus pharetra blandit ut ac odio. Vivamus nec dui quis sem convallis pulvinar. Maecenas sodales sollicitudin leo a faucibus.",
    "The quick red fox jumped over the lazy brown dog"]

INPUTS = [
    "I'm a hypothesis reference, from which the aligner  will compare against.",
    "Lorem ipsum sit amet, consectetur adipiscing elit. Curabitur condimentum velit id velit posuere dictum. Fusce blandit euismod tortor massa, nec euismod sapien blandit laoreet non. Donec vulputate mi velit, eu ultricies nibh iaculis vel. Aenean posuere urna nec sapien consectetur, vitae porttitor sapien finibus. Duis nec libero convallis lectus pharetra blandit ut ac odio. Vivamus nec dui quis sem convallis pulvinar. Maecenas sodales sollicitudin leo a faucibus.",
    "dog brown lazy the over jumped fox red quick The"]


def test_default_tokenizer():
    """Test default tokenizer"""
    results = [4/7, 1/26, 1]
    for i, (reference, hypothesis) in enumerate(zip(REFERENCES, INPUTS)):
        wer = word_error_rate(reference, hypothesis).wer
        assert math.isclose(wer, results[i], rel_tol=tolerance), \
            f"WER for {reference}, {hypothesis} was {wer}, expected ~{results[i]}."


def test_commons_stringtokenizer():
    """Test Apache Commons StringTokenizer"""
    from trustyai.utils.tokenizers import CommonsStringTokenizer
    results = [8 / 12., 3 / 66., 1.0]
    def tokenizer(text: str) -> List[str]:
        return CommonsStringTokenizer(text).getTokenList()
    for i, (reference, hypothesis) in enumerate(zip(REFERENCES, INPUTS)):
        wer = word_error_rate(reference, hypothesis, tokenizer=tokenizer).wer
        assert math.isclose(wer, results[i], rel_tol=tolerance), \
            f"WER for {reference}, {hypothesis} was {wer}, expected ~{results[i]}."

def test_opennlp_tokenizer():
    """Test Apache Commons StringTokenizer"""
    from trustyai.utils.tokenizers import OpenNLPTokenizer
    results = [9 / 14., 3 / 78., 1.0]
    tokenizer = OpenNLPTokenizer()
    for i, (reference, hypothesis) in enumerate(zip(REFERENCES, INPUTS)):
        wer = word_error_rate(reference, hypothesis, tokenizer=tokenizer).wer
        assert math.isclose(wer, results[i], rel_tol=tolerance), \
            f"WER for {reference}, {hypothesis} was {wer}, expected ~{results[i]}."
