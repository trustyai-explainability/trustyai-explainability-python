""""Group fairness metrics"""
from dataclasses import dataclass
# pylint: disable = import-error
from typing import List, Optional, Any, Union, Callable

from org.kie.trustyai.metrics.language.wer import (
    WordErrorRate as _WordErrorRate,
    WordErrorRateResult as _WordErrorRateResult,
)

from opennlp.tools.tokenize import Tokenizer


@dataclass
class TokenSequenceAlignmentCounters:
    substitutions: int
    insertions: int
    deletions: int
    correct: int


@dataclass
class WordErrorRateResult:
    """Word Error Rate Result"""

    wer: float
    aligned_reference: str
    aligned_input: str
    alignment_counters: TokenSequenceAlignmentCounters

    @staticmethod
    def convert(wer_result: _WordErrorRateResult):
        wer = wer_result.getWordErrorRate()
        aligned_reference = wer_result.getAlignedReferenceString()
        aligned_input = wer_result.getAlignedInputString()
        alignment_counters = wer_result.getAlignmentCounters()
        return WordErrorRateResult(wer=wer,
                                   aligned_reference=aligned_reference,
                                   aligned_input=aligned_input,
                                   alignment_counters=alignment_counters)


def word_error_rate(
        reference: str,
        hypothesis: str,
        tokenizer: Optional[Union[Tokenizer, Callable[[str], List[str]]]] = None,
) -> WordErrorRateResult:
    """Calculate Word Error Rate between reference and hypothesis strings"""
    if not tokenizer:
        _wer = _WordErrorRate()
    elif isinstance(tokenizer, Tokenizer):
        _wer = _WordErrorRate(tokenizer)
    elif callable(tokenizer):
        tokenized_reference = tokenizer(reference)
        tokenized_hypothesis = tokenizer(hypothesis)
        _wer = _WordErrorRate()
        return WordErrorRateResult.convert(_wer.calculate(tokenized_reference, tokenized_hypothesis))
    else:
        raise ValueError("Unsupported tokenizer")
    return WordErrorRateResult.convert(_wer.calculate(reference, hypothesis))
