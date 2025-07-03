""" "Language metrics"""

# pylint: disable = import-error
from dataclasses import dataclass

from typing import List, Optional, Union, Callable

from org.kie.trustyai.metrics.language.levenshtein import (
    WordErrorRate as _WordErrorRate,
    ErrorRateResult as _ErrorRateResult,
)
from opennlp.tools.tokenize import Tokenizer
from trustyai import _default_initializer  # pylint: disable=unused-import

from .distance import LevenshteinCounters


@dataclass
class ErrorRateResult:
    """Word Error Rate Result"""

    value: float
    alignment_counters: LevenshteinCounters

    @staticmethod
    def convert(result: _ErrorRateResult):
        """Converts a Java ErrorRateResult to a Python ErrorRateResult"""
        value = result.getValue()
        alignment_counters = result.getAlignmentCounters()
        return ErrorRateResult(
            value=value,
            alignment_counters=alignment_counters,
        )


def word_error_rate(
    reference: str,
    hypothesis: str,
    tokenizer: Optional[Union[Tokenizer, Callable[[str], List[str]]]] = None,
) -> ErrorRateResult:
    """Calculate Word Error Rate between reference and hypothesis strings"""
    if not tokenizer:
        _wer = _WordErrorRate()
    elif isinstance(tokenizer, Tokenizer):
        _wer = _WordErrorRate(tokenizer)
    elif callable(tokenizer):
        tokenized_reference = tokenizer(reference)
        tokenized_hypothesis = tokenizer(hypothesis)
        _wer = _WordErrorRate()
        return ErrorRateResult.convert(
            _wer.calculate(tokenized_reference, tokenized_hypothesis)
        )
    else:
        raise ValueError("Unsupported tokenizer")
    return ErrorRateResult.convert(_wer.calculate(reference, hypothesis))
