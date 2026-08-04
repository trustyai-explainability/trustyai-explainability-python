""" "Default tokenizers for TrustyAI."""

# pylint: disable = import-error

from org.apache.commons.text import StringTokenizer as _StringTokenizer
from opennlp.tools.tokenize import SimpleTokenizer as _SimpleTokenizer

CommonsStringTokenizer = _StringTokenizer


def OpenNLPTokenizer():  # pylint: disable=invalid-name
    """Return the OpenNLP SimpleTokenizer singleton (2.x API)."""
    return _SimpleTokenizer.INSTANCE
