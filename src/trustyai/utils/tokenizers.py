""" "Default tokenizers for TrustyAI."""

# pylint: disable = import-error

from org.apache.commons.text import StringTokenizer as _StringTokenizer
from opennlp.tools.tokenize import SimpleTokenizer as _SimpleTokenizer

CommonsStringTokenizer = _StringTokenizer
OpenNLPTokenizer = _SimpleTokenizer
