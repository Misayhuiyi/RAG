"""Text cleaning utilities for document preprocessing.

Removes extra whitespace, control characters, common header/footer
patterns, and normalizes Chinese/English punctuation spacing.
"""

import re
from typing import List


class TextCleaner:
    """Clean and normalize raw document text before chunking.

    Usage:
        cleaner = TextCleaner()
        cleaned = cleaner.clean(raw_text)
    """

    # Common header/footer regex patterns for Chinese enterprise documents
    HEADER_FOOTER_PATTERNS: List[re.Pattern] = [
        re.compile(r"第\s*\d+\s*页\s*共\s*\d+\s*页"),       # "第 X 页 共 Y 页"
        re.compile(r"^\d+/\d+$", re.MULTILINE),               # "1/10"
        re.compile(r"^[=\-—]{3,}$", re.MULTILINE),            # separator lines
        re.compile(r"^版权所有.*$", re.MULTILINE),             # copyright
        re.compile(r"^机密.*$", re.MULTILINE),                 # confidentiality
        re.compile(r"^\s*[Vv]\d+\.\d+.*$", re.MULTILINE),     # version numbers
    ]

    def __init__(self, remove_headers_footers: bool = True, normalize_punct: bool = True):
        self._remove_hf = remove_headers_footers
        self._normalize_punct = normalize_punct

    def clean(self, text: str) -> str:
        """Apply all cleaning steps to the input text."""
        if not text or not text.strip():
            return ""

        text = self._remove_excess_whitespace(text)
        text = self._remove_control_chars(text)
        if self._remove_hf:
            text = self._strip_headers_footers(text)
        if self._normalize_punct:
            text = self._normalize_punctuation(text)
        return text.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of text strings."""
        return [self.clean(t) for t in texts]

    # ── Internal steps ───────────────────────────────────────────────

    @staticmethod
    def _remove_excess_whitespace(text: str) -> str:
        """Collapse multiple whitespace chars into one, strip blank lines."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ consecutive newlines into exactly 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        """Remove non-printable control characters (keep newlines and tabs)."""
        # Remove all control chars except \n and \t
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        return text

    def _strip_headers_footers(self, text: str) -> str:
        """Apply header/footer removal patterns."""
        for pattern in self.HEADER_FOOTER_PATTERNS:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        """Normalize spacing around Chinese and English punctuation.

        - Add space between Chinese text and English letters/digits.
        - Unify full-width/half-width punctuation.
        """
        # Add space between CJK and Latin/Digit
        text = re.sub(
            r"([一-鿿㐀-䶿])([a-zA-Z0-9])",
            r"\1 \2", text
        )
        text = re.sub(
            r"([a-zA-Z0-9])([一-鿿㐀-䶿])",
            r"\1 \2", text
        )
        # Unify Chinese punctuation
        text = text.replace("，", ",").replace("；", ";")
        # Remove duplicated punctuation at line starts (list artifacts)
        text = re.sub(r"^\s*[,;.]\s*", "", text, flags=re.MULTILINE)
        return text
