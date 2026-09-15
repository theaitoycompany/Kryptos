"""Child/parent conversation de-identification.

Two independent privacy problems, deliberately kept separate:

  * content de-identification - names, school, address, teacher, DOB, the
    parent's workplace: this package
  * voice anonymisation - preventing identification from the voice itself:
     a sanitized transcript does NOT make the source audio
    anonymous

Typical use:

    from kryptos import Pipeline, Config
    result = Pipeline(Config.load()).process_text(transcript)
    print(result.sanitized_text, result.status)
"""

from .config import Config, Taxonomy
from .pipeline import Pipeline, process_text
from .types import Detection, Document, QAFinding, Result, Span, Turn, Word

__version__ = "0.1.0"
__all__ = [
    "Config",
    "Taxonomy",
    "Pipeline",
    "process_text",
    "Detection",
    "Document",
    "QAFinding",
    "Result",
    "Span",
    "Turn",
    "Word",
]
