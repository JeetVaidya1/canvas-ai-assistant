# exam_generator - Advanced exam generation from past papers and course materials
#
# Facade package: re-exports the exact public API of the former single-module
# `exam_generator.py` so every `import exam_generator` / `from exam_generator
# import X` keeps working identically. The implementation is split across
# focused internal modules for cohesion (schemas, pdf_utils, question_parsing,
# analysis, generation, solving, generator).
from dotenv import load_dotenv

from .schemas import EXAM_DIFFICULTY_SPECS, EXAM_QUESTION_SCHEMA
from .pdf_utils import OCR_OK
from .generator import ExamGenerator

# Preserve the original module-level import-time side effect.
load_dotenv()

__all__ = [
    "ExamGenerator",
    "EXAM_DIFFICULTY_SPECS",
    "EXAM_QUESTION_SCHEMA",
    "OCR_OK",
]
