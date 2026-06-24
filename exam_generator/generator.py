# exam_generator/generator.py - Composed ExamGenerator facade over the focused mixins
from providers import make_client
from vector_store import VectorStore

from .analysis import AnalysisMixin
from .generation import GenerationMixin
from .pdf_utils import PdfMixin
from .question_parsing import QuestionParsingMixin
from .solving import SolvingMixin


class ExamGenerator(
    AnalysisMixin,
    GenerationMixin,
    QuestionParsingMixin,
    SolvingMixin,
    PdfMixin,
):
    """Generate practice exams from past papers and course materials"""

    def __init__(self):
        self.openai_client = make_client()
        self.vector_store = VectorStore()
