from dataclasses import dataclass
from typing import List


@dataclass
class Notes:

    topic: str

    introduction: str

    key_concepts: List[str]

    detailed_explanation: str

    examples: List[str]

    advantages: List[str]

    disadvantages: List[str]

    important_points: List[str]

    interview_questions: List[str]

    quick_revision: str