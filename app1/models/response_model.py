from pydantic import BaseModel
from typing import List, Optional

class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    sentiment: Optional[str]

class AudioAnalysisResponse(BaseModel):
    language: str
    transcription: str
    segments: List[Segment]
    overall_sentiment: str
    summary: Optional[str]
    speaker_stats: Optional[dict]


