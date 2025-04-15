import os
import whisper
import torchaudio
import numpy as np
from sklearn.cluster import KMeans
from app1.services.diarization import diarize_with_speechbrain
from app1.utils.audio_utils import LANGUAGE_MAP, get_speaker_embedding
from app1.models.response_model import AudioAnalysisResponse, Segment
from app1.services.summarization import generate_summary_with_groq
from app1.services.sentiment import ensemble_sentiment
from app1.services.speaker_stats import calculate_speaker_stats


# Load Whisper model
whisper_model = whisper.load_model("base")

async def transcribe_audio_file(file):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    try:
        # Transcribe audio
        result = whisper_model.transcribe(temp_file_path)
        language_code = result["language"]
        language_name = LANGUAGE_MAP.get(language_code, language_code)
        segments = result["segments"]

        # Load audio file
        waveform, sample_rate = torchaudio.load(temp_file_path)

        # Extract embeddings and cluster speakers
        embeddings = []
        for seg in segments:
            start_sample = int(seg['start'] * sample_rate)
            end_sample = int(seg['end'] * sample_rate)
            segment_audio = waveform[:, start_sample:end_sample]
            emb = get_speaker_embedding(segment_audio, sample_rate)
            embeddings.append(emb)

        embeddings = np.vstack(embeddings)
        labels = KMeans(n_clusters=2, random_state=0).fit_predict(embeddings)

        # Add speaker and sentiment to segments
        segments_with_metadata = [
            Segment(
                start=seg["start"],
                end=seg["end"],
                speaker=f"Speaker_{labels[i]}",
                text=seg["text"],
                sentiment=ensemble_sentiment(seg["text"])
            )
            for i, seg in enumerate(segments)
        ]

        # Calculate speaker stats
        speaker_stats = calculate_speaker_stats(segments, waveform, sample_rate, labels, num_speakers=2)

        # Create response using AudioAnalysisResponse model
        response = AudioAnalysisResponse(
            language=language_name,
            transcription=result["text"],
            segments=segments_with_metadata,
            overall_sentiment=ensemble_sentiment(result["text"]),
            speaker_stats=speaker_stats,
            summary=generate_summary_with_groq(result["text"])
        )

        return response.dict()
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


