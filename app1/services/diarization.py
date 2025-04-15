import os
import numpy as np
from sklearn.cluster import KMeans
import torchaudio
from app1.services.sentiment import  ensemble_sentiment
from app1.utils.audio_utils import get_speaker_embedding, whisper_model

def diarize_with_speechbrain(audio_path, num_speakers=2):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    result = whisper_model.transcribe(audio_path)
    segments = result["segments"]
    waveform, sample_rate = torchaudio.load(audio_path)

    embeddings = []
    for seg in segments:
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_audio = waveform[:, start_sample:end_sample]
        emb = get_speaker_embedding(segment_audio, sample_rate)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    labels = KMeans(n_clusters=num_speakers, random_state=0).fit_predict(embeddings)

    diarized_segments = []

    for i, seg in enumerate(segments):
        speaker = f"Speaker_{labels[i]}"
        text = seg["text"]
        combined_sentiment = ensemble_sentiment(text)

        diarized_segments.append({
             "start": seg["start"],
             "end": seg["end"],
             "speaker": speaker,
             "text": text,
             "sentiment": combined_sentiment
               })

    return diarized_segments



