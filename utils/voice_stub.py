"""
Voice input stub — simulates voice transcription for demo purposes.
In production, replace transcribe_stub with a real Whisper/SpeechRecognition call.
"""

# Pre-defined demo voice queries mapped by filename keyword or index
_DEMO_QUERIES = [
    "Find me the best wireless headphones under $300",
    "Show me sustainable clothing options",
    "Compare prices for air purifiers",
    "What laptops are good for students?",
    "Recommend eco-friendly home products",
]


def transcribe_stub(audio_path: str = "") -> str:
    """
    Simulate voice-to-text transcription.

    For demo purposes, cycles through pre-defined example queries.
    Pass audio_path as a digit string ("0"-"4") to pick a specific demo query,
    or pass any string to get the default demo query.

    Parameters
    ----------
    audio_path : str
        Path to an audio file (ignored in stub mode) or a digit index.

    Returns
    -------
    str
        A simulated transcript string.
    """
    if audio_path.strip().isdigit():
        idx = int(audio_path.strip()) % len(_DEMO_QUERIES)
        return _DEMO_QUERIES[idx]
    return _DEMO_QUERIES[0]


def get_demo_queries() -> list:
    """Return the list of available demo voice queries."""
    return list(_DEMO_QUERIES)
