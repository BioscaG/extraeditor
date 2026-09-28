"""Word-level transcription, only on speech segments (§9).

Two rules make ASR usable in an editing pipeline rather than a liability:

1. **Only transcribe what VAD called speech.** Whisper hallucinates confidently on
   music and silence (§28) — it will invent a sentence over a crowd roar and the
   captions will show it. Gating on VAD is the single highest-value guard here.
2. **Drop low-confidence output.** Segments above `no_speech_prob_max` or below
   `min_avg_logprob` are discarded rather than shown, because a wrong caption is
   worse than no caption.

Word timings are what the caption component animates against, so words are emitted
individually with absolute source times.

`mlx-whisper` runs on Apple Silicon's GPU; `faster-whisper` is the fallback. Neither
being present is not an error — it means no captions, and the plan validates fine
without them.
"""

from __future__ import annotations

import logging

from montaje import ffmpeg
from montaje.config import AsrConfig
from montaje.models.asset import Asset
from montaje.models.events import Event

log = logging.getLogger(__name__)

# Repository ids for the MLX-converted Whisper weights, largest first.
MLX_MODELS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
}

# Pad the VAD segment before transcribing: Whisper's first word is often clipped when
# the audio starts exactly on the attack.
SEGMENT_PAD_S = 0.25
# Segments shorter than this rarely contain an intelligible word.
MIN_SEGMENT_S = 0.35


class AsrUnavailable(RuntimeError):
    pass


def _load_backend(model: str):
    """Return `(name, transcribe_callable)` for the best available backend."""
    try:
        import mlx_whisper

        repo = MLX_MODELS.get(model, model)

        def transcribe_mlx(path: str, language: str | None):
            return mlx_whisper.transcribe(
                path,
                path_or_hf_repo=repo,
                word_timestamps=True,
                language=language,
                condition_on_previous_text=False,  # stops one hallucination seeding the next
                verbose=False,
            )

        return "mlx-whisper", transcribe_mlx
    except ImportError:
        pass

    try:
        from faster_whisper import WhisperModel

        engine = WhisperModel(model, compute_type="int8")

        def transcribe_faster(path: str, language: str | None):
            segments, _ = engine.transcribe(
                path, word_timestamps=True, language=language,
                condition_on_previous_text=False,
            )
            return {
                "segments": [
                    {
                        "text": s.text,
                        "no_speech_prob": s.no_speech_prob,
                        "avg_logprob": s.avg_logprob,
                        "words": [
                            {"word": w.word, "start": w.start, "end": w.end,
                             "probability": w.probability}
                            for w in (s.words or [])
                        ],
                    }
                    for s in segments
                ]
            }

        return "faster-whisper", transcribe_faster
    except ImportError as e:
        raise AsrUnavailable(
            "no ASR backend; install mlx-whisper (Apple Silicon) or faster-whisper"
        ) from e


def _extract_segment(src, dest, t0: float, t1: float) -> None:
    ffmpeg.run([
        "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest),
    ])


class AsrAnalyzer:
    """Transcribes speech segments to word events. Depends on `vad` having run."""

    name = "asr"
    version = 1

    def __init__(self, cfg: AsrConfig | None = None, language: str | None = None):
        self.cfg = cfg or AsrConfig()
        self.language = language

    def params(self) -> dict:
        return {
            "model": self.cfg.model,
            "language": self.language,
            "no_speech_prob_max": self.cfg.no_speech_prob_max,
            "min_avg_logprob": self.cfg.min_avg_logprob,
            "pad_s": SEGMENT_PAD_S,
        }

    def run(self, asset: Asset, ws) -> list[Event]:
        from montaje.index.store import Store

        wav = ws.audio_16k_path(asset.asset_id)
        if not wav.exists():
            return []

        with Store(ws.db_path) as store:
            speech = [
                e for e in store.get_events(asset.asset_id, "vad")
                if e.type == "speech" and e.t1 - e.t0 >= MIN_SEGMENT_S
            ]
        if not speech:
            # Nothing VAD called speech: emit nothing rather than transcribing the
            # whole clip, which is exactly how hallucinated captions get in.
            return []

        backend_name, transcribe = _load_backend(self.cfg.model)
        analyzer = f"{self.name}@{self.version}"
        events: list[Event] = []
        tmp = ws.asset_cache(asset.asset_id) / "asr_segment.wav"

        for segment in speech:
            t0 = max(0.0, segment.t0 - SEGMENT_PAD_S)
            t1 = min(asset.duration_s, segment.t1 + SEGMENT_PAD_S)
            _extract_segment(wav, tmp, t0, t1)
            try:
                result = transcribe(str(tmp), self.language)
            except Exception as e:
                log.warning("ASR failed on %s %.2f-%.2f: %s", asset.asset_id, t0, t1, e)
                continue

            for chunk in result.get("segments", []):
                if not _confident(chunk, self.cfg):
                    continue
                text = (chunk.get("text") or "").strip()
                words = chunk.get("words") or []
                if not words:
                    continue
                # Times inside the segment are relative to the extracted clip.
                events.append(Event(
                    asset_id=asset.asset_id, analyzer=analyzer, type="segment",
                    t0=round(t0 + float(words[0]["start"]), 3),
                    t1=round(t0 + float(words[-1]["end"]), 3),
                    score=round(1.0 - float(chunk.get("no_speech_prob", 0.0)), 3),
                    data={"text": text, "backend": backend_name,
                          "avg_logprob": chunk.get("avg_logprob")},
                ))
                for word in words:
                    token = str(word.get("word", "")).strip()
                    if not token:
                        continue
                    events.append(Event(
                        asset_id=asset.asset_id, analyzer=analyzer, type="word",
                        t0=round(t0 + float(word["start"]), 3),
                        t1=round(t0 + float(word["end"]), 3),
                        score=round(float(word.get("probability", 1.0)), 3),
                        data={"text": token},
                    ))
        tmp.unlink(missing_ok=True)
        events.sort(key=lambda e: (e.t0, e.type))
        return events


def _confident(chunk: dict, cfg: AsrConfig) -> bool:
    """Reject output the model itself is unsure about — a wrong caption is worse than none."""
    if float(chunk.get("no_speech_prob", 0.0) or 0.0) > cfg.no_speech_prob_max:
        return False
    logprob = chunk.get("avg_logprob")
    if logprob is not None and float(logprob) < cfg.min_avg_logprob:
        return False
    return True


def transcript_text(events: list[Event]) -> str:
    """Join word events into readable text, for prompts and SRT."""
    return " ".join(
        str(e.data.get("text", "")).strip()
        for e in sorted(events, key=lambda e: e.t0)
        if e.type == "word"
    ).strip()
