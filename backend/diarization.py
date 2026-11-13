"""
Local Speaker Diarization and Voice Identification Engine.
Uses Sherpa-ONNX SpeakerEmbeddingExtractor (ERes2Net / 3D-Speaker ONNX model)
with fallback MFCC spectral voiceprint features if the ONNX model hasn't been downloaded yet.
Supports both:
1. Online / Incremental speaker identification (matching each new speech segment against known speakers in the session)
2. Full-file Agglomerative / Cosine clustering with automatic speaker count estimation.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

from models_manager import ensure_speaker_model, is_speaker_model_downloaded

SAMPLE_RATE = 16000


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class LocalSpeakerDiarizer:
    """Extracts voice embeddings and clusters/assigns Speaker IDs (speaker_1, speaker_2, ...)."""

    def __init__(self, similarity_threshold: float = 0.44):
        self.similarity_threshold = similarity_threshold
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is not None:
            return self._extractor
        if not HAS_SHERPA:
            return None
        try:
            model_path = ensure_speaker_model()
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path),
                num_threads=2,
                debug=False,
                provider="cpu",
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            return self._extractor
        except Exception as exc:
            print(f"[Diarizer] Could not load Sherpa-ONNX speaker model, using spectral voiceprint fallback: {exc}")
            return None

    def extract_embedding(self, audio_16k: np.ndarray) -> Optional[np.ndarray]:
        """Extract a normalized 1D speaker voice embedding vector from 16kHz mono float32 audio."""
        if len(audio_16k) < int(SAMPLE_RATE * 0.35):
            # Pad short utterances so embedding extractor has enough frames
            pad_len = int(SAMPLE_RATE * 0.5) - len(audio_16k)
            if pad_len > 0:
                audio_16k = np.pad(audio_16k, (0, pad_len), mode="wrap" if len(audio_16k) > 100 else "constant")

        extractor = self._get_extractor()
        if extractor is not None:
            try:
                stream = extractor.create_stream()
                stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=audio_16k.tolist())
                stream.input_finished()
                if extractor.is_ready(stream):
                    emb = np.array(extractor.compute(stream), dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-8:
                        emb = emb / norm
                    return emb
            except Exception as exc:
                print(f"[Diarizer] ONNX embedding computation warning: {exc}")

        # Fallback: Pitch + Log-Mel Spectral Voiceprint
        return self._extract_spectral_voiceprint(audio_16k)

    def _extract_spectral_voiceprint(self, audio_16k: np.ndarray) -> np.ndarray:
        """Fallback deterministic acoustic voiceprint when ONNX model is unavailable."""
        frame_size = 512
        hop = 256
        if len(audio_16k) < frame_size:
            audio_16k = np.pad(audio_16k, (0, frame_size - len(audio_16k)))
        num_frames = max(1, (len(audio_16k) - frame_size) // hop)
        window = np.hanning(frame_size)
        specs = []
        for i in range(min(num_frames, 200)):
            frame = audio_16k[i * hop : i * hop + frame_size] * window
            mag = np.abs(np.fft.rfft(frame))[:128]
            specs.append(np.log1p(mag))
        mat = np.vstack(specs)
        mean_vec = np.mean(mat, axis=0)
        std_vec = np.std(mat, axis=0)
        emb = np.concatenate([mean_vec, std_vec]).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb /= norm
        return emb

    def assign_speaker_online(
        self,
        audio_16k: np.ndarray,
        session_embeddings: Dict[str, List[float]],
        max_speakers: int = 8,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """
        Match an utterance's audio against existing speakers in `session_embeddings`.
        Updates the centroid for the matched speaker or creates `speaker_N` if it's a new voice.
        """
        emb = self.extract_embedding(audio_16k)
        if emb is None:
            spk_id = "speaker_1"
            return spk_id, session_embeddings

        if not session_embeddings:
            spk_id = "speaker_1"
            session_embeddings[spk_id] = emb.tolist()
            return spk_id, session_embeddings

        best_spk = None
        best_sim = -1.0
        for spk_id, centroid_list in session_embeddings.items():
            centroid = np.array(centroid_list, dtype=np.float32)
            if centroid.shape != emb.shape:
                continue
            sim = _cosine_similarity(emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_spk = spk_id

        if best_spk is not None and (
            best_sim >= self.similarity_threshold or len(session_embeddings) >= max_speakers
        ):
            # Update running centroid (exponential moving average)
            old_centroid = np.array(session_embeddings[best_spk], dtype=np.float32)
            updated = 0.8 * old_centroid + 0.2 * emb
            norm = np.linalg.norm(updated)
            if norm > 1e-8:
                updated /= norm
            session_embeddings[best_spk] = updated.tolist()
            return best_spk, session_embeddings

        # Create a new speaker ID
        next_idx = len(session_embeddings) + 1
        new_spk_id = f"speaker_{next_idx}"
        while new_spk_id in session_embeddings:
            next_idx += 1
            new_spk_id = f"speaker_{next_idx}"

        session_embeddings[new_spk_id] = emb.tolist()
        return new_spk_id, session_embeddings

    def cluster_segments_batch(
        self,
        full_audio_16k: np.ndarray,
        segments_times: List[Tuple[float, float]],
        num_speakers: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Given a full recording and a list of (start_sec, end_sec) segments,
        extract embeddings for all segments and cluster them into speaker_1, speaker_2, etc.
        """
        if not segments_times:
            return [], {}

        embeddings: List[np.ndarray] = []
        valid_indices: List[int] = []

        for idx, (start_s, end_s) in enumerate(segments_times):
            s_idx = max(0, int(start_s * SAMPLE_RATE))
            e_idx = min(len(full_audio_16k), int(end_s * SAMPLE_RATE))
            clip = full_audio_16k[s_idx:e_idx]
            emb = self.extract_embedding(clip)
            if emb is not None:
                embeddings.append(emb)
                valid_indices.append(idx)

        if len(embeddings) <= 1:
            centroids = {"speaker_1": embeddings[0].tolist()} if embeddings else {}
            return ["speaker_1" for _ in segments_times], centroids

        emb_matrix = np.vstack(embeddings)

        # Use AgglomerativeClustering with cosine distance
        try:
            from sklearn.cluster import AgglomerativeClustering

            if num_speakers is not None and num_speakers > 1:
                n_clusters = min(num_speakers, len(embeddings))
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric="cosine",
                    linkage="average",
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1.0 - self.similarity_threshold,
                    metric="cosine",
                    linkage="average",
                )
            raw_labels = clustering.fit_predict(emb_matrix)
        except Exception as exc:
            print(f"[Diarizer] Fallback to online clustering: {exc}")
            assigned = []
            centroids: Dict[str, List[float]] = {}
            for start_s, end_s in segments_times:
                s_idx = max(0, int(start_s * SAMPLE_RATE))
                e_idx = min(len(full_audio_16k), int(end_s * SAMPLE_RATE))
                spk, centroids = self.assign_speaker_online(
                    full_audio_16k[s_idx:e_idx], centroids
                )
                assigned.append(spk)
            return assigned, centroids

        # Remap cluster labels in order of first appearance so first voice is always speaker_1
        label_to_speaker: Dict[int, str] = {}
        centroids_accum: Dict[str, List[np.ndarray]] = {}
        ordered_speakers: List[str] = ["speaker_1"] * len(segments_times)

        for emb_idx, seg_idx in enumerate(valid_indices):
            raw_lbl = int(raw_labels[emb_idx])
            if raw_lbl not in label_to_speaker:
                spk_num = len(label_to_speaker) + 1
                label_to_speaker[raw_lbl] = f"speaker_{spk_num}"
            spk_id = label_to_speaker[raw_lbl]
            ordered_speakers[seg_idx] = spk_id
            centroids_accum.setdefault(spk_id, []).append(emb_matrix[emb_idx])

        centroids_out: Dict[str, List[float]] = {}
        for spk_id, vecs in centroids_accum.items():
            mean_v = np.mean( np.vstack(vecs), axis=0 )
            norm = np.linalg.norm(mean_v)
            if norm > 1e-8:
                mean_v /= norm
            centroids_out[spk_id] = mean_v.tolist()

        return ordered_speakers, centroids_out
