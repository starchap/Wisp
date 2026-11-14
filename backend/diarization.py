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

from models_manager import is_speaker_model_downloaded
import config as app_config

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
        if not HAS_SHERPA or not is_speaker_model_downloaded():
            return None
        try:
            model_path = app_config.SPEAKER_EMBEDDING_MODEL_PATH
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path),
                num_threads=2,
                debug=False,
                provider="cpu",
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            return self._extractor
        except Exception as exc:
            print(f"[Diarizer] Could not load local Sherpa-ONNX speaker model, using spectral voiceprint fallback: {exc}")
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
        max_speakers: int = 6,
    ) -> Tuple[str, Dict[str, List[float]]]:
        """
        Match an utterance's audio against existing speakers in `session_embeddings` (isolated per session).
        Updates the centroid for the matched speaker or creates `speaker_N` if it's a genuinely distinct voice.
        """
        duration_sec = len(audio_16k) / float(SAMPLE_RATE)
        emb = self.extract_embedding(audio_16k)
        if emb is None:
            spk_id = next(iter(session_embeddings.keys()), "speaker_1")
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

        # Short clips (< 1.5s) have noisier embeddings: match existing speaker more generously
        # instead of spawning a new speaker for every brief "Yes" / "Okay"
        effective_threshold = (
            self.similarity_threshold * 0.65
            if duration_sec < 1.5
            else self.similarity_threshold * 0.85
        )

        if best_spk is not None and (
            best_sim >= effective_threshold or len(session_embeddings) >= max_speakers
        ):
            # Only update running centroid if clip is long enough to be a clean voice sample
            if duration_sec >= 1.2:
                old_centroid = np.array(session_embeddings[best_spk], dtype=np.float32)
                updated = 0.82 * old_centroid + 0.18 * emb
                norm = np.linalg.norm(updated)
                if norm > 1e-8:
                    updated /= norm
                session_embeddings[best_spk] = updated.tolist()
            return best_spk, session_embeddings

        # Create a new speaker ID within this session
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
        Given a full recording and a list of (start_sec, end_sec) segments for a single session,
        extract embeddings for all segments and cluster them into isolated per-session
        speaker_1, speaker_2, etc., consolidating noisy short utterances into core speaker centroids.
        """
        if not segments_times:
            return [], {}

        embeddings: List[np.ndarray] = []
        durations: List[float] = []
        valid_indices: List[int] = []

        for idx, (start_s, end_s) in enumerate(segments_times):
            s_idx = max(0, int(start_s * SAMPLE_RATE))
            e_idx = min(len(full_audio_16k), int(end_s * SAMPLE_RATE))
            clip = full_audio_16k[s_idx:e_idx]
            emb = self.extract_embedding(clip)
            if emb is not None:
                embeddings.append(emb)
                durations.append(max(0.1, end_s - start_s))
                valid_indices.append(idx)

        if len(embeddings) <= 1:
            centroids = {"speaker_1": embeddings[0].tolist()} if embeddings else {}
            return ["speaker_1" for _ in segments_times], centroids

        emb_matrix = np.vstack(embeddings)
        dur_array = np.array(durations, dtype=np.float32)

        # Stage 1: Identify anchor segments (>= 1.5s) with reliable voiceprints
        anchor_mask = dur_array >= 1.5
        if np.sum(anchor_mask) < 2:
            anchor_mask = np.ones(len(embeddings), dtype=bool)

        anchor_indices = np.where(anchor_mask)[0]
        anchor_embs = emb_matrix[anchor_indices]

        # Cluster anchor segments using SciPy hierarchical average-linkage cosine clustering
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist

            dists = np.clip(pdist(anchor_embs, metric="cosine"), 0.0, 2.0)
            z_mat = linkage(dists, method="average")
            if num_speakers is not None and num_speakers > 1:
                n_clusters = min(num_speakers, len(anchor_embs))
                anchor_labels = fcluster(z_mat, t=n_clusters, criterion="maxclust")
            else:
                # Cosine distance threshold ~0.64 (similarity ~0.36) prevents over-splitting the same voice
                anchor_labels = fcluster(z_mat, t=0.64, criterion="distance")
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

        # Build initial cluster centroids from anchor segments (weighted by segment duration)
        cluster_vecs: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        for a_idx, lbl in zip(anchor_indices, anchor_labels):
            cluster_vecs.setdefault(int(lbl), []).append(
                (emb_matrix[a_idx], float(dur_array[a_idx]))
            )

        def _compute_centroid(items: List[Tuple[np.ndarray, float]]) -> np.ndarray:
            weights = np.array([w for _, w in items], dtype=np.float32)[:, None]
            vecs = np.vstack([v for v, _ in items])
            c = np.sum(vecs * weights, axis=0)
            norm = np.linalg.norm(c)
            return c / norm if norm > 1e-8 else c

        # Stage 2: Iteratively merge clusters whose centroids are similar (>= 0.40)
        # or absorb tiny 1-segment outlier clusters when auto-detecting speakers
        target_max = num_speakers if (num_speakers and num_speakers > 0) else 6
        while len(cluster_vecs) > 1:
            c_ids = sorted(cluster_vecs.keys())
            centroids_map = {cid: _compute_centroid(cluster_vecs[cid]) for cid in c_ids}

            best_pair = None
            best_sim = -1.0
            for i, c1 in enumerate(c_ids):
                for c2 in c_ids[i + 1 :]:
                    sim = _cosine_similarity(centroids_map[c1], centroids_map[c2])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (c1, c2)

            if best_pair is None:
                break

            # Merge if centroid similarity >= 0.40, or if we have more clusters than target_max
            if best_sim >= 0.40 or len(cluster_vecs) > target_max:
                c_keep, c_merge = best_pair
                cluster_vecs[c_keep].extend(cluster_vecs.pop(c_merge))
                continue

            # If auto-detecting and a cluster has only 1 segment (< 5.0s),
            # absorb the singleton into the closest multi-segment speaker cluster so brief
            # interjections don't spawn extra speakers (Speaker 4..16)
            if not num_speakers:
                singleton_merged = False
                for cid in list(c_ids):
                    items = cluster_vecs[cid]
                    total_dur = sum(w for _, w in items)
                    if len(items) == 1 and total_dur < 5.0 and len(cluster_vecs) > 1:
                        other_ids = [o for o in c_ids if o != cid and len(cluster_vecs[o]) >= 2]
                        if not other_ids:
                            other_ids = [o for o in c_ids if o != cid]
                        if other_ids:
                            best_other = max(
                                other_ids,
                                key=lambda o: _cosine_similarity(centroids_map[cid], centroids_map[o]),
                            )
                            cluster_vecs[best_other].extend(cluster_vecs.pop(cid))
                            singleton_merged = True
                            break
                if singleton_merged:
                    continue
            break

        # Final core centroids
        core_ids = sorted(cluster_vecs.keys())
        core_centroids = {cid: _compute_centroid(cluster_vecs[cid]) for cid in core_ids}

        # Stage 3: Assign every segment to the best matching core speaker centroid
        # (with a small temporal continuity bonus for very short < 1.4s segments)
        raw_assigned: List[int] = []
        prev_cid: Optional[int] = None
        for emb_idx in range(len(embeddings)):
            emb = emb_matrix[emb_idx]
            dur = dur_array[emb_idx]
            best_cid = core_ids[0]
            best_score = -2.0
            for cid in core_ids:
                sim = _cosine_similarity(emb, core_centroids[cid])
                if dur < 1.4 and prev_cid is not None and cid == prev_cid:
                    sim += 0.08
                if sim > best_score:
                    best_score = sim
                    best_cid = cid
            raw_assigned.append(best_cid)
            prev_cid = best_cid

        # Stage 4: Remap cluster labels in order of first appearance (speaker_1, speaker_2, ...)
        label_to_speaker: Dict[int, str] = {}
        centroids_accum: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        ordered_speakers: List[str] = ["speaker_1"] * len(segments_times)

        for emb_idx, seg_idx in enumerate(valid_indices):
            raw_lbl = raw_assigned[emb_idx]
            if raw_lbl not in label_to_speaker:
                spk_num = len(label_to_speaker) + 1
                label_to_speaker[raw_lbl] = f"speaker_{spk_num}"
            spk_id = label_to_speaker[raw_lbl]
            ordered_speakers[seg_idx] = spk_id
            centroids_accum.setdefault(spk_id, []).append(
                (emb_matrix[emb_idx], float(dur_array[emb_idx]))
            )

        centroids_out: Dict[str, List[float]] = {}
        for spk_id, items in centroids_accum.items():
            centroids_out[spk_id] = _compute_centroid(items).tolist()

        return ordered_speakers, centroids_out
