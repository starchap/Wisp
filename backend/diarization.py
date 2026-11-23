"""
Local Speaker Diarization and Voice Identification Engine.
Uses Microsoft ONNX Runtime (`onnxruntime.InferenceSession`) directly with the 3D-Speaker
ERes2Net ONNX model (and fallback spectral voiceprint features if the ONNX model hasn't been downloaded yet).
Zero unsigned native C++/Fortran extensions (no `sherpa_onnx` or `scipy` required).
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import onnxruntime as ort

from models_manager import is_speaker_model_downloaded
import config as app_config

SAMPLE_RATE = 16000


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_fbank_matrix_80(sr: int = SAMPLE_RATE, n_fft: int = 512, n_mels: int = 80) -> np.ndarray:
    """Exact Kaldi / kaldifeat 80-bin Mel filterbank (20 Hz to 7600 Hz, Mel-space triangular interpolation)."""
    mel_low = 1127.0 * np.log(1.0 + 20.0 / 700.0)
    mel_high = 1127.0 * np.log(1.0 + 7600.0 / 700.0)
    mels = np.linspace(mel_low, mel_high, n_mels + 2)
    freqs = (float(sr) / float(n_fft)) * np.arange(n_fft // 2 + 1)
    f_mel = 1127.0 * np.log(1.0 + freqs / 700.0)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = mels[m], mels[m + 1], mels[m + 2]
        fb[m] = np.maximum(0.0, np.minimum((f_mel - l) / (c - l), (r - f_mel) / (r - c)))
    fb[:, -1] = 0.0  # Kaldi ignores Nyquist bin
    return fb


_FBANK_80 = _build_fbank_matrix_80()
_POVEY_400 = np.power(0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(400) / 399.0), 0.85).astype(np.float32)


def _extract_kaldi_fbank_80(audio_16k: np.ndarray) -> np.ndarray:
    """
    Compute [1, T, 80] mean-normalized 80-dim Kaldi log-Mel filterbank features for 3D-Speaker ONNX.
    Applies per-frame DC offset removal, 0.97 pre-emphasis, and Povey window to match `kaldifeat` (>0.992 cosine match).
    """
    if len(audio_16k) < 400:
        audio_16k = np.pad(audio_16k, (0, 400 - len(audio_16k)))
    x = np.ascontiguousarray(audio_16k, dtype=np.float32)
    n_frames = max(1, (len(x) - 400) // 160 + 1)
    strides = (x.strides[0] * 160, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=(n_frames, 400), strides=strides).copy()
    frames -= np.mean(frames, axis=1, keepdims=True)
    frames[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
    frames[:, 0] *= 0.03
    frames *= _POVEY_400
    power = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
    fbank = np.log(np.maximum(power @ _FBANK_80.T, 1.1920929e-7)).astype(np.float32)
    fbank -= np.mean(fbank, axis=0, keepdims=True)
    return fbank[np.newaxis, :, :]


def _agglomerative_average_cosine_numpy(
    embs: np.ndarray,
    num_speakers: Optional[int] = None,
    distance_threshold: float = 0.64,
) -> np.ndarray:
    """
    Pure-NumPy average-linkage hierarchical cosine clustering.
    Eliminates `scipy.cluster.hierarchy` dependency completely.
    """
    n = len(embs)
    if n <= 1:
        return np.ones(n, dtype=int)

    # Normalize embeddings so dot product == cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.maximum(norms, 1e-8)
    sim_mat = np.clip(normed @ normed.T, -1.0, 1.0)
    dist_mat = 1.0 - sim_mat

    clusters: List[List[int]] = [[i] for i in range(n)]
    target_k = min(num_speakers, n) if (num_speakers is not None and num_speakers > 1) else 1

    while len(clusters) > target_k:
        best_dist = 999.0
        best_pair = (-1, -1)
        m = len(clusters)
        for i in range(m):
            ci = clusters[i]
            for j in range(i + 1, m):
                cj = clusters[j]
                avg_d = float(np.mean(dist_mat[np.ix_(ci, cj)]))
                if avg_d < best_dist:
                    best_dist = avg_d
                    best_pair = (i, j)

        if best_pair[0] < 0:
            break
        if (num_speakers is None or num_speakers <= 1) and best_dist > distance_threshold:
            break

        i_keep, j_merge = best_pair
        clusters[i_keep].extend(clusters.pop(j_merge))

    labels = np.ones(n, dtype=int)
    for cluster_idx, member_indices in enumerate(clusters, start=1):
        for idx in member_indices:
            labels[idx] = cluster_idx
    return labels


class LocalSpeakerDiarizer:
    """Extracts voice embeddings and clusters/assigns Speaker IDs (speaker_1, speaker_2, ...)."""

    def __init__(self, similarity_threshold: float = 0.44):
        self.similarity_threshold = similarity_threshold
        self._extractor: Optional[ort.InferenceSession] = None

    def _get_extractor(self) -> Optional[ort.InferenceSession]:
        if self._extractor is not None:
            return self._extractor
        if not is_speaker_model_downloaded():
            return None
        try:
            model_path = app_config.SPEAKER_EMBEDDING_MODEL_PATH
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 2
            self._extractor = ort.InferenceSession(
                str(model_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            return self._extractor
        except Exception as exc:
            print(f"[Diarizer] Could not load local ONNX speaker model, using spectral voiceprint fallback: {exc}")
            return None

    def extract_embedding(self, audio_16k: np.ndarray) -> Optional[np.ndarray]:
        """Extract a normalized 1D speaker voice embedding vector from 16kHz mono float32 audio."""
        if len(audio_16k) < int(SAMPLE_RATE * 0.35):
            pad_len = int(SAMPLE_RATE * 0.5) - len(audio_16k)
            if pad_len > 0:
                audio_16k = np.pad(audio_16k, (0, pad_len), mode="wrap" if len(audio_16k) > 100 else "constant")

        extractor = self._get_extractor()
        if extractor is not None:
            try:
                fbank = _extract_kaldi_fbank_80(audio_16k)
                inp_name = extractor.get_inputs()[0].name
                raw_out = extractor.run(None, {inp_name: fbank})[0]
                emb = np.asarray(raw_out, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(emb))
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

    def split_segments_on_speaker_change(
        self,
        full_audio_16k: np.ndarray,
        segments_times: List[Tuple[float, float]],
        win_sec: float = 1.6,
        hop_sec: float = 0.8,
        change_sim_thresh: float = 0.22,
    ) -> List[Tuple[float, float]]:
        """
        Detect intra-segment speaker changes in continuous TV/interview/meeting audio
        where one person finishes speaking and another person starts with <0.4s silence.
        Splits any segment >= 3.6s where the sliding-window 3D-Speaker voice embedding
        shifts abruptly (cosine similarity < `change_sim_thresh`), snapping the cut point
        to the local minimum acoustic energy within ±0.40s.
        """
        if not segments_times or len(full_audio_16k) == 0:
            return segments_times

        total_dur = len(full_audio_16k) / float(SAMPLE_RATE)
        refined: List[Tuple[float, float]] = []

        for s_sec, e_sec in segments_times:
            dur = e_sec - s_sec
            if dur < 3.6:
                refined.append((s_sec, e_sec))
                continue

            times: List[float] = []
            embs: List[np.ndarray] = []
            t = s_sec
            while t + win_sec <= e_sec + 0.05:
                c_start = max(0, int(t * SAMPLE_RATE))
                c_end = min(len(full_audio_16k), int((t + win_sec) * SAMPLE_RATE))
                clip = full_audio_16k[c_start:c_end]
                if len(clip) >= int(SAMPLE_RATE * 0.4) and float(np.sqrt(np.mean(clip**2))) >= 0.008:
                    emb = self.extract_embedding(clip)
                    if emb is not None:
                        times.append(t + win_sec / 2.0)
                        embs.append(emb)
                t += hop_sec

            if len(embs) < 3:
                refined.append((s_sec, e_sec))
                continue

            cur_start = s_sec
            last_split_idx = 0
            for i in range(1, len(embs) - 1):
                if (times[i] - cur_start) < 1.8 or (e_sec - times[i]) < 1.8:
                    continue
                prev_mean = np.mean(embs[max(last_split_idx, i - 2) : i], axis=0)
                next_mean = np.mean(embs[i : min(len(embs), i + 2)], axis=0)
                sim = _cosine_similarity(prev_mean, next_mean)
                if sim < change_sim_thresh:
                    search_s = max(cur_start + 1.2, times[i] - 0.40)
                    search_e = min(e_sec - 1.2, times[i] + 0.40)
                    best_t = round(float(times[i]), 2)
                    min_e = 999.0
                    for cand_t in np.arange(search_s, search_e, 0.05):
                        w_s = max(0, int((cand_t - 0.05) * SAMPLE_RATE))
                        w_e = min(len(full_audio_16k), int((cand_t + 0.05) * SAMPLE_RATE))
                        w = full_audio_16k[w_s:w_e]
                        if len(w) > 0:
                            e_val = float(np.sqrt(np.mean(w**2)))
                            if e_val < min_e:
                                min_e = e_val
                                best_t = round(min(total_dur, max(0.0, float(cand_t))), 2)
                    if (best_t - cur_start) >= 1.0 and (e_sec - best_t) >= 1.0:
                        refined.append((cur_start, best_t))
                        cur_start = best_t
                        last_split_idx = i
            refined.append((cur_start, e_sec))

        return refined

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

        # Cluster anchor segments using pure-NumPy hierarchical average-linkage cosine clustering
        try:
            anchor_labels = _agglomerative_average_cosine_numpy(
                anchor_embs,
                num_speakers=num_speakers,
                distance_threshold=0.64,
            )
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
