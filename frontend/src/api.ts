import {
  AppConfig,
  LiveState,
  MeetingSession,
  ModelsStatus,
  SessionSummary,
} from './types';

export const API_BASE = 'http://127.0.0.1:8765';
export const WS_URL = 'ws://127.0.0.1:8765/ws';

export async function fetchConfig(): Promise<AppConfig> {
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) throw new Error('Failed to load backend config');
  return res.json();
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const res = await fetch(`${API_BASE}/api/models/status`);
  if (!res.ok) throw new Error('Failed to load model status');
  return res.json();
}

export async function updateStoragePath(
  storagePath: string
): Promise<ModelsStatus> {
  const res = await fetch(`${API_BASE}/api/settings/storage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ storage_path: storagePath }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Invalid path' }));
    throw new Error(err.detail || 'Failed to update storage path');
  }
  return res.json();
}

export async function triggerModelDownload(
  modelType:
    | 'whisper'
    | 'speaker'
    | 'clarification'
    | 'opus'
    | 'nllb'
    | 'all_recommended'
    | 'onboarding',
  whisperId?: string,
  translationId?: string,
  clarificationId?: string
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/models/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_type: modelType,
      whisper_id: whisperId,
      clarification_id: clarificationId,
      translation_id: translationId,
    }),
  });
  if (!res.ok) throw new Error('Failed to start model download');
}

export async function deleteLocalModel(
  modelType:
    | 'whisper'
    | 'speaker'
    | 'clarification'
    | 'opus'
    | 'nllb'
    | 'translation',
  modelId?: string
): Promise<ModelsStatus> {
  const res = await fetch(`${API_BASE}/api/models/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_type: modelType,
      model_id: modelId,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Delete failed' }));
    throw new Error(err.detail || 'Failed to delete model');
  }
  const data = await res.json();
  return data.models_status;
}

export async function reprocessSessionAudio(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/reprocess`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Reprocess failed' }));
    throw new Error(err.detail || 'Failed to reprocess session audio');
  }
}

export async function fetchSessions(): Promise<SessionSummary[]> {
  const res = await fetch(`${API_BASE}/api/sessions`);
  if (!res.ok) throw new Error('Failed to list sessions');
  const data = await res.json();
  return data.sessions;
}

export async function createSession(
  title: string,
  targetLanguage: string,
  whisperModel: string,
  translationModel: string = 'opus-mt-da-en',
  sessionType: 'file' | 'live' = 'file',
  clarificationModel: string = 'whisper-context',
  spokenLanguages: string[] = []
): Promise<MeetingSession> {
  const res = await fetch(`${API_BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title,
      session_type: sessionType,
      target_language: targetLanguage,
      whisper_model: whisperModel,
      clarification_model: clarificationModel,
      translation_model: translationModel,
      spoken_languages: spokenLanguages,
    }),
  });
  if (!res.ok) throw new Error('Failed to create session');
  return res.json();
}

export async function fetchSession(sessionId: string): Promise<MeetingSession> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`);
  if (!res.ok) throw new Error('Failed to fetch session');
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/api/sessions/${sessionId}`, { method: 'DELETE' });
}

export async function updateSession(
  sessionId: string,
  patch: {
    title?: string;
    target_language?: string;
    whisper_model?: string;
    clarification_model?: string;
    translation_model?: string;
    spoken_languages?: string[];
    retranslate_existing?: boolean;
    retranscribe_existing?: boolean;
  }
): Promise<MeetingSession> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error('Failed to update session');
  return res.json();
}

export async function updateSpeaker(
  sessionId: string,
  speakerId: string,
  patch: {
    name?: string;
    language?: string;
    color?: string;
    reprocess_segments?: boolean;
  }
): Promise<MeetingSession> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speakers/${speakerId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }
  );
  if (!res.ok) throw new Error('Failed to update speaker');
  return res.json();
}

export async function addSpeaker(
  sessionId: string,
  name: string,
  language: string
): Promise<MeetingSession> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/speakers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, language }),
  });
  if (!res.ok) throw new Error('Failed to add speaker');
  return res.json();
}

export async function mergeSpeakers(
  sessionId: string,
  sourceSpeakerId: string,
  targetSpeakerId: string
): Promise<MeetingSession> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speakers/merge`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_speaker_id: sourceSpeakerId,
        target_speaker_id: targetSpeakerId,
        reprocess_with_target_language: true,
      }),
    }
  );
  if (!res.ok) throw new Error('Failed to merge speakers');
  return res.json();
}

export async function reassignSegmentSpeaker(
  sessionId: string,
  segmentId: string,
  speakerId: string
): Promise<MeetingSession> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/segments/${segmentId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        speaker_id: speakerId,
        reprocess_with_speaker_language: true,
      }),
    }
  );
  if (!res.ok) throw new Error('Failed to reassign segment');
  return res.json();
}

function encodeMonoFloat32ToWavBlob(
  samples: Float32Array,
  sampleRate: number
): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format = 1
  view.setUint16(22, 1, true); // Mono = 1 channel
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // Byte rate
  view.setUint16(32, 2, true); // Block align
  view.setUint16(34, 16, true); // 16-bit
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

async function transcodeContainerAudioIfNeeded(file: File): Promise<File> {
  const lower = file.name.toLowerCase();
  const needsWebAudioDecode =
    lower.endsWith('.m4a') ||
    lower.endsWith('.mp4') ||
    lower.endsWith('.aac') ||
    lower.endsWith('.webm');

  if (!needsWebAudioDecode) {
    return file;
  }

  try {
    const arrayBuf = await file.arrayBuffer();
    const AudioCtx =
      window.AudioContext || (window as any).webkitAudioContext;
    const ctx: AudioContext = new AudioCtx({ sampleRate: 16000 });
    const decoded = await ctx.decodeAudioData(arrayBuf);
    await ctx.close().catch(() => {});

    const length = decoded.length;
    const numChannels = decoded.numberOfChannels;
    const mono = new Float32Array(length);
    for (let ch = 0; ch < numChannels; ch++) {
      const chData = decoded.getChannelData(ch);
      for (let i = 0; i < length; i++) {
        mono[i] += chData[i] / numChannels;
      }
    }

    const wavBlob = encodeMonoFloat32ToWavBlob(mono, decoded.sampleRate);
    return new File(
      [wavBlob],
      file.name.replace(/\.[^/.]+$/, '') + '.wav',
      { type: 'audio/wav' }
    );
  } catch (_) {
    // Fallback to raw file if WebAudio decode fails
    return file;
  }
}

export async function uploadAudioRecording(
  sessionId: string,
  file: File,
  numSpeakers?: number
): Promise<void> {
  const w = typeof window !== 'undefined' ? (window as any) : null;
  let localPath = '';
  try {
    if (w?.wispElectron?.getPathForFile) {
      localPath = w.wispElectron.getPathForFile(file) || '';
    } else if ((file as any)?.path) {
      localPath = (file as any).path || '';
    }
  } catch (_) {}

  const form = new FormData();
  form.append('original_filename', file.name);
  if (numSpeakers && numSpeakers > 0) {
    form.append('num_speakers', String(numSpeakers));
  }

  // In the Electron desktop app, pass the direct local/network file path so huge 2-hour MP4 files
  // are decoded directly from disk via Windows Media Foundation without loading GBs into RAM!
  if (localPath) {
    form.append('local_file_path', localPath);
  } else {
    // Only use in-browser WebAudio transcode for small container files (< 60 MB)
    const preparedFile =
      file.size <= 60 * 1024 * 1024
        ? await transcodeContainerAudioIfNeeded(file)
        : file;
    form.append('file', preparedFile);
  }

  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
}

export async function startLiveCapture(params: {
  session_id: string;
  loopback_id?: string | null;
  mic_id?: string | null;
  chunk_seconds?: number;
}): Promise<LiveState> {
  const res = await fetch(`${API_BASE}/api/live/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('Failed to start live recording');
  return res.json();
}

export async function stopLiveCapture(): Promise<LiveState> {
  const res = await fetch(`${API_BASE}/api/live/stop`, { method: 'POST' });
  if (!res.ok) throw new Error('Failed to stop live recording');
  return res.json();
}

export async function fetchExportText(
  sessionId: string,
  mode: 'translated' | 'dual' | 'original' = 'translated',
  format: 'clean' | 'txt' | 'md' = 'clean'
): Promise<string> {
  const res = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/export?mode=${mode}&format=${format}`
  );
  if (!res.ok) throw new Error('Failed to export transcript');
  return res.text();
}

export function getSessionAudioUrl(sessionId: string, cacheBust?: number): string {
  return `${API_BASE}/api/sessions/${sessionId}/audio${
    cacheBust ? `?t=${cacheBust}` : ''
  }`;
}

export function getSessionAudioDownloadUrl(sessionId: string): string {
  return `${API_BASE}/api/sessions/${sessionId}/audio?download=1`;
}

export async function openSessionAudioFolder(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/sessions/${sessionId}/open-folder`, {
    method: 'POST',
  });
  if (!res.ok) {
    throw new Error('Could not open session folder');
  }
}
