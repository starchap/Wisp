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
    | 'opus'
    | 'nllb'
    | 'all_recommended'
    | 'onboarding',
  whisperId?: string,
  translationId?: string
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/models/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_type: modelType,
      whisper_id: whisperId,
      translation_id: translationId,
    }),
  });
  if (!res.ok) throw new Error('Failed to start model download');
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
  sessionType: 'file' | 'live' = 'file'
): Promise<MeetingSession> {
  const res = await fetch(`${API_BASE}/api/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title,
      session_type: sessionType,
      target_language: targetLanguage,
      whisper_model: whisperModel,
      translation_model: translationModel,
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
    translation_model?: string;
    retranslate_existing?: boolean;
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

export async function uploadAudioRecording(
  sessionId: string,
  file: File,
  numSpeakers?: number
): Promise<void> {
  const form = new FormData();
  form.append('file', file);
  if (numSpeakers && numSpeakers > 0) {
    form.append('num_speakers', String(numSpeakers));
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
