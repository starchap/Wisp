export interface SupportedLanguage {
  code: string;
  name: string;
  nllb: string;
}

export interface WhisperModelInfo {
  id: string;
  name: string;
  description: string;
  size_mb: string;
  downloaded?: boolean;
  local_path?: string;
}

export interface TranslationModelInfo {
  id: string;
  name: string;
  description: string;
  size_mb: string;
  type: string;
  downloaded?: boolean;
  local_path?: string;
}

export interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
  type: 'microphone' | 'loopback';
}

export interface ModelsStatus {
  storage_directory?: string;
  models_directory: string;
  sessions_directory?: string;
  needs_onboarding?: boolean;
  whisper_models: WhisperModelInfo[];
  translation_models: TranslationModelInfo[];
  speaker_diarization: {
    id: string;
    name: string;
    size_mb: string;
    downloaded: boolean;
    local_path: string;
  };
  translator: {
    id: string;
    name: string;
    size_mb: string;
    downloaded: boolean;
    local_path: string;
  };
}

export interface SpeakerProfile {
  id: string;
  name: string;
  language: string; // 'auto', 'da', 'es', 'en', etc.
  color: string;
  segment_count: number;
  total_duration: number;
}

export interface TranscriptSegment {
  id: string;
  speaker_id: string;
  start: number;
  end: number;
  source_language: string;
  original_text: string;
  translated_text: string;
  target_language: string;
  confidence: number;
  is_reprocessing?: boolean;
}

export interface MeetingSession {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  session_type: 'file' | 'live';
  source_file_name?: string;
  target_language: string;
  whisper_model: string;
  translation_model: string;
  duration: number;
  audio_filename: string;
  speakers: Record<string, SpeakerProfile>;
  segments: TranscriptSegment[];
}

export interface SessionSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  session_type: 'file' | 'live';
  source_file_name?: string;
  target_language: string;
  whisper_model?: string;
  translation_model?: string;
  duration: number;
  speaker_count: number;
  segment_count: number;
}

export interface LiveState {
  is_recording: boolean;
  session_id: string | null;
  loopback_id: string | null;
  mic_id: string | null;
  rms_level: number;
}

export interface AppConfig {
  languages: SupportedLanguage[];
  whisper_models: WhisperModelInfo[];
  default_whisper_model: string;
  translation_models: TranslationModelInfo[];
  default_translation_model: string;
  device: string;
  compute_type: string;
  models_status: ModelsStatus;
  audio_devices: {
    microphones: AudioDevice[];
    loopbacks: AudioDevice[];
  };
  live_state: LiveState;
}
