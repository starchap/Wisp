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
  language_group?: 'multilingual' | 'da' | 'custom' | string;
  is_custom?: boolean;
}

export interface LocalUnmanagedModelInfo {
  folder_name: string;
  custom_whisper_id?: string;
  kind: 'custom_stt' | 'legacy' | string;
  engine?: string;
  size_mb: string;
  local_path: string;
}

export interface ClarificationModelInfo {
  id: string;
  name: string;
  description: string;
  size_mb: string;
  type: string;
  downloaded?: boolean;
  local_path?: string;
  language_group?: string;
  supported_languages?: string;
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

export interface SpeakerDiarizationModelInfo {
  id: string;
  name: string;
  description: string;
  size_mb: string;
  type: string;
  downloaded?: boolean;
  local_path?: string;
}

export interface OpusPairStatusInfo {
  pair: string;
  repo: string;
  size_mb: string;
  downloaded: boolean;
  local_path: string;
}

export interface AudioDevice {
  id: string;
  name: string;
  is_default: boolean;
  type: 'microphone' | 'loopback';
}

export interface SystemSpecsRecommendation {
  cpu_name: string;
  cpu_threads: number;
  total_ram_gb: number;
  avail_ram_gb: number;
  gpu_name?: string;
  gpu_detected?: boolean;
  gpu_provider?: string;
  gpu_acceleration_enabled?: boolean;
  tier: 'high' | 'balanced' | 'light' | string;
  tier_label: string;
  recommended_multilingual_id: string;
  recommended_danish_id: string;
  recommended_clarifier_id: string;
  recommended_whisper_ids: string[];
  summary: string;
}

export interface ModelsStatus {
  storage_directory?: string;
  models_directory: string;
  sessions_directory?: string;
  needs_onboarding?: boolean;
  system_specs?: SystemSpecsRecommendation;
  whisper_models: WhisperModelInfo[];
  clarification_models?: ClarificationModelInfo[];
  translation_models: TranslationModelInfo[];
  opus_pairs_status?: Record<string, OpusPairStatusInfo>;
  local_unmanaged_models?: LocalUnmanagedModelInfo[];
  active_diarization_model?: string;
  speaker_diarization_models?: SpeakerDiarizationModelInfo[];
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
  raw_text?: string;
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
  clarification_model?: string;
  translation_model: string;
  spoken_languages?: string[];
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
  clarification_model?: string;
  translation_model?: string;
  spoken_languages?: string[];
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
  active_speaker_id?: string | null;
}

export interface AppConfig {
  languages: SupportedLanguage[];
  whisper_models: WhisperModelInfo[];
  default_whisper_model: string;
  clarification_models?: ClarificationModelInfo[];
  default_clarification_model?: string;
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
