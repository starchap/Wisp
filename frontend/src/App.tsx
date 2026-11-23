import React, { useEffect, useRef, useState } from 'react';
import {
  Clock,
  Download,
  Edit2,
  FileAudio,
  FolderOpen,
  MessageSquare,
  Plus,
  Radio,
  Trash2,
  Users,
} from 'lucide-react';
import {
  WS_URL,
  createSession,
  deleteSession,
  fetchConfig,
  fetchSession,
  fetchSessions,
  getSessionAudioUrl,
  mergeSpeakers,
  reprocessSessionAudio,
  startLiveCapture,
  stopLiveCapture,
  updateSession,
  updateSpeaker,
  uploadAudioRecording,
} from './api';
import { AudioPlayerBar } from './components/AudioPlayerBar';
import { HeaderBar } from './components/HeaderBar';
import { LiveAndUploadBar } from './components/LiveAndUploadBar';
import {
  ExportModal,
  MergeSpeakersModal,
  OnboardingWizardModal,
  SessionDialogModal,
  SettingsModal,
  SettingsTab,
} from './components/Modals';
import { SpeakerPanel } from './components/SpeakerPanel';
import {
  TranscriptFeed,
  TranscriptViewMode,
} from './components/TranscriptFeed';
import {
  I18nContext,
  TRANSLATIONS,
  ThemeMode,
  UILanguage,
} from './i18n';
import {
  AppConfig,
  LiveState,
  MeetingSession,
  ModelsStatus,
  SessionSummary,
  TranscriptSegment,
} from './types';

export function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus | null>(null);
  const [sessionsList, setSessionsList] = useState<SessionSummary[]>([]);
  const [activeSession, setActiveSession] = useState<MeetingSession | null>(
    null
  );
  const activeSessionIdRef = useRef<string | null>(null);
  useEffect(() => {
    activeSessionIdRef.current = activeSession?.id || null;
  }, [activeSession?.id]);
  const [liveState, setLiveState] = useState<LiveState>({
    is_recording: false,
    session_id: null,
    loopback_id: null,
    mic_id: null,
    rms_level: 0,
  });

  const [transcriptViewMode, setTranscriptViewMode] =
    useState<TranscriptViewMode>('translated');
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [seekTarget, setSeekTarget] = useState<{
    time: number;
    endTime?: number;
    segmentId?: string;
    ts: number;
  } | null>(null);
  const [audioCacheBust, setAudioCacheBust] = useState<number>(Date.now());

  const [uploadProgress, setUploadProgress] = useState<{
    message: string;
    percent: number;
  } | null>(null);
  const [reprocessingSpeakers, setReprocessingSpeakers] = useState<
    Record<string, boolean>
  >({});

  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isOnboardingOpen, setIsOnboardingOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('appearance');
  const [uiLanguage, setUiLanguageState] = useState<UILanguage>(
    () => (localStorage.getItem('wisp_ui_lang') as UILanguage) || 'en'
  );
  const [theme, setThemeState] = useState<ThemeMode>(
    () => (localStorage.getItem('wisp_theme') as ThemeMode) || 'system'
  );
  const [defaultOutputLang, setDefaultOutputLang] = useState<string>(
    () => localStorage.getItem('wisp_default_output_lang') || 'en'
  );

  const setUiLanguage = (lang: UILanguage) => {
    setUiLanguageState(lang);
    localStorage.setItem('wisp_ui_lang', lang);
  };

  const setTheme = (nextTheme: ThemeMode) => {
    setThemeState(nextTheme);
    localStorage.setItem('wisp_theme', nextTheme);
  };

  const t = (key: string): string =>
    TRANSLATIONS[uiLanguage]?.[key] || TRANSLATIONS.en[key] || key;

  // Apply Dark / Light / System theme to document root
  useEffect(() => {
    const root = document.documentElement;
    const mq = window.matchMedia('(prefers-color-scheme: light)');

    const applyTheme = () => {
      const isLight =
        theme === 'light' || (theme === 'system' && mq.matches);
      if (isLight) {
        root.classList.add('theme-light');
        root.classList.remove('theme-dark');
      } else {
        root.classList.add('theme-dark');
        root.classList.remove('theme-light');
      }
    };

    applyTheme();
    mq.addEventListener('change', applyTheme);
    return () => mq.removeEventListener('change', applyTheme);
  }, [theme]);
  const [selectedLoopback, setSelectedLoopback] = useState<string>('');
  const [selectedMic, setSelectedMic] = useState<string>('');
  const [modelProgressText, setModelProgressText] = useState<string | null>(
    null
  );
  const [isModelBusy, setIsModelBusy] = useState(false);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [exportSessionTarget, setExportSessionTarget] =
    useState<MeetingSession | null>(null);
  const [mergeModalSourceId, setMergeModalSourceId] = useState<string | null>(
    null
  );
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const [sessionDialog, setSessionDialog] = useState<{
    isOpen: boolean;
    mode: 'create' | 'edit';
    sessionId?: string;
    initialTitle?: string;
    initialSessionType?: 'file' | 'live';
    currentSourceFileName?: string;
    initialWhisperModel?: string;
    initialClarificationModel?: string;
    initialTranslationModel?: string;
    initialTargetLanguage?: string;
    initialSpokenLanguages?: string[];
    hasExistingAudio?: boolean;
  }>({
    isOpen: false,
    mode: 'create',
  });

  // Initial load of config & sessions (with automatic retry while embedded AI engine boots)
  useEffect(() => {
    let mounted = true;
    let retryTimer: any = null;
    let attempts = 0;

    async function init() {
      attempts += 1;
      try {
        const cfg = await fetchConfig();
        if (!mounted) return;
        setErrorBanner(null);
        setConfig(cfg);
        setModelsStatus(cfg.models_status);
        setLiveState(cfg.live_state);
        if (cfg.models_status?.needs_onboarding) {
          setIsOnboardingOpen(true);
        }

        const defLb =
          cfg.audio_devices.loopbacks.find((d) => d.is_default)?.id ||
          cfg.audio_devices.loopbacks[0]?.id ||
          '';
        setSelectedLoopback((prev) => prev || defLb);

        const defMic =
          cfg.audio_devices.microphones.find((d) => d.is_default)?.id ||
          cfg.audio_devices.microphones[0]?.id ||
          '';
        setSelectedMic((prev) => prev || defMic);

        const list = await fetchSessions();
        if (!mounted) return;
        setSessionsList(list);

        if (list.length > 0) {
          const first = await fetchSession(list[0].id);
          if (mounted) setActiveSession((prev) => prev || first);
        } else {
          if (mounted) setActiveSession(null);
        }
      } catch (err: any) {
        if (mounted) {
          if (attempts >= 8) {
            setErrorBanner('Starting embedded local AI engine...');
          }
          retryTimer = setTimeout(init, 800);
        }
      }
    }
    init();
    return () => {
      mounted = false;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, []);

  // Real-time WebSocket subscription
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: any = null;

    function connect() {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => {
        setErrorBanner(null);
      };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'session_updated') {
            const updated: MeetingSession = msg.data;
            setActiveSession((prev) =>
              !prev || prev.id === updated.id ? updated : prev
            );
            setAudioCacheBust(Date.now());
            fetchSessions().then(setSessionsList).catch(() => {});
          } else if (msg.type === 'live_state') {
            setLiveState(msg.data);
          } else if (msg.type === 'live_level') {
            setLiveState((prev) => ({
              ...prev,
              rms_level: msg.data.rms,
              active_speaker_id:
                msg.data.active_speaker_id !== undefined
                  ? msg.data.active_speaker_id
                  : prev.active_speaker_id,
            }));
          } else if (msg.type === 'live_speaker') {
            setLiveState((prev) => ({
              ...prev,
              active_speaker_id: msg.data.speaker_id,
            }));
          } else if (msg.type === 'upload_progress') {
            if (
              msg.data.session_id &&
              activeSessionIdRef.current &&
              msg.data.session_id !== activeSessionIdRef.current
            ) {
              return;
            }
            if (msg.data.error) {
              setErrorBanner(msg.data.error);
              setUploadProgress(null);
            } else if (msg.data.percent >= 100) {
              setTimeout(() => setUploadProgress(null), 800);
            } else {
              setUploadProgress({
                message: msg.data.message,
                percent: msg.data.percent,
              });
            }
          } else if (msg.type === 'speaker_reprocessing') {
            const { session_id, speaker_id, status } = msg.data;
            if (
              session_id &&
              activeSessionIdRef.current &&
              session_id !== activeSessionIdRef.current
            ) {
              return;
            }
            setReprocessingSpeakers((prev) => ({
              ...prev,
              [speaker_id]: status === 'running',
            }));
          } else if (msg.type === 'model_progress') {
            setModelProgressText(msg.data.status);
            setIsModelBusy(!!msg.data.busy);
            if (msg.data.models_status) {
              setModelsStatus(msg.data.models_status);
            }
          }
        } catch (_) {}
      };
      ws.onclose = () => {
        reconnectTimer = setTimeout(connect, 2500);
      };
    }

    connect();
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    };
  }, []);

  const openNewSessionDialog = () => {
    const num = sessionsList.length + 1;
    setSessionDialog({
      isOpen: true,
      mode: 'create',
      initialTitle: `Sales Meeting #${num}`,
      initialSessionType: 'file',
      currentSourceFileName: '',
      initialWhisperModel:
        activeSession?.whisper_model || config?.default_whisper_model || 'small',
      initialClarificationModel:
        activeSession?.clarification_model ||
        config?.default_clarification_model ||
        'whisper-context',
      initialTranslationModel:
        activeSession?.translation_model ||
        config?.default_translation_model ||
        'opus-mt-da-en',
      initialTargetLanguage:
        defaultOutputLang || activeSession?.target_language || 'en',
      initialSpokenLanguages: activeSession?.spoken_languages || ['da', 'en'],
      hasExistingAudio: false,
    });
  };

  const openEditSessionDialog = async (s: SessionSummary | MeetingSession) => {
    const full =
      activeSession && activeSession.id === s.id
        ? activeSession
        : await fetchSession(s.id).catch(() => null);
    setSessionDialog({
      isOpen: true,
      mode: 'edit',
      sessionId: s.id,
      initialTitle: s.title,
      initialSessionType: s.session_type || 'file',
      currentSourceFileName: s.source_file_name || '',
      initialWhisperModel:
        full?.whisper_model ||
        s.whisper_model ||
        config?.default_whisper_model ||
        'small',
      initialClarificationModel:
        full?.clarification_model ||
        s.clarification_model ||
        config?.default_clarification_model ||
        'whisper-context',
      initialTranslationModel:
        full?.translation_model ||
        s.translation_model ||
        config?.default_translation_model ||
        'opus-mt-da-en',
      initialTargetLanguage:
        full?.target_language || s.target_language || defaultOutputLang || 'en',
      initialSpokenLanguages:
        full?.spoken_languages || s.spoken_languages || [],
      hasExistingAudio:
        (full ? full.segments.length > 0 || full.duration > 0 : false) ||
        ('segment_count' in s && (s.segment_count > 0 || s.duration > 0)),
    });
  };

  const handleSaveSessionDialog = async (params: {
    title: string;
    sessionType: 'file' | 'live';
    file: File | null;
    expectedSpeakers?: number;
    whisperModel: string;
    clarificationModel: string;
    translationModel: string;
    targetLanguage: string;
    spokenLanguages: string[];
    retranscribe?: boolean;
  }) => {
    if (sessionDialog.mode === 'create') {
      const created = await createSession(
        params.title,
        params.targetLanguage ||
          defaultOutputLang ||
          activeSession?.target_language ||
          'en',
        params.whisperModel ||
          activeSession?.whisper_model ||
          config?.default_whisper_model ||
          'tiny',
        params.translationModel ||
          activeSession?.translation_model ||
          config?.default_translation_model ||
          'opus-mt-da-en',
        params.sessionType,
        params.clarificationModel || 'whisper-context',
        params.spokenLanguages || []
      );
      setActiveSession(created);
      setSessionsList(await fetchSessions());
      setAudioCacheBust(Date.now());

      if (params.sessionType === 'file' && params.file) {
        setUploadProgress({
          message: `Uploading ${params.file.name}...`,
          percent: 5,
        });
        try {
          await uploadAudioRecording(
            created.id,
            params.file,
            params.expectedSpeakers
          );
        } catch (err: any) {
          setUploadProgress(null);
          setErrorBanner(err.message || 'Upload failed');
        }
      }
    } else if (sessionDialog.mode === 'edit' && sessionDialog.sessionId) {
      const targetId = sessionDialog.sessionId;
      const shouldRetranscribe = !params.file && !!params.retranscribe;
      if (shouldRetranscribe) {
        setUploadProgress({
          message: `Re-transcribing, clarifying & translating with ${params.whisperModel}...`,
          percent: 8,
        });
      }
      const updated = await updateSession(targetId, {
        title: params.title,
        target_language: params.targetLanguage,
        whisper_model: params.whisperModel,
        clarification_model: params.clarificationModel,
        translation_model: params.translationModel,
        spoken_languages: params.spokenLanguages,
        retranscribe_existing: shouldRetranscribe,
      });
      setActiveSession(updated);
      setSessionsList(await fetchSessions());

      if (params.sessionType === 'file' && params.file) {
        setUploadProgress({
          message: `Uploading ${params.file.name}...`,
          percent: 5,
        });
        try {
          await uploadAudioRecording(
            targetId,
            params.file,
            params.expectedSpeakers
          );
        } catch (err: any) {
          setUploadProgress(null);
          setErrorBanner(err.message || 'Upload failed');
        }
      }
    }
  };

  const handleSelectSession = async (sessionId: string) => {
    activeSessionIdRef.current = sessionId;
    setActiveSegmentId(null);
    setReprocessingSpeakers({});
    setUploadProgress(null);
    const loaded = await fetchSession(sessionId);
    if (activeSessionIdRef.current === sessionId) {
      setActiveSession(loaded);
      setAudioCacheBust(Date.now());
    }
  };

  const handleDeleteSession = async (
    e: React.MouseEvent,
    sessionId: string
  ) => {
    e.stopPropagation();
    await deleteSession(sessionId);
    const remaining = await fetchSessions();
    setSessionsList(remaining);
    if (activeSession?.id === sessionId) {
      if (remaining.length > 0) {
        setActiveSession(await fetchSession(remaining[0].id));
      } else {
        activeSessionIdRef.current = null;
        setActiveSession(null);
      }
    }
  };

  const handleDeleteAllSessions = async () => {
    for (const s of sessionsList) {
      await deleteSession(s.id);
    }
    setSessionsList([]);
    activeSessionIdRef.current = null;
    setActiveSession(null);
  };

  const handlePlaySegment = (segment: TranscriptSegment) => {
    setActiveSegmentId(segment.id);
    setSeekTarget({
      time: segment.start,
      endTime: segment.end,
      segmentId: segment.id,
      ts: Date.now(),
    });
  };

  const handlePlaySpeakerSample = (speakerId: string) => {
    if (!activeSession) return;
    const firstSeg = activeSession.segments.find(
      (s) => s.speaker_id === speakerId
    );
    if (firstSeg) {
      handlePlaySegment(firstSeg);
    }
  };

  const audioUrl = activeSession
    ? getSessionAudioUrl(activeSession.id, audioCacheBust)
    : null;

  return (
    <I18nContext.Provider
      value={{ uiLanguage, setUiLanguage, theme, setTheme, t }}
    >
      <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100 overflow-hidden">
        {/* Top Bar */}
        <HeaderBar
          onOpenSettingsModal={(tab = 'appearance') => {
            setSettingsTab(tab);
            setIsSettingsOpen(true);
          }}
        />

        {errorBanner && (
          <div className="bg-rose-500/20 border-b border-rose-500/40 px-6 py-2 text-xs text-rose-200 flex items-center justify-between">
            <span>{errorBanner}</span>
            <button
              onClick={() => setErrorBanner(null)}
              className="text-rose-300 hover:text-white underline"
            >
              {t('dismiss')}
            </button>
          </div>
        )}

        {/* File Processing Progress Bar */}
        <LiveAndUploadBar uploadProgress={uploadProgress} />

        {/* Main 3-Column Workspace: Left Sessions Sidebar | Center Transcript Feed | Right Speaker & Language Panel */}
        <div className="flex-1 flex min-h-0 overflow-hidden">
          {/* Left Sidebar: Saved Recordings / Meetings */}
          <aside className="w-64 border-r border-slate-800/80 bg-slate-900/40 flex flex-col shrink-0 select-none">
            <div className="p-3.5 border-b border-slate-800/80 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                <FolderOpen className="w-4 h-4 text-indigo-400 shrink-0" />
                <span>
                  {t('recordedSessions')} ({sessionsList.length})
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                {sessionsList.length > 0 && (
                  <button
                    type="button"
                    onClick={handleDeleteAllSessions}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/15 transition"
                    title="Delete all sessions"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  type="button"
                  onClick={openNewSessionDialog}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-600/20 transition"
                  title={t('newRecordingSession')}
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>{t('newBtn')}</span>
                </button>
              </div>
            </div>

          <div className="flex-1 overflow-y-auto p-2.5 space-y-1.5">
            {sessionsList.map((s) => {
              const isSelected = activeSession?.id === s.id;
              const isLive = s.session_type === 'live';

              return (
                <div
                  key={s.id}
                  onClick={() => handleSelectSession(s.id)}
                  className={`group rounded-xl p-2.5 cursor-pointer border transition flex items-start justify-between gap-2 ${
                    isSelected
                      ? 'bg-blue-600/15 border-blue-500/40 text-white'
                      : 'bg-slate-900/40 hover:bg-slate-800/70 border-transparent text-slate-300'
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      {isLive ? (
                        <Radio className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                      ) : (
                        <FileAudio className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                      )}
                      <div className="text-xs font-semibold truncate">
                        {s.title}
                      </div>
                    </div>
                    <div className="flex items-center gap-2.5 text-[10px] text-slate-400 mt-1">
                      <span className="flex items-center gap-1">
                        <Users className="w-3 h-3" />
                        {s.speaker_count}
                      </span>
                      <span className="flex items-center gap-1">
                        <MessageSquare className="w-3 h-3" />
                        {s.segment_count}
                      </span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {Math.round(s.duration)}s
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-0.5 shrink-0">
                    <button
                      type="button"
                      onClick={async (e) => {
                        e.stopPropagation();
                        const targetSess =
                          activeSession?.id === s.id
                            ? activeSession
                            : await fetchSession(s.id);
                        setExportSessionTarget(targetSess);
                        setIsExportModalOpen(true);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-emerald-500/20 text-slate-400 hover:text-emerald-400 transition"
                      title="Export session transcript"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        openEditSessionDialog(s);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-slate-700/60 text-slate-400 hover:text-white transition"
                      title="Edit session name or audio file"
                    >
                      <Edit2 className="w-3.5 h-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => handleDeleteSession(e, s.id)}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-rose-500/20 text-slate-500 hover:text-rose-400 transition"
                      title="Delete session"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        {/* Center: Multi-Speaker Transcript & Translation Feed */}
        <main className="flex-1 flex min-w-0 overflow-hidden">
          {activeSession ? (
            <TranscriptFeed
              key={`feed-${activeSession.id}`}
              session={activeSession}
              languages={config?.languages || []}
              activeSegmentId={activeSegmentId}
              viewMode={transcriptViewMode}
              onChangeViewMode={setTranscriptViewMode}
              onPlaySegment={handlePlaySegment}
              onOpenEditSession={() => openEditSessionDialog(activeSession)}
              onReprocessSession={async () => {
                if (!activeSession) return;
                setUploadProgress({
                  message: `Re-transcribing & translating with ${activeSession.whisper_model}...`,
                  percent: 8,
                });
                try {
                  await reprocessSessionAudio(activeSession.id);
                } catch (err: any) {
                  setUploadProgress(null);
                  setErrorBanner(err.message || 'Re-transcribe failed');
                }
              }}
              devices={
                config?.audio_devices || { microphones: [], loopbacks: [] }
              }
              selectedLoopback={selectedLoopback}
              selectedMic={selectedMic}
              onChangeLoopback={setSelectedLoopback}
              onChangeMic={setSelectedMic}
              liveState={liveState}
              onStartLive={async (loopbackId, micId) => {
                if (!activeSession) return;
                const st = await startLiveCapture({
                  session_id: activeSession.id,
                  loopback_id: loopbackId,
                  mic_id: micId,
                  chunk_seconds: 4.0,
                });
                setLiveState(st);
              }}
              onStopLive={async () => {
                const st = await stopLiveCapture();
                setLiveState(st);
                setAudioCacheBust(Date.now());
              }}
            />
          ) : (
            <div
              key="empty-sessions-view"
              className="flex-1 flex flex-col items-center justify-center p-8 text-center bg-slate-950/60"
            >
              <div className="max-w-sm rounded-2xl border border-slate-800 bg-slate-900/40 p-8 space-y-3 shadow-sm">
                <FolderOpen className="w-10 h-10 text-indigo-400 mx-auto" />
                <div className="text-sm font-semibold text-white">
                  {t('recordedSessions')} (0)
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  {t('newSessionSub')}
                </p>
                <button
                  type="button"
                  onClick={openNewSessionDialog}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-600/20 transition"
                >
                  <Plus className="w-4 h-4" />
                  <span>{t('newRecordingSession')}</span>
                </button>
              </div>
            </div>
          )}
        </main>

        {/* Right: Speaker Recognition & Per-Speaker Language Assignment Panel */}
        {activeSession && (
          <SpeakerPanel
            key={`speakers-${activeSession.id}`}
            session={activeSession}
            languages={config?.languages || []}
            reprocessingSpeakers={reprocessingSpeakers}
            activeSpeakerId={
              liveState.is_recording
                ? liveState.active_speaker_id ||
                  (liveState.rms_level > 0.005 &&
                  activeSession.segments.length > 0
                    ? activeSession.segments[activeSession.segments.length - 1]
                        .speaker_id
                    : null)
                : activeSegmentId
                ? activeSession.segments.find((s) => s.id === activeSegmentId)
                    ?.speaker_id || null
                : null
            }
            isLiveSpeaking={
              liveState.is_recording && (liveState.rms_level || 0) > 0.004
            }
            onUpdateSpeaker={async (speakerId, patch) => {
              const updated = await updateSpeaker(
                activeSession.id,
                speakerId,
                patch
              );
              setActiveSession(updated);
            }}
            onOpenMergeModal={(sourceSpeakerId) =>
              setMergeModalSourceId(sourceSpeakerId)
            }
            onPlaySpeakerSample={handlePlaySpeakerSample}
          />
        )}
      </div>

      {/* Bottom Audio Player Bar for Backtracking */}
      <AudioPlayerBar
        session={activeSession}
        audioUrl={audioUrl}
        isLiveRecording={liveState.is_recording}
        seekTarget={seekTarget}
        onActiveSegmentChange={setActiveSegmentId}
      />

      {/* Modals */}
      <SessionDialogModal
        isOpen={sessionDialog.isOpen}
        mode={sessionDialog.mode}
        initialTitle={sessionDialog.initialTitle}
        initialSessionType={sessionDialog.initialSessionType}
        currentSourceFileName={sessionDialog.currentSourceFileName}
        modelsStatus={modelsStatus}
        languages={config?.languages || []}
        initialWhisperModel={sessionDialog.initialWhisperModel}
        initialClarificationModel={sessionDialog.initialClarificationModel}
        initialTranslationModel={sessionDialog.initialTranslationModel}
        initialTargetLanguage={sessionDialog.initialTargetLanguage}
        initialSpokenLanguages={sessionDialog.initialSpokenLanguages}
        hasExistingAudio={sessionDialog.hasExistingAudio}
        onSave={handleSaveSessionDialog}
        onClose={() =>
          setSessionDialog((prev) => ({ ...prev, isOpen: false }))
        }
      />

      <SettingsModal
        isOpen={isSettingsOpen}
        initialTab={settingsTab}
        modelsStatus={modelsStatus}
        languages={config?.languages || []}
        audioDevices={
          config?.audio_devices || { microphones: [], loopbacks: [] }
        }
        defaultOutputLanguage={defaultOutputLang}
        onChangeDefaultOutputLanguage={async (langCode) => {
          setDefaultOutputLang(langCode);
          localStorage.setItem('wisp_default_output_lang', langCode);
          if (activeSession) {
            const updated = await updateSession(activeSession.id, {
              target_language: langCode,
              retranslate_existing: true,
            });
            setActiveSession(updated);
          }
        }}
        selectedLoopback={selectedLoopback}
        selectedMic={selectedMic}
        onChangeLoopback={setSelectedLoopback}
        onChangeMic={setSelectedMic}
        activeWhisperModel={activeSession?.whisper_model}
        activeClarificationModel={activeSession?.clarification_model}
        activeTranslationModel={activeSession?.translation_model}
        progressText={modelProgressText}
        isBusy={isModelBusy}
        onSelectWhisperModel={async (modelId) => {
          if (!activeSession) return;
          const updated = await updateSession(activeSession.id, {
            whisper_model: modelId,
          });
          setActiveSession(updated);
        }}
        onSelectClarificationModel={async (modelId) => {
          if (!activeSession) return;
          const updated = await updateSession(activeSession.id, {
            clarification_model: modelId,
            retranslate_existing: true,
          });
          setActiveSession(updated);
        }}
        onSelectTranslationModel={async (modelId) => {
          if (!activeSession) return;
          const updated = await updateSession(activeSession.id, {
            translation_model: modelId,
            target_language:
              modelId === 'none'
                ? 'none'
                : activeSession.target_language === 'none'
                ? defaultOutputLang || 'en'
                : activeSession.target_language,
            retranslate_existing: true,
          });
          setActiveSession(updated);
        }}
        onModelsStatusUpdated={setModelsStatus}
        onOpenOnboardingWizard={() => setIsOnboardingOpen(true)}
        onClose={() => setIsSettingsOpen(false)}
      />

      <OnboardingWizardModal
        isOpen={isOnboardingOpen}
        modelsStatus={modelsStatus}
        progressText={modelProgressText}
        isBusy={isModelBusy}
        onStorageUpdated={async (newStatus) => {
          setModelsStatus(newStatus);
          const list = await fetchSessions();
          setSessionsList(list);
          if (list.length > 0) {
            setActiveSession(await fetchSession(list[0].id));
          }
        }}
        onComplete={async (whisperId, translationId, clarificationId) => {
          if (activeSession) {
            const updated = await updateSession(activeSession.id, {
              whisper_model: whisperId,
              clarification_model: clarificationId || 'whisper-context',
              translation_model: translationId,
            });
            setActiveSession(updated);
          }
          setIsOnboardingOpen(false);
        }}
      />

      <ExportModal
        isOpen={isExportModalOpen}
        session={exportSessionTarget || activeSession}
        onClose={() => {
          setIsExportModalOpen(false);
          setExportSessionTarget(null);
        }}
      />

      <MergeSpeakersModal
        isOpen={!!mergeModalSourceId}
        session={activeSession}
        initialSourceSpeakerId={mergeModalSourceId}
        onMerge={async (sourceSpeakerId, targetSpeakerId) => {
          if (!activeSession) return;
          const updated = await mergeSpeakers(
            activeSession.id,
            sourceSpeakerId,
            targetSpeakerId
          );
          setActiveSession(updated);
        }}
        onPlaySample={handlePlaySpeakerSample}
        onClose={() => setMergeModalSourceId(null)}
      />
      </div>
    </I18nContext.Provider>
  );
}
export default App;
