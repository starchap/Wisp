import React, { useEffect, useState } from 'react';
import {
  ArrowRight,
  Check,
  CheckCircle2,
  Copy,
  Cpu,
  Download,
  FileAudio,
  FolderOpen,
  GitMerge,
  Globe,
  Headphones,
  Info,
  Languages,
  Loader2,
  Mic,
  Monitor,
  Moon,
  Palette,
  Radio,
  RefreshCw,
  Shield,
  Sparkles,
  Sun,
  Trash2,
  Upload,
  Users,
  Volume2,
  X,
} from 'lucide-react';
import {
  deleteLocalModel,
  fetchExportText,
  triggerModelDownload,
  updateStoragePath,
} from '../api';
import { ThemeMode, UILanguage, UI_LANGUAGES, useI18n } from '../i18n';
import {
  AudioDevice,
  MeetingSession,
  ModelsStatus,
  SpeakerProfile,
  SupportedLanguage,
} from '../types';

export type SettingsTab =
  | 'appearance'
  | 'models'
  | 'languages'
  | 'audio'
  | 'about';

interface SettingsModalProps {
  isOpen: boolean;
  initialTab?: SettingsTab;
  modelsStatus: ModelsStatus | null;
  languages: SupportedLanguage[];
  audioDevices: {
    microphones: AudioDevice[];
    loopbacks: AudioDevice[];
  };
  defaultOutputLanguage: string;
  onChangeDefaultOutputLanguage: (langCode: string) => void;
  selectedLoopback: string;
  selectedMic: string;
  onChangeLoopback: (id: string) => void;
  onChangeMic: (id: string) => void;
  activeWhisperModel?: string;
  activeTranslationModel?: string;
  progressText: string | null;
  isBusy: boolean;
  onSelectWhisperModel?: (modelId: string) => void;
  onSelectTranslationModel?: (modelId: string) => void;
  onModelsStatusUpdated?: (status: ModelsStatus) => void;
  onOpenOnboardingWizard?: () => void;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  initialTab = 'models',
  modelsStatus,
  languages,
  audioDevices,
  defaultOutputLanguage,
  onChangeDefaultOutputLanguage,
  selectedLoopback,
  selectedMic,
  onChangeLoopback,
  onChangeMic,
  activeWhisperModel,
  activeTranslationModel,
  progressText,
  isBusy,
  onSelectWhisperModel,
  onSelectTranslationModel,
  onModelsStatusUpdated,
  onOpenOnboardingWizard,
  onClose,
}) => {
  const { t, uiLanguage, setUiLanguage, theme, setTheme } = useI18n();
  const [activeTab, setActiveTab] = useState<SettingsTab>(initialTab);

  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab);
    }
  }, [isOpen, initialTab]);

  if (!isOpen || !modelsStatus) return null;

  const outputLanguages = languages.filter((l) => l.code !== 'auto');

  const navItems: {
    id: SettingsTab;
    label: string;
    icon: React.ReactNode;
    badgeColor: string;
  }[] = [
    {
      id: 'appearance',
      label: t('tabAppearance'),
      icon: <Palette className="w-4 h-4 text-white" />,
      badgeColor: 'bg-purple-600',
    },
    {
      id: 'models',
      label: t('tabModels'),
      icon: <Cpu className="w-4 h-4 text-white" />,
      badgeColor: 'bg-blue-600',
    },
    {
      id: 'languages',
      label: t('tabLanguages'),
      icon: <Globe className="w-4 h-4 text-white" />,
      badgeColor: 'bg-emerald-600',
    },
    {
      id: 'audio',
      label: t('tabAudio'),
      icon: <Headphones className="w-4 h-4 text-white" />,
      badgeColor: 'bg-rose-600',
    },
    {
      id: 'about',
      label: t('tabAbout'),
      icon: <Info className="w-4 h-4 text-white" />,
      badgeColor: 'bg-indigo-600',
    },
  ];

  const renderUiAndThemeSection = () => (
    <div className="rounded-2xl bg-slate-950/60 border border-slate-800 divide-y divide-slate-800/80">
      {/* 1. Interface Language (UI) */}
      <div className="p-4 flex items-center justify-between gap-4">
        <div>
          <div className="text-xs font-bold text-white">
            {t('uiLanguageTitle')}
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {t('uiLanguageDesc')}
          </div>
        </div>
        <select
          value={uiLanguage}
          onChange={(e) => setUiLanguage(e.target.value as UILanguage)}
          className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500"
        >
          {UI_LANGUAGES.map((ul) => (
            <option key={ul.code} value={ul.code}>
              {ul.name}
            </option>
          ))}
        </select>
      </div>

      {/* 2. Appearance Theme Picker (Dark / Light / System Match) */}
      <div className="p-4 space-y-3">
        <div>
          <div className="text-xs font-bold text-white">{t('themeTitle')}</div>
          <div className="text-[11px] text-slate-400 mt-0.5">
            {t('themeDesc')}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3">
          {(
            [
              {
                id: 'dark' as ThemeMode,
                label: t('themeDark'),
                icon: <Moon className="w-4 h-4" />,
              },
              {
                id: 'light' as ThemeMode,
                label: t('themeLight'),
                icon: <Sun className="w-4 h-4" />,
              },
              {
                id: 'system' as ThemeMode,
                label: t('themeSystem'),
                icon: <Monitor className="w-4 h-4" />,
              },
            ] as const
          ).map((opt) => {
            const isSelected = theme === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => setTheme(opt.id)}
                className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl border text-xs font-semibold transition ${
                  isSelected
                    ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-600/20'
                    : 'bg-slate-900 border-slate-700/80 text-slate-300 hover:border-slate-600'
                }`}
              >
                {opt.icon}
                <span>{opt.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-md flex items-center justify-center p-4 select-none">
      {/* macOS System Settings Window Frame */}
      <div className="bg-slate-900/95 border border-slate-700/80 rounded-2xl max-w-6xl w-full h-[82vh] max-h-[760px] shadow-2xl overflow-hidden flex">
        {/* Left macOS-Style Sidebar */}
        <aside className="w-60 bg-slate-950/80 border-r border-slate-800/90 flex flex-col justify-between p-3.5 shrink-0">
          <div className="space-y-4">
            {/* macOS Window Traffic Lights + Title */}
            <div className="flex items-center gap-3 px-2 pt-1">
              <div className="flex items-center gap-1.5">
                <button
                  onClick={onClose}
                  className="w-3 h-3 rounded-full bg-rose-500 hover:bg-rose-400 transition"
                  title="Close Settings"
                />
                <span className="w-3 h-3 rounded-full bg-amber-500/80" />
                <span className="w-3 h-3 rounded-full bg-emerald-500/80" />
              </div>
              <span className="text-xs font-bold text-slate-300 tracking-tight">
                {t('systemSettings')}
              </span>
            </div>

            {/* App Mini Profile Card */}
            <div className="flex items-center gap-2.5 px-2.5 py-2 rounded-xl bg-slate-900/90 border border-slate-800/80">
              <img
                src="./wisp-icon.png"
                alt="Wisp"
                className="w-8 h-8 rounded-lg object-cover shrink-0"
              />
              <div className="min-w-0">
                <div className="text-xs font-bold text-white truncate">
                  Wisp Desktop
                </div>
                <div className="text-[10px] text-emerald-400 font-medium">
                  {t('offlineAiEngine')}
                </div>
              </div>
            </div>

            {/* Navigation Items */}
            <nav className="space-y-1">
              {navItems.map((item) => {
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setActiveTab(item.id)}
                    className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-xl text-xs font-semibold transition ${
                      isActive
                        ? 'bg-blue-600 text-white shadow-md shadow-blue-600/20'
                        : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                    }`}
                  >
                    <span
                      className={`w-6 h-6 rounded-lg flex items-center justify-center shadow-sm shrink-0 ${
                        isActive ? 'bg-white/20' : item.badgeColor
                      }`}
                    >
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>

          <div className="px-2 py-1.5 text-[10px] text-slate-500 border-t border-slate-800/80 pt-3 flex items-center justify-between">
            <span>Wisp v1.0.0</span>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-white font-medium"
            >
              {t('done')}
            </button>
          </div>
        </aside>

        {/* Right Content Pane */}
        <div className="flex-1 flex flex-col min-w-0 bg-slate-900/70 overflow-hidden">
          {/* Top Bar of Active Settings Pane */}
          <div className="h-14 px-6 border-b border-slate-800/80 flex items-center justify-between shrink-0">
            <h3 className="text-sm font-bold text-white">
              {navItems.find((n) => n.id === activeTab)?.label}
            </h3>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Scrollable Settings Body */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* TAB 0: APPEARANCE & UI LANGUAGE */}
            {activeTab === 'appearance' && (
              <div className="max-w-2xl space-y-6">
                {renderUiAndThemeSection()}
              </div>
            )}

            {/* TAB 1: LOCAL AI MODELS (3-Column Pipeline Layout) */}
            {activeTab === 'models' && (
              <div className="space-y-5">
                {/* Overarching Top Banner: Local Storage Path + Setup Wizard + Download All Recommended */}
                <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/25 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5 text-xs text-emerald-300">
                    <Shield className="w-4 h-4 shrink-0 text-emerald-400" />
                    <span>
                      <strong>{t('localStoragePath')}</strong>{' '}
                      <code className="text-emerald-200 font-mono bg-slate-950/60 px-2 py-0.5 rounded border border-emerald-500/20">
                        {modelsStatus.models_directory}
                      </code>
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    {onOpenOnboardingWizard && (
                      <button
                        type="button"
                        onClick={() => {
                          onClose();
                          onOpenOnboardingWizard();
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold shrink-0 transition"
                      >
                        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                        <span>{t('runSetupWizard')}</span>
                      </button>
                    )}
                    <button
                      disabled={isBusy}
                      onClick={() =>
                        triggerModelDownload('all_recommended', 'small')
                      }
                      className="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold shrink-0 transition shadow-md shadow-emerald-600/20"
                    >
                      {t('downloadAllRecommended')}
                    </button>
                  </div>
                </div>

                {progressText && (
                  <div className="p-3 rounded-xl bg-blue-500/15 border border-blue-500/30 flex items-center gap-2.5 text-xs text-blue-200">
                    {isBusy ? (
                      <Loader2 className="w-4 h-4 animate-spin text-blue-400 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    )}
                    <span>{progressText}</span>
                  </div>
                )}

                {/* 3-Column Left-to-Right Pipeline Layout */}
                <div className="grid grid-cols-1 lg:grid-cols-11 gap-4 items-stretch">
                  {/* COLUMN 1: Step 1 — Audio In & Speakers */}
                  <div className="lg:col-span-3 rounded-2xl bg-slate-950/60 border border-slate-800 p-4 flex flex-col justify-between">
                    <div className="space-y-3">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                        <Users className="w-3.5 h-3.5" />
                        Step 1: Audio In & Speakers
                      </span>

                      <p className="text-xs text-slate-400 leading-relaxed">
                        Captures MP3 recordings or live Teams/Huddle audio,
                        splits speech segments, and identifies who is speaking (
                        <code className="text-slate-200">Speaker 1</code>,{' '}
                        <code className="text-slate-200">Speaker 2</code>, etc.).
                      </p>

                      <div className="p-3.5 rounded-xl bg-slate-900 border border-indigo-500/30 space-y-2.5">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="text-xs font-bold text-white">
                              {modelsStatus.speaker_diarization.name}
                            </div>
                            <div className="text-[11px] text-indigo-300 mt-0.5">
                              Active Diarization Model
                            </div>
                          </div>
                          <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded shrink-0">
                            ~{modelsStatus.speaker_diarization.size_mb} MB
                          </span>
                        </div>

                        <p className="text-[11px] text-slate-400 leading-relaxed">
                          Extracts acoustic voice embeddings locally via
                          Sherpa-ONNX to cluster distinct voices.
                        </p>

                        <div className="pt-1 flex items-center justify-between gap-2">
                          {modelsStatus.speaker_diarization.downloaded ? (
                            <div className="flex items-center justify-between w-full gap-2">
                              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-emerald-500/15 text-emerald-400 text-xs font-semibold border border-emerald-500/30">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                {t('readyOffline')}
                              </span>
                              <button
                                type="button"
                                disabled={isBusy}
                                title={t('deleteModelTooltip')}
                                onClick={async (e) => {
                                  e.stopPropagation();
                                  try {
                                    const next =
                                      await deleteLocalModel('speaker');
                                    onModelsStatusUpdated?.(next);
                                  } catch (_) {}
                                }}
                                className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-800/80 hover:bg-rose-600/20 text-slate-400 hover:text-rose-400 border border-slate-700/80 hover:border-rose-500/40 text-[11px] font-semibold disabled:opacity-40 transition"
                              >
                                <Trash2 className="w-3 h-3" />
                                <span>{t('deleteModel')}</span>
                              </button>
                            </div>
                          ) : (
                            <button
                              disabled={isBusy}
                              onClick={() => triggerModelDownload('speaker')}
                              className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold"
                            >
                              {t('download')}
                            </button>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
                      <span>Output: Speaker ID + Audio Slice</span>
                      <ArrowRight className="w-4 h-4 text-indigo-400" />
                    </div>
                  </div>

                  {/* COLUMN 2: Step 2 — Speech-to-Text Models */}
                  <div className="lg:col-span-4 rounded-2xl bg-slate-950/60 border border-slate-800 p-4 flex flex-col justify-between">
                    <div className="space-y-3">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-blue-500/15 text-blue-300 border border-blue-500/30">
                        <Cpu className="w-3.5 h-3.5" />
                        Step 2: Speech-to-Text Models
                      </span>

                      <p className="text-xs text-slate-400 leading-relaxed">
                        Transcribes each speaker&apos;s audio in their spoken
                        language. Click a downloaded model to make it active.
                      </p>

                      <div className="space-y-2">
                        {modelsStatus.whisper_models.map((wm) => {
                          const isSelected = activeWhisperModel === wm.id;
                          return (
                            <div
                              key={wm.id}
                              onClick={() => {
                                if (wm.downloaded && onSelectWhisperModel) {
                                  onSelectWhisperModel(wm.id);
                                }
                              }}
                              className={`p-3 rounded-xl border transition ${
                                isSelected
                                  ? 'bg-blue-950/30 border-blue-500/60 shadow-md shadow-blue-500/10'
                                  : 'bg-slate-900/90 border-slate-800 hover:border-slate-700'
                              } ${wm.downloaded ? 'cursor-pointer' : ''}`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <div className="flex items-center gap-2 min-w-0">
                                  <span className="text-xs font-bold text-white truncate">
                                    {wm.name}
                                  </span>
                                  {isSelected && (
                                    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-600 text-white shrink-0">
                                      ACTIVE
                                    </span>
                                  )}
                                </div>

                                {wm.downloaded ? (
                                  <div className="flex items-center gap-1.5 shrink-0">
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-400 text-[11px] font-semibold border border-emerald-500/30">
                                      <CheckCircle2 className="w-3 h-3" />
                                      {t('downloaded')}
                                    </span>
                                    <button
                                      type="button"
                                      disabled={isBusy}
                                      title={t('deleteModelTooltip')}
                                      onClick={async (e) => {
                                        e.stopPropagation();
                                        try {
                                          const next = await deleteLocalModel(
                                            'whisper',
                                            wm.id
                                          );
                                          onModelsStatusUpdated?.(next);
                                          if (
                                            isSelected &&
                                            onSelectWhisperModel
                                          ) {
                                            const fallback =
                                              next.whisper_models.find(
                                                (w) =>
                                                  w.downloaded && w.id !== wm.id
                                              );
                                            if (fallback) {
                                              onSelectWhisperModel(fallback.id);
                                            }
                                          }
                                        } catch (_) {}
                                      }}
                                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-800/80 hover:bg-rose-600/20 text-slate-400 hover:text-rose-400 border border-slate-700/80 hover:border-rose-500/40 text-[11px] font-semibold disabled:opacity-40 transition"
                                    >
                                      <Trash2 className="w-3 h-3" />
                                      <span>{t('deleteModel')}</span>
                                    </button>
                                  </div>
                                ) : (
                                  <button
                                    disabled={isBusy}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      triggerModelDownload('whisper', wm.id);
                                    }}
                                    className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-blue-600 disabled:opacity-50 text-white text-[11px] font-semibold border border-slate-700 shrink-0 transition"
                                  >
                                    {t('download')} (~{wm.size_mb} MB)
                                  </button>
                                )}
                              </div>
                              <p className="text-[11px] text-slate-400 mt-1 leading-snug">
                                {wm.description}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
                      <span>Output: Spoken Language Text</span>
                      <ArrowRight className="w-4 h-4 text-blue-400" />
                    </div>
                  </div>

                  {/* COLUMN 3: Step 3 — Local AI Translation Models */}
                  <div className="lg:col-span-4 rounded-2xl bg-slate-950/60 border border-slate-800 p-4 flex flex-col justify-between">
                    <div className="space-y-3">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold uppercase tracking-wider bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                        <Languages className="w-3.5 h-3.5" />
                        Step 3: Local AI Translation Models
                      </span>

                      <p className="text-xs text-slate-400 leading-relaxed">
                        Translates each speaker&apos;s transcribed text into
                        your single chosen Output Language.
                      </p>

                      <div className="space-y-2.5">
                        {(modelsStatus.translation_models || []).map((tm) => {
                          const isSelected = activeTranslationModel === tm.id;
                          return (
                            <div
                              key={tm.id}
                              onClick={() => {
                                if (
                                  tm.downloaded &&
                                  onSelectTranslationModel
                                ) {
                                  onSelectTranslationModel(tm.id);
                                }
                              }}
                              className={`p-3.5 rounded-xl border transition ${
                                isSelected
                                  ? 'bg-emerald-950/25 border-emerald-500/60 shadow-md shadow-emerald-500/10'
                                  : 'bg-slate-900/90 border-slate-800 hover:border-slate-700'
                              } ${tm.downloaded ? 'cursor-pointer' : ''}`}
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className="text-xs font-bold text-white">
                                    {tm.name}
                                  </span>
                                  {isSelected && (
                                    <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-600 text-white shrink-0">
                                      ACTIVE
                                    </span>
                                  )}
                                </div>

                                {tm.downloaded ? (
                                  <div className="flex items-center gap-1.5 shrink-0">
                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/15 text-emerald-400 text-[11px] font-semibold border border-emerald-500/30">
                                      <CheckCircle2 className="w-3 h-3" />
                                      {t('readyOffline')}
                                    </span>
                                    {tm.id !== 'whisper-native' && (
                                      <button
                                        type="button"
                                        disabled={isBusy}
                                        title={t('deleteModelTooltip')}
                                        onClick={async (e) => {
                                          e.stopPropagation();
                                          try {
                                            const next = await deleteLocalModel(
                                              'translation',
                                              tm.id
                                            );
                                            onModelsStatusUpdated?.(next);
                                            if (
                                              isSelected &&
                                              onSelectTranslationModel
                                            ) {
                                              const fallback = (
                                                next.translation_models || []
                                              ).find(
                                                (m) =>
                                                  m.downloaded && m.id !== tm.id
                                              );
                                              if (fallback) {
                                                onSelectTranslationModel(
                                                  fallback.id
                                                );
                                              }
                                            }
                                          } catch (_) {}
                                        }}
                                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-slate-800/80 hover:bg-rose-600/20 text-slate-400 hover:text-rose-400 border border-slate-700/80 hover:border-rose-500/40 text-[11px] font-semibold disabled:opacity-40 transition"
                                      >
                                        <Trash2 className="w-3 h-3" />
                                        <span>{t('deleteModel')}</span>
                                      </button>
                                    )}
                                  </div>
                                ) : (
                                  <button
                                    disabled={isBusy}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      triggerModelDownload(
                                        tm.id === 'opus-mt-da-en'
                                          ? 'opus'
                                          : 'nllb'
                                      );
                                    }}
                                    className="px-2.5 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-[11px] font-semibold shrink-0 transition"
                                  >
                                    {t('download')} (~{tm.size_mb} MB)
                                  </button>
                                )}
                              </div>
                              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                                {tm.description}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800/80 text-[11px] text-emerald-400 font-medium flex items-center justify-between">
                      <span>Final Output: Plain Text Ready for Word</span>
                      <CheckCircle2 className="w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: LANGUAGE & OUTPUT DEFAULTS (macOS grouped rows style) */}
            {activeTab === 'languages' && (
              <div className="max-w-2xl space-y-6">
                <div className="rounded-2xl bg-slate-950/60 border border-slate-800 divide-y divide-slate-800/80">
                  <div className="p-4 flex items-center justify-between gap-4">
                    <div>
                      <div className="text-xs font-bold text-white">
                        {t('defaultOutputLangTitle')}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {t('defaultOutputLangDesc')}
                      </div>
                    </div>
                    <select
                      value={defaultOutputLanguage}
                      onChange={(e) =>
                        onChangeDefaultOutputLanguage(e.target.value)
                      }
                      className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500"
                    >
                      {outputLanguages.map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.name} ({lang.code.toUpperCase()})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="p-4 flex items-center justify-between gap-4">
                    <div>
                      <div className="text-xs font-bold text-white">
                        {t('defaultSttTitle')}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {t('defaultSttDesc')}
                      </div>
                    </div>
                    <select
                      value={activeWhisperModel || 'tiny'}
                      onChange={(e) =>
                        onSelectWhisperModel &&
                        onSelectWhisperModel(e.target.value)
                      }
                      className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500"
                    >
                      {modelsStatus.whisper_models
                        .filter((w) => w.downloaded)
                        .map((w) => (
                          <option key={w.id} value={w.id}>
                            {w.name}
                          </option>
                        ))}
                    </select>
                  </div>

                  <div className="p-4 flex items-center justify-between gap-4">
                    <div>
                      <div className="text-xs font-bold text-white">
                        {t('defaultTranslationTitle')}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-0.5">
                        {t('defaultTranslationDesc')}
                      </div>
                    </div>
                    <select
                      value={activeTranslationModel || 'opus-mt-da-en'}
                      onChange={(e) =>
                        onSelectTranslationModel &&
                        onSelectTranslationModel(e.target.value)
                      }
                      className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500"
                    >
                      {(modelsStatus.translation_models || [])
                        .filter((tm) => tm.downloaded)
                        .map((tm) => (
                          <option key={tm.id} value={tm.id}>
                            {tm.name}
                          </option>
                        ))}
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 3: SOUND & AUDIO INPUT / OUTPUT SETTINGS */}
            {activeTab === 'audio' && (
              <div className="max-w-2xl space-y-6">
                <div className="rounded-2xl bg-slate-950/60 border border-slate-800 divide-y divide-slate-800/80">
                  <div className="p-4 space-y-2.5">
                    <div className="flex items-center gap-2">
                      <Headphones className="w-4 h-4 text-emerald-400" />
                      <div className="text-xs font-bold text-white">
                        {t('systemLoopbackTitle')}
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      {t('systemLoopbackDesc')}
                    </p>
                    <select
                      value={selectedLoopback}
                      onChange={(e) => onChangeLoopback(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-medium text-white focus:outline-none focus:border-emerald-500"
                    >
                      <option value="">{t('noneDisableSystemAudio')}</option>
                      {audioDevices.loopbacks.map((lb) => (
                        <option key={lb.id} value={lb.id}>
                          {lb.name}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="p-4 space-y-2.5">
                    <div className="flex items-center gap-2">
                      <Mic className="w-4 h-4 text-blue-400" />
                      <div className="text-xs font-bold text-white">
                        {t('micInputDeviceTitle')}
                      </div>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      {t('micInputDeviceDesc')}
                    </p>
                    <select
                      value={selectedMic}
                      onChange={(e) => onChangeMic(e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-medium text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="">{t('noneSystemLoopbackOnly')}</option>
                      {audioDevices.microphones.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 4: ABOUT US / ABOUT WISP */}
            {activeTab === 'about' && (
              <div className="max-w-xl mx-auto py-6 text-center space-y-5">
                <img
                  src="./wisp-icon.png"
                  alt="Wisp"
                  className="w-24 h-24 rounded-3xl mx-auto object-cover shadow-2xl shadow-indigo-500/30 border border-white/10"
                />
                <div>
                  <h2 className="text-2xl font-bold text-white tracking-tight">
                    Wisp
                  </h2>
                  <p className="text-xs text-indigo-300 font-medium mt-0.5">
                    {t('aboutSubtitle')}
                  </p>
                </div>

                <div className="p-5 rounded-2xl bg-slate-950/70 border border-slate-800 text-left space-y-3 text-xs text-slate-300 leading-relaxed">
                  <p>{t('aboutP1')}</p>
                  <p>{t('aboutP2')}</p>
                </div>

                <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
                  <Shield className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{t('zeroCloudBadge')}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

interface ExportModalProps {
  isOpen: boolean;
  session: MeetingSession | null;
  onClose: () => void;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  session,
  onClose,
}) => {
  const [mode, setMode] = useState<'translated' | 'dual' | 'original'>(
    'translated'
  );
  const [format, setFormat] = useState<'clean' | 'txt' | 'md'>('clean');
  const [content, setContent] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!isOpen || !session) return;
    fetchExportText(session.id, mode, format)
      .then(setContent)
      .catch(() => setContent('Failed to generate export text.'));
  }, [isOpen, session, mode, format]);

  if (!isOpen || !session) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadFile = () => {
    const ext = format === 'md' ? 'md' : 'txt';
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${session.title.replace(/[^a-z0-9-_]/gi, '_')}_${mode}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-3xl w-full shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white">
              Export Meeting Plain Text & Translation
            </h3>
            <p className="text-xs text-slate-400">
              Ready for sales notes, Microsoft Word, CRM paste, or documentation
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="px-6 py-3 bg-slate-800/40 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Content:</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as any)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              <option value="translated">
                Single Output Language Only ({session.target_language.toUpperCase()})
              </option>
              <option value="dual">
                Both Spoken Language + Translated Output
              </option>
              <option value="original">Original Spoken Languages Only</option>
            </select>

            <span className="text-xs text-slate-400 ml-2">Format:</span>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value as any)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            >
              <option value="clean">
                Word Copy-Paste Format (Speaker: + Paragraph)
              </option>
              <option value="txt">Timestamped Log (.txt)</option>
              <option value="md">Markdown (.md)</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-white border border-slate-700 transition"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy to Clipboard</span>
                </>
              )}
            </button>

            <button
              onClick={handleDownloadFile}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white transition"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Save File</span>
            </button>
          </div>
        </div>

        <div className="p-6 flex-1 overflow-hidden flex flex-col">
          <textarea
            readOnly
            value={content}
            className="w-full h-80 bg-slate-950 border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-200 focus:outline-none resize-none"
          />
        </div>
      </div>
    </div>
  );
};

interface MergeSpeakersModalProps {
  isOpen: boolean;
  session: MeetingSession | null;
  initialSourceSpeakerId: string | null;
  onMerge: (sourceSpeakerId: string, targetSpeakerId: string) => void;
  onPlaySample: (speakerId: string) => void;
  onClose: () => void;
}

export const MergeSpeakersModal: React.FC<MergeSpeakersModalProps> = ({
  isOpen,
  session,
  initialSourceSpeakerId,
  onMerge,
  onPlaySample,
  onClose,
}) => {
  const speakersList: SpeakerProfile[] = session
    ? Object.values(session.speakers)
    : [];

  const [sourceId, setSourceId] = useState<string>('');
  const [targetId, setTargetId] = useState<string>('');

  useEffect(() => {
    if (!isOpen || speakersList.length < 2) return;
    const defaultSrc =
      initialSourceSpeakerId && session?.speakers[initialSourceSpeakerId]
        ? initialSourceSpeakerId
        : speakersList[1]?.id || speakersList[0].id;
    setSourceId(defaultSrc);
    const firstOther = speakersList.find((s) => s.id !== defaultSrc);
    if (firstOther) setTargetId(firstOther.id);
  }, [isOpen, initialSourceSpeakerId, session]);

  if (!isOpen || !session || speakersList.length < 2) return null;

  const sourceSpk = session.speakers[sourceId];
  const targetSpk = session.speakers[targetId];
  const targetOptions = speakersList.filter((s) => s.id !== sourceId);

  const handleSourceChange = (newSrcId: string) => {
    setSourceId(newSrcId);
    if (newSrcId === targetId) {
      const other = speakersList.find((s) => s.id !== newSrcId);
      if (other) setTargetId(other.id);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 select-none">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-lg w-full shadow-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
              <GitMerge className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">
                Merge Duplicate Speakers
              </h3>
              <p className="text-xs text-slate-400">
                Combine two speaker profiles when the same person was detected twice
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-5">
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">
                1. Duplicate Speaker to Merge & Remove:
              </span>
              {sourceSpk && sourceSpk.segment_count > 0 && (
                <button
                  type="button"
                  onClick={() => onPlaySample(sourceSpk.id)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-medium text-slate-200 transition"
                >
                  <Volume2 className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Play Voice Sample</span>
                </button>
              )}
            </div>
            <select
              value={sourceId}
              onChange={(e) => handleSourceChange(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-indigo-500"
            >
              {speakersList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.segment_count} segments — Language: {s.language.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center justify-center text-xs text-indigo-300 font-semibold gap-2">
            <ArrowRight className="w-4 h-4 rotate-90 text-indigo-400" />
            <span>will be merged into</span>
          </div>

          <div className="p-4 rounded-xl bg-indigo-950/25 border border-indigo-500/30 space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-indigo-200">
                2. Primary Speaker to Keep:
              </span>
              {targetSpk && targetSpk.segment_count > 0 && (
                <button
                  type="button"
                  onClick={() => onPlaySample(targetSpk.id)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[11px] font-medium text-slate-200 transition"
                >
                  <Volume2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Play Voice Sample</span>
                </button>
              )}
            </div>
            <select
              value={targetId}
              onChange={(e) => setTargetId(e.target.value)}
              className="w-full bg-slate-900 border border-indigo-500/50 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-indigo-400"
            >
              {targetOptions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.segment_count} segments — Language: {s.language.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          {sourceSpk && targetSpk && (
            <div className="p-3 rounded-xl bg-slate-800/50 border border-slate-700/60 text-xs text-slate-300 leading-relaxed">
              All <strong>{sourceSpk.segment_count} segments</strong> from{' '}
              <strong className="text-white">{sourceSpk.name}</strong> will be
              reassigned to{' '}
              <strong className="text-white">{targetSpk.name}</strong>, and both
              voiceprints will be combined so future speech is recognized as{' '}
              <strong className="text-white">{targetSpk.name}</strong>.
            </div>
          )}
        </div>

        <div className="px-6 py-4 bg-slate-950/60 border-t border-slate-800 flex items-center justify-end gap-2.5">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              if (sourceId && targetId && sourceId !== targetId) {
                onMerge(sourceId, targetId);
                onClose();
              }
            }}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/25 transition"
          >
            <GitMerge className="w-3.5 h-3.5" />
            <span>Merge Speakers</span>
          </button>
        </div>
      </div>
    </div>
  );
};

interface SessionDialogModalProps {
  isOpen: boolean;
  mode: 'create' | 'edit';
  initialTitle?: string;
  initialSessionType?: 'file' | 'live';
  currentSourceFileName?: string;
  modelsStatus?: ModelsStatus | null;
  languages?: SupportedLanguage[];
  initialWhisperModel?: string;
  initialTranslationModel?: string;
  initialTargetLanguage?: string;
  hasExistingAudio?: boolean;
  onSave: (params: {
    title: string;
    sessionType: 'file' | 'live';
    file: File | null;
    expectedSpeakers?: number;
    whisperModel: string;
    translationModel: string;
    targetLanguage: string;
    retranscribe?: boolean;
  }) => void;
  onClose: () => void;
}

export const SessionDialogModal: React.FC<SessionDialogModalProps> = ({
  isOpen,
  mode,
  initialTitle = 'Sales Meeting',
  initialSessionType = 'file',
  currentSourceFileName = '',
  modelsStatus,
  languages = [],
  initialWhisperModel = 'small',
  initialTranslationModel = 'opus-mt-da-en',
  initialTargetLanguage = 'en',
  hasExistingAudio = false,
  onSave,
  onClose,
}) => {
  const { t } = useI18n();
  const [title, setTitle] = useState(initialTitle);
  const [sessionType, setSessionType] = useState<'file' | 'live'>(
    initialSessionType
  );
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [expectedSpeakers, setExpectedSpeakers] = useState<string>('0');
  const [whisperModel, setWhisperModel] = useState<string>(initialWhisperModel);
  const [translationModel, setTranslationModel] = useState<string>(
    initialTranslationModel
  );
  const [targetLanguage, setTargetLanguage] = useState<string>(
    initialTargetLanguage
  );
  const [retranscribe, setRetranscribe] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setTitle(initialTitle);
      setSessionType(initialSessionType || 'file');
      setSelectedFile(null);
      setExpectedSpeakers('0');
      setWhisperModel(initialWhisperModel || 'small');
      setTranslationModel(initialTranslationModel || 'opus-mt-da-en');
      setTargetLanguage(initialTargetLanguage || 'en');
      setRetranscribe(false);
    }
  }, [
    isOpen,
    initialTitle,
    initialSessionType,
    initialWhisperModel,
    initialTranslationModel,
    initialTargetLanguage,
  ]);

  if (!isOpen) return null;

  const isEdit = mode === 'edit';
  const downloadedWhispers =
    modelsStatus?.whisper_models.filter((w) => w.downloaded) || [];
  const whisperOptions =
    downloadedWhispers.length > 0
      ? downloadedWhispers
      : modelsStatus?.whisper_models || [
          { id: 'tiny', name: 'Whisper Tiny', size_mb: 110, description: '', downloaded: true },
          { id: 'small', name: 'Whisper Small', size_mb: 610, description: '', downloaded: true },
        ];

  const downloadedTranslators =
    (modelsStatus?.translation_models || []).filter((tm) => tm.downloaded);
  const translatorOptions =
    downloadedTranslators.length > 0
      ? downloadedTranslators
      : modelsStatus?.translation_models || [
          { id: 'opus-mt-da-en', name: 'OPUS-MT (DA → EN)', size_mb: 195, description: '', downloaded: true },
        ];

  const outputLanguages = languages.filter((l) => l.code !== 'auto');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanTitle = title.trim() || 'Untitled Meeting';
    const spkCount = parseInt(expectedSpeakers, 10);
    onSave({
      title: cleanTitle,
      sessionType,
      file: sessionType === 'file' ? selectedFile : null,
      expectedSpeakers: spkCount > 0 ? spkCount : undefined,
      whisperModel,
      translationModel,
      targetLanguage,
      retranscribe: isEdit && hasExistingAudio && !selectedFile ? retranscribe : false,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 select-none">
      <form
        onSubmit={handleSubmit}
        className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-lg w-full shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white">
              {isEdit ? t('editRecordingSession') : t('newRecordingSession')}
            </h3>
            <p className="text-xs text-slate-400">
              {isEdit ? t('editSessionSub') : t('newSessionSub')}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-4 max-h-[75vh] overflow-y-auto">
          {/* 1. Session Type Selector (Locked when editing an existing session) */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300">
                {t('sessionType')}
              </label>
              {isEdit && (
                <span className="text-[10px] text-slate-500">
                  {t('lockedAfterCreation')}
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                disabled={isEdit}
                onClick={() => setSessionType('file')}
                className={`p-3.5 rounded-xl border text-left transition flex items-start gap-3 ${
                  sessionType === 'file'
                    ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-md shadow-indigo-600/10'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
                } ${isEdit ? 'opacity-75 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <FileAudio
                  className={`w-5 h-5 shrink-0 mt-0.5 ${
                    sessionType === 'file'
                      ? 'text-indigo-400'
                      : 'text-slate-500'
                  }`}
                />
                <div>
                  <div className="text-xs font-bold">{t('mp3AudioFile')}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                    {t('mp3AudioFileDesc')}
                  </div>
                </div>
              </button>

              <button
                type="button"
                disabled={isEdit}
                onClick={() => setSessionType('live')}
                className={`p-3.5 rounded-xl border text-left transition flex items-start gap-3 ${
                  sessionType === 'live'
                    ? 'bg-emerald-600/20 border-emerald-500 text-white shadow-md shadow-emerald-600/10'
                    : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700'
                } ${isEdit ? 'opacity-75 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <Radio
                  className={`w-5 h-5 shrink-0 mt-0.5 ${
                    sessionType === 'live'
                      ? 'text-emerald-400'
                      : 'text-slate-500'
                  }`}
                />
                <div>
                  <div className="text-xs font-bold">{t('liveMeeting')}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5 leading-snug">
                    {t('liveMeetingDesc')}
                  </div>
                </div>
              </button>
            </div>
          </div>

          {/* 2. Session Name */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-300">
              {t('sessionName')}
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Q4 Sales Call with Nordic Client"
              autoFocus
              className="w-full bg-slate-950 border border-slate-700 focus:border-blue-500 rounded-xl px-3.5 py-2.5 text-xs font-medium text-white focus:outline-none"
            />
          </div>

          {/* 3. If File Session: File Picker + Expected Speakers */}
          {sessionType === 'file' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-300">
                  {isEdit ? t('audioFileLabelEdit') : t('audioFileLabelCreate')}
                </label>
                {isEdit && currentSourceFileName && !selectedFile && (
                  <span className="text-[11px] text-indigo-300 font-medium truncate max-w-[200px]">
                    {t('currentFile')} {currentSourceFileName}
                  </span>
                )}
              </div>

              <label
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  const f = e.dataTransfer.files?.[0];
                  if (f) {
                    setSelectedFile(f);
                    if (
                      !isEdit &&
                      (title.startsWith('Sales Meeting') || !title.trim())
                    ) {
                      setTitle(f.name.replace(/\.[^/.]+$/, ''));
                    }
                  }
                }}
                className={`w-full rounded-xl border-2 border-dashed p-5 cursor-pointer transition flex flex-col items-center text-center ${
                  isDragging
                    ? 'border-indigo-400 bg-indigo-500/15'
                    : selectedFile
                    ? 'border-emerald-500/60 bg-emerald-950/20'
                    : 'border-slate-700 hover:border-indigo-500/60 bg-slate-950/50'
                }`}
              >
                <input
                  type="file"
                  accept=".mp3,.m4a,.wav,.flac,.ogg,.aac,.webm,.mp4,audio/mpeg,audio/mp3,audio/mp4,audio/x-m4a,audio/aac,audio/wav,audio/*"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) {
                      setSelectedFile(f);
                      if (
                        !isEdit &&
                        (title.startsWith('Sales Meeting') || !title.trim())
                      ) {
                        setTitle(f.name.replace(/\.[^/.]+$/, ''));
                      }
                    }
                  }}
                  className="hidden"
                />
                <Upload
                  className={`w-6 h-6 mb-2 ${
                    selectedFile ? 'text-emerald-400' : 'text-indigo-400'
                  }`}
                />
                {selectedFile ? (
                  <div>
                    <div className="text-xs font-bold text-emerald-300">
                      {selectedFile.name}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="text-xs font-semibold text-slate-200">
                      {t('browseOrDropAudio')}
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5">
                      {isEdit
                        ? t('leaveEmptyKeepAudio')
                        : t('supportsFormats')}
                    </div>
                  </div>
                )}
              </label>

              <div className="flex items-center justify-between bg-slate-950/60 border border-slate-800 rounded-xl px-3.5 py-2.5">
                <span className="text-xs text-slate-400 font-medium">
                  {t('expectedSpeakers')}
                </span>
                <select
                  value={expectedSpeakers}
                  onChange={(e) => setExpectedSpeakers(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white font-semibold focus:outline-none"
                >
                  <option value="0">{t('autoDetectSpeakers')}</option>
                  <option value="2">2 {t('speakersCount')}</option>
                  <option value="3">3 {t('speakersCount')}</option>
                  <option value="4">4 {t('speakersCount')}</option>
                  <option value="5">5 {t('speakersCount')}</option>
                </select>
              </div>
            </div>
          )}

          {/* 4. AI Models & Output Language Selection for this Session */}
          <div className="rounded-xl bg-slate-950/60 border border-slate-800 p-3.5 space-y-3">
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-blue-400" />
              <span>{t('sessionModelsHeader')}</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <div className="space-y-1">
                <label className="text-[11px] font-medium text-slate-400">
                  {t('speechToTextModel')}
                </label>
                <select
                  value={whisperModel}
                  onChange={(e) => {
                    const val = e.target.value;
                    setWhisperModel(val);
                    if (isEdit && hasExistingAudio) {
                      setRetranscribe(true);
                    }
                  }}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-semibold focus:outline-none focus:border-blue-500"
                >
                  {whisperOptions.map((wm) => (
                    <option key={wm.id} value={wm.id}>
                      {wm.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[11px] font-medium text-slate-400">
                  {t('localTranslationModel')}
                </label>
                <select
                  value={translationModel}
                  onChange={(e) => {
                    const val = e.target.value;
                    setTranslationModel(val);
                    if (isEdit && hasExistingAudio) {
                      setRetranscribe(true);
                    }
                  }}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-semibold focus:outline-none focus:border-emerald-500"
                >
                  {translatorOptions.map((tm) => (
                    <option key={tm.id} value={tm.id}>
                      {tm.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {outputLanguages.length > 0 && (
              <div className="flex items-center justify-between pt-1 border-t border-slate-800/80">
                <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1.5">
                  <Globe className="w-3.5 h-3.5 text-indigo-400" />
                  <span>{t('outputLanguage')}:</span>
                </span>
                <select
                  value={targetLanguage}
                  onChange={(e) => {
                    setTargetLanguage(e.target.value);
                    if (isEdit && hasExistingAudio) {
                      setRetranscribe(true);
                    }
                  }}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white font-semibold focus:outline-none focus:border-indigo-500"
                >
                  {outputLanguages.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.name} ({lang.code.toUpperCase()})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Re-transcribe & Translate checkbox when editing a session with existing audio */}
            {isEdit && hasExistingAudio && !selectedFile && (
              <label className="flex items-center gap-2.5 pt-2 border-t border-slate-800/80 cursor-pointer">
                <input
                  type="checkbox"
                  checked={retranscribe}
                  onChange={(e) => setRetranscribe(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-xs font-semibold text-blue-300 flex items-center gap-1.5">
                  <RefreshCw className="w-3.5 h-3.5 text-blue-400" />
                  <span>{t('retranscribeOnSave')}</span>
                </span>
              </label>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-950/60 border-t border-slate-800 flex items-center justify-end gap-2.5">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            {t('cancel')}
          </button>
          <button
            type="submit"
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-600/25 transition"
          >
            {isEdit ? t('saveChanges') : t('createSession')}
          </button>
        </div>
      </form>
    </div>
  );
};

interface OnboardingWizardModalProps {
  isOpen: boolean;
  modelsStatus: ModelsStatus | null;
  progressText: string | null;
  isBusy: boolean;
  onStorageUpdated: (status: ModelsStatus) => void;
  onComplete: (whisperId: string, translationId: string) => void;
}

export const OnboardingWizardModal: React.FC<OnboardingWizardModalProps> = ({
  isOpen,
  modelsStatus,
  progressText,
  isBusy,
  onStorageUpdated,
  onComplete,
}) => {
  const { t, uiLanguage, setUiLanguage, theme, setTheme } = useI18n();
  const [step, setStep] = useState<1 | 2>(1);
  const [storagePathInput, setStoragePathInput] = useState<string>('');
  const [pathSavedNotice, setPathSavedNotice] = useState(false);
  const [pathError, setPathError] = useState<string | null>(null);
  const [selectedWhisper, setSelectedWhisper] = useState<string>('tiny');
  const [selectedTranslator, setSelectedTranslator] =
    useState<string>('opus-mt-da-en');

  useEffect(() => {
    if (isOpen && modelsStatus) {
      setStoragePathInput(
        modelsStatus.storage_directory || modelsStatus.models_directory
      );
      const firstDlWhisper = modelsStatus.whisper_models.find(
        (w) => w.downloaded
      );
      if (firstDlWhisper) {
        setSelectedWhisper(firstDlWhisper.id);
      }
    }
  }, [isOpen, modelsStatus?.storage_directory, modelsStatus?.models_directory]);

  if (!isOpen || !modelsStatus) return null;

  const spkDownloaded = modelsStatus.speaker_diarization.downloaded;
  const chosenWhisperObj = modelsStatus.whisper_models.find(
    (w) => w.id === selectedWhisper
  );
  const chosenTranslatorObj = (modelsStatus.translation_models || []).find(
    (tm) => tm.id === selectedTranslator
  );

  const isChosenSelectionReady =
    spkDownloaded &&
    !!chosenWhisperObj?.downloaded &&
    !!chosenTranslatorObj?.downloaded;

  const handleSaveStoragePath = async (customPath?: string) => {
    const target = (customPath ?? storagePathInput).trim();
    if (!target) return;
    setPathError(null);
    try {
      const updated = await updateStoragePath(target);
      setStoragePathInput(
        updated.storage_directory || updated.models_directory
      );
      onStorageUpdated(updated);
      setPathSavedNotice(true);
      setTimeout(() => setPathSavedNotice(false), 2000);
    } catch (err: any) {
      setPathError(err.message || 'Could not save storage path');
    }
  };

  const handleBrowseNativeFolder = async () => {
    const w = window as any;
    if (w.wispElectron?.selectDirectory) {
      const picked = await w.wispElectron.selectDirectory(storagePathInput);
      if (picked) {
        setStoragePathInput(picked);
        await handleSaveStoragePath(picked);
      }
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-md flex items-center justify-center p-4 select-none">
      <div className="bg-slate-900 border border-slate-700/80 rounded-2xl max-w-3xl w-full shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-6 py-5 border-b border-slate-800 flex items-center justify-between gap-4 bg-slate-950/60">
          <div className="flex items-center gap-3.5">
            <img
              src="./wisp-icon.png"
              alt="Wisp"
              className="w-11 h-11 rounded-2xl object-cover shadow-lg shadow-indigo-500/25 shrink-0"
            />
            <div>
              <h2 className="text-base font-bold text-white">
                {t('onboardingTitle')}
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                {t('onboardingSub')}
              </p>
            </div>
          </div>

          {/* Step Indicator Pills */}
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setStep(1)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition ${
                step === 1
                  ? 'bg-blue-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              {t('onboardingStep1')}
            </button>
            <button
              type="button"
              onClick={() => setStep(2)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition ${
                step === 2
                  ? 'bg-blue-600 text-white shadow'
                  : 'bg-slate-800 text-slate-400 hover:text-white'
              }`}
            >
              {t('onboardingStep2')}
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto flex-1 space-y-5">
          {step === 1 ? (
            <>
              {/* 1. Storage Folder Picker */}
              <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-5 space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 mt-0.5">
                    <FolderOpen className="w-5 h-5" />
                  </div>
                  <div className="flex-1">
                    <div className="text-xs font-bold text-white">
                      {t('storageLocationTitle')}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {t('storageLocationDesc')}
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2 pt-1">
                  <input
                    type="text"
                    value={storagePathInput}
                    onChange={(e) => setStoragePathInput(e.target.value)}
                    className="flex-1 min-w-[240px] bg-slate-900 border border-slate-700 focus:border-blue-500 rounded-xl px-3.5 py-2 text-xs font-mono text-white focus:outline-none"
                    placeholder="C:\Users\...\WispData"
                  />

                  {typeof window !== 'undefined' &&
                    (window as any).wispElectron?.selectDirectory && (
                      <button
                        type="button"
                        onClick={handleBrowseNativeFolder}
                        className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition"
                      >
                        <FolderOpen className="w-3.5 h-3.5 text-indigo-400" />
                        <span>{t('browseFolder')}</span>
                      </button>
                    )}

                  <button
                    type="button"
                    onClick={() => handleSaveStoragePath()}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-xs font-semibold text-white transition"
                  >
                    {pathSavedNotice ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span>{t('pathSaved')}</span>
                      </>
                    ) : (
                      <span>{t('applyPath')}</span>
                    )}
                  </button>
                </div>

                {pathError && (
                  <div className="text-xs text-rose-400 font-medium">
                    {pathError}
                  </div>
                )}

                <div className="flex flex-wrap items-center gap-4 pt-1 text-[11px] text-slate-400">
                  <span>
                    Models folder:{' '}
                    <code className="text-slate-300 font-mono">
                      {modelsStatus.models_directory}
                    </code>
                  </span>
                  {modelsStatus.sessions_directory && (
                    <span>
                      Sessions folder:{' '}
                      <code className="text-slate-300 font-mono">
                        {modelsStatus.sessions_directory}
                      </code>
                    </span>
                  )}
                </div>
              </div>

              {/* 2. Interface Language & Theme */}
              <div className="rounded-2xl bg-slate-950/60 border border-slate-800 divide-y divide-slate-800/80">
                <div className="p-4 flex items-center justify-between gap-4">
                  <div>
                    <div className="text-xs font-bold text-white">
                      {t('uiLanguageTitle')}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {t('uiLanguageDesc')}
                    </div>
                  </div>
                  <select
                    value={uiLanguage}
                    onChange={(e) =>
                      setUiLanguage(e.target.value as UILanguage)
                    }
                    className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-white focus:outline-none focus:border-blue-500"
                  >
                    {UI_LANGUAGES.map((ul) => (
                      <option key={ul.code} value={ul.code}>
                        {ul.name}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="p-4 space-y-2.5">
                  <div>
                    <div className="text-xs font-bold text-white">
                      {t('themeTitle')}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {t('themeDesc')}
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    {(
                      [
                        {
                          id: 'dark' as ThemeMode,
                          label: t('themeDark'),
                          icon: <Moon className="w-4 h-4" />,
                        },
                        {
                          id: 'light' as ThemeMode,
                          label: t('themeLight'),
                          icon: <Sun className="w-4 h-4" />,
                        },
                        {
                          id: 'system' as ThemeMode,
                          label: t('themeSystem'),
                          icon: <Monitor className="w-4 h-4" />,
                        },
                      ] as const
                    ).map((opt) => (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setTheme(opt.id)}
                        className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl border text-xs font-semibold transition ${
                          theme === opt.id
                            ? 'bg-blue-600 border-blue-500 text-white shadow-md shadow-blue-600/20'
                            : 'bg-slate-900 border-slate-700/80 text-slate-300 hover:border-slate-600'
                        }`}
                      >
                        {opt.icon}
                        <span>{opt.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-4">
              {progressText && (
                <div className="p-3.5 rounded-xl bg-blue-500/15 border border-blue-500/30 flex items-center gap-2.5 text-xs text-blue-200">
                  {isBusy ? (
                    <Loader2 className="w-4 h-4 animate-spin text-blue-400 shrink-0" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  )}
                  <span className="font-medium">{progressText}</span>
                </div>
              )}

              {/* Stage 1: Speaker Diarization */}
              <div className="p-4 rounded-2xl bg-slate-950/60 border border-slate-800 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
                    <Users className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-bold text-white">
                      {t('requiredSpeakerModel')} —{' '}
                      {modelsStatus.speaker_diarization.name}
                    </div>
                    <div className="text-[11px] text-slate-400">
                      ~{modelsStatus.speaker_diarization.size_mb} MB &bull;{' '}
                      {t('step1ModelDesc')}
                    </div>
                  </div>
                </div>

                {spkDownloaded ? (
                  <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-emerald-500/15 text-emerald-400 text-xs font-semibold border border-emerald-500/30 shrink-0">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    {t('readyOffline')}
                  </span>
                ) : (
                  <span className="text-xs text-indigo-300 font-semibold shrink-0">
                    Included in setup
                  </span>
                )}
              </div>

              {/* Stage 2 & Stage 3 side-by-side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Stage 2: Whisper Speech-to-Text */}
                <div className="p-4 rounded-2xl bg-slate-950/60 border border-slate-800 space-y-2.5">
                  <div className="flex items-center gap-2 text-xs font-bold text-white">
                    <Cpu className="w-4 h-4 text-blue-400" />
                    <span>{t('chooseSttModel')}</span>
                  </div>
                  <div className="space-y-2">
                    {modelsStatus.whisper_models.map((wm) => {
                      const isSel = selectedWhisper === wm.id;
                      return (
                        <div
                          key={wm.id}
                          onClick={() => setSelectedWhisper(wm.id)}
                          className={`p-3 rounded-xl border cursor-pointer transition flex items-start justify-between gap-2 ${
                            isSel
                              ? 'bg-blue-950/30 border-blue-500'
                              : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                          }`}
                        >
                          <div>
                            <div className="text-xs font-bold text-white">
                              {wm.name}
                            </div>
                            <div className="text-[11px] text-slate-400 mt-0.5">
                              {wm.description}
                            </div>
                          </div>
                          {wm.downloaded ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 text-[10px] font-semibold border border-emerald-500/30 shrink-0">
                              <CheckCircle2 className="w-3 h-3" />
                              {t('downloaded')}
                            </span>
                          ) : (
                            <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded shrink-0">
                              ~{wm.size_mb} MB
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Stage 3: Local Translation AI Model */}
                <div className="p-4 rounded-2xl bg-slate-950/60 border border-slate-800 space-y-2.5">
                  <div className="flex items-center gap-2 text-xs font-bold text-white">
                    <Languages className="w-4 h-4 text-emerald-400" />
                    <span>{t('chooseTranslationModel')}</span>
                  </div>
                  <div className="space-y-2">
                    {(modelsStatus.translation_models || [])
                      .filter((tm) => tm.id !== 'whisper-native')
                      .map((tm) => {
                        const isSel = selectedTranslator === tm.id;
                        return (
                          <div
                            key={tm.id}
                            onClick={() => setSelectedTranslator(tm.id)}
                            className={`p-3 rounded-xl border cursor-pointer transition flex items-start justify-between gap-2 ${
                              isSel
                                ? 'bg-emerald-950/25 border-emerald-500'
                                : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                            }`}
                          >
                            <div>
                              <div className="text-xs font-bold text-white">
                                {tm.name}
                              </div>
                              <div className="text-[11px] text-slate-400 mt-0.5">
                                {tm.description}
                              </div>
                            </div>
                            {tm.downloaded ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 text-[10px] font-semibold border border-emerald-500/30 shrink-0">
                                <CheckCircle2 className="w-3 h-3" />
                                {t('readyOffline')}
                              </span>
                            ) : (
                              <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded shrink-0">
                                ~{tm.size_mb} MB
                              </span>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-950/60 border-t border-slate-800 flex items-center justify-between gap-3">
          {step === 1 ? (
            <>
              <div className="text-[11px] text-slate-400">
                {t('zeroCloudBadge')}
              </div>
              <button
                type="button"
                onClick={async () => {
                  await handleSaveStoragePath();
                  setStep(2);
                }}
                className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white shadow-lg shadow-blue-600/25 transition"
              >
                <span>{t('nextDownloadModels')}</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setStep(1)}
                className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                {t('back')}
              </button>

              <div className="flex items-center gap-2.5">
                {!isChosenSelectionReady && (
                  <button
                    type="button"
                    disabled={isBusy}
                    onClick={() =>
                      triggerModelDownload(
                        'onboarding',
                        selectedWhisper,
                        selectedTranslator
                      )
                    }
                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-xs font-semibold text-white shadow-lg shadow-indigo-600/25 transition"
                  >
                    {isBusy ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Download className="w-4 h-4" />
                    )}
                    <span>{t('downloadSelectedModels')}</span>
                  </button>
                )}

                <button
                  type="button"
                  disabled={isBusy || !isChosenSelectionReady}
                  onClick={() =>
                    onComplete(selectedWhisper, selectedTranslator)
                  }
                  className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-xs font-semibold text-white shadow-lg shadow-emerald-600/25 transition"
                >
                  <CheckCircle2 className="w-4 h-4" />
                  <span>{t('startUsingWisp')}</span>
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
