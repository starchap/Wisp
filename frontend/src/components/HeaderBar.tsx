import React from 'react';
import {
  Cpu,
  Globe,
  Languages,
  Settings,
  Shield,
} from 'lucide-react';
import { useI18n } from '../i18n';
import {
  MeetingSession,
  ModelsStatus,
  SupportedLanguage,
  TranslationModelInfo,
  WhisperModelInfo,
} from '../types';
import { SettingsTab } from './Modals';

interface HeaderBarProps {
  session: MeetingSession | null;
  languages: SupportedLanguage[];
  whisperModels: WhisperModelInfo[];
  translationModels: TranslationModelInfo[];
  device: string;
  computeType: string;
  modelsStatus: ModelsStatus | null;
  onChangeTargetLanguage: (langCode: string) => void;
  onChangeWhisperModel: (modelId: string) => void;
  onChangeTranslationModel: (modelId: string) => void;
  onOpenSettingsModal: (initialTab?: SettingsTab) => void;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  session,
  languages,
  whisperModels,
  translationModels,
  device,
  modelsStatus,
  onChangeTargetLanguage,
  onChangeWhisperModel,
  onChangeTranslationModel,
  onOpenSettingsModal,
}) => {
  const { t } = useI18n();
  const outputLanguages = languages.filter((l) => l.code !== 'auto');

  const downloadedWhisperModels =
    modelsStatus?.whisper_models.filter((w) => w.downloaded) ||
    whisperModels.filter((w) => w.downloaded);

  const downloadedTranslationModels =
    modelsStatus?.translation_models.filter((tm) => tm.downloaded) ||
    translationModels.filter((tm) => tm.downloaded);

  return (
    <header className="h-16 border-b border-slate-800/80 bg-slate-900/90 backdrop-blur px-5 flex items-center justify-between gap-3 select-none shrink-0">
      {/* Left: Brand + 100% Local & Offline Guarantee Badge */}
      <div className="flex items-center gap-3 min-w-0">
        <div className="flex items-center gap-2.5">
          <img
            src="./wisp-icon.png"
            alt="WispAI"
            className="w-9 h-9 rounded-xl object-cover shadow-lg shadow-indigo-500/25 shrink-0"
          />
          <span className="font-bold tracking-tight text-white text-lg">
            WispAI
          </span>
        </div>

        <button
          type="button"
          onClick={() => onOpenSettingsModal('about')}
          title={t('privacyGuaranteeSub')}
          className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-[11px] font-semibold transition"
        >
          <Shield className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span>{t('privacyGuaranteeBadge')}</span>
        </button>
      </div>

      {/* Center: Speech-to-Text Model -> Local Translation AI Model -> Output Language */}
      <div className="flex items-center gap-2.5">
        {session && (
          <>
            {/* 1. Local Speech-to-Text ASR Model Selector (Only Downloaded Models) */}
            <div className="flex items-center gap-2 bg-slate-800/70 border border-slate-700/70 rounded-xl px-3 py-1.5">
              <Cpu className="w-4 h-4 text-indigo-400 shrink-0" />
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold leading-none">
                  {t('speechToTextModel')} ({device.toUpperCase()})
                </span>
                <select
                  value={session.whisper_model}
                  onChange={(e) => {
                    if (e.target.value === '__open_models__') {
                      onOpenSettingsModal('models');
                    } else {
                      onChangeWhisperModel(e.target.value);
                    }
                  }}
                  className="bg-transparent text-slate-200 text-xs font-medium focus:outline-none cursor-pointer mt-0.5 max-w-[180px] truncate"
                >
                  {downloadedWhisperModels.map((m) => (
                    <option
                      key={m.id}
                      value={m.id}
                      className="bg-slate-900 text-slate-100"
                    >
                      {m.name}
                    </option>
                  ))}
                  <option
                    value="__open_models__"
                    className="bg-slate-900 text-blue-400 font-semibold"
                  >
                    {t('downloadMoreModels')}
                  </option>
                </select>
              </div>
            </div>

            {/* 2. Local Translation AI Model Selector (Only Downloaded Models) */}
            <div className="flex items-center gap-2 bg-slate-800/80 border border-emerald-500/30 rounded-xl px-3 py-1.5">
              <Languages className="w-4 h-4 text-emerald-400 shrink-0" />
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-emerald-300 font-semibold leading-none">
                  {t('localTranslationModel')}
                </span>
                <select
                  value={session.translation_model || 'opus-mt-da-en'}
                  onChange={(e) => {
                    if (e.target.value === '__open_models__') {
                      onOpenSettingsModal('models');
                    } else {
                      onChangeTranslationModel(e.target.value);
                    }
                  }}
                  className="bg-transparent text-slate-100 text-xs font-medium focus:outline-none cursor-pointer mt-0.5 max-w-[230px] truncate"
                >
                  {downloadedTranslationModels.map((tm) => (
                    <option
                      key={tm.id}
                      value={tm.id}
                      className="bg-slate-900 text-slate-100"
                    >
                      {tm.name}
                    </option>
                  ))}
                  <option
                    value="__open_models__"
                    className="bg-slate-900 text-emerald-400 font-semibold"
                  >
                    {t('downloadMoreModels')}
                  </option>
                </select>
              </div>
            </div>

            {/* 3. Single Target Output Language */}
            <div className="flex items-center gap-2 bg-slate-800/90 border border-blue-500/30 rounded-xl px-3 py-1.5 shadow-inner">
              <Globe className="w-4 h-4 text-blue-400 shrink-0" />
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-blue-300 font-semibold leading-none">
                  {t('outputLanguage')}
                </span>
                <select
                  value={session.target_language}
                  onChange={(e) => onChangeTargetLanguage(e.target.value)}
                  className="bg-transparent text-white text-xs font-semibold focus:outline-none cursor-pointer mt-0.5"
                >
                  {outputLanguages.map((lang) => (
                    <option
                      key={lang.code}
                      value={lang.code}
                      className="bg-slate-900 text-slate-100"
                    >
                      {lang.name} ({lang.code.toUpperCase()})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Right: Settings Button */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onOpenSettingsModal('models')}
          className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition shadow-sm"
        >
          <Settings className="w-4 h-4 text-slate-300" />
          <span>{t('settings')}</span>
        </button>
      </div>
    </header>
  );
};
