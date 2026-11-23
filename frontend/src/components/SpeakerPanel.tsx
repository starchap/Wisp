import React, { useState } from 'react';
import {
  Check,
  Edit2,
  GitMerge,
  Globe2,
  RefreshCw,
  Users,
  Volume2,
} from 'lucide-react';
import { useI18n } from '../i18n';
import {
  MeetingSession,
  SpeakerProfile,
  SupportedLanguage,
} from '../types';

interface SpeakerPanelProps {
  session: MeetingSession;
  languages: SupportedLanguage[];
  reprocessingSpeakers: Record<string, boolean>;
  activeSpeakerId?: string | null;
  isLiveSpeaking?: boolean;
  onUpdateSpeaker: (
    speakerId: string,
    patch: { name?: string; language?: string; reprocess_segments?: boolean }
  ) => void;
  onOpenMergeModal: (sourceSpeakerId: string) => void;
  onPlaySpeakerSample: (speakerId: string) => void;
}

export const SpeakerPanel: React.FC<SpeakerPanelProps> = ({
  session,
  languages,
  reprocessingSpeakers,
  activeSpeakerId = null,
  isLiveSpeaking = false,
  onUpdateSpeaker,
  onOpenMergeModal,
  onPlaySpeakerSample,
}) => {
  const { t } = useI18n();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');

  const activeSpeakerIdsInSession = new Set(
    session.segments.map((seg) => seg.speaker_id)
  );
  const speakersList: SpeakerProfile[] = Object.values(session.speakers)
    .filter(
      (spk) =>
        session.segments.length === 0 ||
        spk.segment_count > 0 ||
        activeSpeakerIdsInSession.has(spk.id)
    )
    .sort((a, b) => {
      const numA = parseInt(a.id.replace('speaker_', ''), 10) || 999;
      const numB = parseInt(b.id.replace('speaker_', ''), 10) || 999;
      return numA - numB;
    });
  const isTranslationSkipped =
    !session.target_language ||
    session.target_language === 'none' ||
    session.translation_model === 'none';

  const targetLangObj = languages.find(
    (l) => l.code === session.target_language && l.code !== 'none'
  );
  const spokenLanguageOptions = languages.filter((l) => l.code !== 'none');

  const handleStartEdit = (spk: SpeakerProfile) => {
    setEditingId(spk.id);
    setEditName(spk.name);
  };

  const handleSaveEdit = (spkId: string) => {
    if (editName.trim()) {
      onUpdateSpeaker(spkId, { name: editName.trim(), reprocess_segments: false });
    }
    setEditingId(null);
  };

  return (
    <aside className="w-80 border-l border-slate-800/80 bg-slate-900/60 flex flex-col shrink-0 select-none">
      {/* Header */}
      <div className="p-4 border-b border-slate-800/80 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Users className="w-4 h-4 text-blue-400 shrink-0" />
          <h2 className="text-sm font-semibold text-white truncate">
            {t('speakersAndLanguages')} ({speakersList.length})
          </h2>
        </div>
      </div>

      {/* Info Banner explaining per-speaker language -> output language */}
      <div className="px-4 py-2.5 bg-slate-800/40 border-b border-slate-800/60 text-[11px] text-slate-300 leading-relaxed">
        {t('speakerPanelHint')}
      </div>

      {/* Speakers List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {speakersList.length === 0 ? (
          <div className="text-center py-10 px-4">
            <Users className="w-8 h-8 text-slate-600 mx-auto mb-2" />
            <p className="text-xs text-slate-400 font-medium">
              {t('noSpeakersDetected')}
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              {t('noSpeakersSub')}
            </p>
          </div>
        ) : (
          speakersList.map((spk) => {
            const isReprocessing = !!reprocessingSpeakers[spk.id];
            const isActiveSpeaker = activeSpeakerId === spk.id;
            const otherSpeakers = speakersList.filter((s) => s.id !== spk.id);

            return (
              <div
                key={spk.id}
                style={
                  isActiveSpeaker
                    ? {
                        borderColor: spk.color,
                        boxShadow: `0 0 18px -2px ${spk.color}55`,
                      }
                    : undefined
                }
                className={`rounded-xl p-3 space-y-2.5 transition-all duration-200 ${
                  isActiveSpeaker
                    ? 'bg-slate-800/95 border-2 scale-[1.01]'
                    : 'bg-slate-800/60 border border-slate-700/70 hover:border-slate-600'
                }`}
              >
                {/* Top row: Color badge, Name / Edit, Active Speaking Badge, Merge button, Sample audio playback */}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    <span className="relative flex h-3.5 w-3.5 shrink-0 items-center justify-center">
                      {isActiveSpeaker && (
                        <span
                          className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                          style={{ backgroundColor: spk.color }}
                        />
                      )}
                      <span
                        className="relative inline-flex rounded-full h-3 w-3 shadow-sm"
                        style={{ backgroundColor: spk.color }}
                      />
                    </span>
                    {editingId === spk.id ? (
                      <div className="flex items-center gap-1 flex-1">
                        <input
                          type="text"
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleSaveEdit(spk.id);
                          }}
                          autoFocus
                          className="w-full bg-slate-900 text-white text-xs font-semibold px-2 py-1 rounded border border-blue-500 focus:outline-none"
                        />
                        <button
                          onClick={() => handleSaveEdit(spk.id)}
                          className="p-1 text-emerald-400 hover:text-emerald-300"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <div
                        onClick={() => handleStartEdit(spk)}
                        className="flex items-center gap-1.5 group cursor-pointer truncate"
                      >
                        <span className="text-xs font-semibold text-white truncate">
                          {spk.name}
                        </span>
                        <Edit2 className="w-3 h-3 text-slate-500 group-hover:text-slate-300 shrink-0" />
                        {isActiveSpeaker && (
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 animate-pulse shrink-0">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                            {isLiveSpeaking ? 'LIVE' : 'ACTIVE'}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <span className="text-[10px] text-slate-400 bg-slate-900/80 px-1.5 py-0.5 rounded">
                      {spk.segment_count} segs
                    </span>
                    {otherSpeakers.length > 0 && (
                      <button
                        onClick={() => onOpenMergeModal(spk.id)}
                        className="p-1 rounded-lg bg-slate-700/60 hover:bg-indigo-600/30 text-slate-300 hover:text-indigo-200 transition"
                      >
                        <GitMerge className="w-3.5 h-3.5" />
                      </button>
                    )}
                    {spk.segment_count > 0 && (
                      <button
                        onClick={() => onPlaySpeakerSample(spk.id)}
                        className="p-1 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 hover:text-white transition"
                      >
                        <Volume2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Spoken Language selector -> Output Language */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span className="flex items-center gap-1">
                      <Globe2 className="w-3 h-3 text-slate-400" />
                      {t('spokenLanguage')}
                    </span>
                    {!isTranslationSkipped && (
                      <span className="text-blue-400 font-medium">
                        &rarr;{' '}
                        {targetLangObj?.code.toUpperCase() ||
                          session.target_language.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 min-w-0">
                    <select
                      value={spk.language}
                      disabled={isReprocessing}
                      onChange={(e) =>
                        onUpdateSpeaker(spk.id, {
                          language: e.target.value,
                          reprocess_segments: true,
                        })
                      }
                      className="flex-1 min-w-0 bg-slate-900/90 border border-slate-700 hover:border-slate-600 focus:border-blue-500 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none cursor-pointer truncate"
                    >
                      {spokenLanguageOptions.map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.name}
                        </option>
                      ))}
                    </select>
                    {spk.segment_count > 0 && (
                      <button
                        onClick={() =>
                          onUpdateSpeaker(spk.id, {
                            language: spk.language,
                            reprocess_segments: true,
                          })
                        }
                        disabled={isReprocessing}
                        className="p-1.5 rounded-lg bg-slate-700/60 hover:bg-slate-700 text-slate-300 hover:text-white disabled:opacity-50 transition shrink-0"
                        title={t('retranscribeBtn')}
                      >
                        <RefreshCw
                          className={`w-3.5 h-3.5 ${
                            isReprocessing ? 'animate-spin text-blue-400' : ''
                          }`}
                        />
                      </button>
                    )}
                  </div>
                </div>

                {isReprocessing && (
                  <div className="text-[11px] text-blue-400 flex items-center gap-1.5 bg-blue-500/10 border border-blue-500/20 rounded-lg px-2.5 py-1">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    <span>{t('retranscribingSegments')}</span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};
