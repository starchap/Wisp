import React, { useState } from 'react';
import {
  Check,
  Copy,
  Edit2,
  FileAudio,
  FolderOpen,
  Headphones,
  Languages,
  Mic,
  Play,
  Radio,
  RefreshCw,
  Square,
  X,
} from 'lucide-react';
import { openSessionAudioFolder, updateTurnText } from '../api';
import { useI18n } from '../i18n';
import {
  AudioDevice,
  LiveState,
  MeetingSession,
  SupportedLanguage,
  TranscriptSegment,
} from '../types';

export type TranscriptViewMode = 'original' | 'corrected' | 'translated';

interface TranscriptFeedProps {
  session: MeetingSession;
  languages: SupportedLanguage[];
  activeSegmentId: string | null;
  viewMode: TranscriptViewMode;
  onChangeViewMode: (mode: TranscriptViewMode) => void;
  onPlaySegment: (segment: TranscriptSegment) => void;
  onSessionUpdated?: (updated: MeetingSession) => void;
  onOpenEditSession?: () => void;
  onReprocessSession?: () => void;
  onResumeSession?: () => void;
  devices?: {
    microphones: AudioDevice[];
    loopbacks: AudioDevice[];
  };
  selectedLoopback?: string;
  selectedMic?: string;
  onChangeLoopback?: (id: string) => void;
  onChangeMic?: (id: string) => void;
  liveState?: LiveState;
  onStartLive?: (loopbackId: string | null, micId: string | null) => void;
  onStopLive?: () => void;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

interface GroupedTurn {
  speakerId: string;
  speakerName: string;
  speakerColor: string;
  firstSegment: TranscriptSegment;
  segments: TranscriptSegment[];
  rawOriginalParagraph: string;
  correctedParagraph: string;
  translatedParagraph: string;
}

function groupConsecutiveTurns(session: MeetingSession): GroupedTurn[] {
  const turns: GroupedTurn[] = [];
  for (const seg of session.segments) {
    const spk = session.speakers[seg.speaker_id];
    const spkName = spk ? spk.name : seg.speaker_id;
    const spkColor = spk ? spk.color : '#3B82F6';
    const rawText = (seg.raw_text || seg.original_text || '').trim();
    const corrText = (seg.original_text || rawText).trim();
    const transText = (seg.translated_text || corrText).trim();
    const prev = turns[turns.length - 1];
    if (prev && prev.speakerId === seg.speaker_id) {
      prev.segments.push(seg);
      prev.rawOriginalParagraph = `${prev.rawOriginalParagraph} ${rawText}`.trim();
      prev.correctedParagraph = `${prev.correctedParagraph} ${corrText}`.trim();
      prev.translatedParagraph = `${prev.translatedParagraph} ${transText}`.trim();
    } else {
      turns.push({
        speakerId: seg.speaker_id,
        speakerName: spkName,
        speakerColor: spkColor,
        firstSegment: seg,
        segments: [seg],
        rawOriginalParagraph: rawText,
        correctedParagraph: corrText,
        translatedParagraph: transText,
      });
    }
  }
  return turns;
}

export function buildWordCopyText(
  session: MeetingSession,
  viewMode: TranscriptViewMode = 'translated'
): string {
  const turns = groupConsecutiveTurns(session);
  return turns
    .map((t) => {
      const body =
        viewMode === 'original'
          ? t.rawOriginalParagraph
          : viewMode === 'corrected'
          ? t.correctedParagraph
          : t.translatedParagraph;
      return `${t.speakerName}:\n${body}`;
    })
    .join('\n\n\n');
}

export const TranscriptFeed: React.FC<TranscriptFeedProps> = ({
  session,
  languages,
  activeSegmentId,
  viewMode,
  onChangeViewMode,
  onPlaySegment,
  onSessionUpdated,
  onOpenEditSession,
  onReprocessSession,
  onResumeSession,
  devices = { microphones: [], loopbacks: [] },
  selectedLoopback = '',
  selectedMic = '',
  onChangeLoopback,
  onChangeMic,
  liveState,
  onStartLive,
  onStopLive,
}) => {
  const { t } = useI18n();
  const [copiedWord, setCopiedWord] = useState(false);
  const [editingTurnId, setEditingTurnId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState<string>('');
  const [editingSpeakerId, setEditingSpeakerId] = useState<string>('');
  const [isSavingEdit, setIsSavingEdit] = useState(false);

  const speakersList = Object.values(session.speakers).sort((a, b) => {
    const numA = parseInt(a.id.replace('speaker_', ''), 10) || 999;
    const numB = parseInt(b.id.replace('speaker_', ''), 10) || 999;
    return numA - numB;
  });

  const handleStartEditTurn = (turn: GroupedTurn, currentText: string) => {
    setEditingTurnId(turn.firstSegment.id);
    setEditingText(currentText);
    setEditingSpeakerId(turn.speakerId);
  };

  const handleSaveEditTurn = async (turn: GroupedTurn) => {
    if (!editingTurnId || isSavingEdit) return;
    setIsSavingEdit(true);
    try {
      const updated = await updateTurnText(session.id, {
        segment_ids: turn.segments.map((s) => s.id),
        text: editingText,
        view_mode: viewMode,
        speaker_id: editingSpeakerId || turn.speakerId,
      });
      onSessionUpdated?.(updated);
      setEditingTurnId(null);
    } catch (_) {
    } finally {
      setIsSavingEdit(false);
    }
  };

  const isTranslationSkipped =
    session.target_language === 'none' || session.translation_model === 'none';

  const targetLangObj = languages.find(
    (l) => l.code === session.target_language && l.code !== 'none'
  );
  const targetLangLabel = isTranslationSkipped
    ? ''
    : targetLangObj
    ? targetLangObj.name.split(' ')[0]
    : session.target_language.toUpperCase();

  const groupedTurns = groupConsecutiveTurns(session);
  const isLiveSession = session.session_type === 'live';
  const isRecording = !!liveState?.is_recording;
  const levelPct = Math.min(100, Math.round((liveState?.rms_level || 0) * 900));
  const isVoiceActive = isRecording && (liveState?.rms_level || 0) > 0.005;

  const handleCopyForWord = () => {
    const text = buildWordCopyText(session, viewMode);
    navigator.clipboard.writeText(text);
    setCopiedWord(true);
    setTimeout(() => setCopiedWord(false), 2000);
  };

  return (
    <div className="flex-1 flex flex-col min-w-0 bg-slate-950/60 overflow-hidden">
      {/* Clean Sub-toolbar: [Original | Corrected | Translated] Tabs + Actions */}
      <div className="px-6 py-2.5 border-b border-slate-800/60 bg-slate-900/30 flex flex-wrap items-center justify-between gap-3 text-xs shrink-0 select-none">
        {/* Left: Clean 3-Tab Toggle: Original | Corrected | Translated (English) */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
          <button
            type="button"
            onClick={() => onChangeViewMode('original')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              viewMode === 'original'
                ? 'bg-blue-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {t('originalTab')}
          </button>

          <button
            type="button"
            onClick={() => onChangeViewMode('corrected')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              viewMode === 'corrected'
                ? 'bg-blue-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {t('correctedTab')}
          </button>

          <button
            type="button"
            onClick={() => onChangeViewMode('translated')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
              viewMode === 'translated'
                ? 'bg-blue-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Languages className="w-3.5 h-3.5" />
            <span>
              {t('translated')}
              {targetLangLabel ? ` (${targetLangLabel})` : ''}
            </span>
          </button>
        </div>

        {/* Right Controls: Open Folder + Copy for Word + Re-transcribe + Live Audio In/Out */}
        <div className="flex flex-wrap items-center gap-2">
          {(session.duration > 0 || session.segments.length > 0) && (
            <button
              type="button"
              onClick={() => openSessionAudioFolder(session.id).catch(() => {})}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 border border-slate-700/70 text-xs font-medium transition"
              title="Open folder where this session's audio and transcript are saved"
            >
              <FolderOpen className="w-3.5 h-3.5 text-amber-400" />
              <span>Open Folder</span>
            </button>
          )}

          {session.segments.length > 0 && (
            <button
              type="button"
              onClick={handleCopyForWord}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 font-semibold transition"
            >
              {copiedWord ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{t('copiedForWord')}</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>{t('copyForWord')}</span>
                </>
              )}
            </button>
          )}

          {(() => {
            const maxEnd =
              session.segments.length > 0
                ? Math.max(...session.segments.map((s) => s.end))
                : 0;
            const isPartiallyTranscribed =
              session.segments.length > 0 &&
              session.duration > 30 &&
              maxEnd < session.duration - 20;
            if (!isPartiallyTranscribed || !onResumeSession || isRecording) {
              return null;
            }
            const pctDone = Math.min(
              99,
              Math.max(1, Math.round((maxEnd / session.duration) * 100))
            );
            return (
              <button
                type="button"
                onClick={onResumeSession}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition"
                title="Continue transcribing the remaining audio from where it stopped"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>
                  Resume from {formatTime(maxEnd)} ({pctDone}%)
                </span>
              </button>
            );
          })()}

          {(session.segments.length > 0 ||
            session.duration > 0 ||
            !!session.source_file_name) &&
            onReprocessSession &&
            !isRecording && (
              <button
                type="button"
                onClick={onReprocessSession}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-blue-600/15 hover:bg-blue-600/25 text-blue-300 border border-blue-500/30 font-semibold transition"
                title={t('retranscribeOnSave')}
              >
                <RefreshCw className="w-3.5 h-3.5 text-blue-400" />
                <span>{t('retranscribeBtn')}</span>
              </button>
            )}

          {isLiveSession && (
            <>
              <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700/70 rounded-xl px-2.5 py-1">
                <Headphones className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <div className="flex flex-col">
                  <span className="text-[9px] text-slate-400 font-medium leading-none">
                    {t('systemAudioLabel')}
                  </span>
                  <select
                    value={selectedLoopback}
                    disabled={isRecording}
                    onChange={(e) => onChangeLoopback?.(e.target.value)}
                    className="bg-transparent text-[11px] text-slate-100 font-medium focus:outline-none cursor-pointer mt-0.5 max-w-[180px] truncate"
                  >
                    <option value="" className="bg-slate-900 text-slate-400">
                      {t('noneDisableSystemAudio')}
                    </option>
                    {devices.loopbacks.map((lb) => (
                      <option
                        key={lb.id}
                        value={lb.id}
                        className="bg-slate-900 text-slate-100"
                      >
                        {lb.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700/70 rounded-xl px-2.5 py-1">
                <Mic className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                <div className="flex flex-col">
                  <span className="text-[9px] text-slate-400 font-medium leading-none">
                    {t('micInputLabel')}
                  </span>
                  <select
                    value={selectedMic}
                    disabled={isRecording}
                    onChange={(e) => onChangeMic?.(e.target.value)}
                    className="bg-transparent text-[11px] text-slate-100 font-medium focus:outline-none cursor-pointer mt-0.5 max-w-[165px] truncate"
                  >
                    <option value="" className="bg-slate-900 text-slate-400">
                      {t('noneSystemLoopbackOnly')}
                    </option>
                    {devices.microphones.map((m) => (
                      <option
                        key={m.id}
                        value={m.id}
                        className="bg-slate-900 text-slate-100"
                      >
                        {m.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </>
          )}

          {isLiveSession && (
            <>
              {!isRecording ? (
                <button
                  type="button"
                  onClick={() =>
                    onStartLive?.(selectedLoopback || null, selectedMic || null)
                  }
                  className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 transition"
                >
                  <Radio className="w-3.5 h-3.5" />
                  <span>{t('startLiveListening')}</span>
                </button>
              ) : (
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onStopLive?.()}
                    className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 animate-pulse transition"
                  >
                    <Square className="w-3 h-3 fill-current" />
                    <span>{t('stopLiveCapture')}</span>
                  </button>

                  <div
                    className={`flex items-center gap-2 bg-slate-800/95 px-3 py-1.5 rounded-xl border transition-all duration-150 ${
                      isVoiceActive
                        ? 'border-emerald-400/70 shadow-md shadow-emerald-500/20'
                        : 'border-rose-500/30'
                    }`}
                  >
                    <span
                      className={`w-2 h-2 rounded-full ${
                        isVoiceActive
                          ? 'bg-emerald-400 animate-ping'
                          : 'bg-rose-500 animate-pulse'
                      }`}
                    />
                    <span
                      className={`text-[10px] font-bold tracking-wider ${
                        isVoiceActive ? 'text-emerald-300' : 'text-rose-300'
                      }`}
                    >
                      {isVoiceActive ? 'SPEAKING' : 'LIVE'}
                    </span>

                    {/* 5-bar real-time voice level equalizer */}
                    <div className="flex items-end gap-0.5 h-3.5 px-0.5">
                      {[0.35, 0.65, 1.0, 0.75, 0.45].map((mult, idx) => {
                        const barH = Math.max(
                          20,
                          Math.min(100, Math.round(levelPct * mult))
                        );
                        return (
                          <span
                            key={idx}
                            className={`w-1 rounded-full transition-all duration-100 ${
                              isVoiceActive
                                ? 'bg-emerald-400'
                                : 'bg-slate-600'
                            }`}
                            style={{ height: `${barH}%` }}
                          />
                        );
                      })}
                    </div>

                    <div className="w-16 h-1.5 bg-slate-900 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-emerald-400 to-blue-500 transition-all duration-100"
                        style={{ width: `${Math.max(5, levelPct)}%` }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Scrollable Plain Text Document Feed */}
      <div className="flex-1 overflow-y-auto p-6">
        {session.segments.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto py-12 select-none">
            {isLiveSession ? (
              <div className="w-full rounded-2xl border border-slate-800 bg-slate-900/40 p-10 flex flex-col items-center">
                <div className="w-14 h-14 rounded-2xl bg-emerald-600/20 border border-emerald-500/30 flex items-center justify-center mb-4 text-emerald-400">
                  <Radio className="w-7 h-7" />
                </div>
                <h3 className="text-base font-bold text-white mb-1">
                  {t('readyForLiveTitle')}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed max-w-md">
                  {t('readyForLiveDesc')}
                </p>
              </div>
            ) : session.duration > 0 || !!session.source_file_name ? (
              <div className="w-full rounded-2xl border border-slate-800 bg-slate-900/50 p-10 flex flex-col items-center">
                <div className="w-14 h-14 rounded-2xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center mb-4 text-blue-400">
                  <FileAudio className="w-7 h-7" />
                </div>
                <h3 className="text-base font-bold text-white mb-1">
                  {session.source_file_name || session.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-5 max-w-md">
                  Audio file is loaded and ready. Click below to run local AI
                  transcription, or edit session settings to choose another
                  model.
                </p>
                <div className="flex flex-wrap items-center justify-center gap-3">
                  {onReprocessSession && (
                    <button
                      type="button"
                      onClick={onReprocessSession}
                      className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-lg shadow-blue-600/25 transition"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>{t('retranscribeBtn')}</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={onOpenEditSession}
                    className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold transition"
                  >
                    <Edit2 className="w-3.5 h-3.5" />
                    <span>{t('editSessionModelsBtn')}</span>
                  </button>
                </div>
              </div>
            ) : (
              <div
                onClick={onOpenEditSession}
                className="w-full rounded-2xl border-2 border-dashed border-slate-800 hover:border-indigo-500/60 bg-slate-900/40 hover:bg-slate-900/70 p-10 cursor-pointer transition flex flex-col items-center"
              >
                <div className="w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center mb-4 text-indigo-400">
                  <FileAudio className="w-7 h-7" />
                </div>
                <h3 className="text-base font-bold text-white mb-1">
                  {t('noAudioLoadedTitle')} &ldquo;{session.title}&rdquo;
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-5 max-w-md">
                  {t('noAudioLoadedDesc')}
                </p>
                <span className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/25 transition">
                  <Edit2 className="w-3.5 h-3.5" />
                  <span>{t('selectAudioInSettings')}</span>
                </span>
              </div>
            )}
          </div>
        ) : (
          /* PLAIN TEXT DOCUMENT VIEW: Directly selectable & copy-pasteable into Word */
          <div className="max-w-4xl mx-auto rounded-2xl bg-slate-900/80 border border-slate-800 p-8 shadow-xl select-text space-y-7">
            {groupedTurns.map((turn, idx) => {
              const isTurnPlaying = turn.segments.some(
                (s) => s.id === activeSegmentId
              );
              const paragraphText =
                viewMode === 'original'
                  ? turn.rawOriginalParagraph
                  : viewMode === 'corrected'
                  ? turn.correctedParagraph
                  : turn.translatedParagraph;

              const isEditingThisTurn =
                editingTurnId === turn.firstSegment.id;

              return (
                <div
                  key={`${turn.firstSegment.id}-${idx}`}
                  className={`group rounded-xl transition ${
                    isEditingThisTurn
                      ? 'bg-slate-800/90 border border-blue-500/60 -mx-3 px-4 py-3 shadow-lg'
                      : isTurnPlaying
                      ? 'bg-blue-950/30 -mx-3 px-3 py-2'
                      : 'hover:bg-slate-800/40 -mx-3 px-3 py-1.5'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2">
                      {isEditingThisTurn && speakersList.length > 0 ? (
                        <select
                          value={editingSpeakerId}
                          onChange={(e) => setEditingSpeakerId(e.target.value)}
                          className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs font-bold text-white focus:outline-none focus:border-blue-500"
                        >
                          {speakersList.map((spk) => (
                            <option key={spk.id} value={spk.id}>
                              {spk.name}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <span className="font-bold text-sm text-white tracking-tight">
                          {turn.speakerName}:
                        </span>
                      )}

                      <button
                        type="button"
                        onClick={() => onPlaySegment(turn.firstSegment)}
                        className={`${
                          isEditingThisTurn
                            ? 'opacity-100'
                            : 'opacity-0 group-hover:opacity-100'
                        } select-none inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-blue-600 text-[10px] text-slate-300 hover:text-white transition`}
                        title="Play audio from this paragraph"
                      >
                        <Play className="w-2.5 h-2.5 fill-current" />
                        <span>{formatTime(turn.firstSegment.start)}</span>
                      </button>

                      {!isEditingThisTurn && (
                        <button
                          type="button"
                          onClick={() =>
                            handleStartEditTurn(turn, paragraphText)
                          }
                          className="opacity-0 group-hover:opacity-100 select-none inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800 hover:bg-indigo-600 text-[10px] text-slate-300 hover:text-white transition"
                          title="Edit this paragraph text"
                        >
                          <Edit2 className="w-2.5 h-2.5" />
                          <span>Edit</span>
                        </button>
                      )}
                    </div>

                    {isEditingThisTurn && (
                      <div className="flex items-center gap-1.5 select-none">
                        <button
                          type="button"
                          disabled={isSavingEdit}
                          onClick={() => handleSaveEditTurn(turn)}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold transition shadow-sm"
                        >
                          <Check className="w-3.5 h-3.5" />
                          <span>Save</span>
                        </button>
                        <button
                          type="button"
                          disabled={isSavingEdit}
                          onClick={() => setEditingTurnId(null)}
                          className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition"
                        >
                          <X className="w-3.5 h-3.5" />
                          <span>Cancel</span>
                        </button>
                      </div>
                    )}
                  </div>

                  {isEditingThisTurn ? (
                    <div className="space-y-1.5 mt-1.5">
                      <textarea
                        value={editingText}
                        onChange={(e) => setEditingText(e.target.value)}
                        onKeyDown={(e) => {
                          if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                            e.preventDefault();
                            handleSaveEditTurn(turn);
                          } else if (e.key === 'Escape') {
                            e.preventDefault();
                            setEditingTurnId(null);
                          }
                        }}
                        rows={Math.max(
                          2,
                          Math.min(10, Math.ceil(editingText.length / 85))
                        )}
                        autoFocus
                        className="w-full bg-slate-950 border border-blue-500/60 focus:border-blue-500 rounded-xl p-3 text-sm text-slate-100 leading-relaxed focus:outline-none resize-y"
                      />
                      <div className="text-[10px] text-slate-400 select-none flex items-center justify-between">
                        <span>
                          Press <strong>Ctrl + Enter</strong> to save or{' '}
                          <strong>Esc</strong> to cancel
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p
                      onDoubleClick={() =>
                        handleStartEditTurn(turn, paragraphText)
                      }
                      title="Double-click or click Edit to modify text"
                      className="text-sm text-slate-100 leading-relaxed whitespace-pre-wrap"
                    >
                      {paragraphText}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
