import React, { useEffect, useRef, useState } from 'react';
import {
  Pause,
  Play,
  RotateCcw,
  RotateCw,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { MeetingSession, TranscriptSegment } from '../types';

interface AudioPlayerBarProps {
  session: MeetingSession | null;
  audioUrl: string | null;
  isLiveRecording?: boolean;
  seekTarget: { time: number; endTime?: number; segmentId?: string; ts: number } | null;
  onActiveSegmentChange: (segmentId: string | null) => void;
}

function formatTime(sec: number): string {
  if (!isFinite(sec) || sec < 0) return '00:00';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export const AudioPlayerBar: React.FC<AudioPlayerBarProps> = ({
  session,
  audioUrl,
  isLiveRecording = false,
  seekTarget,
  onActiveSegmentChange,
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.0);
  const stopAtRef = useRef<number | null>(null);

  // Reset player state when switching sessions
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(session?.duration || 0);
    stopAtRef.current = null;
  }, [session?.id, session?.duration]);

  useEffect(() => {
    if (!seekTarget || !audioRef.current) return;
    const el = audioRef.current;
    el.currentTime = seekTarget.time;
    stopAtRef.current = seekTarget.endTime ?? null;
    el.play()
      .then(() => setIsPlaying(true))
      .catch(() => {});
  }, [seekTarget]);

  const handleTimeUpdate = () => {
    const el = audioRef.current;
    if (!el) return;
    const now = el.currentTime;
    setCurrentTime(now);

    if (stopAtRef.current !== null && now >= stopAtRef.current) {
      el.pause();
      setIsPlaying(false);
      stopAtRef.current = null;
    }

    if (session && session.segments.length > 0) {
      const active: TranscriptSegment | undefined = session.segments.find(
        (seg) => now >= seg.start && now <= seg.end
      );
      onActiveSegmentChange(active ? active.id : null);
    }
  };

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    stopAtRef.current = null;
    if (isPlaying) {
      el.pause();
      setIsPlaying(false);
    } else {
      el.play()
        .then(() => setIsPlaying(true))
        .catch(() => {});
    }
  };

  const skip = (delta: number) => {
    const el = audioRef.current;
    if (!el) return;
    stopAtRef.current = null;
    el.currentTime = Math.max(0, Math.min(el.duration || 0, el.currentTime + delta));
  };

  const changeSpeed = () => {
    const speeds = [1.0, 1.25, 1.5, 2.0, 0.75];
    const next = speeds[(speeds.indexOf(playbackRate) + 1) % speeds.length];
    setPlaybackRate(next);
    if (audioRef.current) {
      audioRef.current.playbackRate = next;
    }
  };

  const effectiveDuration = Math.max(duration, session?.duration || 0);
  const hideBottomBar =
    !session ||
    session.session_type === 'live' ||
    isLiveRecording ||
    effectiveDuration <= 0;

  return (
    <>
      <audio
        ref={audioRef}
        src={audioUrl || undefined}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={() => {
          if (audioRef.current && isFinite(audioRef.current.duration)) {
            setDuration(audioRef.current.duration);
          }
        }}
        onEnded={() => {
          setIsPlaying(false);
          stopAtRef.current = null;
          onActiveSegmentChange(null);
        }}
        className="hidden"
      />

      {!hideBottomBar && (
        <footer className="h-16 border-t border-slate-800/80 bg-slate-900/95 px-6 flex items-center justify-between gap-6 select-none shrink-0">
          {/* Left: Transport controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => skip(-5)}
              disabled={!session || effectiveDuration <= 0}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition"
              title="Backtrack 5 seconds"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <button
              onClick={togglePlay}
              disabled={!session || effectiveDuration <= 0}
              className="w-10 h-10 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white flex items-center justify-center shadow-lg shadow-blue-600/20 transition"
              title={isPlaying ? 'Pause' : 'Play recorded meeting audio'}
            >
              {isPlaying ? (
                <Pause className="w-4 h-4 fill-current" />
              ) : (
                <Play className="w-4 h-4 fill-current ml-0.5" />
              )}
            </button>

            <button
              onClick={() => skip(5)}
              disabled={!session || effectiveDuration <= 0}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-40 transition"
              title="Forward 5 seconds"
            >
              <RotateCw className="w-4 h-4" />
            </button>
          </div>

          {/* Center: Timeline Scrubber with Segment Markers */}
          <div className="flex-1 flex items-center gap-3">
            <span className="font-mono text-xs text-slate-300 w-12 text-right">
              {formatTime(currentTime)}
            </span>

            <div className="relative flex-1 flex items-center">
              <input
                type="range"
                min={0}
                max={effectiveDuration > 0 ? effectiveDuration : 1}
                step={0.1}
                value={currentTime}
                disabled={!session || effectiveDuration <= 0}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setCurrentTime(val);
                  stopAtRef.current = null;
                  if (audioRef.current) {
                    audioRef.current.currentTime = val;
                  }
                }}
                className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>

            <span className="font-mono text-xs text-slate-400 w-12">
              {formatTime(effectiveDuration)}
            </span>
          </div>

          {/* Right: Speed & Mute */}
          <div className="flex items-center gap-3">
            <button
              onClick={changeSpeed}
              className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition"
              title="Playback speed"
            >
              {playbackRate}x
            </button>

            <button
              onClick={() => {
                const next = !isMuted;
                setIsMuted(next);
                if (audioRef.current) audioRef.current.muted = next;
              }}
              className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
            >
              {isMuted ? (
                <VolumeX className="w-4 h-4 text-rose-400" />
              ) : (
                <Volume2 className="w-4 h-4" />
              )}
            </button>
          </div>
        </footer>
      )}
    </>
  );
};
