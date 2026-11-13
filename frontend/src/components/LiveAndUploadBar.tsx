import React from 'react';
import { Loader2 } from 'lucide-react';

interface LiveAndUploadBarProps {
  uploadProgress: { message: string; percent: number } | null;
}

export const LiveAndUploadBar: React.FC<LiveAndUploadBarProps> = ({
  uploadProgress,
}) => {
  if (!uploadProgress) {
    return null;
  }

  return (
    <div className="border-b border-slate-800/80 bg-slate-900/60 px-5 py-2.5 flex items-center justify-center select-none shrink-0">
      <div className="flex items-center gap-3 bg-indigo-500/15 border border-indigo-500/30 rounded-xl px-4 py-2">
        <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />
        <div className="flex flex-col">
          <span className="text-xs font-medium text-indigo-200">
            {uploadProgress.message}
          </span>
          <div className="w-64 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1">
            <div
              className="h-full bg-indigo-500 transition-all duration-300"
              style={{ width: `${uploadProgress.percent}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
