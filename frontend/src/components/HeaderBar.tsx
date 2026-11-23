import React from 'react';
import {
  Settings,
  Shield,
} from 'lucide-react';
import { useI18n } from '../i18n';
import { SettingsTab } from './Modals';

interface HeaderBarProps {
  onOpenSettingsModal: (initialTab?: SettingsTab) => void;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  onOpenSettingsModal,
}) => {
  const { t } = useI18n();

  return (
    <header className="h-14 border-b border-slate-800/80 bg-slate-900/90 backdrop-blur px-5 flex items-center justify-between gap-3 select-none shrink-0">
      {/* Left: Brand + 100% Local & Offline Guarantee Badge */}
      <div className="flex items-center gap-3 min-w-0">
        <div className="flex items-center gap-2.5">
          <img
            src="./wisp-icon.png"
            alt="WispAI"
            className="w-8 h-8 rounded-xl object-cover shadow-lg shadow-indigo-500/25 shrink-0"
          />
          <span className="font-bold tracking-tight text-white text-lg">
            WispAI
          </span>
        </div>

        <button
          type="button"
          onClick={() => onOpenSettingsModal('about')}
          title={t('privacyGuaranteeSub')}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-[11px] font-semibold transition whitespace-nowrap"
        >
          <Shield className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
          <span>{t('privacyGuaranteeBadge')}</span>
        </button>
      </div>

      {/* Right: Settings Button */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => onOpenSettingsModal('models')}
          className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition shadow-sm"
        >
          <Settings className="w-4 h-4 text-slate-300" />
          <span>{t('settings')}</span>
        </button>
      </div>
    </header>
  );
};
