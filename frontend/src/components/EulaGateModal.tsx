import React, { useEffect, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  Check,
  FileText,
  Globe,
  Languages,
  ShieldCheck,
  Sparkles,
  Users,
  X,
} from 'lucide-react';
import { EULA_DOCUMENTS } from '../eulaContent';
import { UI_LANGUAGES, UILanguage, useI18n } from '../i18n';

interface EulaGateModalProps {
  onAccept: () => void;
  readOnly?: boolean;
  onClose?: () => void;
}

export const EulaGateModal: React.FC<EulaGateModalProps> = ({
  onAccept,
  readOnly = false,
  onClose,
}) => {
  const { uiLanguage, setUiLanguage, t } = useI18n();
  const [step, setStep] = useState<'welcome' | 'eula'>(
    readOnly ? 'eula' : 'welcome'
  );
  const [activeFeature, setActiveFeature] = useState<number>(0);
  const [agreed, setAgreed] = useState(false);
  const [declined, setDeclined] = useState(false);
  const doc = EULA_DOCUMENTS[uiLanguage] || EULA_DOCUMENTS.en;

  // Gently auto-advance the animated Welcome showcase every 4.5 seconds
  useEffect(() => {
    if (step !== 'welcome') return;
    const timer = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 4);
    }, 4500);
    return () => clearInterval(timer);
  }, [step]);

  const handleDecline = () => {
    setDeclined(true);
    try {
      window.close();
    } catch (_) {}
  };

  const features = [
    {
      id: 0,
      tab: t('welcomeTabOffline'),
      title: t('welcomeStep1Title'),
      desc: t('welcomeStep1Desc'),
      pillActive: 'bg-blue-500/20 text-blue-300 border-blue-500/50',
      icon: ShieldCheck,
    },
    {
      id: 1,
      tab: t('welcomeTabSpeakers'),
      title: t('welcomeStep2Title'),
      desc: t('welcomeStep2Desc'),
      pillActive: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/50',
      icon: Users,
    },
    {
      id: 2,
      tab: t('welcomeTabGrammar'),
      title: t('welcomeStep3Title'),
      desc: t('welcomeStep3Desc'),
      pillActive: 'bg-blue-500/20 text-blue-300 border-blue-500/50',
      icon: Sparkles,
    },
    {
      id: 3,
      tab: t('welcomeTabTranslate'),
      title: t('welcomeStep4Title'),
      desc: t('welcomeStep4Desc'),
      pillActive: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50',
      icon: Languages,
    },
  ];

  const currentFeat = features[activeFeature];

  // =========================================================================
  // STEP 1: NINTENDO-INSPIRED ANIMATED WELCOME SCREEN
  // =========================================================================
  if (step === 'welcome') {
    return (
      <div className="h-screen w-screen flex items-center justify-center bg-slate-950 text-slate-100 p-4 sm:p-6 overflow-y-auto relative">
        {/* Subtle WispAI-branded ambient glow */}
        <div className="pointer-events-none fixed inset-0 overflow-hidden">
          <div className="absolute -top-24 -left-24 w-96 h-96 rounded-full bg-blue-600/10 blur-3xl" />
          <div className="absolute -bottom-24 -right-24 w-96 h-96 rounded-full bg-emerald-500/10 blur-3xl" />
        </div>

        <div className="relative w-full max-w-2xl rounded-3xl bg-slate-900 border border-slate-800 shadow-2xl overflow-hidden animate-wisp-pop">
          {/* Top Bar: Language Switcher */}
          <div className="px-6 pt-5 pb-2 flex items-center justify-end gap-3">
            <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-700/80 rounded-lg px-2.5 py-1.5">
              <Globe className="w-3.5 h-3.5 text-blue-400 shrink-0" />
              <select
                aria-label={t('eulaLanguageLabel')}
                value={uiLanguage}
                onChange={(e) => setUiLanguage(e.target.value as UILanguage)}
                className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
              >
                {UI_LANGUAGES.map((lang) => (
                  <option
                    key={lang.code}
                    value={lang.code}
                    className="bg-slate-900 text-slate-100"
                  >
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Hero: WispAI Logo with animated audio waves inside + Welcome Title */}
          <div className="px-6 pt-3 pb-5 text-center">
            <div className="relative inline-flex items-center justify-center mb-4 animate-wisp-float">
              <div className="relative w-20 h-20 rounded-2xl overflow-hidden shadow-xl shadow-blue-600/25 border border-blue-500/35 bg-slate-950">
                <img
                  src="./wisp-icon.png"
                  alt="WispAI"
                  className="w-full h-full object-cover opacity-90"
                />
                {/* Animated soundwave bars inside the logo */}
                <div className="absolute inset-0 bg-gradient-to-t from-slate-950/80 via-slate-950/20 to-transparent flex items-end justify-center pb-2.5 gap-1">
                  {[0.45, 0.9, 0.65, 1.0, 0.55].map((h, i) => (
                    <span
                      key={i}
                      className="w-1 rounded-full bg-gradient-to-t from-blue-400 to-emerald-300 shadow-sm animate-wisp-wave"
                      style={{
                        height: `${Math.round(h * 18)}px`,
                        animationDelay: `${i * 0.14}s`,
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>

            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              {t('welcomeTitle')}
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 mt-1.5 max-w-lg mx-auto leading-relaxed">
              {t('welcomeTagline')}
            </p>
          </div>

          {/* Interactive Animated Feature Stage */}
          <div className="px-6 pb-5">
            {/* Standard WispAI Tab Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-1.5 mb-4">
              {features.map((feat) => {
                const Icon = feat.icon;
                const isActive = feat.id === activeFeature;
                return (
                  <button
                    key={feat.id}
                    type="button"
                    onClick={() => setActiveFeature(feat.id)}
                    className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition active:scale-95 ${
                      isActive
                        ? `${feat.pillActive} shadow-sm`
                        : 'bg-slate-800/70 text-slate-300 border-slate-700/80 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{feat.tab}</span>
                  </button>
                );
              })}
            </div>

            {/* Animated Stage Canvas */}
            <div
              key={activeFeature}
              className="rounded-2xl bg-slate-950/80 border border-slate-800/90 p-5 min-h-[172px] flex flex-col sm:flex-row items-center gap-5 animate-wisp-pop"
            >
              {/* Left: Live Mini-Animation */}
              <div className="w-full sm:w-52 shrink-0 flex items-center justify-center">
                {activeFeature === 0 && (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-600/20 to-indigo-600/20 border border-blue-500/40 flex items-center justify-center text-blue-400 animate-wisp-float shadow-lg shadow-blue-600/10">
                      <ShieldCheck className="w-7 h-7" />
                    </div>
                    <div className="flex items-end gap-1 h-5">
                      {[0.5, 0.9, 0.6, 1, 0.7, 0.85, 0.45].map((h, i) => (
                        <span
                          key={i}
                          className="w-1.5 rounded-full bg-blue-400/80 animate-wisp-wave"
                          style={{
                            height: `${Math.round(h * 18)}px`,
                            animationDelay: `${i * 0.12}s`,
                          }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {activeFeature === 1 && (
                  <div className="w-full space-y-2.5">
                    <div className="rounded-xl bg-blue-500/15 border border-blue-500/35 px-3 py-2 text-left animate-wisp-joy-left">
                      <div className="flex items-center gap-1.5 text-[10px] font-bold text-blue-300">
                        <span className="w-2 h-2 rounded-full bg-blue-400" />
                        Speaker 1
                      </div>
                      <div className="h-1.5 w-28 rounded-full bg-blue-300/40 mt-1.5" />
                    </div>
                    <div className="rounded-xl bg-emerald-500/15 border border-emerald-500/35 px-3 py-2 text-left ml-5 animate-wisp-joy-right">
                      <div className="flex items-center gap-1.5 text-[10px] font-bold text-emerald-300">
                        <span className="w-2 h-2 rounded-full bg-emerald-400" />
                        Speaker 2
                      </div>
                      <div className="h-1.5 w-24 rounded-full bg-emerald-300/40 mt-1.5" />
                    </div>
                  </div>
                )}

                {activeFeature === 2 && (
                  <div className="w-full rounded-xl bg-slate-900 border border-indigo-500/30 p-3 text-center space-y-2">
                    <div className="inline-flex items-center gap-1 text-[10px] font-bold text-indigo-300">
                      <Sparkles className="w-3 h-3 animate-pulse" />
                      Context AI
                    </div>
                    <div className="flex items-center justify-center gap-2 text-xs font-semibold">
                      <span className="line-through text-slate-500">
                        musikerne
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-blue-400" />
                      <span className="px-2 py-0.5 rounded-md bg-blue-500/20 text-blue-200 border border-blue-500/40">
                        musklerne,
                      </span>
                    </div>
                    <div className="text-[10px] text-slate-400">
                      . , ? &nbsp;✓
                    </div>
                  </div>
                )}

                {activeFeature === 3 && (
                  <div className="flex flex-col items-center gap-3">
                    <div className="flex items-center gap-2">
                      <span className="px-2.5 py-1 rounded-lg bg-blue-500/20 border border-blue-500/40 text-xs font-bold text-blue-200 animate-wisp-joy-left">
                        DA
                      </span>
                      <ArrowRight className="w-4 h-4 text-emerald-400 animate-pulse" />
                      <span className="px-2.5 py-1 rounded-lg bg-emerald-500/20 border border-emerald-500/40 text-xs font-bold text-emerald-200 animate-wisp-joy-right">
                        EN / DE / ES
                      </span>
                    </div>
                    <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                      {t('copyForWord')} ✓
                    </span>
                  </div>
                )}
              </div>

              {/* Right: Feature Title & Description */}
              <div className="flex-1 text-center sm:text-left">
                <h2 className="text-base font-bold text-white">
                  {currentFeat.title}
                </h2>
                <p className="text-xs text-slate-300 mt-1.5 leading-relaxed">
                  {currentFeat.desc}
                </p>
              </div>
            </div>
          </div>

          {/* Warm EULA Heads-up + Nintendo-Tactile Continue Button */}
          <div className="px-6 py-5 border-t border-slate-800/80 bg-slate-900/95 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-slate-400 leading-relaxed text-center sm:text-left">
              {t('welcomeEulaNote')}
            </p>

            <button
              type="button"
              onClick={() => setStep('eula')}
              className="shrink-0 inline-flex items-center gap-2 px-5 py-2 rounded-xl text-xs font-semibold bg-blue-600 hover:bg-blue-500 active:scale-95 text-white shadow-lg shadow-blue-600/20 transition"
            >
              <span>{t('welcomeContinueBtn')}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  // =========================================================================
  // STEP 2: PLAIN-TEXT PROFESSIONAL EULA DOCUMENT
  // =========================================================================
  return (
    <div
      className={`${
        readOnly
          ? 'fixed inset-0 z-[70] bg-slate-950/80 backdrop-blur-sm'
          : 'h-screen w-screen bg-slate-950'
      } flex items-center justify-center text-slate-100 p-4 sm:p-6 overflow-y-auto`}
    >
      <div className="w-full max-w-3xl rounded-2xl bg-slate-900 border border-slate-800 shadow-2xl overflow-hidden flex flex-col max-h-[90vh] animate-wisp-pop">
        {/* Header with Back Button, Title & UI Language Switcher */}
        <div className="px-6 py-4 border-b border-slate-800 bg-slate-900 flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            {!readOnly && (
              <button
                type="button"
                onClick={() => setStep('welcome')}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition active:scale-95 shrink-0"
                title={t('welcomeBackBtn')}
              >
                <ArrowLeft className="w-4 h-4" />
              </button>
            )}
            <FileText className="w-5 h-5 text-blue-400 shrink-0" />
            <div className="min-w-0">
              <h1 className="text-base font-bold text-white tracking-tight">
                {t('eulaTitle')}
              </h1>
              <p className="text-xs text-slate-400 mt-0.5">
                {t('eulaSubtitle')} — v{doc.version} ({doc.effectiveDate})
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-950 border border-slate-700/80 rounded-lg px-2.5 py-1.5 shrink-0">
            <Globe className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            <span className="text-[11px] font-medium text-slate-400">
              {t('eulaLanguageLabel')}:
            </span>
            <select
              aria-label={t('eulaLanguageLabel')}
              value={uiLanguage}
              onChange={(e) => setUiLanguage(e.target.value as UILanguage)}
              className="bg-transparent text-xs font-semibold text-white focus:outline-none cursor-pointer"
            >
              {UI_LANGUAGES.map((lang) => (
                <option
                  key={lang.code}
                  value={lang.code}
                  className="bg-slate-900 text-slate-100"
                >
                  {lang.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Plain-text scrollable legal document (no cards) */}
        <div className="flex-1 overflow-y-auto px-7 py-6 bg-slate-950/60 text-[13px] leading-relaxed text-slate-300 select-text space-y-5">
          {declined && (
            <p className="text-amber-300 font-medium border-l-2 border-amber-400 pl-3 py-0.5">
              {t('eulaDeclinedNotice')}
            </p>
          )}

          <p className="text-slate-200">{t('eulaIntro')}</p>

          <p className="text-slate-200">
            <strong className="text-white">{t('eulaBackupTitle')}: </strong>
            {t('eulaBackupBody')}
          </p>

          <hr className="border-slate-800" />

          <div className="space-y-5">
            {doc.sections.map((sec) => (
              <div key={sec.title} className="space-y-2">
                <h2 className="text-[13px] font-bold text-white tracking-tight">
                  {sec.title}
                </h2>
                {sec.paragraphs &&
                  sec.paragraphs.map((p, pIdx) => (
                    <p key={pIdx} className="text-slate-300">
                      {p}
                    </p>
                  ))}
                {sec.items && sec.items.length > 0 && (
                  <ul className="list-disc pl-5 space-y-1 text-slate-300">
                    {sec.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Footer: Mandatory Checkbox + Accept / Decline (or Close in readOnly mode) */}
        <div className="px-6 py-4 border-t border-slate-800 bg-slate-900 space-y-4 shrink-0">
          {readOnly ? (
            <div className="flex items-center justify-end">
              <button
                type="button"
                onClick={onClose}
                className="inline-flex items-center gap-1.5 px-5 py-2 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition active:scale-95"
              >
                <span>{t('done')}</span>
              </button>
            </div>
          ) : (
            <>
              <label className="flex items-start gap-3 cursor-pointer select-none group">
                <input
                  type="checkbox"
                  checked={agreed}
                  onChange={(e) => {
                    setAgreed(e.target.checked);
                    if (e.target.checked) setDeclined(false);
                  }}
                  className="mt-0.5 h-4 w-4 rounded border-slate-600 bg-slate-950 text-blue-600 focus:ring-blue-500 cursor-pointer shrink-0"
                />
                <span className="text-xs text-slate-200 group-hover:text-white leading-relaxed">
                  {t('eulaCheckboxLabel')}
                </span>
              </label>

              <div className="flex flex-wrap items-center justify-end gap-2.5">
                <button
                  type="button"
                  onClick={handleDecline}
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition"
                >
                  <X className="w-3.5 h-3.5" />
                  <span>{t('eulaDeclineBtn')}</span>
                </button>

                <button
                  type="button"
                  disabled={!agreed}
                  onClick={onAccept}
                  className="inline-flex items-center gap-1.5 px-5 py-2 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white shadow-sm transition active:scale-95"
                >
                  <Check className="w-4 h-4" />
                  <span>{t('eulaAcceptBtn')}</span>
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
