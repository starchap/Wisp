import React, { createContext, useContext } from 'react';

export type UILanguage = 'en' | 'da' | 'es' | 'de';
export type ThemeMode = 'dark' | 'light' | 'system';

export const UI_LANGUAGES: { code: UILanguage; name: string }[] = [
  { code: 'en', name: 'English' },
  { code: 'da', name: 'Dansk (Danish)' },
  { code: 'es', name: 'Español (Spanish)' },
  { code: 'de', name: 'Deutsch (German)' },
];

export const TRANSLATIONS: Record<UILanguage, Record<string, string>> = {
  en: {
    // Top HeaderBar
    speechToTextModel: 'Speech-to-Text Model',
    contextClarifierModel: 'Context Word Clarifier',
    localTranslationModel: 'Local Translation AI Model',
    outputLanguage: 'Output Language',
    downloadMoreModels: '+ Download more models...',
    settings: 'Settings',
    meetingSpokenLanguagesLabel: 'Meeting / Interview Spoken Languages',
    meetingSpokenLanguagesDesc:
      'Select the languages spoken in this interview (e.g. Danish + English) so Speech-to-Text & Clarification know what to look for, or leave on Auto (All Languages).',
    autoAllLanguages: 'Auto (All Languages)',
    stepClarifyBadge: 'Step 3: Context Word Clarifier',
    stepClarifyDesc:
      'Validates and corrects misheard words in each speaker section based on conversation context without restructuring sentences.',
    stepClarifyFooter: 'Output: Context-Validated Text',
    step4TranslateBadge: 'Step 4: Local AI Translation Models',
    chooseClarifierModel: '3. Choose Context Word Clarifier (Optional)',
    chooseTranslationModelStep4: '4. Choose Local Translation AI Model',
    defaultClarifierTitle: 'Default Context Word Clarifier Model',
    defaultClarifierDesc:
      'Validates misheard words against conversation context before translation (without restructuring sentences)',

    // Left Sidebar (Recorded Sessions)
    recordedSessions: 'Recorded Sessions',
    newBtn: 'New',
    exportSessionTooltip: 'Export session transcript',
    editSessionTooltip: 'Edit session name or audio file',
    deleteSessionTooltip: 'Delete session',
    dismiss: 'Dismiss',

    // Center TranscriptFeed
    plainTextTranscript: 'Plain Text Transcript',
    speakerTurns: 'speaker turns',
    copyForWord: 'Copy Plain Text for Word',
    copiedForWord: 'Copied for Word!',
    translated: 'Translated',
    originalSpoken: 'Original Spoken (Untranslated)',
    systemAudioLabel: 'System Audio (Teams / Huddle / Zoom)',
    noneDisableSystemAudio: 'None (Disable System Audio)',
    micInputLabel: 'Microphone Input (Optional Dual Mix)',
    noneSystemLoopbackOnly: 'None (System Loopback Only)',
    startLiveListening: 'Start Live Listening',
    stopLiveCapture: 'Stop Live Capture',
    readyForLiveTitle: 'Ready for Live Meeting Capture',
    readyForLiveDesc:
      'Select your System Audio or Microphone above and click Start Live Listening to transcribe and translate speakers in real time.',
    noAudioLoadedTitle: 'No Audio File Loaded for',
    noAudioLoadedDesc:
      'Click below (or click the Edit icon on the session card) to select an MP3 or WAV recording file for this session.',
    selectAudioInSettings: 'Select Audio File in Session Settings...',

    // Right SpeakerPanel
    speakersAndLanguages: 'Speakers & Languages',
    merge: 'Merge',
    speakerPanelHint:
      "Assign each speaker's spoken language below (e.g. Danish or Spanish). If one person was split into two speakers, click Merge to combine them.",
    noSpeakersDetected: 'No speakers detected yet.',
    noSpeakersSub:
      'Speakers will appear automatically as Speaker 1, Speaker 2, etc. during live capture or audio import.',
    spokenLanguage: 'Spoken Language:',
    retranscribingSegments: 'Re-transcribing speaker segments...',

    // SessionDialogModal (+ New / Edit)
    newRecordingSession: 'New Recording Session',
    editRecordingSession: 'Edit Recording Session',
    newSessionSub:
      'Choose between importing an audio recording file or live meeting capture',
    editSessionSub:
      'Update the session name or replace its audio recording file',
    sessionType: 'Session Type',
    lockedAfterCreation: 'Locked after creation',
    mp3AudioFile: 'MP3 / Audio File',
    mp3AudioFileDesc: 'Transcribe & translate a recorded file',
    liveMeeting: 'Live Meeting',
    liveMeetingDesc: 'Listen in on Teams, Huddle, or Mic',
    sessionName: 'Session Name',
    audioFileLabelCreate: 'Audio Recording File (.mp3, .m4a, .wav, .flac, .ogg)',
    audioFileLabelEdit: 'Audio Recording File (Optional Replace)',
    currentFile: 'Current:',
    browseOrDropAudio:
      'Click to browse or drag & drop an MP3 / M4A / WAV file here',
    leaveEmptyKeepAudio: 'Leave empty to keep the current recording audio',
    supportsFormats: 'Supports .mp3, .m4a, .wav, .flac, .ogg',
    expectedSpeakers: 'Expected Speakers in File:',
    autoDetectSpeakers: 'Auto-Detect Speakers',
    speakersCount: 'Speakers',
    cancel: 'Cancel',
    createSession: 'Create Session',
    saveChanges: 'Save Changes',

    // SettingsModal
    systemSettings: 'System Settings',
    offlineAiEngine: '100% Offline AI Engine',
    tabAppearance: 'Appearance & UI',
    tabModels: 'Local AI Models',
    tabLanguages: 'Language & Output',
    tabAudio: 'Sound & Audio Input',
    tabAbout: 'About WispAI',
    done: 'Done',
    uiLanguageTitle: 'Interface Language (UI)',
    uiLanguageDesc:
      'Change the language used across all menus, buttons, and dialogs in WispAI',
    themeTitle: 'Appearance Theme',
    themeDesc:
      'Choose Dark, Light, or automatically match your operating system theme',
    themeDark: 'Dark',
    themeLight: 'Light',
    themeSystem: 'System Match',
    localStoragePath: 'Local storage path:',
    downloadAllRecommended: 'Download All Recommended',
    step1Badge: 'Step 1: Audio In & Speakers',
    step1Desc:
      'Captures MP3 recordings or live Teams/Huddle audio, splits speech segments, and identifies who is speaking (Speaker 1, Speaker 2, etc.).',
    activeDiarizationModel: 'Active Diarization Model',
    step1ModelDesc:
      'Extracts acoustic voice embeddings locally via Sherpa-ONNX to cluster distinct voices.',
    readyOffline: 'Ready Offline',
    download: 'Download',
    step1Footer: 'Output: Speaker ID + Audio Slice',
    step2Badge: 'Step 2: Speech-to-Text Models',
    step2Desc:
      "Transcribes each speaker's audio in their spoken language. Click a downloaded model to make it active.",
    downloaded: 'Downloaded',
    step2Footer: 'Output: Spoken Language Text',
    step3Badge: 'Step 3: Local AI Translation Models',
    step3Desc:
      "Translates each speaker's transcribed text into your single chosen Output Language.",
    step3Footer: 'Final Output: Plain Text Ready for Word',
    defaultOutputLangTitle: 'Default Target Output Language',
    defaultOutputLangDesc:
      'Automatically selected when creating new recording sessions (and applied to your current session)',
    defaultSttTitle: 'Default Speech-to-Text Model',
    defaultSttDesc: 'Local Whisper model used for transcribing each speaker',
    defaultTranslationTitle: 'Default Local Translation AI Model',
    defaultTranslationDesc:
      'Specialized neural translator for converting speaker text into the output language',
    systemLoopbackTitle:
      'System Audio Output Loopback (Teams / Huddle / Zoom)',
    systemLoopbackDesc:
      'Select which speaker/headphone output WispAI should listen in on during live meetings.',
    micInputDeviceTitle: 'Microphone Input Device',
    micInputDeviceDesc:
      'Select your microphone to record your own voice alongside the meeting participants.',
    aboutSubtitle: 'Version 1.0.0 • Cross-Platform Local AI Desktop App',
    aboutP1:
      'WispAI is built for sales and project teams who need fast, accurate multi-speaker meeting transcriptions and translations without ever sending confidential customer audio to the cloud.',
    aboutP2:
      'Every step — from 3D-Speaker voice diarization to Whisper speech recognition and OPUS-MT / NLLB-200 neural translation — runs 100% locally on your Windows, macOS, or Linux machine, even completely offline.',
    zeroCloudBadge: 'Zero Cloud • 100% On-Device Privacy',
    onboardingTitle: 'Welcome to WispAI — Initial Setup',
    onboardingSub:
      'Set where WispAI stores your files and download the required offline AI models to get started.',
    onboardingStep1: '1. Storage & Preferences',
    onboardingStep2: '2. Download AI Models',
    storageLocationTitle: 'Where should WispAI store AI models & session recordings?',
    storageLocationDesc:
      'Choose a folder on your computer. WispAI will create "models/" and "sessions/" inside it.',
    browseFolder: 'Browse Folder...',
    applyPath: 'Save Path',
    pathSaved: 'Saved!',
    nextDownloadModels: 'Next: Choose & Download Models',
    back: 'Back',
    requiredSpeakerModel: '1. Speaker Recognition (Required)',
    chooseSttModel: '2. Choose Speech-to-Text Model (Whisper)',
    chooseTranslationModel: '3. Choose Local Translation AI Model',
    downloadSelectedModels: 'Download Selected Models & Setup',
    allRequiredReady: 'All required local AI models are downloaded and ready!',
    startUsingWisp: 'Start Using WispAI',
    runSetupWizard: 'Setup Wizard',
    changeFolder: 'Change Folder...',
    deleteModel: 'Delete',
    deleteModelTooltip: 'Delete model from local storage',
    retranscribeBtn: 'Re-transcribe & Translate',
    retranscribeOnSave: 'Re-transcribe & translate audio with selected models',
    sessionModelsHeader: 'AI Models & Output Language for this Session',
    privacyGuaranteeBadge: '100% Local & Offline',
    privacyGuaranteeTitle: 'Data Privacy & 100% Offline Guarantee',
    privacyGuaranteeSub:
      'Technical guarantee that your audio, transcripts, and voiceprints never leave your computer.',
    guarantee1Title: '1. Zero Data Ever Leaves This App',
    guarantee1Desc:
      'All meeting audio, speaker voiceprints, transcripts, and translations are processed strictly in local memory and stored only on your own disk. Nothing is ever uploaded to any cloud, server, or third party.',
    guarantee2Title: '2. Works 100% Offline (Air-Gapped Ready)',
    guarantee2Desc:
      'Once your chosen AI models are downloaded locally, WispAI operates completely without an internet connection. The speech and translation pipeline never initiates network calls during use.',
    guarantee3Title: '3. Enforced Localhost Firewall & Zero Telemetry',
    guarantee3Desc:
      'Enforced via Electron network request filtering and Content Security Policy (CSP): the app is technically locked to local loopback (127.0.0.1) with all telemetry disabled (DO_NOT_TRACK=1).',
    guarantee4Title: '4. Full Local Storage Ownership',
    guarantee4Desc:
      'All recordings (.wav) and transcripts (.json) live in your chosen local folder and are permanently removed from disk when deleted.',
    offlineReadyStatus: 'Offline Status: Ready for 100% Internet-Free Operation',
    offlineMissingModels: 'Offline Status: Download models in Local AI Models tab to enable offline use',
    sponsorTitle: 'Support WispAI Development',
    sponsorDesc:
      'WispAI is free, open-source, and 100% private. If WispAI saves you time in meetings, you can support its development via GitHub Sponsors.',
    sponsorBtn: 'Sponsor on GitHub',
  },
  da: {
    // Top HeaderBar
    speechToTextModel: 'Tale-til-Tekst Model',
    contextClarifierModel: 'Kontekst Ord-Validering',
    localTranslationModel: 'Lokal AI Oversættelsesmodel',
    outputLanguage: 'Outputsprog',
    downloadMoreModels: '+ Hent flere modeller...',
    settings: 'Indstillinger',
    meetingSpokenLanguagesLabel: 'Talte sprog i mødet / interviewet',
    meetingSpokenLanguagesDesc:
      'Vælg hvilke sprog samtalen foregår på (f.eks. Dansk + Engelsk), så talegenkendelse og ord-validering ved præcis, hvad de skal kigge efter – eller vælg Auto (Alle sprog).',
    autoAllLanguages: 'Auto (Alle sprog)',
    stepClarifyBadge: 'Trin 3: Kontekst Ord-Validering',
    stepClarifyDesc:
      'Analyserer hvad samtalen handler om og retter fejlhørte ord i hvert afsnit ud fra konteksten – uden at omstrukturere sætningerne.',
    stepClarifyFooter: 'Output: Kontekst-valideret tekst',
    step4TranslateBadge: 'Trin 4: Lokale AI Oversættelsesmodeller',
    chooseClarifierModel: '3. Vælg Kontekst Ord-Valideringsmodel (Valgfri)',
    chooseTranslationModelStep4: '4. Vælg Lokal AI Oversættelsesmodel',
    defaultClarifierTitle: 'Standard Kontekst Ord-Valideringsmodel',
    defaultClarifierDesc:
      'Validerer fejlhørte ord ud fra samtalens kontekst før oversættelse (uden at ændre sætningsstrukturen)',

    // Left Sidebar (Recorded Sessions)
    recordedSessions: 'Optagede Sessioner',
    newBtn: 'Ny',
    exportSessionTooltip: 'Eksporter udskrift',
    editSessionTooltip: 'Rediger sessionsnavn eller lydfil',
    deleteSessionTooltip: 'Slet session',
    dismiss: 'Luk',

    // Center TranscriptFeed
    plainTextTranscript: 'Ren Tekstudskrift',
    speakerTurns: 'taler-indlæg',
    copyForWord: 'Kopier ren tekst til Word',
    copiedForWord: 'Kopieret til Word!',
    translated: 'Oversat',
    originalSpoken: 'Originalt talt sprog (Uoversat)',
    systemAudioLabel: 'Systemlyd (Teams / Huddle / Zoom)',
    noneDisableSystemAudio: 'Ingen (Slå systemlyd fra)',
    micInputLabel: 'Mikrofonindgang (Valgfri)',
    noneSystemLoopbackOnly: 'Ingen (Kun systemlyd)',
    startLiveListening: 'Start Live Lytning',
    stopLiveCapture: 'Stop Live Optagelse',
    readyForLiveTitle: 'Klar til Live Mødeoptagelse',
    readyForLiveDesc:
      'Vælg systemlyd eller mikrofon ovenfor og klik på Start Live Lytning for at transskribere og oversætte talere i realtid.',
    noAudioLoadedTitle: 'Ingen lydfil indlæst for',
    noAudioLoadedDesc:
      'Klik nedenfor (eller på Rediger-ikonet på sessionskortet) for at vælge en MP3- eller WAV-optagelse til denne session.',
    selectAudioInSettings: 'Vælg lydfil i sessionsindstillinger...',

    // Right SpeakerPanel
    speakersAndLanguages: 'Talere & Sprog',
    merge: 'Flet',
    speakerPanelHint:
      'Tildel hver talers talte sprog nedenfor (f.eks. Dansk eller Spansk). Hvis samme person blev delt i to talere, klik Flet for at samle dem.',
    noSpeakersDetected: 'Ingen talere fundet endnu.',
    noSpeakersSub:
      'Talere vises automatisk som Speaker 1, Speaker 2 osv. under live-optagelse eller filimport.',
    spokenLanguage: 'Talt sprog:',
    retranscribingSegments: 'Gen-transskriberer talerens segmenter...',

    // SessionDialogModal (+ New / Edit)
    newRecordingSession: 'Ny Optagelsessession',
    editRecordingSession: 'Rediger Optagelsessession',
    newSessionSub:
      'Vælg mellem at importere en lydfil eller lytte med på et live-møde',
    editSessionSub: 'Opdater sessionsnavnet eller udskift lydfilen',
    sessionType: 'Sessionstype',
    lockedAfterCreation: 'Låst efter oprettelse',
    mp3AudioFile: 'MP3 / Lydfil',
    mp3AudioFileDesc: 'Transskriber og oversæt en optaget lydfil',
    liveMeeting: 'Live Møde',
    liveMeetingDesc: 'Lyt med på Teams, Huddle eller mikrofon',
    sessionName: 'Sessionsnavn',
    audioFileLabelCreate: 'Lydoptagelsesfil (.mp3, .m4a, .wav, .flac, .ogg)',
    audioFileLabelEdit: 'Lydoptagelsesfil (Valgfri udskiftning)',
    currentFile: 'Nuværende:',
    browseOrDropAudio:
      'Klik for at vælge eller træk og slip en MP3 / M4A / WAV-fil her',
    leaveEmptyKeepAudio: 'Lad feltet være tomt for at beholde nuværende lydfil',
    supportsFormats: 'Understøtter .mp3, .m4a, .wav, .flac, .ogg',
    expectedSpeakers: 'Forventet antal talere i filen:',
    autoDetectSpeakers: 'Registrer automatisk talere',
    speakersCount: 'Talere',
    cancel: 'Annuller',
    createSession: 'Opret Session',
    saveChanges: 'Gem Ændringer',

    // SettingsModal
    systemSettings: 'Systemindstillinger',
    offlineAiEngine: '100% Lokal Offline AI',
    tabAppearance: 'Udseende & UI-sprog',
    tabModels: 'Lokale AI Modeller',
    tabLanguages: 'Sprog & Output',
    tabAudio: 'Lyd Ind & Ud',
    tabAbout: 'Om WispAI',
    done: 'Færdig',
    uiLanguageTitle: 'Brugerfladesprog (UI)',
    uiLanguageDesc:
      'Vælg sproget for alle menuer, knapper og dialoger i WispAI',
    themeTitle: 'Tema & Udseende',
    themeDesc:
      'Vælg Mørk, Lys eller følg automatisk dit operativsystems tema',
    themeDark: 'Mørk (Dark)',
    themeLight: 'Lys (Light)',
    themeSystem: 'Systemstandard',
    localStoragePath: 'Lokal modelmappe:',
    downloadAllRecommended: 'Hent Alle Anbefalede',
    step1Badge: 'Trin 1: Lyd Ind & Talere',
    step1Desc:
      'Opfanger MP3-optagelser eller live Teams/Huddle-lyd, opdeler talesegmenter og genkender hvem der taler (Speaker 1, Speaker 2 osv.).',
    activeDiarizationModel: 'Aktiv Talergenkendelsesmodel',
    step1ModelDesc:
      'Udtrækker stemmeaftryk lokalt via Sherpa-ONNX for at adskille forskellige stemmer.',
    readyOffline: 'Klar Offline',
    download: 'Hent',
    step1Footer: 'Output: Taler-ID + Lydbid',
    step2Badge: 'Trin 2: Tale-til-Tekst Modeller',
    step2Desc:
      'Transskriberer hver talers lyd på deres talte sprog. Klik på en hentet model for at aktivere den.',
    downloaded: 'Hentet',
    step2Footer: 'Output: Tekst på talt sprog',
    step3Badge: 'Trin 3: Lokale AI Oversættelsesmodeller',
    step3Desc:
      'Oversætter hver talers transskriberede tekst til dit valgte outputsprog.',
    step3Footer: 'Færdigt Output: Ren tekst klar til Word',
    defaultOutputLangTitle: 'Standard Outputsprog',
    defaultOutputLangDesc:
      'Vælges automatisk ved oprettelse af nye sessioner (og anvendes på den aktuelle session)',
    defaultSttTitle: 'Standard Tale-til-Tekst Model',
    defaultSttDesc: 'Lokal Whisper-model til transskribering af hver taler',
    defaultTranslationTitle: 'Standard Lokal AI Oversættelsesmodel',
    defaultTranslationDesc:
      'Specialiseret neural oversætter til at oversætte talernes tekst til outputsproget',
    systemLoopbackTitle: 'Systemlyd Loopback (Teams / Huddle / Zoom)',
    systemLoopbackDesc:
      'Vælg hvilken højttaler/hovedtelefon WispAI skal lytte med på under live-møder.',
    micInputDeviceTitle: 'Mikrofonindgang',
    micInputDeviceDesc:
      'Vælg din mikrofon for at optage din egen stemme sammen med mødedeltagerne.',
    aboutSubtitle: 'Version 1.0.0 • Lokal AI Desktop App',
    aboutP1:
      'WispAI er bygget til salgs- og projektteams, der har brug for hurtig og præcis transskribering og oversættelse af møder med flere talere – uden nogensinde at sende fortrolig kundelyd i skyen.',
    aboutP2:
      'Hvert trin – fra 3D-Speaker stemmegenkendelse til Whisper talegenkendelse og OPUS-MT / NLLB-200 neural oversættelse – kører 100% lokalt på din computer, selv helt uden internet.',
    zeroCloudBadge: 'Ingen Cloud • 100% Lokal Privatlivsbeskyttelse',
    onboardingTitle: 'Velkommen til WispAI — Førstegangsopsætning',
    onboardingSub:
      'Vælg hvor WispAI skal gemme indhold og hent de nødvendige lokale AI-modeller for at komme i gang.',
    onboardingStep1: '1. Lagringsmappe & Udseende',
    onboardingStep2: '2. Hent AI-Modeller',
    storageLocationTitle: 'Hvor skal WispAI gemme AI-modeller og mødeoptagelser?',
    storageLocationDesc:
      'Vælg en mappe på din computer. WispAI opretter automatisk "models/" og "sessions/" i mappen.',
    browseFolder: 'Vælg mappe...',
    applyPath: 'Gem sti',
    pathSaved: 'Gemt!',
    nextDownloadModels: 'Næste: Vælg & Hent Modeller',
    back: 'Tilbage',
    requiredSpeakerModel: '1. Talergenkendelse (Påkrævet)',
    chooseSttModel: '2. Vælg Tale-til-Tekst Model (Whisper)',
    chooseTranslationModel: '3. Vælg Lokal AI Oversættelsesmodel',
    downloadSelectedModels: 'Hent Valgte Modeller & Gør Klar',
    allRequiredReady: 'Alle nødvendige lokale AI-modeller er hentet og klar!',
    startUsingWisp: 'Start WispAI',
    runSetupWizard: 'Opsætningsguide',
    changeFolder: 'Skift mappe...',
    deleteModel: 'Slet',
    deleteModelTooltip: 'Slet model fra lokal lagring',
    retranscribeBtn: 'Transskriber & oversæt på ny',
    retranscribeOnSave: 'Transskriber & oversæt lyden på ny med valgte modeller',
    sessionModelsHeader: 'AI-Modeller & Outputsprog for denne session',
    privacyGuaranteeBadge: '100% Lokal & Offline',
    privacyGuaranteeTitle: 'Privatlivs- & 100% Offline-Garanti',
    privacyGuaranteeSub:
      'Teknisk garanti for at dine lydfiler, udskrifter og stemmeaftryk aldrig forlader din computer.',
    guarantee1Title: '1. Ingen data forlader nogensinde appen',
    guarantee1Desc:
      'Al møde-lyd, stemmeaftryk, transskriberinger og oversættelser behandles udelukkende lokalt på din maskine og gemmes kun på din egen disk. Intet sendes til skyen, eksterne servere eller tredjeparter.',
    guarantee2Title: '2. Kan bruges 100% Offline uden internet',
    guarantee2Desc:
      'Så længe dine valgte AI-modeller ligger lokalt på computeren, fungerer WispAI helt uden internetforbindelse (f.eks. i flytilstand). Talegenkendelses- og oversættelsesmotoren foretager aldrig netværkskald under brug.',
    guarantee3Title: '3. Teknisk Netværksspærring & Nul Telemetri',
    guarantee3Desc:
      'Håndhævet via Electron netværksfilter og Content Security Policy (CSP): appen er teknisk låst til lokal loopback (127.0.0.1), og al telemetri/sporing er slået helt fra (DO_NOT_TRACK=1).',
    guarantee4Title: '4. Fuld kontrol over egne lokale filer',
    guarantee4Desc:
      'Alle optagelser (.wav) og udskrifter (.json) ligger i din egen valgte mappe på computeren og slettes permanent fra disken, når du sletter dem i appen.',
    offlineReadyStatus: 'Offline-status: Klar til 100% offline brug uden internet',
    offlineMissingModels: 'Offline-status: Hent modeller under Lokale AI Modeller for 100% offline brug',
    sponsorTitle: 'Støt udviklingen af WispAI',
    sponsorDesc:
      'WispAI er gratis, open-source og 100% privat. Hvis WispAI sparer dig tid i møder, kan du støtte projektet via GitHub Sponsors.',
    sponsorBtn: 'Støt via GitHub Sponsors',
  },
  es: {
    speechToTextModel: 'Modelo de Voz a Texto',
    contextClarifierModel: 'Validador Contextual de Palabras',
    localTranslationModel: 'Modelo IA de Traducción Local',
    outputLanguage: 'Idioma de Salida',
    downloadMoreModels: '+ Descargar más modelos...',
    settings: 'Ajustes',
    meetingSpokenLanguagesLabel: 'Idiomas hablados en la reunión / entrevista',
    meetingSpokenLanguagesDesc:
      'Seleccione los idiomas hablados en la entrevista (p. ej. Danés + Inglés) o Auto (Todos los idiomas).',
    autoAllLanguages: 'Auto (Todos los idiomas)',
    stepClarifyBadge: 'Paso 3: Validador Contextual de Palabras',
    stepClarifyDesc:
      'Valida y corrige palabras mal escuchadas según el contexto de la conversación sin reestructurar las frases.',
    stepClarifyFooter: 'Salida: Texto validado por contexto',
    step4TranslateBadge: 'Paso 4: Modelos IA de Traducción Local',
    chooseClarifierModel: '3. Elegir Validador Contextual de Palabras',
    chooseTranslationModelStep4: '4. Elegir Modelo IA de Traducción Local',
    defaultClarifierTitle: 'Modelo Validador Contextual Predeterminado',
    defaultClarifierDesc:
      'Valida palabras mal escuchadas según el contexto antes de traducir',
    recordedSessions: 'Sesiones Grabadas',
    newBtn: 'Nueva',
    exportSessionTooltip: 'Exportar transcripción',
    editSessionTooltip: 'Editar nombre o archivo de audio',
    deleteSessionTooltip: 'Eliminar sesión',
    dismiss: 'Cerrar',
    plainTextTranscript: 'Transcripción en Texto Plano',
    speakerTurns: 'turnos de palabra',
    copyForWord: 'Copiar Texto para Word',
    copiedForWord: '¡Copiado para Word!',
    translated: 'Traducido',
    originalSpoken: 'Idioma Original (Sin traducir)',
    systemAudioLabel: 'Audio del Sistema (Teams / Zoom)',
    noneDisableSystemAudio: 'Ninguno (Desactivar audio del sistema)',
    micInputLabel: 'Entrada de Micrófono',
    noneSystemLoopbackOnly: 'Ninguno (Solo audio del sistema)',
    startLiveListening: 'Iniciar Escucha en Vivo',
    stopLiveCapture: 'Detener Captura en Vivo',
    readyForLiveTitle: 'Listo para Captura de Reunión en Vivo',
    readyForLiveDesc:
      'Seleccione el audio del sistema o micrófono arriba y haga clic en Iniciar Escucha en Vivo.',
    noAudioLoadedTitle: 'Sin archivo de audio para',
    noAudioLoadedDesc:
      'Haga clic abajo (o en el icono Editar de la sesión) para elegir un archivo MP3 o WAV.',
    selectAudioInSettings: 'Seleccionar Archivo de Audio...',
    speakersAndLanguages: 'Hablantes e Idiomas',
    merge: 'Fusionar',
    speakerPanelHint:
      'Asigne el idioma hablado de cada participante abajo. Si una persona fue dividida en dos, haga clic en Fusionar.',
    noSpeakersDetected: 'Aún no se han detectado hablantes.',
    noSpeakersSub:
      'Los hablantes aparecerán automáticamente como Speaker 1, Speaker 2, etc.',
    spokenLanguage: 'Idioma hablado:',
    retranscribingSegments: 'Re-transcribiendo segmentos...',
    newRecordingSession: 'Nueva Sesión de Grabación',
    editRecordingSession: 'Editar Sesión de Grabación',
    newSessionSub:
      'Elija entre importar un archivo de audio o capturar una reunión en vivo',
    editSessionSub: 'Actualice el nombre de la sesión o reemplace el archivo',
    sessionType: 'Tipo de Sesión',
    lockedAfterCreation: 'Bloqueado tras creación',
    mp3AudioFile: 'Archivo MP3 / Audio',
    mp3AudioFileDesc: 'Transcribir y traducir un archivo grabado',
    liveMeeting: 'Reunión en Vivo',
    liveMeetingDesc: 'Escuchar Teams, Huddle o Micrófono',
    sessionName: 'Nombre de la Sesión',
    audioFileLabelCreate: 'Archivo de Audio (.mp3, .wav, .flac, .ogg)',
    audioFileLabelEdit: 'Archivo de Audio (Reemplazo opcional)',
    currentFile: 'Actual:',
    browseOrDropAudio: 'Haga clic o arrastre un archivo MP3 / WAV aquí',
    leaveEmptyKeepAudio: 'Dejar vacío para conservar el audio actual',
    supportsFormats: 'Soporta .mp3, .wav, .flac, .ogg',
    expectedSpeakers: 'Hablantes esperados en el archivo:',
    autoDetectSpeakers: 'Detectar automáticamente',
    speakersCount: 'Hablantes',
    cancel: 'Cancelar',
    createSession: 'Crear Sesión',
    saveChanges: 'Guardar Cambios',
    systemSettings: 'Ajustes del Sistema',
    offlineAiEngine: 'Motor IA 100% Local sin Internet',
    tabAppearance: 'Apariencia e Idioma UI',
    tabModels: 'Modelos IA Locales',
    tabLanguages: 'Idioma y Salida',
    tabAudio: 'Audio de Entrada / Salida',
    tabAbout: 'Acerca de WispAI',
    done: 'Listo',
    uiLanguageTitle: 'Idioma de la Interfaz (UI)',
    uiLanguageDesc:
      'Cambie el idioma de todos los menús, botones y diálogos de WispAI',
    themeTitle: 'Tema de Apariencia',
    themeDesc:
      'Elija Oscuro, Claro o sincronizar automáticamente con el sistema',
    themeDark: 'Oscuro',
    themeLight: 'Claro',
    themeSystem: 'Igualar al Sistema',
    localStoragePath: 'Ruta de almacenamiento local:',
    downloadAllRecommended: 'Descargar Recomendados',
    step1Badge: 'Paso 1: Audio y Hablantes',
    step1Desc:
      'Captura grabaciones MP3 o audio en vivo y separa quién habla (Speaker 1, Speaker 2, etc.).',
    activeDiarizationModel: 'Modelo de Diarización Activo',
    step1ModelDesc:
      'Extrae huellas de voz localmente mediante Sherpa-ONNX.',
    readyOffline: 'Listo Offline',
    download: 'Descargar',
    step1Footer: 'Salida: ID de Hablante + Segmento',
    step2Badge: 'Paso 2: Modelos de Voz a Texto',
    step2Desc:
      'Transcribe el audio de cada hablante en su idioma hablado.',
    downloaded: 'Descargado',
    step2Footer: 'Salida: Texto en idioma original',
    step3Badge: 'Paso 3: Modelos IA de Traducción Local',
    step3Desc:
      'Traduce el texto de cada hablante al idioma de salida elegido.',
    step3Footer: 'Salida Final: Texto plano listo para Word',
    defaultOutputLangTitle: 'Idioma de Salida Predeterminado',
    defaultOutputLangDesc:
      'Seleccionado automáticamente al crear nuevas sesiones',
    defaultSttTitle: 'Modelo de Voz a Texto Predeterminado',
    defaultSttDesc: 'Modelo Whisper local para transcribir a cada hablante',
    defaultTranslationTitle: 'Modelo IA de Traducción Predeterminado',
    defaultTranslationDesc: 'Traductor neuronal local hacia el idioma de salida',
    systemLoopbackTitle: 'Audio del Sistema (Teams / Huddle / Zoom)',
    systemLoopbackDesc:
      'Seleccione qué salida de altavoz/auriculares debe escuchar WispAI.',
    micInputDeviceTitle: 'Dispositivo de Micrófono',
    micInputDeviceDesc:
      'Seleccione su micrófono para grabar su propia voz.',
    aboutSubtitle: 'Versión 1.0.0 • App de Escritorio IA 100% Local',
    aboutP1:
      'WispAI está diseñado para equipos que necesitan transcripciones y traducciones de reuniones precisas sin enviar audio confidencial a la nube.',
    aboutP2:
      'Cada paso se ejecuta 100% localmente en su equipo Windows, macOS o Linux, incluso sin conexión a internet.',
    zeroCloudBadge: 'Cero Nube • 100% Privacidad en el Dispositivo',
    onboardingTitle: 'Bienvenido a WispAI — Configuración Inicial',
    onboardingSub:
      'Elija dónde guarda WispAI sus datos y descargue los modelos de IA locales necesarios.',
    onboardingStep1: '1. Carpeta y Preferencias',
    onboardingStep2: '2. Descargar Modelos IA',
    storageLocationTitle: '¿Dónde debe guardar WispAI los modelos y grabaciones?',
    storageLocationDesc:
      'Elija una carpeta en su equipo. WispAI creará "models/" y "sessions/" dentro de ella.',
    browseFolder: 'Elegir carpeta...',
    applyPath: 'Guardar ruta',
    pathSaved: '¡Guardado!',
    nextDownloadModels: 'Siguiente: Elegir y Descargar Modelos',
    back: 'Atrás',
    requiredSpeakerModel: '1. Reconocimiento de Hablantes (Obligatorio)',
    chooseSttModel: '2. Elegir Modelo de Voz a Texto (Whisper)',
    chooseTranslationModel: '3. Elegir Modelo IA de Traducción Local',
    downloadSelectedModels: 'Descargar Modelos Seleccionados',
    allRequiredReady: '¡Todos los modelos locales requeridos están listos!',
    startUsingWisp: 'Empezar a usar WispAI',
    runSetupWizard: 'Asistente de Configuración',
    changeFolder: 'Cambiar carpeta...',
    deleteModel: 'Eliminar',
    deleteModelTooltip: 'Eliminar modelo del almacenamiento local',
    retranscribeBtn: 'Re-transcribir y Traducir',
    retranscribeOnSave: 'Re-transcribir y traducir el audio con los modelos seleccionados',
    sessionModelsHeader: 'Modelos IA e Idioma de Salida para esta sesión',
    privacyGuaranteeBadge: '100% Local y Offline',
    privacyGuaranteeTitle: 'Garantía de Privacidad y Uso 100% Offline',
    privacyGuaranteeSub:
      'Garantía técnica de que sus grabaciones y transcripciones nunca salen de su equipo.',
    guarantee1Title: '1. Ningún dato sale jamás de la aplicación',
    guarantee1Desc:
      'Todo el audio, las voces, transcripciones y traducciones se procesan únicamente en su equipo local y nunca se envían a servidores externos.',
    guarantee2Title: '2. Funciona 100% sin conexión a Internet',
    guarantee2Desc:
      'Mientras tenga los modelos descargados localmente, WispAI funciona completamente sin conexión a internet.',
    guarantee3Title: '3. Bloqueo de Red Local (127.0.0.1) y Cero Telemetría',
    guarantee3Desc:
      'Bloqueado técnicamente mediante filtro de red Electron y CSP para comunicarse solo con 127.0.0.1 sin telemetría.',
    guarantee4Title: '4. Control total de sus archivos locales',
    guarantee4Desc:
      'Todas las grabaciones y sesiones se guardan en su carpeta local elegida.',
    offlineReadyStatus: 'Estado Offline: Listo para uso 100% sin internet',
    offlineMissingModels: 'Estado Offline: Descargue modelos para uso sin internet',
    sponsorTitle: 'Apoyar el desarrollo de WispAI',
    sponsorDesc:
      'WispAI es gratuito, de código abierto y 100% privado. Puedes apoyar el proyecto mediante GitHub Sponsors.',
    sponsorBtn: 'Patrocinar en GitHub',
  },
  de: {
    speechToTextModel: 'Sprache-zu-Text Modell',
    contextClarifierModel: 'Kontext-Wortvalidierung',
    localTranslationModel: 'Lokales KI-Übersetzungsmodell',
    outputLanguage: 'Ausgabesprache',
    downloadMoreModels: '+ Weitere Modelle laden...',
    settings: 'Einstellungen',
    meetingSpokenLanguagesLabel: 'Gesprochene Sprachen im Meeting / Interview',
    meetingSpokenLanguagesDesc:
      'Wählen Sie die gesprochenen Sprachen (z. B. Dänisch + Englisch) oder Auto (Alle Sprachen).',
    autoAllLanguages: 'Auto (Alle Sprachen)',
    stepClarifyBadge: 'Schritt 3: Kontext-Wortvalidierung',
    stepClarifyDesc:
      'Überprüft und korrigiert falsch verstandene Wörter anhand des Gesprächskontexts, ohne Sätze umzustrukturieren.',
    stepClarifyFooter: 'Ausgabe: Kontextvalidierter Text',
    step4TranslateBadge: 'Schritt 4: Lokale KI-Übersetzungsmodelle',
    chooseClarifierModel: '3. Kontext-Wortvalidierungsmodell wählen',
    chooseTranslationModelStep4: '4. Lokales KI-Übersetzungsmodell wählen',
    defaultClarifierTitle: 'Standard Kontext-Wortvalidierungsmodell',
    defaultClarifierDesc:
      'Validiert falsch gehörte Wörter anhand des Gesprächskontexts vor der Übersetzung',
    recordedSessions: 'Aufgezeichnete Sitzungen',
    newBtn: 'Neu',
    exportSessionTooltip: 'Transkript exportieren',
    editSessionTooltip: 'Sitzungsname oder Audiodatei bearbeiten',
    deleteSessionTooltip: 'Sitzung löschen',
    dismiss: 'Schließen',
    plainTextTranscript: 'Klartext-Transkript',
    speakerTurns: 'Sprecherbeiträge',
    copyForWord: 'Text für Word kopieren',
    copiedForWord: 'Für Word kopiert!',
    translated: 'Übersetzt',
    originalSpoken: 'Originalsprache (Unübersetzt)',
    systemAudioLabel: 'Systemaudio (Teams / Huddle / Zoom)',
    noneDisableSystemAudio: 'Keins (Systemaudio deaktivieren)',
    micInputLabel: 'Mikrofoneingang',
    noneSystemLoopbackOnly: 'Keins (Nur Systemaudio)',
    startLiveListening: 'Live-Mithören starten',
    stopLiveCapture: 'Live-Aufnahme stoppen',
    readyForLiveTitle: 'Bereit für Live-Meeting-Aufnahme',
    readyForLiveDesc:
      'Wählen Sie oben Systemaudio oder Mikrofon und klicken Sie auf Live-Mithören starten.',
    noAudioLoadedTitle: 'Keine Audiodatei geladen für',
    noAudioLoadedDesc:
      'Klicken Sie unten (oder auf Bearbeiten), um eine MP3- oder WAV-Datei auszuwählen.',
    selectAudioInSettings: 'Audiodatei in Sitzungseinstellungen wählen...',
    speakersAndLanguages: 'Sprecher & Sprachen',
    merge: 'Zusammenführen',
    speakerPanelHint:
      'Weisen Sie unten jedem Sprecher die gesprochene Sprache zu. Klicken Sie auf Zusammenführen, falls eine Person doppelt erkannt wurde.',
    noSpeakersDetected: 'Noch keine Sprecher erkannt.',
    noSpeakersSub:
      'Sprecher erscheinen automatisch als Speaker 1, Speaker 2 usw.',
    spokenLanguage: 'Gesprochene Sprache:',
    retranscribingSegments: 'Segmente werden neu transkribiert...',
    newRecordingSession: 'Neue Aufnahmesitzung',
    editRecordingSession: 'Aufnahmesitzung bearbeiten',
    newSessionSub:
      'Wählen Sie zwischen dem Import einer Audiodatei oder einer Live-Aufnahme',
    editSessionSub: 'Sitzungsname ändern oder Audiodatei ersetzen',
    sessionType: 'Sitzungstyp',
    lockedAfterCreation: 'Nach Erstellung gesperrt',
    mp3AudioFile: 'MP3 / Audiodatei',
    mp3AudioFileDesc: 'Aufgezeichnete Datei transkribieren & übersetzen',
    liveMeeting: 'Live-Meeting',
    liveMeetingDesc: 'Teams, Huddle oder Mikrofon mithören',
    sessionName: 'Sitzungsname',
    audioFileLabelCreate: 'Audiodatei (.mp3, .wav, .flac, .ogg)',
    audioFileLabelEdit: 'Audiodatei (Optional ersetzen)',
    currentFile: 'Aktuell:',
    browseOrDropAudio: 'Klicken oder MP3 / WAV-Datei hierher ziehen',
    leaveEmptyKeepAudio: 'Leer lassen, um aktuelles Audio zu behalten',
    supportsFormats: 'Unterstützt .mp3, .wav, .flac, .ogg',
    expectedSpeakers: 'Erwartete Sprecher in Datei:',
    autoDetectSpeakers: 'Sprecher automatisch erkennen',
    speakersCount: 'Sprecher',
    cancel: 'Abbrechen',
    createSession: 'Sitzung erstellen',
    saveChanges: 'Änderungen speichern',
    systemSettings: 'Systemeinstellungen',
    offlineAiEngine: '100% Lokale Offline-KI',
    tabAppearance: 'Darstellung & UI-Sprache',
    tabModels: 'Lokale KI-Modelle',
    tabLanguages: 'Sprache & Ausgabe',
    tabAudio: 'Audio Ein- & Ausgang',
    tabAbout: 'Über WispAI',
    done: 'Fertig',
    uiLanguageTitle: 'Oberflächensprache (UI)',
    uiLanguageDesc:
      'Sprache für alle Menüs, Schaltflächen und Dialoge in WispAI ändern',
    themeTitle: 'Erscheinungsbild (Theme)',
    themeDesc:
      'Dunkel, Hell oder automatisch an das Betriebssystem anpassen',
    themeDark: 'Dunkel',
    themeLight: 'Hell',
    themeSystem: 'Systemstandard',
    localStoragePath: 'Lokaler Modellpfad:',
    downloadAllRecommended: 'Alle Empfohlenen laden',
    step1Badge: 'Schritt 1: Audio & Sprecher',
    step1Desc:
      'Erfasst MP3-Aufnahmen oder Live-Audio und erkennt, wer spricht (Speaker 1, Speaker 2 usw.).',
    activeDiarizationModel: 'Aktives Sprechererkennungsmodell',
    step1ModelDesc:
      'Extrahiert Stimmmerkmale lokal über Sherpa-ONNX.',
    readyOffline: 'Offline bereit',
    download: 'Laden',
    step1Footer: 'Ausgabe: Sprecher-ID + Audioabschnitt',
    step2Badge: 'Schritt 2: Sprache-zu-Text Modelle',
    step2Desc:
      'Transkribiert das Audio jedes Sprechers in seiner gesprochenen Sprache.',
    downloaded: 'Geladen',
    step2Footer: 'Ausgabe: Text in Originalsprache',
    step3Badge: 'Schritt 3: Lokale KI-Übersetzungsmodelle',
    step3Desc:
      'Übersetzt den Text jedes Sprechers in Ihre gewählte Ausgabesprache.',
    step3Footer: 'Endergebnis: Klartext bereit für Word',
    defaultOutputLangTitle: 'Standard-Ausgabesprache',
    defaultOutputLangDesc:
      'Wird beim Erstellen neuer Sitzungen automatisch ausgewählt',
    defaultSttTitle: 'Standard Sprache-zu-Text Modell',
    defaultSttDesc: 'Lokales Whisper-Modell zur Transkription',
    defaultTranslationTitle: 'Standard KI-Übersetzungsmodell',
    defaultTranslationDesc: 'Lokaler neuronaler Übersetzer für die Ausgabesprache',
    systemLoopbackTitle: 'Systemaudio-Loopback (Teams / Huddle / Zoom)',
    systemLoopbackDesc:
      'Wählen Sie den Lautsprecher-/Kopfhörerausgang für Live-Meetings.',
    micInputDeviceTitle: 'Mikrofoneingang',
    micInputDeviceDesc:
      'Wählen Sie Ihr Mikrofon, um Ihre eigene Stimme mitaufzuzeichnen.',
    aboutSubtitle: 'Version 1.0.0 • Lokale KI-Desktop-App',
    aboutP1:
      'WispAI wurde für Teams entwickelt, die schnelle Multi-Speaker-Transkriptionen und Übersetzungen benötigen, ohne vertrauliche Audiodaten in die Cloud zu senden.',
    aboutP2:
      'Jeder Schritt läuft zu 100 % lokal auf Ihrem Computer – auch komplett offline.',
    zeroCloudBadge: 'Zero Cloud • 100% Datenschutz auf dem Gerät',
    onboardingTitle: 'Willkommen bei WispAI — Ersteinrichtung',
    onboardingSub:
      'Legen Sie den Speicherort fest und laden Sie die erforderlichen lokalen KI-Modelle herunter.',
    onboardingStep1: '1. Speicherort & Darstellung',
    onboardingStep2: '2. KI-Modelle herunterladen',
    storageLocationTitle: 'Wo soll WispAI KI-Modelle und Sitzungen speichern?',
    storageLocationDesc:
      'Wählen Sie einen Ordner auf Ihrem Computer. WispAI erstellt darin "models/" und "sessions/".',
    browseFolder: 'Ordner wählen...',
    applyPath: 'Pfad speichern',
    pathSaved: 'Gespeichert!',
    nextDownloadModels: 'Weiter: Modelle auswählen & laden',
    back: 'Zurück',
    requiredSpeakerModel: '1. Sprechererkennung (Erforderlich)',
    chooseSttModel: '2. Sprache-zu-Text Modell wählen (Whisper)',
    chooseTranslationModel: '3. Lokales KI-Übersetzungsmodell wählen',
    downloadSelectedModels: 'Ausgewählte Modelle herunterladen',
    allRequiredReady: 'Alle erforderlichen lokalen KI-Modelle sind bereit!',
    startUsingWisp: 'WispAI starten',
    runSetupWizard: 'Einrichtungsassistent',
    changeFolder: 'Ordner ändern...',
    deleteModel: 'Löschen',
    deleteModelTooltip: 'Modell vom lokalen Speicher löschen',
    retranscribeBtn: 'Neu transkribieren & übersetzen',
    retranscribeOnSave: 'Audio mit ausgewählten Modellen neu transkribieren & übersetzen',
    sessionModelsHeader: 'KI-Modelle & Ausgabesprache für diese Sitzung',
    privacyGuaranteeBadge: '100% Lokal & Offline',
    privacyGuaranteeTitle: 'Datenschutz- & 100% Offline-Garantie',
    privacyGuaranteeSub:
      'Technische Garantie, dass Ihre Audiodaten und Transkripte niemals Ihren Computer verlassen.',
    guarantee1Title: '1. Keine Daten verlassen jemals diese App',
    guarantee1Desc:
      'Alle Audioaufnahmen, Stimmprofile, Transkripte und Übersetzungen werden ausschließlich lokal auf Ihrem Gerät verarbeitet.',
    guarantee2Title: '2. Funktioniert zu 100% Offline ohne Internet',
    guarantee2Desc:
      'Sobald Ihre gewählten KI-Modelle lokal gespeichert sind, funktioniert WispAI komplett ohne Internetverbindung.',
    guarantee3Title: '3. Technische Netzwerksperre (127.0.0.1) & Null Telemetrie',
    guarantee3Desc:
      'Durch Electron-Netzwerkfilter und CSP technisch auf 127.0.0.1 beschränkt, jegliche Telemetrie ist deaktiviert.',
    guarantee4Title: '4. Volle Kontrolle über lokale Dateien',
    guarantee4Desc:
      'Alle Aufnahmen werden nur in Ihrem selbst gewählten lokalen Ordner gespeichert.',
    offlineReadyStatus: 'Offline-Status: Bereit für 100% Offline-Betrieb',
    offlineMissingModels: 'Offline-Status: Bitte Modelle für Offline-Betrieb laden',
    sponsorTitle: 'WispAI-Entwicklung unterstützen',
    sponsorDesc:
      'WispAI ist kostenlos, Open-Source und 100% privat. Unterstützen Sie die Weiterentwicklung über GitHub Sponsors.',
    sponsorBtn: 'Auf GitHub sponsern',
  },
};

interface I18nContextValue {
  uiLanguage: UILanguage;
  setUiLanguage: (lang: UILanguage) => void;
  theme: ThemeMode;
  setTheme: (theme: ThemeMode) => void;
  t: (key: string) => string;
}

export const I18nContext = createContext<I18nContextValue>({
  uiLanguage: 'en',
  setUiLanguage: () => {},
  theme: 'system',
  setTheme: () => {},
  t: (k) => TRANSLATIONS.en[k] || k,
});

export function useI18n() {
  return useContext(I18nContext);
}
