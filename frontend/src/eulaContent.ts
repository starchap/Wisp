import { UILanguage } from './i18n';

export interface EulaSection {
  title: string;
  paragraphs?: string[];
  items?: string[];
}

export interface EulaDocument {
  version: string;
  effectiveDate: string;
  sections: EulaSection[];
}

export const EULA_VERSION = '1.0';
export const EULA_STORAGE_KEY = 'wisp_eula_accepted_v1';

export function isEulaAccepted(): boolean {
  try {
    const raw = localStorage.getItem(EULA_STORAGE_KEY);
    if (!raw) return false;
    if (raw === 'true') return true;
    const parsed = JSON.parse(raw);
    return parsed?.accepted === true && parsed?.version === EULA_VERSION;
  } catch {
    return false;
  }
}

export function saveEulaAcceptance(language: UILanguage): void {
  try {
    localStorage.setItem(
      EULA_STORAGE_KEY,
      JSON.stringify({
        accepted: true,
        version: EULA_VERSION,
        acceptedAt: new Date().toISOString(),
        language,
      })
    );
  } catch {
    localStorage.setItem(EULA_STORAGE_KEY, 'true');
  }
}

export const EULA_DOCUMENTS: Record<UILanguage, EulaDocument> = {
  en: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definitions, Scope & Legal Capacity',
        paragraphs: [
          'This End-User License Agreement ("EULA") is a binding legal agreement between you (an individual or legal entity, "User" or "You") and the authors, maintainers, and contributors of WispAI ("Developers" or "We") governing your download, installation, configuration, and use of the packaged WispAI desktop application, embedded local backend, documentation, and related files (collectively, "the Software").',
          'By clicking "Accept & Continue", installing, or using the Software, you represent and warrant that: (a) you have reached the legal age of majority in your jurisdiction (or have obtained parent/guardian consent); (b) if you accept on behalf of a company, employer, or other legal entity, you have full legal authority to bind that entity to this EULA; and (c) you have read, understood, and agree to be bound by all terms herein. If you do not agree, you must not install or use the Software.',
        ],
      },
      {
        title: '2. License Grant & Relationship to the MIT License',
        paragraphs: [
          'WispAI is distributed free of charge. The WispAI source code is licensed under the MIT License (Copyright © 2026 starchap). Subject to your compliance with this EULA and the MIT License, you are granted a free, non-exclusive, worldwide right to download, install, and use the WispAI Software itself for personal, academic, or commercial purposes.',
          'Nothing in this EULA restricts or contradicts any rights granted to you under the MIT License with respect to the WispAI source code. In the event of any conflict concerning source code copying, modification, sublicensing, or redistribution, the MIT License governs source code rights, while this EULA governs end-user use of the packaged WispAI application, disclaimers, and allocation of risk.',
        ],
      },
      {
        title: '3. Data Backup Responsibility, Due Care & Safe Usage',
        paragraphs: [
          'Prior to importing, converting, transcribing, or otherwise processing any audio, video, or document files with WispAI, you are strongly advised to maintain an independent, verified backup copy of all important files on separate storage or backup media.',
          'You agree to exercise ordinary due care when configuring and operating the Software, including:',
        ],
        items: [
          'Never selecting operating system directories (e.g., Windows, Program Files, System32, /System, /usr) or an entire root drive as your WispAI storage directory.',
          'Ensuring your device has sufficient free disk space before importing long recordings (e.g., 1–2 hour MP4/WAV files) or downloading large local AI models (100 MB – 2 GB per model).',
          'Verifying carefully before deleting any session, recording, or downloaded AI model, as deletion permanently removes files from your local storage.',
        ],
      },
      {
        title: '4. Transparent System, Hardware & File Access Disclosure',
        paragraphs: [
          'WispAI is engineered for 100% local, offline operation. The Software accesses only the following local resources:',
        ],
        items: [
          'User-Selected Media Files: WispAI reads only the audio and video files that you explicitly choose to import or open.',
          'User-Approved Storage Directory: WispAI downloads AI models and stores session recordings, transcripts, and settings exclusively inside the local storage directory approved by you in Settings (or the default local WispAI data directory).',
          'Audio Input Devices (Opt-In Only): Microphone and system audio loopback are accessed solely when you actively start Live Listening and cease immediately upon stopping capture.',
          'Local CPU, Memory & Disk Resources: Local AI transcription, speaker diarization, clarification, and translation are computationally intensive and may cause elevated CPU/RAM utilization, fan noise, battery drain, or heat during processing.',
          'Network & Privacy: Internet access is used strictly when you explicitly initiate a download of open-source AI models (e.g., from HuggingFace or GitHub). Your audio recordings, voiceprints, and transcripts never leave your device — zero telemetry, zero analytics, and zero cloud uploads.',
        ],
      },
      {
        title: '5. Third-Party Open-Source Components & AI Model License Restrictions',
        paragraphs: [
          'WispAI integrates and allows optional download of third-party open-source libraries, runtimes, and AI models, including but not limited to OpenAI Whisper, Syv.ai/CoRal Hviske, Alexandra Instituttet Røst, 3D-Speaker, Helsinki-NLP OPUS-MT, Meta NLLB-200, Alibaba Qwen, Microsoft ONNX Runtime, Electron, Python, React, and OS media codecs (Windows Media Foundation / AVFoundation / FFmpeg).',
          'IMPORTANT NOTICE ON COMMERCIAL USE OF THIRD-PARTY MODELS: The commercial use permission in Section 2 applies solely to the WispAI application itself and does NOT extend to optional third-party AI models that you choose to download. Certain optional models — including Meta NLLB-200 (licensed under CC-BY-NC-4.0, Non-Commercial) — carry separate license terms or non-commercial restrictions. You are solely responsible for reviewing and complying with the license of each third-party model before using it, particularly for commercial purposes. The Developers make no warranties regarding the availability, legality, or licensing of third-party models or repositories.',
        ],
      },
      {
        title: '6. Privacy, GDPR & No Data Processing Agreement (DPA)',
        paragraphs: [
          'Because WispAI operates exclusively on your local device and the Developers never receive, transmit, access, host, or store any of your audio recordings, transcripts, or personal data, the Developers are NOT a "Data Processor" (nor a joint controller) under the EU General Data Protection Regulation (GDPR) or equivalent data protection laws.',
          'You remain the sole Data Controller for all personal data processed on your device using WispAI, and no Data Processing Agreement (DPA) is formed or required between you and the Developers.',
        ],
      },
      {
        title: '7. Acceptable Use, Recording Consent & Regulatory Compliance',
        paragraphs: [
          'You are solely responsible for ensuring that your use of WispAI complies with all applicable local, national, and international laws and regulations — including data protection and privacy laws (such as GDPR), wiretapping and call-recording laws, employment/workplace rules, professional confidentiality obligations, intellectual property laws, and applicable export control or sanctions regulations.',
          'You must obtain all legally required notices, permissions, and informed consent from every participant prior to recording, transcribing, diarizing, or translating any meeting, interview, call, or media file. You may not use the Software for unlawful surveillance or any illegal purpose.',
        ],
      },
      {
        title: '8. AI Output Accuracy, No Professional Advice & High-Risk Exclusion',
        paragraphs: [
          'All transcriptions, speaker labels, grammar and punctuation corrections, conversation summaries, and translations are generated automatically by local machine-learning models and may contain inaccuracies, misheard words, hallucinations, incorrect speaker attributions, or omissions. You must always review and verify important transcripts manually before relying on, publishing, or acting upon them.',
          'The Software does not provide legal, medical, clinical, financial, or other professional advice. WispAI is NOT designed, intended, or authorized for use in high-risk, life-critical, or safety-critical environments (such as medical diagnosis, emergency response, aviation, nuclear facilities, or evidentiary legal proceedings where errors could cause injury, death, or severe legal or financial harm).',
        ],
      },
      {
        title: '9. Disclaimer of All Warranties ("AS IS" / "AS AVAILABLE")',
        paragraphs: [
          'THE SOFTWARE, INCLUDING ALL LOCAL AI PIPELINES, SCRIPTS, AND UPDATES, IS PROVIDED "AS IS" AND "AS AVAILABLE", WITH ALL FAULTS AND WITHOUT WARRANTY OF ANY KIND.',
          'TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE AUTHORS, COPYRIGHT HOLDERS, DEVELOPERS, AND CONTRIBUTORS EXPRESSLY DISCLAIM ALL WARRANTIES AND CONDITIONS, WHETHER EXPRESS, IMPLIED, STATUTORY, OR OTHERWISE, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OF MERCHANTABILITY, SATISFACTORY QUALITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, ACCURACY, DATA INTEGRITY, SYSTEM COMPATIBILITY, NON-INFRINGEMENT, OR UNINTERRUPTED, SECURE, AND ERROR-FREE OPERATION.',
        ],
      },
      {
        title: '10. Comprehensive Exclusion & Limitation of Liability',
        paragraphs: [
          'The Developers strive to deliver safe, reliable software that respects your system and never harms your device or files; any damage is strictly unintended, and WispAI is distributed free of charge.',
          'EXCEPT FOR LIABILITY THAT CANNOT LAWFULLY BE EXCLUDED OR LIMITED UNDER APPLICABLE MANDATORY LAW — SPECIFICALLY LIABILITY FOR INTENT (WILLFUL MISCONDUCT), GROSS NEGLIGENCE, DEATH OR PERSONAL INJURY, OR MANDATORY STATUTORY PRODUCT LIABILITY — IN NO EVENT SHALL THE AUTHORS, COPYRIGHT HOLDERS, DEVELOPERS, OR CONTRIBUTORS BE LIABLE TO YOU OR ANY THIRD PARTY FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE, EXEMPLARY, OR CONSEQUENTIAL DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION: LOSS, CORRUPTION, ACCIDENTAL DELETION, OR MODIFICATION OF LOCAL FILES, AUDIO/VIDEO RECORDINGS, TRANSCRIPTS, OR DATA; STORAGE DRIVE OR FILE SYSTEM DAMAGE; OPERATING SYSTEM INSTABILITY OR CRASHES; HARDWARE DAMAGE, OVERHEATING, OR WEAR; LOSS OF PROFITS, REVENUE, GOODWILL, OR BUSINESS; OR BUSINESS INTERRUPTION — ARISING OUT OF OR IN CONNECTION WITH THE DOWNLOAD, INSTALLATION, CONFIGURATION, USE, OR INABILITY TO USE THE SOFTWARE OR ANY THIRD-PARTY MODEL, WHETHER IN CONTRACT, TORT (INCLUDING ORDINARY NEGLIGENCE), STRICT LIABILITY, OR OTHERWISE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.',
          'BECAUSE THE SOFTWARE IS PROVIDED FREE OF CHARGE, TO THE EXTENT ANY RESIDUAL LIABILITY CANNOT BE EXCLUDED UNDER APPLICABLE LAW, THE DEVELOPERS’ TOTAL AGGREGATE LIABILITY FOR ALL CLAIMS COMBINED SHALL BE LIMITED TO THE MINIMUM AMOUNT PERMITTED BY LAW (ZERO / 0 DKK / $0 USD). YOU USE WISPAI AT YOUR OWN RISK.',
        ],
      },
      {
        title: '11. Indemnification & Hold Harmless',
        paragraphs: [
          'To the maximum extent permitted by applicable law, you agree to defend, indemnify, and hold harmless the Developers, authors, copyright holders, and contributors of WispAI from and against any and all third-party claims, demands, actions, damages, losses, liabilities, fines, and expenses (including reasonable legal fees) arising out of or relating to: (a) your use or misuse of the Software; (b) your failure to maintain backups of local files; (c) your failure to obtain required recording or data-processing consent; or (d) your violation of this EULA, third-party model licenses, third-party rights, or any applicable law.',
        ],
      },
      {
        title: '12. Trademarks & No Affiliation',
        paragraphs: [
          'All third-party product names, logos, and trademarks referenced in WispAI or its documentation — including Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace, and GitHub — are the property of their respective owners. WispAI is an independent open-source project and is not affiliated with, sponsored by, or endorsed by any of these organizations.',
        ],
      },
      {
        title: '13. Updates, No Support Obligation & Termination',
        paragraphs: [
          'The Developers may update, modify, suspend, or discontinue WispAI or revise this EULA at any time without notice and have no obligation to provide technical support, maintenance, bug fixes, or future updates. You may cease using and uninstall WispAI at any time. Your right to use the packaged application under this EULA terminates automatically if you breach its terms (without affecting independent source code rights under the MIT License). Sections 5, 6, 7, 8, 9, 10, 11, 12, and 14 survive termination or uninstallation.',
        ],
      },
      {
        title: '14. Governing Law, Jurisdiction, Prevailing Language & Severability',
        paragraphs: [
          'Governing Law & Jurisdiction: This EULA and any dispute arising out of or in connection with it shall be governed by and construed in accordance with the laws of Denmark, excluding its conflict-of-law rules and the UN Convention on Contracts for the International Sale of Goods (CISG). Subject to mandatory statutory consumer jurisdiction rules in your country of residence, any dispute shall be settled exclusively by the competent courts of Denmark.',
          'Prevailing Language: This EULA may be provided in multiple languages for convenience. In the event of any discrepancy or conflict in interpretation between language versions, the English version shall prevail.',
          'Severability & Entire Agreement: If any provision of this EULA is held invalid or unenforceable under mandatory law, that provision shall be enforced to the maximum extent permissible and all remaining provisions shall remain in full force and effect. Nothing in this EULA limits non-waivable statutory consumer rights. Together with the MIT License, this EULA constitutes the entire agreement regarding your use of WispAI.',
        ],
      },
    ],
  },

  da: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definitioner, Omfang & Retlig Handleevne',
        paragraphs: [
          'Denne slutbrugerlicensaftale ("EULA") er en juridisk bindende aftale mellem dig (som privatperson eller juridisk enhed, "Bruger" eller "Du") og forfatterne, vedligeholderne og bidragyderne bag WispAI ("Udviklerne" eller "Vi") vedrørende din download, installation, opsætning og brug af den distribuerede WispAI-skrivebordsapplikation, den indbyggede lokale backend, dokumentation og tilhørende filer (samlet benævnt "Softwaren").',
          'Ved at klikke "Accepter og fortsæt", installere eller anvende Softwaren erklærer og indestår du for, at: (a) du er myndig i din jurisdiktion (eller har samtykke fra forælder/værge); (b) du, såfremt aftalen accepteres på vegne af en virksomhed, arbejdsgiver eller anden juridisk enhed, har fuld bemyndigelse til at forpligte denne enhed; og (c) du har læst, forstået og accepterer samtlige vilkår nedenfor. Hvis du ikke accepterer vilkårene, må du ikke installere eller anvende Softwaren.',
        ],
      },
      {
        title: '2. Licens & Forholdet til MIT-licensen',
        paragraphs: [
          'WispAI stilles vederlagsfrit til rådighed. Kildekoden til WispAI er licenseret under MIT-licensen (Copyright © 2026 starchap). Under forudsætning af overholdelse af denne EULA og MIT-licensen gives du en gratis, ikke-eksklusiv, verdensomspændende ret til at downloade, installere og anvende selve WispAI-applikationen til privat, uddannelsesmæssig eller erhvervsmæssig brug.',
          'Intet i denne EULA begrænser eller strider mod de rettigheder, der tilkommer dig under MIT-licensen vedrørende kildekoden. Ved eventuel uoverensstemmelse om kopiering, ændring, viderelicensering eller videredistribution af kildekode har MIT-licensen forrang, mens denne EULA regulerer slutbrugerens brug af den samlede WispAI-applikation, ansvarsfraskrivelser og risikofordeling.',
        ],
      },
      {
        title: '3. Sikkerhedskopiering, Agtpågivenhed & Sikker Brug',
        paragraphs: [
          'Inden lyd-, video- eller dokumentfiler importeres, konverteres, transskriberes eller på anden måde behandles i WispAI, opfordres brugeren kraftigt til altid at opretholde en selvstændig, verificeret sikkerhedskopi (backup) af samtlige vigtige filer på et separat lagermedie.',
          'Brugeren forpligter sig til at udvise sædvanlig agtpågivenhed ved opsætning og anvendelse af Softwaren, herunder:',
        ],
        items: [
          'Aldrig at vælge styresystemmapper (fx Windows, Program Files, System32, /System, /usr) eller et helt roddrev som din WispAI-lagermappe.',
          'At sikre tilstrækkelig ledig diskplads, før lange optagelser (fx 1–2 timers MP4/WAV-filer) importeres eller store lokale AI-modeller (100 MB – 2 GB pr. model) downloades.',
          'At kontrollere grundigt, før sessioner, optagelser eller downloadede AI-modeller slettes, da sletning permanent fjerner filerne fra dit lokale lager.',
        ],
      },
      {
        title: '4. Gennemsigtig Adgang til System, Hardware & Filer',
        paragraphs: [
          'WispAI er konstrueret til 100% lokal og offline drift og tilgår udelukkende følgende lokale ressourcer:',
        ],
        items: [
          'Brugervalgte mediefiler: WispAI læser udelukkende de lyd- og videofiler, som du udtrykkeligt vælger at importere eller åbne.',
          'Brugergodkendt lagermappe: WispAI downloader kun AI-modeller og gemmer optagelser, transskriberinger og indstillinger i den lokale mappe, som du har godkendt under Indstillinger (eller standardmappen for WispAI).',
          'Lydindgange (kun ved aktivt tilvalg): Mikrofon og systemlyd (loopback) tilgås kun, når du aktivt starter Live-optagelse, og afbrydes straks, når optagelsen stoppes.',
          'Lokal CPU-, RAM- & diskbelastning: Lokal AI-transskribering, talergenkendelse, grammatikrettelse og oversættelse er beregningstungt og kan medføre høj CPU-/RAM-belastning, blæserstøj, batteriforbrug eller varmeudvikling under behandling.',
          'Netværk & privatliv: Internetforbindelse benyttes udelukkende, når du selv initierer download af open-source AI-modeller (fx fra HuggingFace eller GitHub). Dine lydoptagelser, stemmeprofiler og transskriberinger forlader aldrig din enhed — nul telemetri, nul sporing og nul cloud-upload.',
        ],
      },
      {
        title: '5. Tredjepartskomponenter & Licensbegrænsninger for AI-Modeller',
        paragraphs: [
          'WispAI integrerer og giver mulighed for valgfri download af tredjeparts open-source-biblioteker, runtimes og AI-modeller, herunder bl.a. OpenAI Whisper, Syv.ai/CoRal Hviske, Alexandra Instituttet Røst, 3D-Speaker, Helsinki-NLP OPUS-MT, Meta NLLB-200, Alibaba Qwen, Microsoft ONNX Runtime, Electron, Python, React samt styresystemets medie-codecs (Windows Media Foundation / AVFoundation / FFmpeg).',
          'VIGTIG MEDDELELSE OM ERHVERVSMÆSSIG BRUG AF TREDJEPARTSMODELLER: Retten til erhvervsmæssig brug i punkt 2 gælder udelukkende selve WispAI-applikationen og omfatter IKKE valgfrie tredjeparts AI-modeller, som du selv vælger at downloade. Visse valgfrie modeller — herunder Meta NLLB-200 (licenseret under CC-BY-NC-4.0, kun til ikke-kommerciel brug) — er underlagt særskilte licensvilkår eller forbud mod kommerciel udnyttelse. Du er alene ansvarlig for at gennemgå og overholde licensen for hver enkelt downloadet tredjepartsmodel før brug, særligt i erhvervsmæssig sammenhæng. Udviklerne yder ingen garanti for tredjepartsmodellers tilgængelighed, lovlighed eller licensforhold.',
        ],
      },
      {
        title: '6. Privatliv, GDPR & Ingen Databehandleraftale (DPA)',
        paragraphs: [
          'Idet WispAI udelukkende afvikles lokalt på din egen enhed, og Udviklerne på intet tidspunkt modtager, transmitterer, tilgår, hoster eller opbevarer dine lydoptagelser, transskriberinger eller personoplysninger, er Udviklerne IKKE "databehandler" (og ej heller fælles dataansvarlig) i henhold til EU’s databeskyttelsesforordning (GDPR) eller tilsvarende lovgivning.',
          'Du forbliver alene dataansvarlig for enhver behandling af personoplysninger, der udføres på din enhed ved hjælp af WispAI, og der indgås eller kræves ingen databehandleraftale (DPA) mellem dig og Udviklerne.',
        ],
      },
      {
        title: '7. Acceptabel Brug, Optagelsessamtykke & Lovgivning',
        paragraphs: [
          'Du er alene ansvarlig for, at din brug af WispAI overholder al gældende lokal, national og international lovgivning — herunder databeskyttelsesregler (fx GDPR og Databeskyttelsesloven), regler om lyd- og samtaleoptagelse, arbejdspladsregler, tavshedspligt, ophavsret samt gældende eksportkontrol- og sanktionsregler.',
          'Du skal indhente alle lovpligtige oplysninger, tilladelser og informerede samtykker fra samtlige deltagere, før du optager, transskriberer, taleropdeler eller oversætter møder, interviews, opkald eller mediefiler. Softwaren må ikke anvendes til ulovlig overvågning eller retsstridige formål.',
        ],
      },
      {
        title: '8. AI-Nøjagtighed, Ingen Professionel Rådgivning & Højrisiko-undtagelse',
        paragraphs: [
          'Alle transskriberinger, taleropdelinger, grammatik- og tegnsætningsrettelser, samtaleopsummeringer og oversættelser genereres automatisk af lokale maskinlæringsmodeller og kan indeholde fejl, misforståede ord, hallucinationer, forvekslede talere eller udeladelser. Du skal altid gennemgå og verificere vigtige tekster manuelt, før du anvender, offentliggør eller handler på baggrund af dem.',
          'Softwaren yder ikke juridisk, medicinsk, klinisk, finansiel eller anden professionel rådgivning. WispAI er ikke designet, beregnet eller godkendt til brug i højrisiko-, livskritiske eller sikkerhedskritiske miljøer (fx medicinsk diagnosticering, redningstjenester, luftfart eller bevismæssige retssager, hvor fejl kan medføre personskade, tab eller væsentlige juridiske/økonomiske konsekvenser).',
        ],
      },
      {
        title: '9. Garantifraskrivelse ("SOM BESET" / "AS IS")',
        paragraphs: [
          'SOFTWAREN, HERUNDER ALLE LOKALE AI-PIPELINES, SCRIPTS OG OPDATERINGER, LEVERES "SOM BESET" ("AS IS") OG "SOM TILGÆNGELIG" ("AS AVAILABLE"), MED ALLE EVENTUELLE FEJL OG UDEN NOGEN FORM FOR GARANTI.',
          'I VIDEST MULIGT OMFANG EFTER GÆLDENDE LOVGIVNING FRASKRIVER FORFATTERNE, RETTIGHEDSHAVERNE, UDVIKLERNE OG BIDRAGYDERNE SIG UDTRYKKELIGT ENHVER GARANTI — UDTRYKKELIG, UNDERFORSTÅET ELLER LOVBESTEMT — HERUNDER GARANTI FOR SALGBARHED, TILFREDSSTILLENDE KVALITET, EGNETHED TIL ET BESTEMT FORMÅL, ADKOMST, NØJAGTIGHED, DATAINTEGRITET, SYSTEMKOMPATIBILITET, IKKE-KRÆNKELSE SAMT SIKKER, FEJLFRI OG UAFBRUDT DRIFT.',
        ],
      },
      {
        title: '10. Omfattende Ansvarsfraskrivelse & Ansvarsbegrænsning',
        paragraphs: [
          'Udviklerne bestræber sig på at levere sikker og pålidelig software, der respekterer dit system og ikke beskadiger din enhed eller dine filer; enhver skade er utilsigtet, og WispAI stilles vederlagsfrit til rådighed.',
          'MED UNDTAGELSE AF ANSVAR, DER IKKE LOVLIGT KAN FRASKRIVES ELLER BEGRÆNSES EFTER UFRAVIGELIG LOVGIVNING — HERUNDER SÆRLIGT ANSVAR FOR FORSÆT, GROV UAGTSOMHED, DØD ELLER PERSONSKADE SAMT UFRAVIGELIGT PRODUKTANSVAR — HÆFTER FORFATTERNE, RETTIGHEDSHAVERNE, UDVIKLERNE ELLER BIDRAGYDERNE UNDER INGEN OMSTÆNDIGHEDER OVER FOR DIG ELLER TREDJEPART FOR NOGEN DIREKTE, INDIREKTE, TILFÆLDIGE, SÆRLIGE, PØNALE ELLER AFLEDTE SKADER AF NOGEN ART, HERUNDER: TAB, BESKADIGELSE, UTILSIGTET SLETNING ELLER ÆNDRING AF LOKALE FILER, LYD-/VIDEOOPTAGELSER, TRANSSKRIBERINGER ELLER DATA; SKADE PÅ LAGERMEDIER ELLER FILSYSTEM; USTABILITET ELLER NEDBRUD I STYRESYSTEMET; HARDWARESKADE, OVEROPHEDNING ELLER SLITAGE; TABT FORTJENESTE, OMSÆTNINGSTAB, GOODWILLTAB ELLER DRIFTSAFBRYDELSE — SOM OPSTÅR VED ELLER I FORBINDELSE MED DOWNLOAD, INSTALLATION, OPSÆTNING, BRUG ELLER MANGLENDE MULIGHED FOR AT BRUGE SOFTWAREN ELLER TREDJEPARTSMODELLER, UANSET OM KRAVET BYGGER PÅ KONTRAKT, ERSTATNINGSRET (HERUNDER SIMPEL UAGTSOMHED), OBJEKTIVT ANSVAR ELLER ANDET, SELV HVIS DER ER GJORT OPMÆRKSOM PÅ MULIGHEDEN FOR SÅDANNE SKADER.',
          'DA SOFTWAREN LEVERES VEDERLAGSFRIT, BEGRÆNSES UDVIKLERNES SAMLEDE ANSVAR FOR ALLE KRAV UNDER ÉT — I DET OMFANG ET RESTERENDE ANSVAR IKKE LOVLIGT KAN UDELUKKES — TIL DET LOVMÆSSIGE MINIMUM (0 KR.). BRUG AF WISPAI SKER PÅ EGET ANSVAR.',
        ],
      },
      {
        title: '11. Skadesløsholdelse (Indemnification)',
        paragraphs: [
          'I videst muligt omfang efter gældende lovgivning accepterer du at forsvare, skadesløsholde og friholde Udviklerne, forfatterne, rettighedshaverne og bidragyderne bag WispAI mod ethvert tredjepartskrav, søgsmål, tab, erstatningsansvar, bøde og udgift (herunder rimelige advokatomkostninger), som udspringer af eller vedrører: (a) din brug eller misbrug af Softwaren; (b) manglende sikkerhedskopiering af lokale filer; (c) manglende indhentning af lovpligtigt optagelses- eller databehandlingssamtykke; eller (d) din overtrædelse af denne EULA, tredjepartsmodellicenser, tredjepartsrettigheder eller gældende lovgivning.',
        ],
      },
      {
        title: '12. Varemærker & Ingen Tilknytning',
        paragraphs: [
          'Alle tredjeparts produktnavne, logoer og varemærker, der nævnes i WispAI eller tilhørende dokumentation — herunder Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace og GitHub — tilhører deres respektive ejere. WispAI er et uafhængigt open-source-projekt og har ingen tilknytning til, sponsorering fra eller godkendelse af disse organisationer.',
        ],
      },
      {
        title: '13. Opdateringer, Ingen Supportforpligtelse & Ophør',
        paragraphs: [
          'Udviklerne kan til enhver tid og uden varsel opdatere, ændre, midlertidigt indstille eller ophøre med WispAI samt revidere denne EULA og har ingen forpligtelse til at yde teknisk support, vedligeholdelse, fejlrettelser eller fremtidige opdateringer. Du kan til enhver tid stoppe brugen og afinstallere WispAI. Din brugsret til den distribuerede applikation efter denne EULA ophører automatisk ved væsentlig overtrædelse (uden at påvirke selvstændige kildekoderettigheder under MIT-licensen). Punkt 5, 6, 7, 8, 9, 10, 11, 12 og 14 forbliver gældende efter ophør eller afinstallation.',
        ],
      },
      {
        title: '14. Lovvalg, Værneting, Forrangssprog & Delvis Ugyldighed',
        paragraphs: [
          'Lovvalg & værneting: Denne EULA og enhver tvist, der udspringer af eller i forbindelse med denne, er underlagt dansk ret med undtagelse af dansk rets international-privatretlige regler og FN’s konvention om aftaler om internationale køb (CISG). Med forbehold for ufravigelige lovbestemte forbrugerværnetingsregler i dit bopælsland afgøres enhver tvist udelukkende ved de kompetente danske domstole.',
          'Forrangssprog: Denne EULA kan være oversat til flere sprog af hensyn til brugervenligheden. Ved eventuel uoverensstemmelse eller fortolkningstvivl mellem sprogversionerne har den engelske version forrang.',
          'Delvis ugyldighed & hele aftalen: Hvis en bestemmelse i denne EULA kendes ugyldig eller uden retsvirkning efter ufravigelig lovgivning, håndhæves bestemmelsen i videst muligt tilladt omfang, og de øvrige bestemmelser forbliver fuldt gyldige. Intet i denne EULA begrænser ufravigelige forbrugerrettigheder. Sammen med MIT-licensen udgør denne EULA den samlede aftale vedrørende din brug af WispAI.',
        ],
      },
    ],
  },

  es: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definiciones, Alcance y Capacidad Legal',
        paragraphs: [
          'Este Acuerdo de Licencia de Usuario Final ("EULA") es un contrato vinculante entre usted ("Usuario") y los autores, mantenedores y colaboradores de WispAI ("Desarrolladores") que rige la descarga, instalación, configuración y uso de la aplicación WispAI, su motor local integrado y documentación ("el Software").',
          'Al hacer clic en "Aceptar y continuar", instalar o usar el Software, usted declara que: (a) es mayor de edad en su jurisdicción (o cuenta con consentimiento parental); (b) si acepta en nombre de una empresa u organización, posee plena autoridad legal para vincularla; y (c) ha leído y acepta todos los términos aquí expuestos.',
        ],
      },
      {
        title: '2. Concesión de Licencia y Relación con la Licencia MIT',
        paragraphs: [
          'WispAI se distribuye de forma gratuita. Su código fuente está licenciado bajo la Licencia MIT (Copyright © 2026 starchap). Sujeto al cumplimiento de este EULA y de la Licencia MIT, se le concede un derecho gratuito, no exclusivo y mundial para instalar y utilizar la aplicación WispAI con fines personales, académicos o comerciales.',
          'Nada en este EULA limita los derechos sobre el código fuente otorgados por la Licencia MIT. En caso de conflicto sobre redistribución o sublicencia del código fuente, prevalecerá la Licencia MIT.',
        ],
      },
      {
        title: '3. Responsabilidad de Copias de Seguridad y Diligencia Debida',
        paragraphs: [
          'Antes de importar, convertir o procesar cualquier archivo de audio, vídeo o documento con WispAI, se recomienda encarecidamente mantener una copia de seguridad independiente y verificada de todos los archivos críticos.',
          'El Usuario acepta actuar con la debida diligencia al configurar y utilizar el Software, incluyendo:',
        ],
        items: [
          'No seleccionar nunca carpetas del sistema operativo (p. ej., Windows, Program Files, System32, /System) ni unidades raíz completas como carpeta de almacenamiento de WispAI.',
          'Asegurar suficiente espacio libre en disco antes de importar grabaciones extensas o descargar modelos locales de IA (100 MB – 2 GB por modelo).',
          'Verificar cuidadosamente antes de eliminar sesiones, grabaciones o modelos descargados, ya que el borrado elimina los archivos de forma permanente.',
        ],
      },
      {
        title: '4. Acceso Transparente al Sistema, Hardware y Archivos',
        paragraphs: [
          'WispAI opera 100% en local y sin conexión, accediendo únicamente a los recursos necesarios:',
        ],
        items: [
          'Archivos multimedia seleccionados por el usuario: WispAI solo lee los archivos que usted seleccione explícitamente.',
          'Carpeta de almacenamiento aprobada: WispAI solo descarga modelos de IA y guarda grabaciones, transcripciones y ajustes en la carpeta local aprobada por usted.',
          'Entrada de audio (solo activación voluntaria): El micrófono y el audio del sistema solo se utilizan cuando inicia activamente la grabación en vivo.',
          'Uso local de CPU, RAM y disco: El procesamiento local de IA es intensivo y puede provocar alto uso de CPU/RAM, ruido de ventiladores, consumo de batería o calor.',
          'Red y privacidad: Internet se utiliza únicamente cuando descarga modelos open-source (p. ej., desde HuggingFace o GitHub). Su audio, perfiles de voz y transcripciones nunca salen de su dispositivo: cero telemetría y cero subidas a la nube.',
        ],
      },
      {
        title: '5. Componentes de Terceros y Restricciones de Modelos de IA',
        paragraphs: [
          'WispAI integra bibliotecas y modelos open-source de terceros (como Whisper, Hviske, Røst, 3D-Speaker, OPUS-MT, Meta NLLB-200, Qwen, ONNX Runtime, Electron, Python y códecs del sistema).',
          'AVISO IMPORTANTE SOBRE USO COMERCIAL DE MODELOS DE TERCEROS: El permiso de uso comercial de la Sección 2 se aplica únicamente a la aplicación WispAI y NO se extiende a modelos de IA opcionales de terceros que decida descargar. Algunos modelos opcionales — incluido Meta NLLB-200 (bajo licencia CC-BY-NC-4.0, uso no comercial) — conllevan restricciones propias. Usted es el único responsable de revisar y cumplir la licencia de cada modelo antes de utilizarlo comercialmente.',
        ],
      },
      {
        title: '6. Privacidad, RGPD y Ausencia de Acuerdo de Encargo de Tratamiento (DPA)',
        paragraphs: [
          'Dado que WispAI se ejecuta exclusivamente en su dispositivo local y los Desarrolladores nunca reciben, transmiten, acceden ni almacenan sus grabaciones, transcripciones o datos personales, los Desarrolladores NO actúan como "Encargado del Tratamiento" (Data Processor) conforme al RGPD ni legislación equivalente. Usted es el único Responsable del Tratamiento y no se constituye ni requiere ningún DPA.',
        ],
      },
      {
        title: '7. Uso Aceptable, Consentimiento de Grabación y Cumplimiento Legal',
        paragraphs: [
          'Usted es el único responsable de cumplir con todas las leyes aplicables (incluyendo RGPD, privacidad, grabación de llamadas, confidencialidad, propiedad intelectual y normas de control de exportaciones). Debe obtener el consentimiento informado de todos los participantes antes de grabar, transcribir o traducir cualquier reunión o archivo.',
        ],
      },
      {
        title: '8. Precisión de la IA, Ausencia de Asesoramiento y Exclusión de Alto Riesgo',
        paragraphs: [
          'Todas las transcripciones, etiquetas de hablantes, correcciones gramaticales, resúmenes y traducciones son generados automáticamente por modelos locales de IA y pueden contener errores, alucinaciones u omisiones. Revise siempre manualmente las transcripciones importantes.',
          'WispAI no ofrece asesoramiento legal, médico ni financiero, y NO está diseñado ni autorizado para entornos críticos o de alto riesgo (como diagnóstico médico, emergencias o aviación).',
        ],
      },
      {
        title: '9. Exclusión Total de Garantías ("TAL CUAL" / "AS IS")',
        paragraphs: [
          'EL SOFTWARE SE PROPORCIONA "TAL CUAL" ("AS IS") Y "SEGÚN DISPONIBILIDAD", CON TODOS SUS DEFECTOS Y SIN GARANTÍA DE NINGÚN TIPO. EN LA MÁXIMA MEDIDA PERMITIDA POR LA LEY, LOS AUTORES Y DESARROLLADORES RENUNCIAN EXPRESAMENTE A TODA GARANTÍA EXPRESA, IMPLÍCITA O LEGAL, INCLUIDAS COMERCIABILIDAD, IDONEIDAD PARA UN FIN DETERMINADO, INTEGRIDAD DE DATOS, COMPATIBILIDAD, PRECISIÓN O FUNCIONAMIENTO ININTERRUMPIDO Y LIBRE DE ERRORES.',
        ],
      },
      {
        title: '10. Exclusión y Limitación Integral de Responsabilidad',
        paragraphs: [
          'Los Desarrolladores se esfuerzan por ofrecer un software seguro que nunca dañe su dispositivo ni sus archivos; cualquier daño es estrictamente involuntario y WispAI se distribuye gratuitamente.',
          'SALVO AQUELLA RESPONSABILIDAD QUE NO PUEDA EXCLUIRSE LEGALMENTE CONFORME A DERECHO IMPERATIVO — ESPECÍFICAMENTE DOLO, CULPA GRAVE, MUERTE O LESIONES PERSONALES, O RESPONSABILIDAD LEGAL IMPERATIVA POR PRODUCTOS — EN NINGÚN CASO LOS AUTORES, TITULARES DE DERECHOS, DESARROLLADORES O COLABORADORES SERÁN RESPONSABLES DE DAÑOS DIRECTOS, INDIRECTOS, INCIDENTALES, ESPECIALES, PUNITIVOS O CONSECUENTES, INCLUIDOS PÉRDIDA, CORRUPCIÓN, ELIMINACIÓN ACCIDENTAL O MODIFICACIÓN DE ARCHIVOS LOCALES, GRABACIONES, TRANSCRIPCIONES O DATOS; DAÑOS EN DISCOS O SISTEMA OPERATIVO; DAÑOS DE HARDWARE O SOBRECALENTAMIENTO; O PÉRDIDA DE BENEFICIOS O INTERRUPCIÓN DEL NEGOCIO — INCLUSO SI SE ADVIRTIÓ DE TALES DAÑOS. AL SER GRATUITO, LA RESPONSABILIDAD TOTAL AGREGADA RESIDUAL SE LIMITA AL MÍNIMO PERMITIDO POR LA LEY (0 USD).',
        ],
      },
      {
        title: '11. Indemnización',
        paragraphs: [
          'En la máxima medida permitida por la ley, usted acepta defender, indemnizar y mantener indemnes a los Desarrolladores frente a cualquier reclamación de terceros, daño, multa o gasto derivado de su uso o mal uso del Software, la falta de copias de seguridad, la ausencia de consentimiento de grabación o el incumplimiento de este EULA, licencias de modelos de terceros o la ley aplicable.',
        ],
      },
      {
        title: '12. Marcas Comerciales y Ausencia de Afiliación',
        paragraphs: [
          'Todos los nombres de productos y marcas de terceros mencionados (como Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace y GitHub) pertenecen a sus respectivos propietarios. WispAI es un proyecto independiente sin afiliación ni respaldo de dichas entidades.',
        ],
      },
      {
        title: '13. Actualizaciones, Ausencia de Soporte y Terminación',
        paragraphs: [
          'Los Desarrolladores pueden modificar, suspender o descontinuar WispAI o este EULA en cualquier momento sin obligación de ofrecer soporte técnico ni actualizaciones. Su derecho de uso de la aplicación termina automáticamente si incumple este EULA (sin afectar los derechos independientes sobre el código fuente bajo la Licencia MIT). Las Secciones 5 a 12 y 14 sobreviven a la terminación.',
        ],
      },
      {
        title: '14. Ley Aplicable, Jurisdicción, Idioma Prevaleciente y Divisibilidad',
        paragraphs: [
          'Ley aplicable y jurisdicción: Este EULA se rige por las leyes de Dinamarca, con exclusión de sus normas de conflicto de leyes y de la CISG. Sin perjuicio de las normas imperativas de jurisdicción del consumidor de su país de residencia, cualquier disputa será sometida exclusivamente a los tribunales competentes de Dinamarca.',
          'Idioma prevaleciente: En caso de discrepancia entre versiones traducidas de este EULA, prevalecerá la versión en inglés.',
          'Divisibilidad y acuerdo completo: Si alguna cláusula resultara inválida según la ley imperativa, el resto permanecerá en pleno vigor. Nada en este EULA limita los derechos legales irrenunciables del consumidor. Junto con la Licencia MIT, este EULA constituye el acuerdo completo.',
        ],
      },
    ],
  },

  de: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definitionen, Geltungsbereich & Geschäftsfähigkeit',
        paragraphs: [
          'Diese Endbenutzer-Lizenzvereinbarung ("EULA") ist ein rechtsverbindlicher Vertrag zwischen Ihnen ("Nutzer") und den Autoren, Betreuern und Mitwirkenden von WispAI ("Entwickler") über das Herunterladen, Installieren, Konfigurieren und Nutzen der paketierten WispAI-Desktop-Anwendung, des lokalen Backends und der zugehörigen Dateien ("die Software").',
          'Mit dem Klick auf "Akzeptieren & fortfahren" oder durch Installation bzw. Nutzung der Software sichern Sie zu, dass: (a) Sie in Ihrer Rechtsordnung volljährig sind (oder die Zustimmung eines Erziehungsberechtigten vorliegt); (b) Sie bei Annahme im Namen eines Unternehmens zur rechtsverbindlichen Vertretung befugt sind; und (c) Sie sämtliche nachstehenden Bedingungen akzeptieren.',
        ],
      },
      {
        title: '2. Lizenzgewährung & Verhältnis zur MIT-Lizenz',
        paragraphs: [
          'WispAI wird unentgeltlich bereitgestellt. Der Quellcode von WispAI steht unter der MIT-Lizenz (Copyright © 2026 starchap). Vorbehaltlich dieser EULA und der MIT-Lizenz erhalten Sie ein kostenloses, nicht-exklusives, weltweites Recht zur Installation und Nutzung der WispAI-Anwendung für private, akademische oder kommerzielle Zwecke.',
          'Keine Bestimmung dieser EULA schränkt die unter der MIT-Lizenz gewährten Quellcoderechte ein. Bei Widersprüchen hinsichtlich Kopieren, Ändern, Unterlizenzieren oder Weiterverbreiten des Quellcodes hat die MIT-Lizenz Vorrang.',
        ],
      },
      {
        title: '3. Datensicherung, Sorgfaltspflicht & Sichere Nutzung',
        paragraphs: [
          'Vor dem Importieren, Konvertieren oder Verarbeiten von Audio-, Video- oder Dokumentdateien mit WispAI wird ausdrücklich empfohlen, eine unabhängige, geprüfte Sicherungskopie (Backup) aller wichtigen Dateien auf einem separaten Speichermedium vorzuhalten.',
          'Der Nutzer verpflichtet sich, bei der Einrichtung und Verwendung der Software die übliche Sorgfalt anzuwenden, insbesondere:',
        ],
        items: [
          'Niemals Betriebssystemordner (z. B. Windows, Program Files, System32, /System) oder ganze Stammverzeichnisse als WispAI-Speicherordner auszuwählen.',
          'Vor dem Import langer Aufnahmen oder dem Download großer KI-Modelle (100 MB – 2 GB pro Modell) ausreichend freien Speicherplatz sicherzustellen.',
          'Vor dem Löschen von Sitzungen, Aufnahmen oder Modellen sorgfältig zu prüfen, da Dateien dadurch dauerhaft lokal entfernt werden.',
        ],
      },
      {
        title: '4. Transparenter System-, Hardware- & Dateizugriff',
        paragraphs: [
          'WispAI ist für 100% lokalen Offline-Betrieb entwickelt und greift ausschließlich auf folgende lokale Ressourcen zu:',
        ],
        items: [
          'Von Ihnen gewählte Mediendateien: WispAI liest nur Audio- und Videodateien, die Sie ausdrücklich zum Import auswählen.',
          'Von Ihnen genehmigter Speicherordner: WispAI lädt KI-Modelle nur herunter und speichert Aufnahmen, Transkripte und Einstellungen ausschließlich im von Ihnen genehmigten lokalen Ordner.',
          'Audioeingabe (nur auf aktive Anforderung): Mikrofon und Systemaudio (Loopback) werden nur während aktiver Live-Aufnahmen genutzt.',
          'Lokale CPU-, RAM- & Festplattenlast: Lokale KI-Inferenz ist rechenintensiv und kann zu hoher CPU-/RAM-Auslastung, Lüftergeräuschen, Akkuverbrauch oder Wärmeentwicklung führen.',
          'Netzwerk & Datenschutz: Internetzugriff erfolgt ausschließlich, wenn Sie ausdrücklich Open-Source-KI-Modelle herunterladen (z. B. von HuggingFace oder GitHub). Ihre Audiodaten, Stimmprofile und Transkripte verlassen niemals Ihr Gerät — null Telemetrie und null Cloud-Uploads.',
        ],
      },
      {
        title: '5. Komponenten Dritter & Lizenzbeschränkungen für KI-Modelle',
        paragraphs: [
          'WispAI integriert Bibliotheken und optional herunterladbare KI-Modelle Dritter (u. a. Whisper, Hviske, Røst, 3D-Speaker, OPUS-MT, Meta NLLB-200, Qwen, ONNX Runtime, Electron, Python und System-Codecs).',
          'WICHTIGER HINWEIS ZUR KOMMERZIELLEN NUTZUNG VON DRITTMODELLEN: Die kommerzielle Nutzungserlaubnis in Abschnitt 2 gilt ausschließlich für die WispAI-Anwendung selbst und erstreckt sich NICHT auf optionale KI-Modelle Dritter, die Sie herunterladen. Bestimmte optionale Modelle — darunter Meta NLLB-200 (lizenziert unter CC-BY-NC-4.0, nur nicht-kommerzielle Nutzung) — unterliegen eigenen Lizenzbeschränkungen. Sie sind allein dafür verantwortlich, vor der (insbesondere kommerziellen) Nutzung die Lizenz jedes Drittmodells zu prüfen und einzuhalten.',
        ],
      },
      {
        title: '6. Datenschutz, DSGVO & Keine Auftragsverarbeitung (AVV / DPA)',
        paragraphs: [
          'Da WispAI ausschließlich lokal auf Ihrem eigenen Gerät ausgeführt wird und die Entwickler niemals Zugriff auf Ihre Audioaufnahmen, Transkripte oder personenbezogenen Daten erhalten oder diese speichern, sind die Entwickler KEIN "Auftragsverarbeiter" im Sinne der DSGVO. Sie bleiben alleiniger Verantwortlicher, und ein Auftragsverarbeitungsvertrag (AVV/DPA) wird weder geschlossen noch benötigt.',
        ],
      },
      {
        title: '7. Zulässige Nutzung, Aufnahmeeinwilligung & Rechtskonformität',
        paragraphs: [
          'Sie sind allein verantwortlich für die Einhaltung aller geltenden Gesetze (einschließlich DSGVO, Datenschutz-, Persönlichkeits-, Aufnahme-, Urheber- und Exportkontrollrecht). Vor Aufnahme, Transkription oder Übersetzung müssen alle gesetzlich erforderlichen Einwilligungen sämtlicher Teilnehmenden vorliegen.',
        ],
      },
      {
        title: '8. KI-Genauigkeit, Keine Fachberatung & Ausschluss von Hochrisiko-Einsätzen',
        paragraphs: [
          'Alle Transkripte, Sprecherzuordnungen, Grammatikkorrekturen, Zusammenfassungen und Übersetzungen werden automatisch durch lokale KI-Modelle erzeugt und können Fehler, Halluzinationen oder Auslassungen enthalten. Prüfen Sie wichtige Inhalte stets manuell.',
          'WispAI bietet keine Rechts-, Medizin- oder Finanzberatung und ist NICHT für sicherheits- oder lebenskritische Hochrisiko-Bereiche (z. B. medizinische Diagnose, Notdienste, Luftfahrt) bestimmt.',
        ],
      },
      {
        title: '9. Vollständiger Gewährleistungsausschluss ("WIE BESEHEN" / "AS IS")',
        paragraphs: [
          'DIE SOFTWARE WIRD "WIE BESEHEN" ("AS IS") UND "WIE VERFÜGBAR" MIT ALLEN EVENTUELLEN FEHLERN UND OHNE JEGLICHE GEWÄHRLEISTUNG BEREITGESTELLT. SOWEIT GESETZLICH ZULÄSSIG, SCHLIESSEN DIE AUTOREN UND ENTWICKLER JEGLICHE AUSDRÜCKLICHE, STILLSCHWEIGENDE ODER GESETZLICHE GEWÄHRLEISTUNG AUS — EINSCHLIESSLICH MARKTGÄNGIGKEIT, EIGNUNG FÜR EINEN BESTIMMTEN ZWECK, DATENINTEGRITÄT, KOMPATIBILITÄT, GENAUIGKEIT ODER FEHLERFREIEN, UNUNTERBROCHENEN BETRIEB.',
        ],
      },
      {
        title: '10. Umfassender Haftungsausschluss & Haftungsbeschränkung',
        paragraphs: [
          'Die Entwickler sind bestrebt, sichere und zuverlässige Software bereitzustellen, die Ihr Gerät oder Ihre Dateien niemals beschädigt; jeglicher Schaden ist unbeabsichtigt, und WispAI wird unentgeltlich bereitgestellt.',
          'MIT AUSNAHME DER HAFTUNG, DIE NACH ZWINGENDEM RECHT NICHT AUSGESCHLOSSEN ODER BESCHRÄNKT WERDEN KANN — INSBESONDERE BEI VORSATZ, GROBER FAHRLÄSSIGKEIT, VERLETZUNG VON LEBEN, KÖRPER ODER GESUNDHEIT SOWIE NACH ZWINGENDEM PRODUKTHAFTUNGSRECHT — HAFTEN DIE AUTOREN, RECHTEINHABER, ENTWICKLER ODER MITWIRKENDEN IN KEINEM FALL FÜR DIREKTE, INDIREKTE, ZUFÄLLIGE, BESONDERE ODER FOLGESCHÄDEN JEGLICHER ART — EINSCHLIESSLICH VERLUST, BESCHÄDIGUNG, VERSEHENTLICHER LÖSCHUNG ODER VERÄNDERUNG LOKALER DATEIEN, AUFNAHMEN, TRANSKRIPTE ODER DATEN; SCHÄDEN AN SPEICHERMEDIEN ODER BETRIEBSSYSTEM; HARDWARESCHÄDEN ODER ÜBERHITZUNG; SOWIE ENTGANGENEN GEWINN ODER BETRIEBSUNTERBRECHUNG — SELBST WENN AUF DIE MÖGLICHKEIT SOLCHER SCHÄDEN HINGEWIESEN WURDE. DA DIE SOFTWARE UNENTGELTLICH IST, WIRD EINE ETWAIGE RESTHAFTUNG SOWEIT GESETZLICH ZULÄSSIG AUF DAS GESETZLICHE MINIMUM (0 EUR) BEGRENZT.',
        ],
      },
      {
        title: '11. Freistellung (Indemnification)',
        paragraphs: [
          'Soweit gesetzlich zulässig, stellen Sie die Entwickler von sämtlichen Ansprüchen Dritter, Schäden, Bußgeldern oder Kosten frei, die aus Ihrer Nutzung oder missbräuchlichen Nutzung der Software, fehlenden Backups, fehlender Aufnahmeeinwilligung oder einem Verstoß gegen diese EULA, Drittmodell-Lizenzen oder geltendes Recht entstehen.',
        ],
      },
      {
        title: '12. Markenrechte & Keine Zugehörigkeit',
        paragraphs: [
          'Alle genannten Produktnamen und Marken Dritter (z. B. Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace und GitHub) sind Eigentum ihrer jeweiligen Inhaber. WispAI ist ein unabhängiges Open-Source-Projekt ohne Zugehörigkeit oder Unterstützung dieser Organisationen.',
        ],
      },
      {
        title: '13. Updates, Keine Supportpflicht & Beendigung',
        paragraphs: [
          'Die Entwickler können WispAI oder diese EULA jederzeit ohne Vorankündigung ändern oder einstellen und sind nicht zu Support, Wartung oder Updates verpflichtet. Bei Verstoß gegen diese EULA endet Ihr Nutzungsrecht an der paketierten Anwendung automatisch (unabhängige Quellcoderechte unter der MIT-Lizenz bleiben unberührt). Die Abschnitte 5 bis 12 und 14 bestehen fort.',
        ],
      },
      {
        title: '14. Anwendbares Recht, Gerichtsstand, Maßgebliche Sprache & Salvatorische Klausel',
        paragraphs: [
          'Anwendbares Recht & Gerichtsstand: Diese EULA unterliegt dänischem Recht unter Ausschluss des internationalen Privatrechts und des UN-Kaufrechts (CISG). Vorbehaltlich zwingender gesetzlicher Verbrauchergerichtsstände sind ausschließlich die zuständigen Gerichte in Dänemark zuständig.',
          'Maßgebliche Sprache: Bei Abweichungen oder Auslegungsunterschieden zwischen Sprachversionen dieser EULA ist die englische Fassung maßgeblich.',
          'Salvatorische Klausel & Gesamtvereinbarung: Sollte eine Bestimmung nach zwingendem Recht unwirksam sein, bleiben die übrigen Bestimmungen unberührt. Zwingende gesetzliche Verbraucherrechte bleiben vorbehalten. Zusammen mit der MIT-Lizenz bildet diese EULA die gesamte Vereinbarung.',
        ],
      },
    ],
  },

  sv: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definitioner, Omfattning & Rättshandlingsförmåga',
        paragraphs: [
          'Detta slutanvändarlicensavtal ("EULA") är ett juridiskt bindande avtal mellan dig ("Användare") och författarna, underhållarna och bidragsgivarna bakom WispAI ("Utvecklarna") avseende nedladdning, installation, konfiguration och användning av den paketerade WispAI-applikationen, dess lokala backend och dokumentation ("Programvaran").',
          'Genom att klicka på "Godkänn & fortsätt", installera eller använda Programvaran intygar du att: (a) du är myndig i din jurisdiktion (eller har vårdnadshavares samtycke); (b) du, om du accepterar för ett företags räkning, har full behörighet att binda företaget; och (c) du har läst och accepterar samtliga villkor nedan.',
        ],
      },
      {
        title: '2. Licens & Förhållande till MIT-licensen',
        paragraphs: [
          'WispAI tillhandahålls kostnadsfritt. Källkoden är licensierad under MIT-licensen (Copyright © 2026 starchap). Under förutsättning att du följer detta EULA och MIT-licensen ges du en kostnadsfri, icke-exklusiv, världsomspännande rätt att installera och använda WispAI-applikationen för personligt, akademiskt eller kommersiellt bruk.',
          'Inget i detta EULA begränsar dina rättigheter till källkoden enligt MIT-licensen. Vid konflikt rörande källkod gäller MIT-licensen.',
        ],
      },
      {
        title: '3. Säkerhetskopiering, Aktsamhet & Säker Användning',
        paragraphs: [
          'Innan ljud-, video- eller dokumentfiler importeras eller behandlas i WispAI rekommenderas du starkt att alltid ha en oberoende, verifierad säkerhetskopia (backup) av alla viktiga filer på separat lagringsmedia.',
          'Du förbinder dig att iaktta normal aktsamhet vid användning av Programvaran, inklusive:',
        ],
        items: [
          'Att aldrig välja operativsystemmappar (t.ex. Windows, Program Files, System32, /System) eller en hel rotenhet som lagringsmapp för WispAI.',
          'Att säkerställa tillräckligt ledigt diskutrymme före import av långa inspelningar eller nedladdning av stora lokala AI-modeller (100 MB – 2 GB per modell).',
          'Att kontrollera noggrant innan du raderar sessioner, inspelningar eller modeller, eftersom radering tar bort filerna permanent lokalt.',
        ],
      },
      {
        title: '4. Transparent Åtkomst till System, Hårdvara & Filer',
        paragraphs: [
          'WispAI är byggt för 100% lokal offline-drift och använder endast följande resurser:',
        ],
        items: [
          'Användarvalda mediefiler: WispAI läser endast de ljud- och videofiler som du uttryckligen väljer att importera.',
          'Användargodkänd lagringsmapp: WispAI laddar endast ner AI-modeller och sparar inspelningar, transkriptioner och inställningar i den lokala mapp du godkänt.',
          'Ljudingångar (endast vid aktivt val): Mikrofon och systemljud används endast när du aktivt startar live-inspelning.',
          'Lokal CPU-, RAM- & diskbelastning: Lokal AI-bearbetning är beräkningsintensiv och kan medföra hög CPU-/RAM-belastning, fläktljud, batteriförbrukning eller värmeutveckling.',
          'Nätverk & integritet: Internet används enbart när du aktivt laddar ner open-source AI-modeller (t.ex. från HuggingFace eller GitHub). Ljud, röstprofiler och transkriptioner lämnar aldrig din enhet — noll telemetri och noll molnuppladdning.',
        ],
      },
      {
        title: '5. Tredjepartskomponenter & Licensbegränsningar för AI-Modeller',
        paragraphs: [
          'WispAI integrerar bibliotek och valfria AI-modeller från tredje part (bl.a. Whisper, Hviske, Røst, KB-Whisper, NB-Whisper, 3D-Speaker, OPUS-MT, Meta NLLB-200, Qwen, ONNX Runtime och Electron).',
          'VIKTIGT OM KOMMERSIELL ANVÄNDNING AV TREDJEPARTSMODELLER: Rätten till kommersiell användning i punkt 2 gäller enbart själva WispAI-applikationen och omfattar INTE valfria tredjepartsmodeller du väljer att ladda ner. Vissa modeller — bl.a. Meta NLLB-200 (licensierad under CC-BY-NC-4.0, icke-kommersiell) — har egna licensbegränsningar. Du ansvarar själv för att granska och följa varje modells licens.',
        ],
      },
      {
        title: '6. Integritet, GDPR & Inget Personuppgiftsbiträdesavtal (DPA)',
        paragraphs: [
          'Eftersom WispAI körs helt lokalt på din egen enhet och Utvecklarna aldrig tar emot, överför, får åtkomst till eller lagrar dina inspelningar, transkriptioner eller personuppgifter, är Utvecklarna INTE personuppgiftsbiträde enligt GDPR. Du är ensam personuppgiftsansvarig och inget biträdesavtal (DPA) krävs eller ingås.',
        ],
      },
      {
        title: '7. Acceptabel Användning, Samtycke & Lagstiftning',
        paragraphs: [
          'Du ansvarar ensam för att din användning av WispAI följer all tillämplig lagstiftning (inkl. GDPR, inspelningsregler, sekretess, upphovsrätt och exportkontroll). Du måste inhämta lagstadgat samtycke från samtliga deltagare innan du spelar in eller transkriberar möten eller filer.',
        ],
      },
      {
        title: '8. AI-Noggrannhet, Ingen Professionell Rådgivning & Högriskundantag',
        paragraphs: [
          'Alla transkriptioner, talaruppdelningar, grammatikrättningar, sammanfattningar och översättningar genereras automatiskt av lokala AI-modeller och kan innehålla fel, hallucinationer eller utelämnanden. Granska alltid viktiga texter manuellt.',
          'Programvaran utgör inte juridisk, medicinsk eller finansiell rådgivning och är INTE avsedd för säkerhets- eller livskritiska högriskmiljöer.',
        ],
      },
      {
        title: '9. Garantifriskrivning ("I BEFINTLIGT SKICK" / "AS IS")',
        paragraphs: [
          'PROGRAMVARAN TILLHANDAHÅLLS "I BEFINTLIGT SKICK" ("AS IS") UTAN NÅGON SOM HELST GARANTI. I DEN UTSTRÄCKNING LAG TILLÅTER FRISKRIVER SIG UTVECKLARNA FRÅN ALLA UTTRYCKLIGA OCH UNDERFÖRSTÅDDA GARANTIER, INKLUSIVE SÄLJBARHET, LÄMPLIGHET FÖR VISST ÄNDAMÅL, DATAINTEGRITET, KOMPATIBILITET OCH FELFRI DRIFT.',
        ],
      },
      {
        title: '10. Omfattande Ansvarsfriskrivning & Ansvarsbegränsning',
        paragraphs: [
          'Utvecklarna strävar efter att leverera säker programvara som aldrig skadar din enhet eller dina filer; eventuell skada är helt oavsiktlig, och WispAI tillhandahålls kostnadsfritt.',
          'MED UNDANTAG FÖR ANSVAR SOM INTE LAGLIGEN KAN FRISKRIVAS ENLIGT TVINGANDE RÄTT — SÄRSKILT UPPSÅT, GROV VÅRDSLÖSHET, DÖDSFALL ELLER PERSONSKADA SAMT TVINGANDE PRODUKTANSVAR — ANSVARAR UTVECKLARNA UNDER INGA OMSTÄNDIGHETER FÖR DIREKTA, INDIREKTA ELLER FÖLJDSKADOR, INKLUSIVE FÖRLUST, FÖRVANSKNING ELLER RADERING AV LOKALA FILER, INSPELNINGAR, TRANSKRIPTIONER, LAGRINGSMEDIA, OPERATIVSYSTEM, HÅRDVARUSKADOR, ÖVERHETTNING ELLER UTEBLIVEN VINST. EVENTUELLT KVARVARANDE ANSVAR BEGRÄNSAS TILL LAGENS MINIMUM (0 KR).',
        ],
      },
      {
        title: '11. Skadeslöshet (Indemnification)',
        paragraphs: [
          'I den utsträckning lag tillåter åtar du dig att hålla Utvecklarna skadeslösa mot tredjepartskrav, skador eller kostnader som uppstår till följd av din användning av Programvaran, utebliven säkerhetskopiering, bristande inspelningssamtycke eller brott mot detta EULA eller tillämplig lag.',
        ],
      },
      {
        title: '12. Varumärken & Ingen Anknytning',
        paragraphs: [
          'Alla omnämnda tredjepartsvarumärken (bl.a. Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace och GitHub) tillhör respektive ägare. WispAI är ett oberoende open-source-projekt utan anknytning till dessa organisationer.',
        ],
      },
      {
        title: '13. Uppdateringar, Ingen Supportskyldighet & Upphörande',
        paragraphs: [
          'Utvecklarna kan när som helst ändra eller upphöra med WispAI eller detta EULA utan skyldighet att erbjuda support eller uppdateringar. Vid avtalsbrott upphör rätten att använda den paketerade applikationen automatiskt (utan påverkan på källkodsrättigheter under MIT-licensen). Punkt 5–12 och 14 fortsätter att gälla.',
        ],
      },
      {
        title: '14. Tillämplig Lag, Jurisdiktion, Företrädesspråk & Ogiltighet',
        paragraphs: [
          'Tillämplig lag & jurisdiktion: Detta EULA regleras av dansk rätt (exkl. lagvalsregler och CISG). Med förbehåll för tvingande konsumentforumregler i ditt bosättningsland avgörs tvister exklusivt av behörig dansk domstol.',
          'Företrädesspråk: Vid eventuell avvikelse mellan språkversioner har den engelska versionen företräde.',
          'Delvis ogiltighet & hela avtalet: Om en bestämmelse befinns ogiltig förblir övriga bestämmelser gällande. Tvingande konsumenträttigheter begränsas inte. Tillsammans med MIT-licensen utgör detta EULA hela avtalet.',
        ],
      },
    ],
  },

  no: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definisjoner, Omfang & Rettslig Handleevne',
        paragraphs: [
          'Denne sluttbrukerlisensavtalen ("EULA") er en juridisk bindende avtale mellom deg ("Bruker") og forfatterne, vedlikeholderne og bidragsyterne bak WispAI ("Utviklerne") om nedlasting, installasjon, oppsett og bruk av den pakkede WispAI-applikasjonen, dens lokale backend og dokumentasjon ("Programvaren").',
          'Ved å klikke "Godta & fortsett", installere eller bruke Programvaren bekrefter du at: (a) du er myndig i din jurisdiksjon (eller har foresattes samtykke); (b) du har fullmakt til å binde virksomheten dersom du aksepterer på vegne av en virksomhet; og (c) du har lest og aksepterer alle vilkår nedenfor.',
        ],
      },
      {
        title: '2. Lisens & Forholdet til MIT-lisensen',
        paragraphs: [
          'WispAI leveres kostnadsfritt. Kildekoden er lisensiert under MIT-lisensen (Copyright © 2026 starchap). Under forutsetning av at du overholder denne EULA og MIT-lisensen gis du en gratis, ikke-eksklusiv, verdensomspennende rett til å installere og bruke WispAI-applikasjonen til privat, akademisk eller kommersiell bruk.',
          'Ingenting i denne EULA begrenser rettigheter til kildekoden under MIT-lisensen. Ved konflikt om kildekode gjelder MIT-lisensen.',
        ],
      },
      {
        title: '3. Sikkerhetskopiering, Aktsomhet & Sikker Bruk',
        paragraphs: [
          'Før lyd-, video- eller dokumentfiler importeres eller behandles i WispAI, anbefales du sterkt å alltid ha en uavhengig, verifisert sikkerhetskopi (backup) av alle viktige filer på et separat lagringsmedium.',
          'Du forplikter deg til å utvise alminnelig aktsomhet ved bruk av Programvaren, herunder:',
        ],
        items: [
          'Aldri velge operativsystemmapper (f.eks. Windows, Program Files, System32, /System) eller en hel rotdisk som lagringsmappe for WispAI.',
          'Sikre tilstrekkelig ledig diskplass før import av lange opptak eller nedlasting av store lokale AI-modeller (100 MB – 2 GB per modell).',
          'Kontrollere nøye før sletting av økter, opptak eller modeller, da sletting fjerner filene permanent lokalt.',
        ],
      },
      {
        title: '4. Gjennomsiktig Tilgang til System, Maskinvare & Filer',
        paragraphs: [
          'WispAI er bygget for 100% lokal offline-drift og benytter kun følgende lokale ressurser:',
        ],
        items: [
          'Brukervalgte mediefiler: WispAI leser kun lyd- og videofilene du uttrykkelig velger å importere.',
          'Brukergodkjent lagringsmappe: WispAI laster kun ned AI-modeller og lagrer opptak, transkripsjoner og innstillinger i den lokale mappen du har godkjent.',
          'Lydinnganger (kun ved aktivt valg): Mikrofon og systemlyd brukes kun når du aktivt starter live-opptak.',
          'Lokal CPU-, RAM- & diskbelastning: Lokal AI-behandling er beregningstung og kan gi høy CPU-/RAM-bruk, viftestøy, batteriforbruk eller varmeutvikling.',
          'Nettverk & personvern: Internett brukes kun når du aktivt laster ned open-source AI-modeller (f.eks. fra HuggingFace eller GitHub). Lyd, stemmeprofiler og transkripsjoner forlater aldri enheten din — null telemetri og null skyopplasting.',
        ],
      },
      {
        title: '5. Tredjepartskomponenter & Lisensbegrensninger for AI-Modeller',
        paragraphs: [
          'WispAI integrerer biblioteker og valgfrie AI-modeller fra tredjepart (bl.a. Whisper, Hviske, Røst, KB-Whisper, NB-Whisper, 3D-Speaker, OPUS-MT, Meta NLLB-200, Qwen, ONNX Runtime og Electron).',
          'VIKTIG OM KOMMERSIELL BRUK AV TREDJEPARTSMODELLER: Retten til kommersiell bruk i punkt 2 gjelder kun selve WispAI-applikasjonen og omfatter IKKE valgfrie tredjepartsmodeller du laster ned. Enkelte modeller — bl.a. Meta NLLB-200 (lisensiert under CC-BY-NC-4.0, ikke-kommersiell) — har egne begrensninger. Du er selv ansvarlig for å kontrollere og overholde lisensen til hver modell.',
        ],
      },
      {
        title: '6. Personvern, GDPR & Ingen Databehandleravtale (DPA)',
        paragraphs: [
          'Fordi WispAI kjører utelukkende lokalt på din egen enhet og Utviklerne aldri mottar, overfører, får tilgang til eller lagrer dine opptak, transkripsjoner eller personopplysninger, er Utviklerne IKKE databehandler etter GDPR. Du er alene behandlingsansvarlig, og ingen databehandleravtale (DPA) inngås eller kreves.',
        ],
      },
      {
        title: '7. Akseptabel Bruk, Samtykke & Lovgivning',
        paragraphs: [
          'Du er alene ansvarlig for at bruken av WispAI overholder all gjeldende lovgivning (inkl. GDPR, opptaksregler, taushetsplikt, opphavsrett og eksportkontroll). Du må innhente lovpålagt samtykke fra alle deltakere før opptak eller transkribering.',
        ],
      },
      {
        title: '8. AI-Nøyaktighet, Ingen Profesjonell Rådgivning & Høyrisikountak',
        paragraphs: [
          'Alle transkripsjoner, taleroppdelinger, grammatikkrettelser, oppsummeringer og oversettelser genereres automatisk av lokale AI-modeller og kan inneholde feil, hallusinasjoner eller utelatelser. Gjennomgå alltid viktige tekster manuelt.',
          'Programvaren gir ikke juridisk, medisinsk eller finansiell rådgivning og er IKKE beregnet for sikkerhets- eller livskritiske høyrisikomiljøer.',
        ],
      },
      {
        title: '9. Garantifraskrivelse ("SOM DEN ER" / "AS IS")',
        paragraphs: [
          'PROGRAMVAREN LEVERES "SOM DEN ER" ("AS IS") UTEN NOEN FORM FOR GARANTI. I VIDEST MULIG OMFANG ETTER GJELDENDE RETT FRASKRIVER UTVIKLERNE SEG ALLE UTTRYKKELIGE OG UNDERFORSTÅTTE GARANTIER, HERUNDER SALGBARHET, EGNETHET FOR BESTEMT FORMÅL, DATAINTEGRITET, KOMPATIBILITET OG FEILFRI DRIFT.',
        ],
      },
      {
        title: '10. Omfattende Ansvarsfraskrivelse & Ansvarsbegrensning',
        paragraphs: [
          'Utviklerne bestreber seg på å levere sikker programvare som aldri skader din enhet eller dine filer; enhver skade er utilsiktet, og WispAI leveres kostnadsfritt.',
          'MED UNNTAK AV ANSVAR SOM IKKE LOVLIG KAN FRASKRIVES ETTER UFRAVIKELIG RETT — SÆRLIG FORSETT, GROV UAKTSOMHET, DØD ELLER PERSONSKADE SAMT UFRAVIKELIG PRODUKTANSVAR — HEFTER UTVIKLERNE UNDER INGEN OMSTENDIGHETER FOR DIREKTE, INDIREKTE ELLER FØLGESKADER, HERUNDER TAP, SKADE ELLER SLETTING AV LOKALE FILER, OPPTAK, TRANSKRIPSJONER, LAGRINGSMEDIER, OPERATIVSYSTEM, MASKINVARESKADE, OVEROPPHETING ELLER TAPT FORTJENESTE. EVENTUELT RESTANSVAR BEGRENSES TIL LOVENS MINIMUM (0 KR).',
        ],
      },
      {
        title: '11. Skadesløsholdelse (Indemnification)',
        paragraphs: [
          'I den grad loven tillater det forplikter du deg til å holde Utviklerne skadesløse mot tredjepartskrav, skader eller kostnader som følge av din bruk av Programvaren, manglende sikkerhetskopiering, manglende opptakssamtykke eller brudd på denne EULA eller gjeldende rett.',
        ],
      },
      {
        title: '12. Varemerker & Ingen Tilknytning',
        paragraphs: [
          'Alle nevnte tredjepartsvaremerker (bl.a. Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace og GitHub) tilhører sine respektive eiere. WispAI er et uavhengig open-source-prosjekt uten tilknytning til disse organisasjonene.',
        ],
      },
      {
        title: '13. Oppdateringer, Ingen Supportplikt & Opphør',
        paragraphs: [
          'Utviklerne kan når som helst endre eller avslutte WispAI eller denne EULA uten plikt til å tilby support eller oppdateringer. Ved avtalebrudd opphører retten til å bruke den pakkede applikasjonen automatisk (uten å påvirke kildekoderettigheter under MIT-lisensen). Punkt 5–12 og 14 består.',
        ],
      },
      {
        title: '14. Lovvalg, Verneting, Forrangsspråk & Delvis Ugyldighet',
        paragraphs: [
          'Lovvalg & verneting: Denne EULA er underlagt dansk rett (ekskl. lovvalgsregler og CISG). Med forbehold for ufravikelige forbrukervernetingsregler i ditt bostedsland avgjøres tvister eksklusivt av kompetente danske domstoler.',
          'Forrangsspråk: Ved eventuell uoverensstemmelse mellom språkversjoner har den engelske versjonen forrang.',
          'Delvis ugyldighet & hele avtalen: Dersom en bestemmelse kjennes ugyldig forblir øvrige bestemmelser gyldige. Ufravikelige forbrukerrettigheter begrenses ikke. Sammen med MIT-lisensen utgjør denne EULA hele avtalen.',
        ],
      },
    ],
  },

  fr: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Définitions, Champ d’application & Capacité Juridique',
        paragraphs: [
          'Le présent Contrat de Licence Utilisateur Final (« EULA ») est un accord juridique contraignant entre vous (« Utilisateur ») et les auteurs, mainteneurs et contributeurs de WispAI (« Développeurs ») régissant le téléchargement, l’installation, la configuration et l’utilisation de l’application WispAI, de son moteur local intégré et de sa documentation (« le Logiciel »).',
          'En cliquant sur « Accepter et continuer », en installant ou en utilisant le Logiciel, vous déclarez : (a) avoir atteint l’âge légal de la majorité dans votre juridiction (ou disposer du consentement d’un responsable légal) ; (b) disposer de l’autorité nécessaire si vous acceptez au nom d’une entreprise ; et (c) accepter l’intégralité des conditions ci-dessous.',
        ],
      },
      {
        title: '2. Concession de Licence & Relation avec la Licence MIT',
        paragraphs: [
          'WispAI est distribué gratuitement. Son code source est sous Licence MIT (Copyright © 2026 starchap). Sous réserve du respect du présent EULA et de la Licence MIT, un droit gratuit, non exclusif et mondial vous est accordé pour installer et utiliser l’application WispAI à des fins personnelles, académiques ou commerciales.',
          'Aucune disposition du présent EULA ne restreint vos droits sur le code source au titre de la Licence MIT. En cas de conflit relatif au code source, la Licence MIT prévaut.',
        ],
      },
      {
        title: '3. Sauvegarde des Données, Diligence & Utilisation Sûre',
        paragraphs: [
          'Avant d’importer, convertir ou traiter tout fichier audio, vidéo ou document avec WispAI, il vous est vivement conseillé de conserver une copie de sauvegarde indépendante et vérifiée de vos fichiers importants sur un support distinct.',
          'Vous vous engagez à faire preuve de diligence raisonnable, notamment :',
        ],
        items: [
          'Ne jamais choisir de dossiers système (ex. Windows, Program Files, System32, /System) ni un disque racine entier comme dossier de stockage WispAI.',
          'Vérifier l’espace disque disponible avant d’importer de longs enregistrements ou de télécharger des modèles IA volumineux (100 Mo – 2 Go par modèle).',
          'Vérifier soigneusement avant de supprimer des sessions, enregistrements ou modèles, la suppression effaçant définitivement les fichiers locaux.',
        ],
      },
      {
        title: '4. Transparence sur l’Accès au Système, au Matériel & aux Fichiers',
        paragraphs: [
          'WispAI fonctionne 100% en local et hors-ligne, n’accédant qu’aux ressources suivantes :',
        ],
        items: [
          'Fichiers multimédias choisis par l’utilisateur : WispAI lit uniquement les fichiers que vous importez explicitement.',
          'Dossier de stockage approuvé : WispAI télécharge les modèles IA et enregistre vos sessions et paramètres uniquement dans le dossier local que vous avez approuvé.',
          'Entrées audio (uniquement sur activation) : Le microphone et l’audio système ne sont utilisés que lorsque vous lancez activement l’écoute en direct.',
          'Ressources locales CPU, RAM & disque : Le traitement IA local peut entraîner une forte utilisation CPU/RAM, du bruit de ventilation, une consommation de batterie ou de la chaleur.',
          'Réseau & confidentialité : Internet n’est utilisé que lorsque vous téléchargez explicitement des modèles IA open-source (ex. HuggingFace ou GitHub). Vos audios, profils vocaux et transcriptions ne quittent jamais votre appareil — zéro télémétrie et zéro cloud.',
        ],
      },
      {
        title: '5. Composants Tiers & Restrictions des Licences de Modèles IA',
        paragraphs: [
          'WispAI intègre des bibliothèques et modèles IA tiers optionnels (notamment Whisper, Hviske, Røst, KB-Whisper, NB-Whisper, 3D-Speaker, OPUS-MT, Meta NLLB-200, Qwen, ONNX Runtime et Electron).',
          'AVIS IMPORTANT SUR L’USAGE COMMERCIAL DES MODÈLES TIERS : L’autorisation d’usage commercial de la Section 2 s’applique uniquement à l’application WispAI et NE s’étend PAS aux modèles IA tiers optionnels téléchargés. Certains modèles — dont Meta NLLB-200 (licence CC-BY-NC-4.0, usage non commercial) — comportent des restrictions propres. Vous êtes seul responsable du respect de la licence de chaque modèle.',
        ],
      },
      {
        title: '6. Vie Privée, RGPD & Absence d’Accord de Sous-traitance (DPA)',
        paragraphs: [
          'WispAI s’exécutant exclusivement sur votre appareil local sans que les Développeurs ne reçoivent, n’accèdent ni ne stockent vos enregistrements ou données personnelles, les Développeurs ne sont PAS « sous-traitant » (Data Processor) au sens du RGPD. Vous restez seul responsable de traitement et aucun DPA n’est conclu ni requis.',
        ],
      },
      {
        title: '7. Usage Acceptable, Consentement d’Enregistrement & Conformité',
        paragraphs: [
          'Vous êtes seul responsable du respect de toutes les lois applicables (RGPD, vie privée, enregistrement de communications, secret professionnel, propriété intellectuelle et contrôle des exportations) et devez obtenir le consentement légal de tous les participants avant d’enregistrer ou de transcrire.',
        ],
      },
      {
        title: '8. Exactitude de l’IA, Absence de Conseil & Exclusion Haut Risque',
        paragraphs: [
          'Les transcriptions, attributions d’intervenants, corrections grammaticales, résumés et traductions sont générés automatiquement par des modèles locaux et peuvent contenir des erreurs ou hallucinations. Vérifiez toujours manuellement les textes importants.',
          'WispAI ne fournit aucun conseil juridique, médical ou financier et N’est PAS destiné aux environnements critiques ou à haut risque (diagnostic médical, urgences, aviation).',
        ],
      },
      {
        title: '9. Exclusion de Garantie (« EN L’ÉTAT » / « AS IS »)',
        paragraphs: [
          'LE LOGICIEL EST FOURNI « EN L’ÉTAT » (« AS IS ») SANS GARANTIE D’AUCUNE SORTE. DANS LA LIMITE AUTORISÉE PAR LA LOI, LES DÉVELOPPEURS EXCLUENT TOUTE GARANTIE EXPRESSE, IMPLICITE OU LÉGALE, Y COMPRIS QUALITÉ MARCHANDE, ADÉQUATION À UN USAGE PARTICULIER, INTÉGRITÉ DES DONNÉES, COMPATIBILITÉ OU FONCTIONNEMENT SANS ERREUR.',
        ],
      },
      {
        title: '10. Limitation & Exclusion de Responsabilité',
        paragraphs: [
          'Les Développeurs s’efforcent de fournir un logiciel sûr qui ne nuise jamais à votre appareil ni à vos fichiers ; tout dommage est involontaire et WispAI est distribué gratuitement.',
          'SAUF RESPONSABILITÉ NE POUVANT ÊTRE LÉGALEMENT EXCLUE EN VERTU D’UNE LOI IMPÉRATIVE — NOTAMMENT FAUTE INTENTIONNELLE, FAUTE LOURDE, DÉCÈS OU DOMMAGE CORPOREL, OU RESPONSABILITÉ LÉGALE DU FAIT DES PRODUITS — LES DÉVELOPPEURS NE POURRONT EN AUCUN CAS ÊTRE TENUS RESPONSABLES DE DOMMAGES DIRECTS, INDIRECTS OU CONSÉCUTIFS, Y COMPRIS PERTE, CORRUPTION OU SUPPRESSION DE FICHIERS LOCAUX, ENREGISTREMENTS, TRANSCRIPTIONS, SUPPORTS DE STOCKAGE, SYSTÈME D’EXPLOITATION, MATÉRIEL, SURCHAUFFE OU PERTE D’EXPLOITATION. TOUTE RESPONSABILITÉ RÉSIDUELLE EST LIMITÉE AU MINIMUM LÉGAL (0 EUR).',
        ],
      },
      {
        title: '11. Indemnisation',
        paragraphs: [
          'Dans la mesure permise par la loi, vous acceptez de garantir et d’indemniser les Développeurs contre toute réclamation de tiers résultant de votre utilisation du Logiciel, d’un défaut de sauvegarde, de l’absence de consentement d’enregistrement ou d’une violation du présent EULA ou de la loi.',
        ],
      },
      {
        title: '12. Marques & Absence d’Affiliation',
        paragraphs: [
          'Tous les noms et marques de tiers mentionnés (Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace, GitHub) appartiennent à leurs propriétaires respectifs. WispAI est un projet indépendant sans affiliation avec ces entités.',
        ],
      },
      {
        title: '13. Mises à Jour, Absence de Support & Résiliation',
        paragraphs: [
          'Les Développeurs peuvent modifier ou interrompre WispAI ou le présent EULA à tout moment sans obligation de support ni de mise à jour. En cas de violation, le droit d’utiliser l’application prend fin automatiquement (sans affecter les droits indépendants sur le code source sous Licence MIT). Les Sections 5 à 12 et 14 survivent.',
        ],
      },
      {
        title: '14. Droit Applicable, Juridiction, Langue Prévalente & Divisibilité',
        paragraphs: [
          'Droit applicable & juridiction : Le présent EULA est régi par le droit danois (hors règles de conflit de lois et CVIM/CISG). Sous réserve des règles impératives de compétence applicables aux consommateurs dans votre pays de résidence, tout litige relève exclusivement des tribunaux compétents du Danemark.',
          'Langue prévalente : En cas de divergence entre versions traduites, la version anglaise prévaut.',
          'Divisibilité & intégralité : Si une clause est jugée invalide, les autres demeurent en vigueur. Les droits impératifs des consommateurs demeurent inchangés. Avec la Licence MIT, le présent EULA constitue l’accord complet.',
        ],
      },
    ],
  },

  it: {
    version: '1.0',
    effectiveDate: '2026',
    sections: [
      {
        title: '1. Definizioni, Ambito & Capacità Giuridica',
        paragraphs: [
          'Il presente Contratto di Licenza con l’Utente Finale ("EULA") è un accordo vincolante tra l’utente ("Utente") e gli autori, manutentori e collaboratori di WispAI ("Sviluppatori") relativo a download, installazione, configurazione e utilizzo dell’applicazione WispAI, del backend locale integrato e della documentazione ("il Software").',
          'Cliccando "Accetta e continua", installando o utilizzando il Software, l’Utente dichiara di: (a) essere maggiorenne nella propria giurisdizione (o disporre del consenso di un tutore); (b) avere piena autorità se accetta per conto di un’azienda; e (c) accettare tutti i termini seguenti.',
        ],
      },
      {
        title: '2. Concessione di Licenza & Rapporto con la Licenza MIT',
        paragraphs: [
          'WispAI è distribuito gratuitamente. Il codice sorgente è rilasciato sotto Licenza MIT (Copyright © 2026 starchap). Nel rispetto del presente EULA e della Licenza MIT, è concesso il diritto gratuito, non esclusivo e mondiale di installare e utilizzare l’applicazione WispAI per scopi personali, accademici o commerciali.',
          'Nulla nel presente EULA limita i diritti sul codice sorgente concessi dalla Licenza MIT. In caso di conflitto sul codice sorgente, prevale la Licenza MIT.',
        ],
      },
      {
        title: '3. Responsabilità di Backup, Diligenza & Uso Sicuro',
        paragraphs: [
          'Prima di importare, convertire o elaborare file audio, video o documenti con WispAI, si raccomanda vivamente di conservare una copia di backup indipendente e verificata di tutti i file importanti su un supporto separato.',
          'L’Utente si impegna ad agire con ordinaria diligenza, tra cui:',
        ],
        items: [
          'Non selezionare mai cartelle di sistema (es. Windows, Program Files, System32, /System) né un intero disco radice come cartella di archiviazione di WispAI.',
          'Assicurarsi di disporre di spazio su disco sufficiente prima di importare lunghe registrazioni o scaricare modelli IA di grandi dimensioni (100 MB – 2 GB per modello).',
          'Verificare attentamente prima di eliminare sessioni, registrazioni o modelli, poiché l’eliminazione rimuove permanentemente i file locali.',
        ],
      },
      {
        title: '4. Trasparenza sull’Accesso a Sistema, Hardware & File',
        paragraphs: [
          'WispAI opera al 100% in locale e offline, accedendo esclusivamente alle seguenti risorse:',
        ],
        items: [
          'File multimediali scelti dall’utente: WispAI legge solo i file che l’Utente importa esplicitamente.',
          'Cartella di archiviazione approvata: WispAI scarica modelli IA e salva registrazioni, trascrizioni e impostazioni esclusivamente nella cartella locale approvata dall’Utente.',
          'Ingressi audio (solo su attivazione): Microfono e audio di sistema vengono utilizzati solo quando l’Utente avvia attivamente l’ascolto dal vivo.',
          'Risorse locali CPU, RAM & disco: L’elaborazione IA locale può comportare elevato utilizzo di CPU/RAM, rumore delle ventole, consumo della batteria o calore.',
          'Rete & privacy: Internet viene utilizzato solo quando l’Utente scarica esplicitamente modelli IA open-source (es. da HuggingFace o GitHub). Audio, profili vocali e trascrizioni non lasciano mai il dispositivo — zero telemetria e zero upload sul cloud.',
        ],
      },
      {
        title: '5. Componenti di Terze Parti & Restrizioni delle Licenze dei Modelli IA',
        paragraphs: [
          'WispAI integra librerie e modelli IA opzionali di terze parti (tra cui Whisper, Hviske, Røst, KB-Whisper, NB-Whisper, 3D-Speaker, OPUS-MT, Meta NLLB-200, Qwen, ONNX Runtime ed Electron).',
          'AVVISO IMPORTANTE SULL’USO COMMERCIALE DEI MODELLI DI TERZE PARTI: Il permesso di uso commerciale nella Sezione 2 si applica esclusivamente all’applicazione WispAI e NON si estende ai modelli IA opzionali di terze parti scaricati dall’Utente. Alcuni modelli — tra cui Meta NLLB-200 (licenza CC-BY-NC-4.0, solo uso non commerciale) — prevedono restrizioni specifiche. L’Utente è l’unico responsabile della verifica e del rispetto della licenza di ciascun modello.',
        ],
      },
      {
        title: '6. Privacy, GDPR & Assenza di Accordo sul Trattamento dei Dati (DPA)',
        paragraphs: [
          'Poiché WispAI opera esclusivamente in locale sul dispositivo dell’Utente e gli Sviluppatori non ricevono, accedono né archiviano registrazioni o dati personali, gli Sviluppatori NON sono "Responsabili del trattamento" (Data Processor) ai sensi del GDPR. L’Utente rimane unico Titolare del trattamento e nessun DPA è stipulato né richiesto.',
        ],
      },
      {
        title: '7. Uso Accettabile, Consenso alla Registrazione & Conformità Legale',
        paragraphs: [
          'L’Utente è l’unico responsabile del rispetto di tutte le leggi applicabili (GDPR, privacy, registrazione di conversazioni, segreto professionale, proprietà intellettuale e controllo delle esportazioni) e deve ottenere il consenso informato di tutti i partecipanti prima di registrare o trascrivere.',
        ],
      },
      {
        title: '8. Accuratezza IA, Nessuna Consulenza Professionale & Esclusione Alto Rischio',
        paragraphs: [
          'Trascrizioni, etichette degli interlocutori, correzioni grammaticali, riassunti e traduzioni sono generati automaticamente da modelli locali e possono contenere errori o allucinazioni. Verificare sempre manualmente i testi importanti.',
          'WispAI non fornisce consulenza legale, medica o finanziaria e NON è destinato ad ambienti critici o ad alto rischio (diagnosi medica, emergenze, aviazione).',
        ],
      },
      {
        title: '9. Esclusione di Garanzia ("COSÌ COM’È" / "AS IS")',
        paragraphs: [
          'IL SOFTWARE È FORNITO "COSÌ COM’È" ("AS IS") SENZA GARANZIA DI ALCUN TIPO. NELLA MISURA MASSIMA CONSENTITA DALLA LEGGE, GLI SVILUPPATORI ESCLUDONO OGNI GARANZIA ESPRESSA, IMPLICITA O LEGALE, INCLUSE COMMERCIABILITÀ, IDONEITÀ A UNO SCOPO PARTICOLARE, INTEGRITÀ DEI DATI, COMPATIBILITÀ E FUNZIONAMENTO PRIVO DI ERRORI.',
        ],
      },
      {
        title: '10. Limitazione & Esclusione di Responsabilità',
        paragraphs: [
          'Gli Sviluppatori si impegnano a fornire software sicuro che non danneggi dispositivi o file; eventuali danni sono del tutto involontari e WispAI è distribuito gratuitamente.',
          'FATTA ECCEZIONE PER LE RESPONSABILITÀ NON ESCLUDIBILI PER LEGGE IMPERATIVA — IN PARTICOLARE DOLO, COLPA GRAVE, DECESSO O LESIONI PERSONALI E RESPONSABILITÀ OBBLIGATORIA DA PRODOTTO — GLI SVILUPPATORI NON SARANNO IN ALCUN CASO RESPONSABILI PER DANNI DIRETTI, INDIRETTI O CONSEQUENZIALI, INCLUSI PERDITA, CORRUZIONE O ELIMINAZIONE DI FILE LOCALI, REGISTRAZIONI, TRASCRIZIONI, SUPPORTI DI ARCHIVIAZIONE, SISTEMA OPERATIVO, HARDWARE, SURRISCALDAMENTO O MANCATO GUADAGNO. EVENTUALI RESPONSABILITÀ RESIDUE SONO LIMITATE AL MINIMO DI LEGGE (0 EUR).',
        ],
      },
      {
        title: '11. Manleva (Indemnification)',
        paragraphs: [
          'Nella misura consentita dalla legge, l’Utente accetta di manlevare e tenere indenni gli Sviluppatori da qualsiasi pretesa di terzi derivante dall’uso del Software, dalla mancata esecuzione di backup, dalla mancanza di consenso alla registrazione o dalla violazione del presente EULA o della legge.',
        ],
      },
      {
        title: '12. Marchi & Nessuna Affiliazione',
        paragraphs: [
          'Tutti i nomi e marchi di terze parti citati (Microsoft Word, Windows, macOS, OpenAI Whisper, Meta NLLB-200, Alibaba Qwen, HuggingFace, GitHub) appartengono ai rispettivi proprietari. WispAI è un progetto open-source indipendente senza alcuna affiliazione.',
        ],
      },
      {
        title: '13. Aggiornamenti, Nessun Obbligo di Supporto & Risoluzione',
        paragraphs: [
          'Gli Sviluppatori possono modificare o interrompere WispAI o il presente EULA in qualsiasi momento senza obbligo di supporto o aggiornamento. In caso di violazione, il diritto d’uso dell’applicazione termina automaticamente (fatti salvi i diritti sul codice sorgente sotto Licenza MIT). Le Sezioni 5–12 e 14 sopravvivono.',
        ],
      },
      {
        title: '14. Legge Applicabile, Foro Competente, Lingua Prevalente & Nullità Parziale',
        paragraphs: [
          'Legge applicabile & foro competente: Il presente EULA è regolato dalla legge danese (escluse le norme sul conflitto di leggi e la CISG). Fatte salve le norme imperative sul foro del consumatore nel paese di residenza dell’Utente, ogni controversia è devoluta in via esclusiva ai tribunali competenti della Danimarca.',
          'Lingua prevalente: In caso di discrepanza tra le versioni tradotte, prevale la versione inglese.',
          'Nullità parziale & accordo completo: Qualora una clausola risulti invalida, le restanti rimangono pienamente efficaci. I diritti inderogabili dei consumatori restano impregiudicati. Insieme alla Licenza MIT, il presente EULA costituisce l’intero accordo.',
        ],
      },
    ],
  },
};
