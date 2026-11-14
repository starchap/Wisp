<p align="center">
  <img src="frontend/public/wisp-icon.png" width="110" height="110" alt="Wisp Logo" />
</p>

<h1 align="center">Wisp</h1>

<p align="center">
  <strong>Turn meetings and voice recordings into written text — privately on your own computer.</strong><br/>
  Available for Windows, macOS, and Linux.
</p>

<p align="center">
  <a href="https://github.com/starchap/Wisp/releases/latest">
    <img src="https://img.shields.io/badge/Download-Windows_%7C_macOS_%7C_Linux-2563eb?style=for-the-badge&logo=github" alt="Download Latest Release" />
  </a>
  &nbsp;
  <a href="https://github.com/sponsors/starchap">
    <img src="https://img.shields.io/badge/Sponsor-GitHub_Sponsors-ea4aaa?style=for-the-badge&logo=githubsponsors" alt="Sponsor on GitHub" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/100%25_Local_After_Setup-10b981?style=flat-square" alt="100% Local" />
  <img src="https://img.shields.io/badge/Free_%26_Open_Source-6366f1?style=flat-square" alt="Free and Open Source" />
  <img src="https://img.shields.io/badge/Made_with-Antigravity-8b5cf6?style=flat-square" alt="Made with Antigravity" />
</p>

---

## What is Wisp?

Wisp is a free desktop app that listens to a conversation and writes down what everyone said.

You can use it in two ways:
1. **Open an audio file** (like an `.mp3` or `.wav` recording from a meeting, interview, or class).
2. **Record a live call** (like a Teams, Zoom, or Google Meet call happening on your computer).

Wisp figures out **who is speaking**, writes down **what they said**, and can **translate** the conversation into your preferred language.

---

## Download & Install

Go to the [**Releases Page**](https://github.com/starchap/Wisp/releases/latest) and download the file for your computer:

- **Windows:** `Wisp-Setup-1.0.0.exe`
- **macOS:** `Wisp-1.0.0-arm64.dmg`
- **Linux (Ubuntu / Debian):** `wisp-desktop_1.0.0_amd64.deb` or `Wisp-1.0.0.AppImage`
- **Linux (Fedora):** `wisp-desktop-1.0.0.x86_64.rpm`

---

## What It Does

- **Separates Speakers Automatically:** Labels voices as `Speaker 1`, `Speaker 2`, etc. You can listen to a short clip of each voice, give them real names (like *Anders* or *Maria*), and merge speakers if one person was split into two.
- **Translates Multi-Language Conversations:** If one person speaks Danish and another speaks Spanish or English, you can set each person's language and read the whole meeting in one language. You can switch between the **Translated** text and the **Original** words anytime.
- **Easy Copy to Word:** Gives you a clean page of text that you can copy and paste straight into Microsoft Word, Google Docs, or an email with one click.
- **Re-Run With Different AI Models:** Want a faster result first and a more detailed transcription later? You can switch AI models per session and re-transcribe anytime.

---

## Internet & Offline Use (How the AI Models Work)

> **Note:** You will need an internet connection **the first time** you use Wisp so the app can download the AI models to your computer. Once the models are downloaded, **Wisp works 100% offline** and never needs the internet again.

Wisp uses trusted open-source AI models that run directly on your computer:
- **Speech-to-Text (Writing down what is said):** `OpenAI Whisper` (`Tiny`, `Base`, `Small`, `Medium`, and `Large-v3-Turbo` via `Sherpa-ONNX`)
- **Speaker Recognition (Telling voices apart):** `Alibaba 3D-Speaker` (`ERes2Net` voice embeddings)
- **Translation (Translating between languages):** `Helsinki-NLP OPUS-MT` (for fast Danish, Spanish, German, and English translation) and `Meta NLLB-200` (for 200 languages)

You can manage or delete downloaded models anytime inside **Settings → Local AI Models**.

---

## Privacy & GDPR

Because all audio processing happens on your own computer:
- Your voice recordings and transcripts are **never uploaded** to any cloud server.
- Everything is saved locally in a folder you choose on your computer.
- After you download the AI models, Wisp blocks outgoing network connections automatically.

This makes Wisp a great fit for privacy-conscious users and teams working under **GDPR** rules.

> **Legal Disclaimer:** Wisp is provided free and *"as-is"* without warranty or formal legal certification. While Wisp keeps all data on your own device, you are still responsible for following local laws and workplace rules (such as asking people for permission before recording a meeting).

---

## Built With Antigravity

Wisp was built using **Antigravity**, combining modern agentic development with local on-device AI.

---

## Support the Project

Wisp is free and open-source, with no subscriptions, ads, or paid tiers.  
If you find Wisp useful in your daily work or studies, you are welcome to support its development through [**GitHub Sponsors**](https://github.com/sponsors/starchap).

---

<details>
<summary><strong>For Developers: Building From Source</strong></summary>

<br/>

### 1. Python Backend (`Python 3.10+`)
```bash
pip install -r backend/requirements.txt
python backend/main.py
```

### 2. Electron + React Desktop App
```bash
cd frontend
npm install
npm run build
npm run dev:electron
```

### 3. Build Desktop Installers
- **Windows:** `cd frontend && npm run dist:win`
- **macOS:** `cd frontend && npm run dist:mac`
- **Linux:** `cd frontend && npm run dist:linux`

</details>
