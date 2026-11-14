<p align="center">
  <img src="frontend/public/wisp-icon.png" width="110" height="110" alt="Wisp Logo" />
</p>

<h1 align="center">Wisp</h1>

<p align="center">
  <strong>Your Private, 100% Offline Meeting Transcriber &amp; Translator</strong><br/>
  Turn recordings and live meetings into clean, translated notes — without a single byte of your audio ever leaving your computer.
</p>

<p align="center">
  <a href="https://github.com/starchap/Wisp/releases/latest">
    <img src="https://img.shields.io/badge/Download_Free-Windows_%7C_macOS_%7C_Linux-2563eb?style=for-the-badge&logo=github" alt="Download Latest Release" />
  </a>
  &nbsp;
  <a href="https://github.com/sponsors/starchap">
    <img src="https://img.shields.io/badge/Support_with_%241-GitHub_Sponsors-ea4aaa?style=for-the-badge&logo=githubsponsors" alt="Sponsor on GitHub" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/100%25_Offline-No_Cloud_Servers-10b981?style=flat-square" alt="100% Offline" />
  <img src="https://img.shields.io/badge/Cost-100%25_Free_%26_Open_Source-6366f1?style=flat-square" alt="Free and Open Source" />
  <img src="https://img.shields.io/badge/Built_With-Antigravity-8b5cf6?style=flat-square" alt="Made with Antigravity" />
</p>

---

## Why Wisp?

Most AI transcription tools upload your private meetings, interviews, and client calls to remote cloud servers — and charge you a monthly subscription for it.

**Wisp is different.** It runs **entirely on your own computer**.
- **No subscriptions or accounts** — download it, open it, and use it for free.
- **No cloud uploads** — your voice recordings and written transcripts stay strictly on your own hard drive.
- **Works completely offline** — once you download the language packs inside the app on your first launch, you can turn off your Wi-Fi and transcribe anywhere.

---

## Download Wisp (No Technical Setup Needed)

Grab the ready-to-run installer for your computer from the [**Latest Releases Page**](https://github.com/starchap/Wisp/releases/latest):

| Operating System | Download Format | How to Install |
| :--- | :--- | :--- |
| **Windows** | `Wisp-Setup-1.0.0.exe` | Download and double-click to install & open |
| **macOS (Apple Silicon)** | `Wisp-1.0.0-arm64.dmg` | Open the `.dmg` and drag **Wisp** to Applications |
| **Linux (Ubuntu / Debian)** | `.deb` or `.AppImage` | Install the `.deb` package or run the `.AppImage` |
| **Linux (Fedora / Red Hat)** | `.rpm` | Install with your package manager |

> **First time opening Wisp?** A friendly step-by-step setup wizard will ask where you want to store your files and download the free AI language models to your computer with one click.

---

## What Can You Do With Wisp?

### 1. Transcribe Recorded Audio Files or Live Meetings
- **Audio Files:** Drop in a recorded meeting, interview, or voice memo (`.mp3`, `.wav`, `.flac`, `.ogg`).
- **Live Meetings:** Let Wisp listen in on your computer's audio and microphone during a live **Microsoft Teams**, **Zoom**, **Slack Huddle**, or **Google Meet** call.

### 2. Automatic Speaker Separation ("Who Said What")
Wisp automatically detects different voices in the conversation (`Speaker 1`, `Speaker 2`, etc.).
- Click the **Play** button next to any speaker to hear a quick sample of their voice.
- Rename speakers to their real names (e.g., *Anders*, *Sarah*, *Client*).
- If the same person was accidentally split into two speakers, click **Merge** on their card to combine them in one second.

### 3. Multi-Language Meetings Translated Into One Language
Have a meeting where one person speaks **Danish**, another speaks **Spanish**, and another speaks **English**?
- Assign each speaker's spoken language in the right-hand panel.
- Pick your preferred **Output Language** at the top.
- Switch anytime between the **Translated** view and the **Original Spoken** words with a single click.

### 4. Ready to Copy-Paste Into Word
Your transcript is formatted like a clean, readable document (`Speaker Name:` followed by paragraphs) — with a **Copy for Word** button so you can paste your meeting notes straight into Microsoft Word, Google Docs, Notion, or an email.

### 5. Switch Models or Re-Transcribe Anytime
Started with a fast, lightweight model and want higher accuracy on an important recording? You can switch to a larger AI model at any time and click **Re-transcribe & Translate** without losing your speaker names.

---

## Privacy, Offline Guarantee & GDPR Note

Wisp was built for people who care deeply about privacy — journalists, students, researchers, consultants, healthcare workers, and teams handling confidential conversations.

### How Wisp Protects Your Data:
1. **Zero Cloud Transmission:** Wisp never sends your audio recordings, voice prints, or transcripts to any external server or third-party AI provider.
2. **Built-In Network Lock:** After you download the open-source AI models from Hugging Face, the desktop app actively blocks outgoing internet connections except when you explicitly click download in Settings.
3. **100% Local Storage:** All recordings and transcripts are saved as plain files on your own computer in the folder you choose — and you can delete any session (or all sessions) with one click.

### Honest Note on GDPR & Legal Compliance:
Because Wisp processes and stores everything locally on your own device without sending personal data to external cloud processors, it is designed to make **GDPR and internal data-privacy workflows much simpler**.

> **Important Disclaimer:** Wisp is free, open-source software provided *"as-is"* without formal legal certification or warranty. Using offline software does not automatically exempt you or your organization from local laws (such as obtaining consent from participants before recording a conversation, or securing the computer where you store your files). Always follow your organization's own legal and compliance guidelines.

---

## Enjoying Wisp? Send a $1 Thank-You! ❤️

Wisp is **100% free**. There are no paywalls, no "Pro" tiers, no ads, and no data harvesting.

If Wisp saved you time on a meeting, helped you transcribe an interview, or spared you from paying $20/month for a cloud transcription service, please consider supporting the project on **GitHub Sponsors** — even **$1** as a virtual coffee or token of appreciation genuinely means a lot and helps keep Wisp actively improved for everyone!

<p align="center">
  <a href="https://github.com/sponsors/starchap">
    <img src="https://img.shields.io/badge/❤️_Sponsor_starchap_on_GitHub_(Even_%241_Helps!)-ea4aaa?style=for-the-badge&logo=githubsponsors" alt="Sponsor starchap on GitHub" />
  </a>
</p>

---

## Built With Antigravity

Wisp was proudly designed and built with **Antigravity**, combining modern agentic development with 100% local, privacy-first on-device AI.

---

<details>
<summary><strong>For Developers: Building From Source (Optional)</strong></summary>

<br/>

If you are a developer and want to run or build Wisp from source code instead of using the pre-built installers:

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

### 3. Packaging Desktop Installers
| Platform | Target Formats | Command |
| :--- | :--- | :--- |
| **Windows** | `.exe` (NSIS Installer) | `cd frontend && npm run dist:win` |
| **macOS** | `.dmg` & `.zip` | `cd frontend && npm run dist:mac` |
| **Linux (Ubuntu / Debian)** | `.deb` & `.AppImage` | `cd frontend && npm run dist:linux` |
| **Linux (Fedora / RHEL)** | `.rpm` | `cd frontend && npm run dist:linux` |

</details>
