# 🆘 Disaster Response & Recovery Assistant

A compact, **offline-first AI assistant** powered by Google's **Gemma 3n**, designed to support first responders, field medics, and disaster relief personnel in **low-connectivity, high-stakes environments**.

---

## 🚀 Features

- 📷 **Image-Based Damage Detection**  
  Upload photos to detect collapsed structures, debris, and hazard zones using AI.
  
- 🎤 **Voice-to-Text Triage with Hazard Detection**  
  Upload or record voice notes that are transcribed via Whisper and scanned for key hazard keywords (e.g., “help”, “gunshot”, “fire”).

- 📝 **Natural Language Situation Analysis**  
  Paste or type in free-form reports to receive AI-assisted recommendations.

- 📄 **Export to PDF**  
  Generate field-ready printable reports with a single click — includes auto-download + toast confirmation.

- 🗂️ **Hazard Warnings Panel**  
  Automatically displays AI-detected threats from transcribed audio in an animated, collapsible panel (🚨).

- 🌐 **Offline-First Functionality**  
  Fully operable in airplane mode after first load; works with local inference and static storage.

- 🔒 **Privacy-First Design**  
  No external calls after page load — all media and text are processed locally using on-device models.

- 📱 **Mobile-Optimized Interface**  
  Large buttons, responsive layout, and offline connectivity indicators.

- 🌓 **Dark Mode & High Contrast Mode**  
  Designed for field visibility and accessibility in challenging lighting conditions.

---

## ✅ Sprint Progress

### 🧩 Sprint 1 – Core Input & Analysis
- ✅ Voice/audio transcription via Whisper
- ✅ Hazard keyword detection in speech
- ✅ Image upload + object detection
- ✅ Text-based disaster analysis engine
- ✅ Severity scoring logic
- ✅ Panic mode UX setup
- ✅ Static fallback map (placeholder)
- ✅ Offline HTML fallback route

---

### 🧩 Sprint 2 – Multimodal UX & Offline Enhancements
- ✅ Dark Mode toggle in settings drawer
- ✅ Bounding box toggle for hazard overlays
- ✅ Detection result download (canvas PNG)
- ✅ Offline-first PWA shell with service worker
- ✅ Export PDF from text triage reports
- ✅ Simulated test hazard UI + mock output

---

### 🧩 Sprint 3 – Smart Report Automation
- ✅ JSON-to-PDF generator endpoint (`/generate-report`)
- ✅ Field-friendly test page for JSON input (`/generate`)
- ✅ Auto-download + success toast for generated PDFs
- ✅ Full integration with `report_utils.py` + image rendering
- ✅ All reports saved locally in `outputs/`

---

## 🧠 Powered By

- 🤖 **Gemma 3n (Google)** — Multimodal Transformer (text/image)
- 🎙️ **OpenAI Whisper** — On-device speech-to-text
- 🐍 **FastAPI** + **Jinja2** — Fast backend and dynamic templating
- 🧠 **ChromaDB / SQLite** (optional) — Emergency document embedding and search
- 🎨 **Custom Tailwind-style CSS** — Optimized for clarity, contrast, and real-world use

---

## 📦 Folder Structure







