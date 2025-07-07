# 🆘 Disaster Response & Recovery Assistant

A comprehensive, **offline-first AI assistant** powered by **Gemma 3n** and advanced computer vision, designed to support first responders, field medics, emergency coordinators, and disaster relief personnel in **low-connectivity, high-stakes environments**.

---

## ✅ Development Progress

### 🧩 Sprint 1 – Core AI & Analysis Engine ✅

- ✅ OpenAI Whisper integration for voice transcription  
- ✅ Intelligent hazard keyword detection in speech  
- ✅ Computer vision object detection with PyTorch/torchvision  
- ✅ Gemma 3n text-based disaster analysis engine  
- ✅ Severity scoring and priority classification  
- ✅ Basic mobile-responsive UI foundation  

---

### 🧩 Sprint 2 – Enhanced UX & Offline Capabilities ✅

- ✅ Dark mode and high contrast accessibility modes  
- ✅ Interactive bounding box visualization for detected hazards  
- ✅ Canvas-based result downloads (annotated images)  
- ✅ Service worker implementation for offline functionality  
- ✅ PDF export system for text-based reports  
- ✅ Comprehensive hazard detection testing suite  

---

### 🧩 Sprint 3 – Professional Reporting System ✅

- ✅ Advanced JSON-to-PDF report generator with Jinja2 + WeasyPrint  
- ✅ Interactive report builder with live preview (`generate.html`)  
- ✅ GPS coordinate integration and dynamic location mapping  
- ✅ Team assignment and checklist management  
- ✅ Professional PDF templates with severity badges and embedded imagery  

---

### 🧩 Sprint 4 – Administration & Data Management ✅

- ✅ Admin dashboard with real-time report filtering  
- ✅ Report archive with tone/escalation filters + keyword search  
- ✅ SQLite + SQLAlchemy database for persistent storage  
- ✅ Role-based authentication using JWT tokens  
- ✅ Status tracking + update workflows  

---

### 🧩 Sprint 5 – Advanced Sync & Coordination ✅

- ✅ Offline submission queue with IndexedDB  
- ✅ Auto-sync via service workers + retry logic  
- ✅ Real-time geolocation broadcast map (Leaflet)  
- ✅ WebRTC fallback for peer-to-peer emergency broadcasting  
- ✅ `/broadcast`, `/broadcasts` API for emergency alerts  
- ✅ Sync queue viewer and recovery support  

---

### 🧩 Sprint 6 – Real-Time Risk & Resilience Features ✅

- ✅ Predictive risk scoring using weather/location inputs  
- ✅ Broadcast triggering based on risk score threshold  
- ✅ Geolocated broadcast pins with color-coded urgency  
- ✅ Decentralized P2P broadcasting for offline resilience  
- ✅ Modular broadcast system with mesh fallback  

---

### 🧩 Sprint 7 – Geospatial Intelligence & Map Reporting ✅

- ✅ Live Leaflet map for crowd reports  
- ✅ Filtering by `tone`, `escalation`, and `keyword`  
- ✅ Cluster markers + escalation-based coloring  
- ✅ Dynamic heatmap overlays for report density  
- ✅ Map snapshot export view using `map_snapshot.html`  
- ✅ Real-time update polling  
- ✅ `/api/crowd-report-locations` with smart query filters  

---

### 🧩 Sprint 8 – Specialized Response Modules ✅

- ✅ Medical Triage UI with patient intake + color-coded severity (`triage_form.html`)  
- ✅ Patient Tracker with filters, update/discharge buttons (`patient_list.html`)  
- ✅ PDF export of triage status + patient logs  
- ✅ Full Jinja2 template refactor using `base.html` + `home.html`  
- ✅ Offline queue integration for medical reports  
- ✅ Edit and discharge views (`edit_patient.html`, status tracking)  

---

### 🧩 Sprint 9 – Crowd Reports, Export & Map Snapshots ✅

- ✅ Filtering of crowd reports by tone, escalation, and keyword (`crowd_reports.html`)  
- ✅ Export options for PDF, CSV, and JSON  
- ✅ Enhanced PDF export with embedded image/audio links  
- ✅ `export_pdf.html` template with styling + timestamp  
- ✅ Bulk ZIP export for selected reports (PDFs + metadata)  
- ✅ `map_snapshot.html` rendered Leaflet export for embedding  
- ✅ Static map PDF snapshots based on coordinates  
- ✅ CrowdReport model updated for full export support  

---

### 🧩 Sprint 10 – Live Report Builder & Modularization ✅

- ✅ `live_generate.html` live report editor with real-time preview  
- ✅ Refactored `generate.html` to split JS into `report-generator.js`  
- ✅ Refactored `hazards.html` to modular `hazards.js` for clarity  
- ✅ Base styles and layout consistent across pages  
- ✅ View and edit reports from admin dashboard or archive  

---

## 🔜 Upcoming Sprints

### 🧩 Sprint 11 – Analytics Dashboards & Visual Insights 🔄
- 📊 Report analytics (tone, severity, escalation over time)  
- 📈 Timeline graphs, heatmaps, and keyword clouds  
- 🧮 Per-user activity & top locations summary  
- 📥 Export analytics as PNG/PDF  
- 📊 Built with Chart.js, Recharts, or Plotly  

---

### 🧩 Sprint 12 – Full Incident Lifecycle 🔄
- 🧾 Multi-stage report lifecycle: Submitted → Reviewed → Resolved  
- 🏷️ Tagging, notes, and attachments per status  
- 🔁 Workflow escalation: auto-prioritize follow-ups  
- 📤 Export lifecycle history with metadata  

---

### 🧩 Sprint 13 – Live Collaboration & Messaging 🔄
- 💬 Internal responder chat per incident  
- 📎 File and image sharing in chat  
- ⏳ Typing indicators, seen/unseen markers  
- 📲 Push notifications (PWA + desktop)  

---

### 🧩 Sprint 14 – Simulation & Training 🔄
- 🎓 Training mode with fake data playback  
- 🕹️ Replay of past incident timelines  
- 💻 VR/AR support hooks for future expansion  
- 📘 Guided scenario checklists  

---

### 🧩 Sprint 15 – Deployment, Hosting, and Monitoring 🔄
- 🌍 Docker + Gunicorn deployment bundle  
- 📡 Prometheus/Grafana metrics  
- 🔐 Hardened OAuth2 & SSL setup  
- ☁️ Multi-region deployment strategy  

---

## 🧠 Long-Term Features (Ideas Vault)
- 🌐 Language translation + voice commands  
- 🛰️ Satellite and drone imagery ingestion  
- 🧬 Predictive health deterioration via vitals  
- 🔗 Blockchain timestamping + report integrity  
- 👁️‍🗨️ Facial recognition and missing persons matching  

---

## 🏗️ Technical Architecture

### Backend Stack

- 🐍 **FastAPI** — High-performance async API framework  
- 🤖 **Gemma 3n** — Google's multimodal transformer for analysis  
- 🎙️ **OpenAI Whisper** — Speech-to-text transcription  
- 🧠 **PyTorch + torchvision** — Computer vision and object detection  
- 📊 **SQLite** — Embedded database for report storage  
- 📄 **WeasyPrint** — Professional PDF generation  
- 🔐 **JWT + bcrypt** — Authentication and security  

---

### Frontend Stack

- 📱 **Progressive Web App (PWA)** — Offline-first architecture  
- 🎨 **Custom CSS Framework** — Optimized for emergency use  
- 🗺️ **Leaflet.js** — Interactive mapping and geolocation  
- 💾 **IndexedDB** — Client-side database for offline storage  
- 🔄 **Service Workers** — Background sync and caching  
- ♿ **WCAG 2.1 AA** — Accessibility compliance  

---

### AI/ML Pipeline

- 🔍 **COCO Object Detection** — Pre-trained hazard recognition  
- 📝 **Custom NLP Models** — Emergency-specific language processing  
- 📊 **Severity Scoring Algorithm** — Multi-factor risk assessment  
- 🎯 **Keyword Detection Engine** — Audio hazard identification  
- 🔄 **Continuous Learning Pipeline** — Model improvement from usage  

---

## 📦 Database Schema

### Reports Table
```sql
CREATE TABLE reports (
   id TEXT PRIMARY KEY,
   timestamp TEXT,
   location TEXT,
   severity REAL,
   filename TEXT,
   user TEXT,
   status TEXT,
   image_url TEXT,
   checklist TEXT,
   coordinates TEXT,
   hazards TEXT
);





