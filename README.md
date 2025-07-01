# 🆘 Disaster Response & Recovery Assistant

A comprehensive, **offline-first AI assistant** powered by **Gemma 3n** and advanced computer vision, designed to support first responders, field medics, emergency coordinators, and disaster relief personnel in **low-connectivity, high-stakes environments**.

---

## 🚀 Core Features

### 🧠 AI-Powered Analysis
- 📷 **Computer Vision Hazard Detection**  
 Upload photos to detect hazards, vehicles, people, and structural damage using pre-trained object detection models with confidence scoring and bounding box visualization.

- 🎤 **Voice-to-Text Triage with Smart Hazard Detection**  
 Record or upload voice notes that are transcribed via OpenAI Whisper and automatically scanned for emergency keywords ("help", "fire", "gunshot", "sirens", etc.).

- 📝 **Natural Language Situation Analysis**  
 Process free-form text reports to extract priority levels, damage types, and generate actionable recommendations using Gemma's language understanding.

### 📋 Advanced Reporting & Documentation
- 📄 **Professional PDF Report Generation**  
 Create field-ready incident reports with embedded images, GPS coordinates, severity scoring, checklists, and team assignments.

- 🧩 **Live JSON-to-PDF Editor**  
 Interactive report builder with real-time preview, GPS auto-fill, severity badges, team assignment, and checklist management.

- 🗂️ **Hazard Warnings Panel**  
 Automatically displays AI-detected threats from transcribed audio in an animated, collapsible interface with priority indicators.

- 📊 **Admin Dashboard & Report Management**  
 Complete report archive with filtering, status updates, bulk export, and analytics dashboard for administrators.

### 🔄 Offline-First Architecture
- 🌐 **Progressive Web App (PWA)**  
 Fully functional offline mode with service worker caching, background sync queuing, and conflict resolution.

- 💾 **Smart Sync Management**  
 Priority-based synchronization, bandwidth-aware uploads, and automatic retry mechanisms for when connectivity returns.

- 📱 **Mobile-Optimized Interface**  
 Touch-friendly controls, responsive design, and optimized for use with gloves in field conditions.

### 🔐 Security & Authentication
- 👤 **Role-Based Access Control**  
 Multi-tier user system (Admin, Responder, Viewer) with JWT authentication and secure endpoints.

- 🔒 **Data Privacy & Security**  
 Local processing for sensitive data, encrypted communications, and audit trail capabilities.

---

## ✅ Development Progress

### 🧩 Sprint 6 – Real-Time Risk & Resilience Features ✅
- ✅ Predictive analytics engine with location, weather, and hazard inputs
- ✅ Risk scoring and severity classification with actionable outputs
- ✅ Conditional broadcast triggering based on threshold (risk_score ≥ 0.8)
- ✅ WebRTC-based P2P fallback for decentralized emergency messaging
- ✅ Emergency broadcast API with `/broadcast` and `/broadcasts` endpoints
- ✅ Live broadcast polling UI with auto-refresh and severity banners
- ✅ Leaflet map integration with broadcast pins and geolocation tracking
- ✅ Service Worker caching for offline viewing and sync
- ✅ IndexedDB queueing for alert storage and replay after reconnection
- ✅ Modular broadcast architecture: `broadcast.js`, `broadcast-map.js`, `fallback-webrtc.js`

---

### 🧩 Sprint 1 – Core AI & Analysis Engine ✅
- ✅ OpenAI Whisper integration for voice transcription
- ✅ Intelligent hazard keyword detection in speech
- ✅ Computer vision object detection with PyTorch/torchvision
- ✅ Gemma 3n text-based disaster analysis engine
- ✅ Severity scoring and priority classification
- ✅ Basic mobile-responsive UI foundation

### 🧩 Sprint 2 – Enhanced UX & Offline Capabilities ✅
- ✅ Dark mode and high contrast accessibility modes
- ✅ Interactive bounding box visualization for detected hazards
- ✅ Canvas-based result downloads (annotated images)
- ✅ Service worker implementation for offline functionality
- ✅ PDF export system for text-based reports
- ✅ Comprehensive hazard detection testing suite

### 🧩 Sprint 3 – Professional Reporting System ✅
- ✅ Advanced JSON-to-PDF report generator with template engine
- ✅ Interactive report builder with live preview
- ✅ GPS coordinate integration and mapping
- ✅ Team assignment and checklist management
- ✅ Professional PDF templates with severity badges and imagery

### 🧩 Sprint 4 – Administration & Data Management ✅
- ✅ Comprehensive admin dashboard with analytics
- ✅ Report archive with advanced filtering and search
- ✅ Status tracking and workflow management
- ✅ Bulk export functionality (.zip archives with metadata)
- ✅ SQLite database integration for persistent storage
- ✅ Role-based authentication system

### 🧩 Sprint 5 – Advanced Sync & Coordination ✅
- ✅ Offline report queue using IndexedDB with auto-retry
- ✅ Background sync via Service Worker and SyncManager
- ✅ Broadcast alert queuing and recovery during network loss
- ✅ Leaflet-based broadcast map with geolocation pins
- ✅ WebRTC-based peer-to-peer broadcast fallback system
- ✅ Live broadcast polling with automatic UI refresh
- ✅ Risk-based conditional emergency alert triggering

---

### 🧩 Sprint 7 – Geospatial Intelligence & 3D Mapping ⏳
- ⏳ 3D damage modeling and real-time visualization
- ⏳ Heatmaps for incident density and severity
- ⏳ Safe route planning with dynamic hazard avoidance
- ⏳ Augmented reality overlays for first responders

### 🧩 Sprint 8 – Specialized Response Modules ⏳
- ⏳ Medical emergency triage and patient tracking
- ⏳ Hospital capacity forecasting and integration
- ⏳ SAR enhancements: thermal imaging and GPS-based coordination
- ⏳ Live responder equipment tracking and task assignments

### 🧩 Sprint 9 – Security, Resilience, and Deployment ⏳
- ⏳ End-to-end encrypted communications
- ⏳ Blockchain-based audit trails for critical reports
- ⏳ Multi-region failover deployment
- ⏳ Edge computing for field deployments

### 🧩 Sprint 10 – Training, Simulation & Accessibility ⏳
- ⏳ VR/AR disaster response training modules
- ⏳ Multi-language and voice command interfaces
- ⏳ Accessibility features for cognitive/visual support
- ⏳ Cultural and regional customization for global response

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

### Frontend Stack
- 📱 **Progressive Web App (PWA)** — Offline-first architecture
- 🎨 **Custom CSS Framework** — Optimized for emergency use
- 🗺️ **Leaflet.js** — Interactive mapping and geolocation
- 💾 **IndexedDB** — Client-side database for offline storage
- 🔄 **Service Workers** — Background sync and caching
- ♿ **WCAG 2.1 AA** — Accessibility compliance

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





