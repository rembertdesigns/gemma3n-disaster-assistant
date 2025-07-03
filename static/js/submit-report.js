// Enhanced submit-report.js - Advanced Emergency Reporting System

// Configuration
const CONFIG = {
  MAX_MESSAGE_LENGTH: 1000,
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  ACCEPTED_IMAGE_TYPES: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
  ACCEPTED_AUDIO_TYPES: ['audio/wav', 'audio/mp3', 'audio/webm', 'audio/ogg'],
  GEOLOCATION_TIMEOUT: 10000,
  AUTO_SAVE_INTERVAL: 2000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
};

// Global state
let isSubmitting = false;
let autoSaveTimer = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let locationData = null;

// Form elements
let form = null;
let statusBox = null;
let messageField = null;
let priorityInputs = null;
let locationField = null;
let contactField = null;
let imageInput = null;
let audioUpload = null;

/**
 * Initialize the emergency reporting system
 */
document.addEventListener("DOMContentLoaded", () => {
  console.log("🚀 Initializing emergency reporting system");
  
  initializeFormElements();
  setupEventListeners();
  setupAutoSave();
  setupValidation();
  loadDraftData();
  
  // Automatically get location
  setTimeout(getCurrentLocation, 500);
  
  console.log("✅ Emergency reporting system ready");
});

/**
 * Initialize form elements
 */
function initializeFormElements() {
  form = document.getElementById("emergency-report-form") || document.getElementById("report-form");
  statusBox = document.getElementById("statusMessage") || document.getElementById("submit-status");
  messageField = document.getElementById("message");
  priorityInputs = document.querySelectorAll('input[name="priority"]');
  locationField = document.getElementById("location");
  contactField = document.getElementById("contact");
  imageInput = document.getElementById("imageInput") || document.getElementById("image");
  audioUpload = document.getElementById("audioUpload");
  
  if (!form || !messageField) {
    console.error("❌ Required form elements not found");
    return;
  }
  
  console.log("📋 Form elements initialized");
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
  // Form submission
  form.addEventListener("submit", handleFormSubmission);
  
  // Message field character counting
  if (messageField) {
    messageField.addEventListener("input", handleMessageInput);
  }
  
  // Priority selection
  priorityInputs.forEach(input => {
    input.addEventListener("change", handlePriorityChange);
  });
  
  // Location button
  const locationBtn = document.getElementById("getLocationBtn");
  if (locationBtn) {
    locationBtn.addEventListener("click", getCurrentLocation);
  }
  
  // Media uploads
  if (imageInput) {
    imageInput.addEventListener("change", handleImageUpload);
    setupDragAndDrop();
  }
  
  if (audioUpload) {
    audioUpload.addEventListener("click", toggleAudioRecording);
  }
  
  // Template buttons
  const templateButtons = document.querySelectorAll('.template-button');
  templateButtons.forEach(button => {
    button.addEventListener("click", handleTemplateSelection);
  });
  
  // Online/offline events
  window.addEventListener("online", handleOnline);
  window.addEventListener("offline", handleOffline);
  
  // Keyboard shortcuts
  document.addEventListener("keydown", handleKeyboardShortcuts);
  
  // Page unload warning
  window.addEventListener("beforeunload", handlePageUnload);
}

/**
 * Enhanced form submission handler
 */
async function handleFormSubmission(e) {
  e.preventDefault();
  
  if (isSubmitting) {
    console.log("🔄 Submission already in progress");
    return;
  }
  
  console.log("📤 Starting report submission");
  
  // Validate form
  if (!validateForm()) {
    return;
  }
  
  isSubmitting = true;
  showStatus("⏳ Submitting emergency report...", "loading");
  
  try {
    const formData = await collectFormData();
    
    let response;
    let attempt = 0;
    
    // Retry logic for network issues
    while (attempt < CONFIG.RETRY_ATTEMPTS) {
      try {
        response = await submitReport(formData);
        break; // Success, exit retry loop
      } catch (error) {
        attempt++;
        if (attempt === CONFIG.RETRY_ATTEMPTS) {
          throw error; // Final attempt failed
        }
        
        console.warn(`⚠️ Attempt ${attempt} failed, retrying...`, error);
        showStatus(`🔄 Connection issue, retrying... (${attempt}/${CONFIG.RETRY_ATTEMPTS})`, "loading");
        await delay(CONFIG.RETRY_DELAY * attempt);
      }
    }
    
    // Handle successful submission
    await handleSubmissionSuccess(response);
    
  } catch (error) {
    await handleSubmissionError(error);
  } finally {
    isSubmitting = false;
  }
}

/**
 * Collect and prepare form data
 */
async function collectFormData() {
  const formData = new FormData();
  
  // Basic form fields
  formData.append("message", messageField.value.trim());
  formData.append("location", locationField ? locationField.value.trim() : "");
  formData.append("contact", contactField ? contactField.value.trim() : "");
  formData.append("user", contactField ? contactField.value.trim() || "Anonymous" : "Anonymous");
  
  // Priority
  const selectedPriority = document.querySelector('input[name="priority"]:checked');
  formData.append("priority", selectedPriority ? selectedPriority.value : "medium");
  
  // Timestamp and metadata
  formData.append("timestamp", new Date().toISOString());
  formData.append("user_agent", navigator.userAgent);
  formData.append("report_source", "web_form");
  
  // Location data
  if (locationData) {
    formData.append("latitude", locationData.lat.toString());
    formData.append("longitude", locationData.lon.toString());
    formData.append("location_accuracy", locationData.accuracy ? locationData.accuracy.toString() : "");
  }
  
  // Image uploads
  if (imageInput && imageInput.files.length > 0) {
    Array.from(imageInput.files).forEach((file, index) => {
      formData.append(`image_${index}`, file);
    });
    formData.append("image_count", imageInput.files.length.toString());
  }
  
  // Audio recording
  if (recordedChunks.length > 0) {
    const audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
    formData.append("audio", audioBlob, "emergency_audio.webm");
  }
  
  return formData;
}

/**
 * Submit report with enhanced error handling
 */
async function submitReport(formData) {
  const endpoint = "/api/submit-emergency-report";
  
  const response = await fetch(endpoint, {
    method: "POST",
    body: formData
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Server error ${response.status}: ${errorText}`);
  }
  
  return await response.json();
}

/**
 * Handle successful submission
 */
async function handleSubmissionSuccess(result) {
  console.log("✅ Report submitted successfully:", result);
  
  // Clear draft data
  clearDraftData();
  
  // Show success message
  const escalation = result.escalation || result.report?.escalation || "processed";
  showStatus(`✅ Emergency report submitted successfully! Priority: ${escalation.toUpperCase()}`, "success");
  
  // Reset form
  resetForm();
  
  // Optional: Redirect or show next steps
  setTimeout(() => {
    if (confirm("Report submitted successfully! Would you like to submit another report?")) {
      // Stay on page for another report
      hideStatus();
    } else {
      // Redirect to dashboard or home
      window.location.href = "/";
    }
  }, 3000);
}

/**
 * Handle submission errors
 */
async function handleSubmissionError(error) {
  console.error("❌ Report submission failed:", error);
  
  // Check if it's a network error (offline)
  if (!navigator.onLine || error.message.includes("fetch")) {
    // Save to offline queue
    try {
      await saveToOfflineQueue();
      showStatus("📱 No internet connection. Report saved locally and will be submitted when connection is restored.", "warning");
    } catch (offlineError) {
      console.error("❌ Failed to save offline:", offlineError);
      showStatus("❌ Failed to submit report and unable to save offline. Please try again.", "error");
    }
  } else {
    // Server error or validation error
    showStatus(`❌ Failed to submit report: ${error.message}`, "error");
  }
}

/**
 * Save report to offline queue
 */
async function saveToOfflineQueue() {
  const reportData = {
    message: messageField.value.trim(),
    priority: document.querySelector('input[name="priority"]:checked')?.value || "medium",
    location: locationField ? locationField.value.trim() : "",
    contact: contactField ? contactField.value.trim() : "",
    timestamp: new Date().toISOString(),
    locationData: locationData,
    // Note: Files will need special handling for offline storage
    hasImages: imageInput && imageInput.files.length > 0,
    hasAudio: recordedChunks.length > 0
  };
  
  // Add to localStorage sync queue
  const syncQueue = JSON.parse(localStorage.getItem('syncQueue') || '[]');
  syncQueue.push({
    id: Date.now().toString(),
    type: 'emergency_report',
    data: reportData,
    timestamp: new Date().toISOString()
  });
  
  localStorage.setItem('syncQueue', JSON.stringify(syncQueue));
  
  console.log("💾 Report saved to offline queue");
}

/**
 * Enhanced form validation
 */
function validateForm() {
  let isValid = true;
  const errors = [];
  
  // Message validation
  const message = messageField.value.trim();
  if (!message) {
    errors.push("Please describe the emergency situation");
    markFieldInvalid(messageField);
    isValid = false;
  } else if (message.length < 10) {
    errors.push("Please provide more details about the emergency (minimum 10 characters)");
    markFieldInvalid(messageField);
    isValid = false;
  } else if (message.length > CONFIG.MAX_MESSAGE_LENGTH) {
    errors.push(`Message too long (maximum ${CONFIG.MAX_MESSAGE_LENGTH} characters)`);
    markFieldInvalid(messageField);
    isValid = false;
  } else {
    markFieldValid(messageField);
  }
  
  // Priority validation
  const selectedPriority = document.querySelector('input[name="priority"]:checked');
  if (!selectedPriority) {
    errors.push("Please select an emergency priority level");
    isValid = false;
  }
  
  // File validation
  if (imageInput && imageInput.files.length > 0) {
    for (const file of imageInput.files) {
      if (!CONFIG.ACCEPTED_IMAGE_TYPES.includes(file.type)) {
        errors.push(`Invalid image type: ${file.name}`);
        isValid = false;
      }
      if (file.size > CONFIG.MAX_FILE_SIZE) {
        errors.push(`Image too large: ${file.name} (max ${CONFIG.MAX_FILE_SIZE / 1024 / 1024}MB)`);
        isValid = false;
      }
    }
  }
  
  // Show validation errors
  if (!isValid) {
    showStatus(`⚠️ Please fix the following issues:\n${errors.join('\n')}`, "error");
  }
  
  return isValid;
}

/**
 * Handle message input with character counting
 */
function handleMessageInput() {
  const message = messageField.value;
  const charCounter = document.getElementById("charCounter");
  
  if (charCounter) {
    const count = message.length;
    const max = CONFIG.MAX_MESSAGE_LENGTH;
    charCounter.textContent = `${count} / ${max} characters`;
    
    // Update counter styling
    if (count > max * 0.9) {
      charCounter.className = "char-counter warning";
    } else if (count >= max) {
      charCounter.className = "char-counter error";
    } else {
      charCounter.className = "char-counter";
    }
  }
  
  // Auto-save draft
  if (autoSaveTimer) {
    clearTimeout(autoSaveTimer);
  }
  autoSaveTimer = setTimeout(saveDraft, CONFIG.AUTO_SAVE_INTERVAL);
}

/**
 * Handle priority change
 */
function handlePriorityChange(e) {
  const priority = e.target.value;
  console.log(`📊 Priority selected: ${priority}`);
  
  // Update UI based on priority
  if (priority === "critical") {
    showStatus("🚨 CRITICAL priority selected. This report will be processed immediately.", "warning");
  }
}

/**
 * Enhanced geolocation
 */
async function getCurrentLocation() {
  const locationBtn = document.getElementById("getLocationBtn");
  
  if (!navigator.geolocation) {
    showStatus("⚠️ Geolocation not supported by this browser", "error");
    return;
  }
  
  if (locationBtn) {
    locationBtn.disabled = true;
    locationBtn.textContent = "📍 Getting location...";
  }
  
  try {
    const position = await new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        resolve,
        reject,
        {
          enableHighAccuracy: true,
          timeout: CONFIG.GEOLOCATION_TIMEOUT,
          maximumAge: 300000 // 5 minutes
        }
      );
    });
    
    locationData = {
      lat: position.coords.latitude,
      lon: position.coords.longitude,
      accuracy: position.coords.accuracy,
      timestamp: Date.now()
    };
    
    // Try to get human-readable address
    try {
      const address = await reverseGeocode(locationData.lat, locationData.lon);
      if (locationField) {
        locationField.value = address || `${locationData.lat.toFixed(6)}, ${locationData.lon.toFixed(6)}`;
      }
    } catch (error) {
      if (locationField) {
        locationField.value = `${locationData.lat.toFixed(6)}, ${locationData.lon.toFixed(6)}`;
      }
    }
    
    if (locationBtn) {
      locationBtn.textContent = "✅ Location Added";
      locationBtn.style.background = "#16a34a";
      
      setTimeout(() => {
        locationBtn.textContent = "📍 Use My Location";
        locationBtn.style.background = "";
        locationBtn.disabled = false;
      }, 3000);
    }
    
    console.log("📍 Location acquired:", locationData);
    
  } catch (error) {
    console.error("❌ Geolocation error:", error);
    showStatus("⚠️ Unable to get your location. Please enter manually.", "warning");
    
    if (locationBtn) {
      locationBtn.textContent = "📍 Use My Location";
      locationBtn.disabled = false;
    }
  }
}

/**
 * Reverse geocoding
 */
async function reverseGeocode(lat, lon) {
  try {
    const response = await fetch(`https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`);
    const data = await response.json();
    
    if (data && data.locality && data.principalSubdivision) {
      return `${data.locality}, ${data.principalSubdivision}`;
    }
  } catch (error) {
    console.warn("Reverse geocoding failed:", error);
  }
  return null;
}

/**
 * Handle image uploads
 */
function handleImageUpload(e) {
  const files = Array.from(e.target.files);
  
  // Validate files
  const validFiles = files.filter(file => {
    if (!CONFIG.ACCEPTED_IMAGE_TYPES.includes(file.type)) {
      showStatus(`⚠️ Invalid file type: ${file.name}`, "warning");
      return false;
    }
    if (file.size > CONFIG.MAX_FILE_SIZE) {
      showStatus(`⚠️ File too large: ${file.name}`, "warning");
      return false;
    }
    return true;
  });
  
  if (validFiles.length > 0) {
    displayImagePreviews(validFiles);
  }
}

/**
 * Display image previews
 */
function displayImagePreviews(files) {
  const preview = document.getElementById("imagePreview") || document.getElementById("imageAnalysisPreview");
  if (!preview) return;
  
  preview.style.display = "block";
  preview.innerHTML = "";
  
  files.forEach((file, index) => {
    const fileInfo = document.createElement("div");
    fileInfo.className = "file-info";
    fileInfo.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <span>📷 ${file.name} (${formatFileSize(file.size)})</span>
        <button type="button" class="remove-file-btn" onclick="removeFile(this)">Remove</button>
      </div>
    `;
    
    // Add image preview
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = document.createElement("img");
        img.src = e.target.result;
        img.style.cssText = "max-width: 200px; max-height: 150px; object-fit: cover; border-radius: 4px; margin-top: 0.5rem;";
        fileInfo.appendChild(img);
      };
      reader.readAsDataURL(file);
    }
    
    preview.appendChild(fileInfo);
  });
}

/**
 * Setup drag and drop for images
 */
function setupDragAndDrop() {
  const dropArea = document.getElementById("fileUploadArea");
  if (!dropArea) return;
  
  ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
  });
  
  ["dragenter", "dragover"].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.add("dragover"), false);
  });
  
  ["dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, () => dropArea.classList.remove("dragover"), false);
  });
  
  dropArea.addEventListener("drop", handleDrop, false);
}

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function handleDrop(e) {
  const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
  
  if (files.length > 0 && imageInput) {
    // Create a new FileList
    const dt = new DataTransfer();
    files.forEach(file => dt.items.add(file));
    imageInput.files = dt.files;
    
    handleImageUpload({ target: { files: dt.files } });
  }
}

/**
 * Audio recording functionality
 */
async function toggleAudioRecording() {
  if (!isRecording) {
    await startAudioRecording();
  } else {
    stopAudioRecording();
  }
}

async function startAudioRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    recordedChunks = [];
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };
    
    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
      displayAudioPreview(audioBlob);
    };
    
    mediaRecorder.start();
    isRecording = true;
    
    // Update UI
    const audioUpload = document.getElementById("audioUpload");
    if (audioUpload) {
      audioUpload.innerHTML = `
        <span class="upload-icon" style="color: #dc2626;">🔴</span>
        <div class="upload-text">Recording... Click to stop</div>
        <div class="upload-hint">Recording audio for emergency report</div>
      `;
      audioUpload.style.background = "#fee2e2";
      audioUpload.style.borderColor = "#dc2626";
    }
    
    console.log("🎤 Audio recording started");
    
  } catch (error) {
    console.error("❌ Failed to start recording:", error);
    showStatus("⚠️ Unable to access microphone. Please check permissions.", "error");
  }
}

function stopAudioRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    
    // Reset UI
    const audioUpload = document.getElementById("audioUpload");
    if (audioUpload) {
      audioUpload.innerHTML = `
        <span class="upload-icon">🎤</span>
        <div class="upload-text">Record Audio</div>
        <div class="upload-hint">Click to start recording</div>
      `;
      audioUpload.style.background = "";
      audioUpload.style.borderColor = "";
    }
    
    console.log("🎤 Audio recording stopped");
  }
}

function displayAudioPreview(audioBlob) {
  const preview = document.getElementById("audioPreview");
  if (!preview) return;
  
  const audioUrl = URL.createObjectURL(audioBlob);
  
  preview.style.display = "block";
  preview.innerHTML = `
    <div class="file-info">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span>🎤 Audio Recording (${formatFileSize(audioBlob.size)})</span>
        <button type="button" class="remove-file-btn" onclick="removeAudio()">Remove</button>
      </div>
      <audio controls style="width: 100%; margin-top: 0.5rem;">
        <source src="${audioUrl}" type="audio/webm">
      </audio>
    </div>
  `;
}

/**
 * Template selection
 */
function handleTemplateSelection(e) {
  const template = e.target.dataset.template;
  if (!template || !messageField) return;
  
  const templates = {
    fire: "🔥 FIRE EMERGENCY: I can see flames and smoke coming from [location]. There appear to be [number] people in the area. The fire seems to be spreading [direction/speed]. Immediate fire response needed.",
    medical: "🚑 MEDICAL EMERGENCY: Someone appears to need immediate medical attention at [location]. The person is [conscious/unconscious] and [breathing/not breathing normally]. Multiple victims: [yes/no].",
    weather: "🌪️ SEVERE WEATHER: Experiencing dangerous weather conditions including [wind/rain/hail/tornado]. Visibility is [poor/zero]. Damage to [buildings/vehicles/infrastructure] observed. People seeking shelter.",
    accident: "🚗 TRAFFIC ACCIDENT: Vehicle collision at [location]. [Number] vehicles involved. Injuries: [apparent/unknown]. Traffic is [blocked/severely impacted]. Emergency vehicles needed.",
    flooding: "🌊 FLOODING: Water levels are rising rapidly at [location]. Current depth approximately [measurement]. [Number] people/vehicles stranded. Water is [rising/stable/receding].",
    other: "📋 EMERGENCY SITUATION: [Describe the situation in detail]. Number of people affected: [number]. Immediate dangers: [list dangers]. Current status: [ongoing/resolved/worsening]."
  };
  
  const template_text = templates[template] || "";
  
  if (messageField.value.trim() === "") {
    messageField.value = template_text;
  } else {
    messageField.value += "\n\n" + template_text;
  }
  
  messageField.focus();
  messageField.dispatchEvent(new Event("input"));
}

/**
 * Auto-save functionality
 */
function setupAutoSave() {
  // Save draft every time user stops typing for 2 seconds
  if (messageField) {
    messageField.addEventListener("input", () => {
      if (autoSaveTimer) {
        clearTimeout(autoSaveTimer);
      }
      autoSaveTimer = setTimeout(saveDraft, CONFIG.AUTO_SAVE_INTERVAL);
    });
  }
}

function saveDraft() {
  if (!messageField || !messageField.value.trim()) return;
  
  const draft = {
    message: messageField.value,
    priority: document.querySelector('input[name="priority"]:checked')?.value,
    location: locationField ? locationField.value : "",
    contact: contactField ? contactField.value : "",
    timestamp: Date.now()
  };
  
  try {
    localStorage.setItem("emergencyReportDraft", JSON.stringify(draft));
    console.log("💾 Draft saved");
  } catch (error) {
    console.warn("⚠️ Failed to save draft:", error);
  }
}

function loadDraftData() {
  try {
    const draft = localStorage.getItem("emergencyReportDraft");
    if (!draft) return;
    
    const data = JSON.parse(draft);
    const age = Date.now() - data.timestamp;
    
    // Only load drafts less than 1 hour old
    if (age > 3600000) {
      localStorage.removeItem("emergencyReportDraft");
      return;
    }
    
    if (confirm("You have a saved draft. Would you like to restore it?")) {
      if (messageField && data.message) messageField.value = data.message;
      if (locationField && data.location) locationField.value = data.location;
      if (contactField && data.contact) contactField.value = data.contact;
      
      if (data.priority) {
        const priorityInput = document.getElementById(data.priority);
        if (priorityInput) priorityInput.checked = true;
      }
      
      // Update character counter
      if (messageField) {
        messageField.dispatchEvent(new Event("input"));
      }
      
      console.log("📝 Draft restored");
    }
    
    // Clean up draft after loading
    localStorage.removeItem("emergencyReportDraft");
    
  } catch (error) {
    console.warn("⚠️ Failed to load draft:", error);
    localStorage.removeItem("emergencyReportDraft");
  }
}

function clearDraftData() {
  localStorage.removeItem("emergencyReportDraft");
}

/**
 * Online/offline handlers
 */
function handleOnline() {
  console.log("🌐 Connection restored");
  showStatus("📡 Connection restored. You can now submit reports normally.", "success");
  setTimeout(hideStatus, 3000);
}

function handleOffline() {
  console.log("📴 Connection lost");
  showStatus("📴 You're offline. Reports will be saved locally and submitted when connection returns.", "warning");
}

/**
 * Keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
  if (e.ctrlKey || e.metaKey) {
    switch (e.key) {
      case "s":
        e.preventDefault();
        saveDraft();
        showStatus("💾 Draft saved", "success");
        setTimeout(hideStatus, 2000);
        break;
      case "Enter":
        if (e.shiftKey) {
          e.preventDefault();
          form.dispatchEvent(new Event("submit"));
        }
        break;
    }
  }
}

/**
 * Page unload warning
 */
function handlePageUnload(e) {
  if (messageField && messageField.value.trim() && !isSubmitting) {
    const message = "You have unsaved changes. Are you sure you want to leave?";
    e.returnValue = message;
    return message;
  }
}

/**
 * Utility functions
 */
function showStatus(message, type) {
  if (!statusBox) return;
  
  statusBox.textContent = message;
  statusBox.className = `status-message status-${type}`;
  statusBox.style.display = "block";
  
  // Auto-hide success messages
  if (type === "success") {
    setTimeout(hideStatus, 5000);
  }
}

function hideStatus() {
  if (statusBox) {
    statusBox.style.display = "none";
  }
}

function resetForm() {
  if (form) {
    form.reset();
  }
  
  // Reset additional state
  recordedChunks = [];
  locationData = null;
  
  // Reset file previews
  const previews = document.querySelectorAll('.file-preview');
  previews.forEach(preview => {
    preview.style.display = "none";
  });
  
  // Reset character counter
  const charCounter = document.getElementById("charCounter");
  if (charCounter) {
    charCounter.textContent = "0 / 1000 characters";
    charCounter.className = "char-counter";
  }
  
  // Clear status
  hideStatus();
}

function markFieldInvalid(field) {
  field.classList.add("error");
  field.classList.remove("valid");
}

function markFieldValid(field) {
  field.classList.add("valid");
  field.classList.remove("error");
}

function setupValidation() {
  // Real-time validation
  if (messageField) {
    messageField.addEventListener("blur", () => {
      const message = messageField.value.trim();
      if (message.length >= 10) {
        markFieldValid(messageField);
      } else if (message.length > 0) {
        markFieldInvalid(messageField);
      }
    });
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Global functions for template compatibility
window.removeFile = function(button) {
  const preview = button.closest('.file-preview');
  const input = document.getElementById('imageInput');
  
  if (input) {
    input.value = '';
  }
  
  if (preview) {
    preview.style.display = 'none';
  }
};

window.removeAudio = function() {
  recordedChunks = [];
  const preview = document.getElementById('audioPreview');
  if (preview) {
    preview.style.display = 'none';
  }
};

// Export functions for external use
window.EmergencyReporting = {
  saveDraft,
  loadDraftData,
  clearDraftData,
  getCurrentLocation,
  toggleAudioRecording,
  showStatus,
  hideStatus,
  resetForm,
  validateForm
};

// Initialize analytics if available
if (window.gtag) {
  window.gtag('event', 'emergency_form_loaded', {
    'event_category': 'emergency',
    'event_label': 'form_initialization'
  });
}

console.log("🚨 Emergency reporting system fully loaded"); 