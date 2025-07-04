// Triage Form JavaScript - Medical Emergency Assessment System
let selectedTriageColor = null;
let formValidation = {
  name: false,
  injury_type: false,
  consciousness: false,
  breathing: false,
  severity: false,
  triage_color: false
};

// Initialize the form when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  initializeForm();
  setupEventListeners();
  updateCurrentTime();
  
  // Update time every second
  setInterval(updateCurrentTime, 1000);
  
  // Auto-save form data every 30 seconds
  setInterval(autoSaveFormData, 30000);
});

function initializeForm() {
  // Clear any existing form data on fresh load
  const form = document.getElementById('triage-form');
  if (form) {
    // Load any saved form data
    loadSavedFormData();
  }
  
  // Initialize vital signs monitoring
  setupVitalSignsMonitoring();
  
  // Setup form validation
  setupFormValidation();
  
  console.log('Triage form initialized');
}

function setupEventListeners() {
  // Triage color selection
  const colorOptions = document.querySelectorAll('.color-option');
  colorOptions.forEach(option => {
    option.addEventListener('click', handleTriageColorSelection);
  });
  
  // Form submission
  const form = document.getElementById('triage-form');
  if (form) {
    form.addEventListener('submit', handleFormSubmission);
  }
  
  // Real-time validation for required fields
  setupFieldValidation();
  
  // Severity assessment auto-suggestions
  setupSeverityAssessment();
  
  // Vital signs change detection
  setupVitalSignsListeners();
  
  // Keyboard shortcuts
  document.addEventListener('keydown', handleKeyboardShortcuts);
}

function handleTriageColorSelection(event) {
  const clickedOption = event.currentTarget;
  const color = clickedOption.dataset.color;
  
  // Remove previous selection
  document.querySelectorAll('.color-option').forEach(option => {
    option.classList.remove('selected');
  });
  
  // Add selection to clicked option
  clickedOption.classList.add('selected');
  
  // Update hidden input
  const hiddenInput = document.getElementById('triage_color');
  if (hiddenInput) {
    hiddenInput.value = color;
    selectedTriageColor = color;
    formValidation.triage_color = true;
  }
  
  // Provide visual feedback
  showMessage(`Triage color set to ${color.toUpperCase()}`, 'success');
  
  // Update form validation
  validateForm();
  
  // Auto-suggest based on triage color
  autoSuggestBasedOnTriageColor(color);
}

function autoSuggestBasedOnTriageColor(color) {
  const severityRadios = document.querySelectorAll('input[name="severity"]');
  
  // Auto-suggest severity based on triage color
  switch(color) {
    case 'red':
      // Suggest critical or severe
      const criticalRadio = document.getElementById('critical');
      if (criticalRadio && !document.querySelector('input[name="severity"]:checked')) {
        criticalRadio.checked = true;
        formValidation.severity = true;
        showMessage('Severity auto-set to CRITICAL based on RED triage', 'info');
      }
      break;
    case 'yellow':
      // Suggest severe or moderate
      const severeRadio = document.getElementById('severe');
      if (severeRadio && !document.querySelector('input[name="severity"]:checked')) {
        severeRadio.checked = true;
        formValidation.severity = true;
        showMessage('Severity auto-set to SEVERE based on YELLOW triage', 'info');
      }
      break;
    case 'green':
      // Suggest mild or moderate
      const mildRadio = document.getElementById('mild');
      if (mildRadio && !document.querySelector('input[name="severity"]:checked')) {
        mildRadio.checked = true;
        formValidation.severity = true;
        showMessage('Severity auto-set to MILD based on GREEN triage', 'info');
      }
      break;
  }
  
  validateForm();
}

function setupFieldValidation() {
  // Required fields validation
  const requiredFields = ['name', 'injury_type', 'consciousness', 'breathing'];
  
  requiredFields.forEach(fieldName => {
    const field = document.getElementById(fieldName);
    if (field) {
      field.addEventListener('input', () => validateField(fieldName));
      field.addEventListener('blur', () => validateField(fieldName));
    }
  });
  
  // Severity radio buttons
  const severityRadios = document.querySelectorAll('input[name="severity"]');
  severityRadios.forEach(radio => {
    radio.addEventListener('change', () => {
      formValidation.severity = true;
      validateForm();
      
      // Auto-suggest triage color based on severity
      autoSuggestTriageColor(radio.value);
    });
  });
}

function autoSuggestTriageColor(severity) {
  if (selectedTriageColor) return; // Don't override manual selection
  
  let suggestedColor = null;
  
  switch(severity) {
    case 'critical':
      suggestedColor = 'red';
      break;
    case 'severe':
      suggestedColor = 'yellow';
      break;
    case 'moderate':
      suggestedColor = 'yellow';
      break;
    case 'mild':
      suggestedColor = 'green';
      break;
  }
  
  if (suggestedColor) {
    const colorOption = document.querySelector(`[data-color="${suggestedColor}"]`);
    if (colorOption) {
      // Highlight suggestion but don't auto-select
      colorOption.style.animation = 'suggest-pulse 2s ease-in-out 3';
      showMessage(`Consider ${suggestedColor.toUpperCase()} triage color based on ${severity} severity`, 'suggestion');
    }
  }
}

function validateField(fieldName) {
  const field = document.getElementById(fieldName);
  const errorElement = document.getElementById(`${fieldName}-error`);
  
  if (!field) return;
  
  let isValid = false;
  let errorMessage = '';
  
  switch(fieldName) {
    case 'name':
      isValid = field.value.trim().length >= 2;
      errorMessage = 'Patient name/ID must be at least 2 characters';
      break;
    case 'injury_type':
      isValid = field.value.trim().length >= 3;
      errorMessage = 'Please describe the primary injury/condition';
      break;
    case 'consciousness':
    case 'breathing':
      isValid = field.value !== '';
      errorMessage = 'This field is required';
      break;
  }
  
  // Update field styling
  field.classList.remove('error', 'valid');
  field.classList.add(isValid ? 'valid' : 'error');
  
  // Update error message
  if (errorElement) {
    errorElement.textContent = isValid ? '' : errorMessage;
  }
  
  // Update validation state
  formValidation[fieldName] = isValid;
  
  // Update overall form validation
  validateForm();
  
  return isValid;
}

function setupVitalSignsMonitoring() {
  const vitalInputs = document.querySelectorAll('.vital-input');
  
  vitalInputs.forEach(input => {
    input.addEventListener('input', function() {
      evaluateVitalSign(this);
    });
  });
}

function evaluateVitalSign(input) {
  const value = parseFloat(input.value);
  const fieldName = input.name;
  
  if (isNaN(value)) {
    input.classList.remove('vital-normal', 'vital-warning', 'vital-critical');
    return;
  }
  
  let classification = 'normal';
  
  switch(fieldName) {
    case 'heart_rate':
      if (value < 50 || value > 120) classification = 'critical';
      else if (value < 60 || value > 100) classification = 'warning';
      break;
    case 'bp_systolic':
      if (value < 80 || value > 180) classification = 'critical';
      else if (value < 90 || value > 140) classification = 'warning';
      break;
    case 'bp_diastolic':
      if (value < 50 || value > 110) classification = 'critical';
      else if (value < 60 || value > 90) classification = 'warning';
      break;
    case 'respiratory_rate':
      if (value < 10 || value > 30) classification = 'critical';
      else if (value < 12 || value > 24) classification = 'warning';
      break;
    case 'temperature':
      if (value < 95 || value > 104) classification = 'critical';
      else if (value < 97 || value > 100.4) classification = 'warning';
      break;
    case 'oxygen_sat':
      if (value < 90) classification = 'critical';
      else if (value < 95) classification = 'warning';
      break;
  }
  
  // Update styling
  input.classList.remove('vital-normal', 'vital-warning', 'vital-critical');
  input.classList.add(`vital-${classification}`);
  
  // Show alert for critical vitals
  if (classification === 'critical') {
    showMessage(`⚠️ Critical vital sign detected: ${fieldName}`, 'warning');
  }
}

function setupVitalSignsListeners() {
  const vitalInputs = document.querySelectorAll('.vital-input');
  
  vitalInputs.forEach(input => {
    input.addEventListener('change', function() {
      // Auto-save when vital signs change
      autoSaveFormData();
      
      // Check for patterns that might suggest severity
      checkVitalSignPatterns();
    });
  });
}

function checkVitalSignPatterns() {
  const vitals = {
    heart_rate: parseFloat(document.getElementById('heart_rate')?.value) || null,
    bp_systolic: parseFloat(document.getElementById('bp_systolic')?.value) || null,
    bp_diastolic: parseFloat(document.getElementById('bp_diastolic')?.value) || null,
    respiratory_rate: parseFloat(document.getElementById('respiratory_rate')?.value) || null,
    temperature: parseFloat(document.getElementById('temperature')?.value) || null,
    oxygen_sat: parseFloat(document.getElementById('oxygen_sat')?.value) || null
  };
  
  let criticalCount = 0;
  let warningCount = 0;
  
  // Count critical and warning vitals
  Object.entries(vitals).forEach(([key, value]) => {
    if (value === null) return;
    
    switch(key) {
      case 'heart_rate':
        if (value < 50 || value > 120) criticalCount++;
        else if (value < 60 || value > 100) warningCount++;
        break;
      case 'oxygen_sat':
        if (value < 90) criticalCount++;
        else if (value < 95) warningCount++;
        break;
      // Add other vital checks...
    }
  });
  
  // Suggest severity based on vital patterns
  if (criticalCount >= 2) {
    showMessage('Multiple critical vitals detected - consider CRITICAL severity', 'warning');
  } else if (criticalCount >= 1 || warningCount >= 3) {
    showMessage('Abnormal vital patterns detected - review severity assessment', 'info');
  }
}

function setupSeverityAssessment() {
  // Add severity change suggestions based on other form inputs
  const consciousnessField = document.getElementById('consciousness');
  const breathingField = document.getElementById('breathing');
  
  if (consciousnessField) {
    consciousnessField.addEventListener('change', function() {
      if (this.value === 'unresponsive') {
        showMessage('Unresponsive patient - consider CRITICAL severity', 'warning');
      }
    });
  }
  
  if (breathingField) {
    breathingField.addEventListener('change', function() {
      if (this.value === 'absent') {
        showMessage('Absent breathing - consider CRITICAL severity', 'warning');
      }
    });
  }
}

function validateForm() {
  const isFormValid = Object.values(formValidation).every(valid => valid);
  const submitButton = document.getElementById('submit-btn');
  
  if (submitButton) {
    submitButton.disabled = !isFormValid;
    
    if (isFormValid) {
      submitButton.classList.remove('disabled');
      submitButton.textContent = '🚑 Submit Triage Assessment';
    } else {
      submitButton.classList.add('disabled');
      submitButton.textContent = '🚑 Complete Required Fields';
    }
  }
  
  return isFormValid;
}

function handleFormSubmission(event) {
  event.preventDefault();
  
  if (!validateForm()) {
    showMessage('Please complete all required fields', 'error');
    return;
  }
  
  const submitButton = document.getElementById('submit-btn');
  const originalText = submitButton.textContent;
  
  // Show loading state
  submitButton.disabled = true;
  submitButton.textContent = '🔄 Submitting Assessment...';
  
  // Collect form data
  const formData = new FormData(event.target);
  
  // Add timestamp
  formData.append('assessment_timestamp', new Date().toISOString());
  
  // Submit to server
  fetch('/submit-triage', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      showMessage('✅ Triage assessment submitted successfully', 'success');
      
      // Clear saved form data
      clearSavedFormData();
      
      // Redirect to dashboard after delay
      setTimeout(() => {
        window.location.href = '/triage-dashboard';
      }, 2000);
    } else {
      throw new Error(data.message || 'Submission failed');
    }
  })
  .catch(error => {
    console.error('Submission error:', error);
    showMessage('❌ Failed to submit assessment. Please try again.', 'error');
  })
  .finally(() => {
    submitButton.disabled = false;
    submitButton.textContent = originalText;
  });
}

function updateCurrentTime() {
  const timeElement = document.getElementById('current-time');
  if (timeElement) {
    const now = new Date();
    const timeString = now.toLocaleString('en-US', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
    timeElement.textContent = timeString;
  }
}

function autoSaveFormData() {
  const form = document.getElementById('triage-form');
  if (!form) return;
  
  const formData = new FormData(form);
  const saveData = {};
  
  for (let [key, value] of formData.entries()) {
    saveData[key] = value;
  }
  
  // Add selected triage color
  saveData.selected_triage_color = selectedTriageColor;
  
  // Save to localStorage (in a real app, might save to server)
  try {
    localStorage.setItem('triage_form_autosave', JSON.stringify({
      data: saveData,
      timestamp: new Date().toISOString()
    }));
  } catch (error) {
    console.warn('Could not auto-save form data:', error);
  }
}

function loadSavedFormData() {
  try {
    const savedData = localStorage.getItem('triage_form_autosave');
    if (!savedData) return;
    
    const { data, timestamp } = JSON.parse(savedData);
    const saveTime = new Date(timestamp);
    const now = new Date();
    
    // Only load if saved within last hour
    if (now - saveTime > 3600000) {
      clearSavedFormData();
      return;
    }
    
    // Ask user if they want to restore
    if (confirm('Found auto-saved form data. Would you like to restore it?')) {
      Object.entries(data).forEach(([key, value]) => {
        const field = document.getElementById(key);
        if (field) {
          if (field.type === 'radio') {
            const radio = document.querySelector(`input[name="${key}"][value="${value}"]`);
            if (radio) radio.checked = true;
          } else {
            field.value = value;
          }
        }
      });
      
      // Restore triage color selection
      if (data.selected_triage_color) {
        const colorOption = document.querySelector(`[data-color="${data.selected_triage_color}"]`);
        if (colorOption) {
          colorOption.click();
        }
      }
      
      showMessage('Form data restored from auto-save', 'info');
    }
  } catch (error) {
    console.warn('Could not load saved form data:', error);
    clearSavedFormData();
  }
}

function clearSavedFormData() {
  try {
    localStorage.removeItem('triage_form_autosave');
  } catch (error) {
    console.warn('Could not clear saved form data:', error);
  }
}

function handleKeyboardShortcuts(event) {
  if (event.ctrlKey || event.metaKey) {
    switch(event.key) {
      case 's':
        event.preventDefault();
        autoSaveFormData();
        showMessage('Form data saved', 'info');
        break;
      case 'Enter':
        if (event.target.tagName !== 'TEXTAREA') {
          event.preventDefault();
          const form = document.getElementById('triage-form');
          if (form && validateForm()) {
            form.dispatchEvent(new Event('submit'));
          }
        }
        break;
    }
  }
  
  // Quick triage color selection
  if (event.altKey) {
    switch(event.key) {
      case '1':
        document.querySelector('[data-color="red"]')?.click();
        break;
      case '2':
        document.querySelector('[data-color="yellow"]')?.click();
        break;
      case '3':
        document.querySelector('[data-color="green"]')?.click();
        break;
      case '4':
        document.querySelector('[data-color="black"]')?.click();
        break;
    }
  }
}

function showMessage(text, type = 'info') {
  const statusElement = document.getElementById('status-message');
  if (!statusElement) {
    // Create floating message if status element doesn't exist
    createFloatingMessage(text, type);
    return;
  }
  
  statusElement.textContent = text;
  statusElement.className = `status-message status-${type}`;
  statusElement.style.display = 'block';
  
  // Auto-hide after 5 seconds
  setTimeout(() => {
    statusElement.style.display = 'none';
  }, 5000);
}

function createFloatingMessage(text, type) {
  const message = document.createElement('div');
  message.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 1rem;
    border-radius: 8px;
    color: white;
    font-weight: bold;
    z-index: 9999;
    max-width: 300px;
    animation: slideInRight 0.3s ease;
    ${type === 'error' ? 'background: #dc2626;' : 
      type === 'warning' ? 'background: #f59e0b;' : 
      type === 'success' ? 'background: #16a34a;' :
      type === 'suggestion' ? 'background: #7c3aed;' :
      'background: #3b82f6;'}
  `;
  message.textContent = text;
  
  document.body.appendChild(message);
  
  setTimeout(() => {
    message.remove();
  }, 5000);
}

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
  
  @keyframes suggest-pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3); }
    50% { transform: scale(1.02); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5); }
  }
`;
document.head.appendChild(style);