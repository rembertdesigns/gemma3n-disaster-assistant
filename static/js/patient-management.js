/**
 * 🏥 Patient Management System - Emergency Department Patient Tracking
 * 
 * Comprehensive patient management utilities for emergency response systems
 * Includes patient tracking, status management, workflow optimization, and reporting
 */

import { openDB } from './idb.mjs';
import { ESI_LEVELS, TRIAGE_COLORS, calculateESILevel, assessVitalSigns } from './triage-utils.js';

// Configuration
const PATIENT_CONFIG = {
  DB_NAME: 'emergency_patients',
  DB_VERSION: 2,
  STORES: {
    PATIENTS: 'patients',
    VISITS: 'patient_visits',
    ASSESSMENTS: 'triage_assessments',
    NOTES: 'clinical_notes',
    MEDICATIONS: 'medications',
    PROCEDURES: 'procedures'
  },
  STATUS_CATEGORIES: [
    'registered', 'triaged', 'in_treatment', 'awaiting_results', 
    'ready_for_discharge', 'discharged', 'admitted', 'transferred'
  ],
  PRIORITY_LEVELS: ['critical', 'urgent', 'standard', 'low'],
  AUTO_SAVE_INTERVAL: 30000, // 30 seconds
  STATS_UPDATE_INTERVAL: 60000 // 1 minute
};

// Global state
let patientDatabase = null;
let activePatients = new Map();
let patientStats = {
  total: 0,
  byStatus: {},
  byPriority: {},
  averageWaitTime: 0,
  lastUpdated: null
};
let statusChangeCallbacks = [];

/**
 * Initialize patient management system
 */
export async function initializePatientManagement() {
  try {
    console.log('🏥 Initializing Patient Management System...');
    
    patientDatabase = await openDB(PATIENT_CONFIG.DB_NAME, PATIENT_CONFIG.DB_VERSION, {
      upgrade(db) {
        // Patients store
        if (!db.objectStoreNames.contains(PATIENT_CONFIG.STORES.PATIENTS)) {
          const patientsStore = db.createObjectStore(PATIENT_CONFIG.STORES.PATIENTS, { 
            keyPath: 'id' 
          });
          patientsStore.createIndex('status', 'status');
          patientsStore.createIndex('priority', 'priority');
          patientsStore.createIndex('triageTime', 'triageTime');
          patientsStore.createIndex('mrn', 'mrn');
        }
        
        // Patient visits store
        if (!db.objectStoreNames.contains(PATIENT_CONFIG.STORES.VISITS)) {
          const visitsStore = db.createObjectStore(PATIENT_CONFIG.STORES.VISITS, { 
            keyPath: 'id', autoIncrement: true 
          });
          visitsStore.createIndex('patientId', 'patientId');
          visitsStore.createIndex('visitDate', 'visitDate');
        }
        
        // Triage assessments store
        if (!db.objectStoreNames.contains(PATIENT_CONFIG.STORES.ASSESSMENTS)) {
          const assessmentsStore = db.createObjectStore(PATIENT_CONFIG.STORES.ASSESSMENTS, { 
            keyPath: 'id', autoIncrement: true 
          });
          assessmentsStore.createIndex('patientId', 'patientId');
          assessmentsStore.createIndex('timestamp', 'timestamp');
        }
        
        // Clinical notes store
        if (!db.objectStoreNames.contains(PATIENT_CONFIG.STORES.NOTES)) {
          const notesStore = db.createObjectStore(PATIENT_CONFIG.STORES.NOTES, { 
            keyPath: 'id', autoIncrement: true 
          });
          notesStore.createIndex('patientId', 'patientId');
          notesStore.createIndex('timestamp', 'timestamp');
        }
        
        // Medications store
        if (!db.objectStoreNames.contains(PATIENT_CONFIG.STORES.MEDICATIONS)) {
          const medsStore = db.createObjectStore(PATIENT_CONFIG.STORES.MEDICATIONS, { 
            keyPath: 'id', autoIncrement: true 
          });
          medsStore.createIndex('patientId', 'patientId');
        }
        
        // Procedures store
        if (!db.objectStoreNames.contains(PATIENT_CONFIG.STORES.PROCEDURES)) {
          const proceduresStore = db.createObjectStore(PATIENT_CONFIG.STORES.PROCEDURES, { 
            keyPath: 'id', autoIncrement: true 
          });
          proceduresStore.createIndex('patientId', 'patientId');
          proceduresStore.createIndex('timestamp', 'timestamp');
        }
      }
    });
    
    // Load active patients
    await loadActivePatients();
    
    // Setup periodic updates
    setupPeriodicUpdates();
    
    console.log('✅ Patient Management System initialized');
    return true;
    
  } catch (error) {
    console.error('❌ Failed to initialize Patient Management System:', error);
    throw error;
  }
}

/**
 * Register a new patient
 */
export async function registerPatient(patientData) {
  try {
    const patient = {
      id: patientData.id || generatePatientId(),
      mrn: patientData.mrn || generateMRN(),
      demographics: {
        firstName: patientData.firstName || '',
        lastName: patientData.lastName || '',
        dateOfBirth: patientData.dateOfBirth || '',
        gender: patientData.gender || '',
        address: patientData.address || {},
        phone: patientData.phone || '',
        emergencyContact: patientData.emergencyContact || {}
      },
      insurance: patientData.insurance || {},
      allergies: patientData.allergies || [],
      medications: patientData.medications || [],
      medicalHistory: patientData.medicalHistory || [],
      
      // Visit information
      visitId: generateVisitId(),
      arrivalTime: new Date().toISOString(),
      chiefComplaint: patientData.chiefComplaint || '',
      
      // Triage information
      status: 'registered',
      priority: 'standard',
      esiLevel: null,
      triageColor: null,
      triageTime: null,
      triageNurse: null,
      
      // Care team
      assignedProvider: null,
      assignedNurse: null,
      roomNumber: null,
      bedNumber: null,
      
      // Tracking
      timestamps: {
        registered: new Date().toISOString(),
        triaged: null,
        seenByProvider: null,
        discharged: null
      },
      
      // Flags
      flags: [],
      isActive: true,
      
      // Metadata
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    
    // Validate required fields
    const validation = validatePatientData(patient);
    if (!validation.isValid) {
      throw new Error(`Patient registration failed: ${validation.errors.join(', ')}`);
    }
    
    // Save to database
    const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.PATIENTS], 'readwrite');
    await tx.objectStore(PATIENT_CONFIG.STORES.PATIENTS).put(patient);
    await tx.complete;
    
    // Add to active patients
    activePatients.set(patient.id, patient);
    
    // Create initial visit record
    await createVisitRecord(patient);
    
    // Update statistics
    await updatePatientStatistics();
    
    // Notify listeners
    notifyStatusChange(patient.id, 'registered', null);
    
    console.log(`👤 Patient registered: ${patient.firstName} ${patient.lastName} (${patient.id})`);
    
    return patient;
    
  } catch (error) {
    console.error('❌ Failed to register patient:', error);
    throw error;
  }
}

/**
 * Update patient status
 */
export async function updatePatientStatus(patientId, newStatus, metadata = {}) {
  try {
    const patient = activePatients.get(patientId);
    if (!patient) {
      throw new Error(`Patient not found: ${patientId}`);
    }
    
    const oldStatus = patient.status;
    
    // Validate status transition
    if (!isValidStatusTransition(oldStatus, newStatus)) {
      throw new Error(`Invalid status transition from ${oldStatus} to ${newStatus}`);
    }
    
    // Update patient record
    patient.status = newStatus;
    patient.updatedAt = new Date().toISOString();
    patient.timestamps[newStatus] = new Date().toISOString();
    
    // Add metadata if provided
    if (metadata.provider) patient.assignedProvider = metadata.provider;
    if (metadata.nurse) patient.assignedNurse = metadata.nurse;
    if (metadata.room) patient.roomNumber = metadata.room;
    if (metadata.bed) patient.bedNumber = metadata.bed;
    if (metadata.notes) {
      await addClinicalNote(patientId, metadata.notes, metadata.provider || 'System');
    }
    
    // Save to database
    const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.PATIENTS], 'readwrite');
    await tx.objectStore(PATIENT_CONFIG.STORES.PATIENTS).put(patient);
    await tx.complete;
    
    // Update active patients map
    activePatients.set(patientId, patient);
    
    // Handle status-specific actions
    await handleStatusChange(patient, oldStatus, newStatus);
    
    // Update statistics
    await updatePatientStatistics();
    
    // Notify listeners
    notifyStatusChange(patientId, newStatus, oldStatus);
    
    console.log(`📋 Patient ${patientId} status updated: ${oldStatus} → ${newStatus}`);
    
    return patient;
    
  } catch (error) {
    console.error('❌ Failed to update patient status:', error);
    throw error;
  }
}

/**
 * Perform triage assessment
 */
export async function performTriage(patientId, triageData) {
  try {
    const patient = activePatients.get(patientId);
    if (!patient) {
      throw new Error(`Patient not found: ${patientId}`);
    }
    
    // Calculate ESI level and triage color
    const esiResult = calculateESILevel({
      vitals: triageData.vitals,
      consciousness: triageData.consciousness,
      breathing: triageData.breathing,
      circulation: triageData.circulation,
      chiefComplaint: patient.chiefComplaint,
      age: calculateAge(patient.demographics.dateOfBirth),
      resourceNeeds: triageData.resourceNeeds || []
    });
    
    // Assess vital signs
    const vitalAssessment = assessVitalSigns(
      triageData.vitals, 
      calculateAge(patient.demographics.dateOfBirth)
    );
    
    // Create triage assessment record
    const assessment = {
      patientId: patientId,
      timestamp: new Date().toISOString(),
      nurseId: triageData.nurseId,
      vitals: triageData.vitals,
      esiLevel: esiResult.level,
      esiReasoning: esiResult.reasoning,
      triageColor: triageData.triageColor,
      painScale: triageData.painScale,
      consciousness: triageData.consciousness,
      breathing: triageData.breathing,
      circulation: triageData.circulation,
      vitalAssessment: vitalAssessment,
      chiefComplaint: triageData.chiefComplaint || patient.chiefComplaint,
      historyPresentIllness: triageData.historyPresentIllness,
      allergies: triageData.allergies,
      currentMedications: triageData.currentMedications,
      medicalHistory: triageData.medicalHistory,
      recommendations: esiResult.recommendations,
      flags: triageData.flags || [],
      notes: triageData.notes
    };
    
    // Save assessment to database
    const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.ASSESSMENTS], 'readwrite');
    await tx.objectStore(PATIENT_CONFIG.STORES.ASSESSMENTS).add(assessment);
    await tx.complete;
    
    // Update patient record
    patient.esiLevel = esiResult.level;
    patient.triageColor = triageData.triageColor;
    patient.priority = determinePriority(esiResult.level, triageData.triageColor);
    patient.triageTime = new Date().toISOString();
    patient.triageNurse = triageData.nurseId;
    patient.flags = [...new Set([...patient.flags, ...triageData.flags])];
    
    // Update patient status to triaged
    await updatePatientStatus(patientId, 'triaged', {
      nurse: triageData.nurseId
    });
    
    console.log(`🏥 Triage completed for patient ${patientId}: ESI ${esiResult.level}, ${triageData.triageColor} triage`);
    
    return {
      patient,
      assessment,
      esiResult,
      vitalAssessment
    };
    
  } catch (error) {
    console.error('❌ Failed to perform triage:', error);
    throw error;
  }
}

/**
 * Add clinical note
 */
export async function addClinicalNote(patientId, noteText, providerId, noteType = 'general') {
  try {
    const note = {
      patientId: patientId,
      timestamp: new Date().toISOString(),
      providerId: providerId,
      noteType: noteType,
      text: noteText,
      isAmended: false,
      amendmentHistory: []
    };
    
    const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.NOTES], 'readwrite');
    const result = await tx.objectStore(PATIENT_CONFIG.STORES.NOTES).add(note);
    await tx.complete;
    
    note.id = result;
    
    console.log(`📝 Clinical note added for patient ${patientId} by ${providerId}`);
    
    return note;
    
  } catch (error) {
    console.error('❌ Failed to add clinical note:', error);
    throw error;
  }
}

/**
 * Get patient by ID
 */
export async function getPatient(patientId) {
  try {
    // Try active patients first
    if (activePatients.has(patientId)) {
      return activePatients.get(patientId);
    }
    
    // Search database
    const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.PATIENTS], 'readonly');
    const patient = await tx.objectStore(PATIENT_CONFIG.STORES.PATIENTS).get(patientId);
    await tx.complete;
    
    return patient || null;
    
  } catch (error) {
    console.error('❌ Failed to get patient:', error);
    return null;
  }
}

/**
 * Search patients
 */
export async function searchPatients(criteria) {
  try {
    const results = [];
    
    // Search active patients first
    for (const [id, patient] of activePatients) {
      if (matchesSearchCriteria(patient, criteria)) {
        results.push(patient);
      }
    }
    
    // If not enough results, search database
    if (results.length < 10) {
      const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.PATIENTS], 'readonly');
      const store = tx.objectStore(PATIENT_CONFIG.STORES.PATIENTS);
      
      // Get all patients if criteria is broad
      const allPatients = await store.getAll();
      
      for (const patient of allPatients) {
        if (results.length >= 50) break; // Limit results
        
        if (matchesSearchCriteria(patient, criteria) && !results.find(p => p.id === patient.id)) {
          results.push(patient);
        }
      }
      
      await tx.complete;
    }
    
    return results.sort((a, b) => new Date(b.arrivalTime) - new Date(a.arrivalTime));
    
  } catch (error) {
    console.error('❌ Failed to search patients:', error);
    return [];
  }
}

/**
 * Get patient visit history
 */
export async function getPatientHistory(patientId) {
  try {
    const tx = patientDatabase.transaction([
      PATIENT_CONFIG.STORES.VISITS,
      PATIENT_CONFIG.STORES.ASSESSMENTS,
      PATIENT_CONFIG.STORES.NOTES,
      PATIENT_CONFIG.STORES.PROCEDURES
    ], 'readonly');
    
    const [visits, assessments, notes, procedures] = await Promise.all([
      tx.objectStore(PATIENT_CONFIG.STORES.VISITS).index('patientId').getAll(patientId),
      tx.objectStore(PATIENT_CONFIG.STORES.ASSESSMENTS).index('patientId').getAll(patientId),
      tx.objectStore(PATIENT_CONFIG.STORES.NOTES).index('patientId').getAll(patientId),
      tx.objectStore(PATIENT_CONFIG.STORES.PROCEDURES).index('patientId').getAll(patientId)
    ]);
    
    await tx.complete;
    
    return {
      visits: visits.sort((a, b) => new Date(b.visitDate) - new Date(a.visitDate)),
      assessments: assessments.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)),
      notes: notes.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp)),
      procedures: procedures.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
    };
    
  } catch (error) {
    console.error('❌ Failed to get patient history:', error);
    return { visits: [], assessments: [], notes: [], procedures: [] };
  }
}

/**
 * Get active patients by status
 */
export function getActivePatientsByStatus(status = null) {
  const patients = Array.from(activePatients.values());
  
  if (status) {
    return patients.filter(p => p.status === status && p.isActive);
  }
  
  return patients.filter(p => p.isActive);
}

/**
 * Get patient wait times
 */
export function calculateWaitTimes() {
  const now = new Date();
  const waitTimes = [];
  
  for (const patient of activePatients.values()) {
    if (!patient.isActive) continue;
    
    const arrivalTime = new Date(patient.arrivalTime);
    const waitTime = now - arrivalTime;
    
    waitTimes.push({
      patientId: patient.id,
      name: `${patient.demographics.firstName} ${patient.demographics.lastName}`,
      status: patient.status,
      esiLevel: patient.esiLevel,
      waitTimeMinutes: Math.floor(waitTime / 60000),
      arrivalTime: patient.arrivalTime,
      isOverdue: isPatientOverdue(patient, waitTime)
    });
  }
  
  return waitTimes.sort((a, b) => b.waitTimeMinutes - a.waitTimeMinutes);
}

/**
 * Get department statistics
 */
export function getDepartmentStatistics() {
  const stats = {
    totalActive: 0,
    byStatus: {},
    byPriority: {},
    byESILevel: {},
    avgWaitTime: 0,
    overduePatients: 0,
    bedOccupancy: calculateBedOccupancy(),
    lastUpdated: new Date().toISOString()
  };
  
  let totalWaitTime = 0;
  const now = new Date();
  
  for (const patient of activePatients.values()) {
    if (!patient.isActive) continue;
    
    stats.totalActive++;
    
    // By status
    stats.byStatus[patient.status] = (stats.byStatus[patient.status] || 0) + 1;
    
    // By priority
    stats.byPriority[patient.priority] = (stats.byPriority[patient.priority] || 0) + 1;
    
    // By ESI level
    if (patient.esiLevel) {
      stats.byESILevel[patient.esiLevel] = (stats.byESILevel[patient.esiLevel] || 0) + 1;
    }
    
    // Wait time calculations
    const arrivalTime = new Date(patient.arrivalTime);
    const waitTime = now - arrivalTime;
    totalWaitTime += waitTime;
    
    if (isPatientOverdue(patient, waitTime)) {
      stats.overduePatients++;
    }
  }
  
  stats.avgWaitTime = stats.totalActive > 0 ? Math.floor(totalWaitTime / stats.totalActive / 60000) : 0;
  
  return stats;
}

/**
 * Export patient data
 */
export async function exportPatientData(patientIds, format = 'json') {
  try {
    const patients = [];
    
    for (const patientId of patientIds) {
      const patient = await getPatient(patientId);
      if (patient) {
        const history = await getPatientHistory(patientId);
        patients.push({
          patient,
          history
        });
      }
    }
    
    switch (format.toLowerCase()) {
      case 'csv':
        return convertPatientsToCSV(patients);
      case 'json':
        return JSON.stringify(patients, null, 2);
      case 'xml':
        return convertPatientsToXML(patients);
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
    
  } catch (error) {
    console.error('❌ Failed to export patient data:', error);
    throw error;
  }
}

// Helper functions

function generatePatientId() {
  return `PT-${Date.now()}-${Math.random().toString(36).substr(2, 6).toUpperCase()}`;
}

function generateMRN() {
  return `MRN${Date.now().toString().slice(-8)}`;
}

function generateVisitId() {
  return `VIS-${Date.now()}-${Math.random().toString(36).substr(2, 4).toUpperCase()}`;
}

function calculateAge(dateOfBirth) {
  if (!dateOfBirth) return null;
  
  const today = new Date();
  const birth = new Date(dateOfBirth);
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) {
    age--;
  }
  
  return age;
}

function determinePriority(esiLevel, triageColor) {
  if (esiLevel <= 2 || triageColor === 'RED') return 'critical';
  if (esiLevel === 3 || triageColor === 'YELLOW') return 'urgent';
  if (esiLevel === 4) return 'standard';
  return 'low';
}

function isValidStatusTransition(fromStatus, toStatus) {
  const validTransitions = {
    'registered': ['triaged', 'left_without_treatment'],
    'triaged': ['in_treatment', 'awaiting_results', 'left_without_treatment'],
    'in_treatment': ['awaiting_results', 'ready_for_discharge', 'admitted', 'transferred'],
    'awaiting_results': ['in_treatment', 'ready_for_discharge', 'admitted'],
    'ready_for_discharge': ['discharged'],
    'discharged': [],
    'admitted': [],
    'transferred': [],
    'left_without_treatment': []
  };
  
  return validTransitions[fromStatus]?.includes(toStatus) || false;
}

function validatePatientData(patient) {
  const errors = [];
  
  if (!patient.demographics.firstName) {
    errors.push('First name is required');
  }
  
  if (!patient.demographics.lastName) {
    errors.push('Last name is required');
  }
  
  if (!patient.chiefComplaint) {
    errors.push('Chief complaint is required');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
}

function matchesSearchCriteria(patient, criteria) {
  if (!criteria) return true;
  
  const searchText = criteria.toLowerCase();
  
  return (
    patient.demographics.firstName.toLowerCase().includes(searchText) ||
    patient.demographics.lastName.toLowerCase().includes(searchText) ||
    patient.mrn.toLowerCase().includes(searchText) ||
    patient.id.toLowerCase().includes(searchText) ||
    (patient.chiefComplaint && patient.chiefComplaint.toLowerCase().includes(searchText))
  );
}

async function createVisitRecord(patient) {
  const visit = {
    patientId: patient.id,
    visitId: patient.visitId,
    visitDate: patient.arrivalTime,
    chiefComplaint: patient.chiefComplaint,
    disposition: null,
    dischargeTime: null,
    lengthOfStay: null
  };
  
  const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.VISITS], 'readwrite');
  await tx.objectStore(PATIENT_CONFIG.STORES.VISITS).add(visit);
  await tx.complete;
}

async function handleStatusChange(patient, oldStatus, newStatus) {
  // Implement status-specific logic
  switch (newStatus) {
    case 'discharged':
    case 'admitted':
    case 'transferred':
      patient.isActive = false;
      activePatients.delete(patient.id);
      break;
  }
}

function isPatientOverdue(patient, waitTime) {
  if (!patient.esiLevel) return false;
  
  const maxWaitMinutes = ESI_LEVELS[patient.esiLevel]?.maxWaitTime || 240;
  const waitMinutes = waitTime / 60000;
  
  return waitMinutes > maxWaitMinutes;
}

function calculateBedOccupancy() {
  // Placeholder - would integrate with actual bed management system
  return {
    total: 50,
    occupied: Array.from(activePatients.values()).filter(p => p.roomNumber).length,
    available: 50 - Array.from(activePatients.values()).filter(p => p.roomNumber).length
  };
}

async function loadActivePatients() {
  try {
    const tx = patientDatabase.transaction([PATIENT_CONFIG.STORES.PATIENTS], 'readonly');
    const allPatients = await tx.objectStore(PATIENT_CONFIG.STORES.PATIENTS).getAll();
    await tx.complete;
    
    for (const patient of allPatients) {
      if (patient.isActive) {
        activePatients.set(patient.id, patient);
      }
    }
    
    console.log(`📊 Loaded ${activePatients.size} active patients`);
    
  } catch (error) {
    console.error('❌ Failed to load active patients:', error);
  }
}

async function updatePatientStatistics() {
  patientStats = getDepartmentStatistics();
  patientStats.lastUpdated = new Date().toISOString();
}

function setupPeriodicUpdates() {
  setInterval(async () => {
    await updatePatientStatistics();
  }, PATIENT_CONFIG.STATS_UPDATE_INTERVAL);
}

function notifyStatusChange(patientId, newStatus, oldStatus) {
  statusChangeCallbacks.forEach(callback => {
    try {
      callback({ patientId, newStatus, oldStatus, timestamp: new Date().toISOString() });
    } catch (error) {
      console.error('❌ Status change callback error:', error);
    }
  });
}

function convertPatientsToCSV(patients) {
  const headers = [
    'Patient ID', 'MRN', 'First Name', 'Last Name', 'Age', 'Gender',
    'Arrival Time', 'Status', 'ESI Level', 'Triage Color', 'Chief Complaint'
  ];
  
  const rows = [headers.join(',')];
  
  patients.forEach(({ patient }) => {
    const age = calculateAge(patient.demographics.dateOfBirth);
    const row = [
      patient.id,
      patient.mrn,
      patient.demographics.firstName,
      patient.demographics.lastName,
      age || '',
      patient.demographics.gender,
      patient.arrivalTime,
      patient.status,
      patient.esiLevel || '',
      patient.triageColor || '',
      patient.chiefComplaint
    ].map(field => `"${field}"`);
    
    rows.push(row.join(','));
  });
  
  return rows.join('\n');
}

function convertPatientsToXML(patients) {
  let xml = '<?xml version="1.0" encoding="UTF-8"?>\n<patients>\n';
  
  patients.forEach(({ patient, history }) => {
    xml += `  <patient id="${patient.id}">\n`;
    xml += `    <demographics>\n`;
    xml += `      <firstName>${escapeXML(patient.demographics.firstName)}</firstName>\n`;
    xml += `      <lastName>${escapeXML(patient.demographics.lastName)}</lastName>\n`;
    xml += `    </demographics>\n`;
    xml += `    <visit>\n`;
    xml += `      <status>${patient.status}</status>\n`;
    xml += `      <arrivalTime>${patient.arrivalTime}</arrivalTime>\n`;
    xml += `    </visit>\n`;
    xml += `  </patient>\n`;
  });
  
  xml += '</patients>';
  return xml;
}

function escapeXML(str) {
  return str.replace(/[<>&'"]/g, (c) => {
    switch (c) {
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '&': return '&amp;';
      case "'": return '&apos;';
      case '"': return '&quot;';
    }
  });
}

/**
 * Subscribe to status changes
 */
export function onStatusChange(callback) {
  statusChangeCallbacks.push(callback);
  
  return () => {
    const index = statusChangeCallbacks.indexOf(callback);
    if (index > -1) {
      statusChangeCallbacks.splice(index, 1);
    }
  };
}

/**
 * Get current patient statistics
 */
export function getPatientStatistics() {
  return { ...patientStats };
}

console.log('🏥 Patient Management utilities loaded successfully');