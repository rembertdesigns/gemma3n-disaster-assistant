/**
 * 🏥 Triage Utilities - Medical Emergency Assessment Tools
 * 
 * Essential utility functions for emergency medical triage systems
 * Based on Emergency Severity Index (ESI) and medical best practices
 */

// Emergency Severity Index (ESI) levels
export const ESI_LEVELS = {
    1: { 
      name: 'Resuscitation', 
      color: '#dc2626', 
      description: 'Life-threatening, requires immediate intervention',
      maxWaitTime: 0
    },
    2: { 
      name: 'Emergent', 
      color: '#f59e0b', 
      description: 'High risk, should not wait',
      maxWaitTime: 10
    },
    3: { 
      name: 'Urgent', 
      color: '#eab308', 
      description: 'Stable but needs multiple resources',
      maxWaitTime: 60
    },
    4: { 
      name: 'Less Urgent', 
      color: '#22c55e', 
      description: 'Stable, needs one resource',
      maxWaitTime: 120
    },
    5: { 
      name: 'Non-urgent', 
      color: '#3b82f6', 
      description: 'Stable, may not need resources',
      maxWaitTime: 240
    }
  };
  
  // Triage color categories (Standard START/JumpSTART)
  export const TRIAGE_COLORS = {
    RED: {
      name: 'Immediate',
      priority: 1,
      description: 'Life-threatening injuries requiring immediate care',
      color: '#dc2626',
      icon: '🔴'
    },
    YELLOW: {
      name: 'Delayed',
      priority: 2,
      description: 'Serious but stable injuries, can wait for treatment',
      color: '#eab308',
      icon: '🟡'
    },
    GREEN: {
      name: 'Minor',
      priority: 3,
      description: 'Minor injuries, can walk and wait',
      color: '#22c55e',
      icon: '🟢'
    },
    BLACK: {
      name: 'Deceased/Expectant',
      priority: 4,
      description: 'Deceased or injuries incompatible with life',
      color: '#1f2937',
      icon: '⚫'
    }
  };
  
  // Vital signs normal ranges (adult)
  export const VITAL_RANGES = {
    heartRate: {
      normal: { min: 60, max: 100 },
      warning: { min: 50, max: 120 },
      critical: { min: 40, max: 150 }
    },
    bloodPressure: {
      systolic: {
        normal: { min: 90, max: 140 },
        warning: { min: 80, max: 160 },
        critical: { min: 70, max: 180 }
      },
      diastolic: {
        normal: { min: 60, max: 90 },
        warning: { min: 50, max: 100 },
        critical: { min: 40, max: 110 }
      }
    },
    respiratory: {
      normal: { min: 12, max: 20 },
      warning: { min: 10, max: 24 },
      critical: { min: 8, max: 30 }
    },
    temperature: {
      normal: { min: 97.0, max: 99.5 },
      warning: { min: 95.0, max: 101.0 },
      critical: { min: 92.0, max: 104.0 }
    },
    oxygenSaturation: {
      normal: { min: 95, max: 100 },
      warning: { min: 90, max: 94 },
      critical: { min: 0, max: 89 }
    }
  };
  
  /**
   * Calculate Emergency Severity Index (ESI) level
   * @param {Object} assessment - Patient assessment data
   * @returns {Object} ESI level and reasoning
   */
  export function calculateESILevel(assessment) {
    const {
      vitals = {},
      consciousness = 'alert',
      breathing = 'normal',
      circulation = 'stable',
      chiefComplaint = '',
      age = 30,
      resourceNeeds = []
    } = assessment;
  
    // ESI Level 1: Life-threatening
    const level1Criteria = [
      vitals.heartRate < 40 || vitals.heartRate > 150,
      vitals.oxygenSat < 90,
      consciousness === 'unresponsive',
      breathing === 'absent' || breathing === 'agonal',
      circulation === 'no_pulse',
      chiefComplaint.toLowerCase().includes('cardiac arrest'),
      chiefComplaint.toLowerCase().includes('respiratory arrest')
    ];
  
    if (level1Criteria.some(criteria => criteria)) {
      return {
        level: 1,
        ...ESI_LEVELS[1],
        reasoning: 'Life-threatening condition requiring immediate intervention',
        recommendations: ['Immediate resuscitation', 'Code team activation', 'ABC assessment']
      };
    }
  
    // ESI Level 2: High risk, should not wait
    const level2Criteria = [
      vitals.heartRate < 50 || vitals.heartRate > 120,
      vitals.oxygenSat < 95,
      vitals.systolic < 90 || vitals.systolic > 180,
      consciousness === 'confused' || consciousness === 'lethargic',
      breathing === 'labored',
      age > 65 && (vitals.heartRate > 100 || vitals.temperature > 101),
      chiefComplaint.toLowerCase().includes('chest pain'),
      chiefComplaint.toLowerCase().includes('shortness of breath')
    ];
  
    if (level2Criteria.some(criteria => criteria)) {
      return {
        level: 2,
        ...ESI_LEVELS[2],
        reasoning: 'High-risk situation requiring prompt evaluation',
        recommendations: ['Rapid assessment', 'Monitor vitals', 'IV access if needed']
      };
    }
  
    // ESI Level 3-5: Based on resource needs
    const resourceCount = resourceNeeds.length;
    
    if (resourceCount >= 2) {
      return {
        level: 3,
        ...ESI_LEVELS[3],
        reasoning: 'Multiple resources needed for care',
        recommendations: ['Standard triage process', 'Monitor for changes']
      };
    } else if (resourceCount === 1) {
      return {
        level: 4,
        ...ESI_LEVELS[4],
        reasoning: 'Single resource needed',
        recommendations: ['Standard care', 'Queue for provider']
      };
    } else {
      return {
        level: 5,
        ...ESI_LEVELS[5],
        reasoning: 'Low acuity, minimal resources needed',
        recommendations: ['Self-care education', 'Follow-up as needed']
      };
    }
  }
  
  /**
   * Assess vital signs and return status
   * @param {Object} vitals - Vital signs object
   * @param {number} age - Patient age
   * @returns {Object} Vital signs assessment
   */
  export function assessVitalSigns(vitals, age = 30) {
    const assessment = {
      overall: 'normal',
      alerts: [],
      score: 0
    };
  
    // Adjust ranges for pediatric patients
    const ranges = age < 18 ? getPediatricVitalRanges(age) : VITAL_RANGES;
  
    // Check each vital sign
    Object.entries(vitals).forEach(([vital, value]) => {
      if (value === null || value === undefined) return;
  
      const range = ranges[vital];
      if (!range) return;
  
      let status = 'normal';
      let points = 0;
  
      if (vital === 'bloodPressure') {
        // Special handling for blood pressure
        const systolicStatus = assessSingleVital(value.systolic, range.systolic);
        const diastolicStatus = assessSingleVital(value.diastolic, range.diastolic);
        
        status = getWorseStatus(systolicStatus.status, diastolicStatus.status);
        points = Math.max(systolicStatus.points, diastolicStatus.points);
      } else {
        const result = assessSingleVital(value, range);
        status = result.status;
        points = result.points;
      }
  
      if (status !== 'normal') {
        assessment.alerts.push({
          vital,
          value,
          status,
          message: `${vital} is ${status}: ${value}`
        });
      }
  
      assessment.score += points;
    });
  
    // Overall assessment based on worst individual vital and total score
    if (assessment.alerts.some(alert => alert.status === 'critical')) {
      assessment.overall = 'critical';
    } else if (assessment.alerts.some(alert => alert.status === 'warning') || assessment.score >= 3) {
      assessment.overall = 'warning';
    }
  
    return assessment;
  }
  
  /**
   * Assess a single vital sign against ranges
   */
  function assessSingleVital(value, ranges) {
    if (value < ranges.critical.min || value > ranges.critical.max) {
      return { status: 'critical', points: 3 };
    }
    if (value < ranges.warning.min || value > ranges.warning.max) {
      return { status: 'warning', points: 2 };
    }
    if (value < ranges.normal.min || value > ranges.normal.max) {
      return { status: 'abnormal', points: 1 };
    }
    return { status: 'normal', points: 0 };
  }
  
  /**
   * Get worse of two statuses
   */
  function getWorseStatus(status1, status2) {
    const severity = { normal: 0, abnormal: 1, warning: 2, critical: 3 };
    return severity[status1] >= severity[status2] ? status1 : status2;
  }
  
  /**
   * Get pediatric vital sign ranges based on age
   */
  function getPediatricVitalRanges(age) {
    // Simplified pediatric ranges - in practice, use more detailed age-based ranges
    if (age < 1) {
      return {
        heartRate: { normal: { min: 100, max: 160 }, warning: { min: 80, max: 180 }, critical: { min: 60, max: 200 } },
        respiratory: { normal: { min: 30, max: 60 }, warning: { min: 20, max: 70 }, critical: { min: 15, max: 80 } }
      };
    } else if (age < 5) {
      return {
        heartRate: { normal: { min: 90, max: 150 }, warning: { min: 70, max: 170 }, critical: { min: 50, max: 190 } },
        respiratory: { normal: { min: 20, max: 40 }, warning: { min: 15, max: 50 }, critical: { min: 10, max: 60 } }
      };
    }
    
    // Use adult ranges for older children (simplified)
    return VITAL_RANGES;
  }
  
  /**
   * Calculate triage color based on START algorithm
   * @param {Object} assessment - Patient assessment
   * @returns {Object} Triage color and priority
   */
  export function calculateTriageColor(assessment) {
    const {
      walking = false,
      breathing = 'normal',
      pulse = 'present',
      capillaryRefill = 'normal',
      consciousness = 'alert',
      vitals = {}
    } = assessment;
  
    // GREEN: Walking wounded
    if (walking) {
      return {
        color: 'GREEN',
        ...TRIAGE_COLORS.GREEN,
        reasoning: 'Patient is ambulatory'
      };
    }
  
    // RED: Immediate
    const redCriteria = [
      breathing === 'absent' && pulse === 'present', // Airway obstruction
      vitals.respiratory > 30 || vitals.respiratory < 10,
      pulse === 'absent',
      capillaryRefill === 'delayed' || vitals.heartRate > 120,
      consciousness === 'unresponsive' || consciousness === 'altered'
    ];
  
    if (redCriteria.some(criteria => criteria)) {
      return {
        color: 'RED',
        ...TRIAGE_COLORS.RED,
        reasoning: 'Life-threatening condition requiring immediate care'
      };
    }
  
    // BLACK: Deceased/Expectant
    if (breathing === 'absent' && pulse === 'absent') {
      return {
        color: 'BLACK',
        ...TRIAGE_COLORS.BLACK,
        reasoning: 'No signs of life'
      };
    }
  
    // YELLOW: Delayed (default for non-walking, stable patients)
    return {
      color: 'YELLOW',
      ...TRIAGE_COLORS.YELLOW,
      reasoning: 'Stable but requires medical attention'
    };
  }
  
  /**
   * Generate triage summary report
   * @param {Object} patientData - Complete patient assessment
   * @returns {Object} Comprehensive triage summary
   */
  export function generateTriageSummary(patientData) {
    const {
      demographics = {},
      vitals = {},
      assessment = {},
      chiefComplaint = '',
      timestamp = new Date().toISOString()
    } = patientData;
  
    const vitalAssessment = assessVitalSigns(vitals, demographics.age);
    const esiLevel = calculateESILevel({ ...assessment, vitals, chiefComplaint, age: demographics.age });
    const triageColor = calculateTriageColor({ ...assessment, vitals });
  
    const summary = {
      patient: {
        name: demographics.name || 'Unknown',
        age: demographics.age || 'Unknown',
        gender: demographics.gender || 'Unknown'
      },
      triage: {
        timestamp,
        esiLevel: esiLevel.level,
        esiDescription: esiLevel.description,
        triageColor: triageColor.color,
        priority: triageColor.priority
      },
      vitals: {
        assessment: vitalAssessment.overall,
        score: vitalAssessment.score,
        alerts: vitalAssessment.alerts
      },
      recommendations: [
        ...esiLevel.recommendations,
        ...(vitalAssessment.overall === 'critical' ? ['Continuous monitoring', 'Prepare for intervention'] : []),
        ...(triageColor.color === 'RED' ? ['Immediate physician evaluation'] : [])
      ],
      disposition: determineDisposition(esiLevel.level, triageColor.color, vitalAssessment.overall),
      estimatedWaitTime: ESI_LEVELS[esiLevel.level].maxWaitTime
    };
  
    return summary;
  }
  
  /**
   * Determine patient disposition based on triage findings
   */
  function determineDisposition(esiLevel, triageColor, vitalStatus) {
    if (esiLevel <= 2 || triageColor === 'RED' || vitalStatus === 'critical') {
      return {
        area: 'Resuscitation Bay',
        urgency: 'Immediate',
        staffing: 'Physician and nurse immediately'
      };
    } else if (esiLevel === 3 || triageColor === 'YELLOW') {
      return {
        area: 'Acute Care Area',
        urgency: 'Urgent',
        staffing: 'Nurse assessment, physician within 30 minutes'
      };
    } else {
      return {
        area: 'Fast Track/Urgent Care',
        urgency: 'Standard',
        staffing: 'Standard triage flow'
      };
    }
  }
  
  /**
   * Validate triage assessment data
   * @param {Object} data - Triage assessment data
   * @returns {Object} Validation result
   */
  export function validateTriageData(data) {
    const errors = [];
    const warnings = [];
  
    // Required fields
    const required = ['name', 'chiefComplaint'];
    required.forEach(field => {
      if (!data[field] || data[field].trim() === '') {
        errors.push(`${field} is required`);
      }
    });
  
    // Vital signs validation
    if (data.vitals) {
      const { vitals } = data;
      
      if (vitals.heartRate && (vitals.heartRate < 20 || vitals.heartRate > 250)) {
        warnings.push('Heart rate seems unusual, please verify');
      }
      
      if (vitals.temperature && (vitals.temperature < 90 || vitals.temperature > 110)) {
        warnings.push('Temperature seems unusual, please verify');
      }
      
      if (vitals.oxygenSat && (vitals.oxygenSat < 50 || vitals.oxygenSat > 100)) {
        errors.push('Oxygen saturation must be between 50-100%');
      }
    }
  
    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }
  
  /**
   * Format time for triage timestamps
   */
  export function formatTriageTime(timestamp) {
    const date = new Date(timestamp);
    return {
      full: date.toLocaleString(),
      time: date.toLocaleTimeString(),
      relative: getTimeAgo(timestamp)
    };
  }
  
  /**
   * Get relative time (e.g., "5 minutes ago")
   */
  function getTimeAgo(timestamp) {
    const now = Date.now();
    const time = new Date(timestamp).getTime();
    const diff = now - time;
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} minutes ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
    return `${Math.floor(diff / 86400000)} days ago`;
  }
  
  /**
   * Export triage data to standardized format
   */
  export function exportTriageData(triageData, format = 'json') {
    const standardized = {
      version: '1.0',
      timestamp: new Date().toISOString(),
      facility: 'Emergency Department',
      data: triageData
    };
  
    switch (format.toLowerCase()) {
      case 'csv':
        return convertToCSV(triageData);
      case 'hl7':
        return convertToHL7(triageData);
      default:
        return JSON.stringify(standardized, null, 2);
    }
  }
  
  /**
   * Convert to CSV format (simplified)
   */
  function convertToCSV(data) {
    const headers = ['Name', 'Age', 'Chief Complaint', 'ESI Level', 'Triage Color', 'Timestamp'];
    const rows = [headers.join(',')];
    
    if (Array.isArray(data)) {
      data.forEach(patient => {
        const row = [
          patient.name || '',
          patient.age || '',
          patient.chiefComplaint || '',
          patient.esiLevel || '',
          patient.triageColor || '',
          patient.timestamp || ''
        ].map(field => `"${field}"`);
        rows.push(row.join(','));
      });
    }
    
    return rows.join('\n');
  }
  
  /**
   * Convert to HL7 format (simplified)
   */
  function convertToHL7(data) {
    // Simplified HL7 ADT message structure
    return `MSH|^~\\&|TRIAGE|ED|HIS|HOSPITAL|${new Date().toISOString().replace(/[-:]/g, '').slice(0, 14)}||ADT^A04|${Date.now()}|P|2.5\r\n` +
           `EVN|A04|${new Date().toISOString().replace(/[-:]/g, '').slice(0, 14)}\r\n` +
           `PID|||${data.id || 'UNKNOWN'}||${data.name || 'UNKNOWN'}|||${data.gender || 'U'}||||||||||||\r\n` +
           `PV1|1|E|ED^${data.triageColor || 'UNKNOWN'}^1||||||||||||${data.esiLevel || 'UNKNOWN'}|\r\n`;
  }
  
  console.log('🏥 Triage utilities loaded successfully');