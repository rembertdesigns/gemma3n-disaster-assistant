const PDF_CONFIG = {
    pageSize: 'letter',
    margins: { top: 72, bottom: 72, left: 72, right: 72 },
    defaultFont: 'Helvetica',
    colors: {
      primary: '#2563eb',
      danger: '#dc2626',
      warning: '#f59e0b',
      success: '#16a34a',
      text: '#374151',
      light: '#f9fafb'
    },
    logoUrl: '/static/images/logo.png'
  };
  
  /**
   * Generate Triage Assessment PDF Report
   * @param {Object} triageData - Triage assessment data
   * @param {Object} options - PDF generation options
   * @returns {Promise<Blob>} PDF blob
   */
  export async function generateTriagePDF(triageData, options = {}) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'pt',
      format: options.pageSize || PDF_CONFIG.pageSize
    });
  
    try {
      // Document setup
      let yPosition = setupDocumentHeader(doc, 'EMERGENCY TRIAGE ASSESSMENT', triageData.timestamp);
      
      // Patient information section
      yPosition = addPatientInfo(doc, triageData.patient, yPosition);
      
      // Vital signs section
      if (triageData.vitals) {
        yPosition = addVitalSigns(doc, triageData.vitals, yPosition);
      }
      
      // Triage results section
      yPosition = addTriageResults(doc, triageData.triage, yPosition);
      
      // Assessment details
      if (triageData.assessment) {
        yPosition = addAssessmentDetails(doc, triageData.assessment, yPosition);
      }
      
      // Recommendations
      if (triageData.recommendations) {
        yPosition = addRecommendations(doc, triageData.recommendations, yPosition);
      }
      
      // Footer
      addDocumentFooter(doc, triageData);
      
      return doc.output('blob');
      
    } catch (error) {
      console.error('❌ Failed to generate triage PDF:', error);
      throw new Error(`PDF generation failed: ${error.message}`);
    }
  }
  
  /**
   * Generate Emergency Report PDF
   * @param {Object} reportData - Emergency report data
   * @param {Object} options - PDF generation options
   * @returns {Promise<Blob>} PDF blob
   */
  export async function generateEmergencyReportPDF(reportData, options = {}) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'pt',
      format: options.pageSize || PDF_CONFIG.pageSize
    });
  
    try {
      let yPosition = setupDocumentHeader(doc, 'EMERGENCY INCIDENT REPORT', reportData.timestamp);
      
      // Incident summary
      yPosition = addIncidentSummary(doc, reportData, yPosition);
      
      // Location information
      if (reportData.location) {
        yPosition = addLocationInfo(doc, reportData.location, yPosition);
      }
      
      // Hazard analysis
      if (reportData.hazards && reportData.hazards.length > 0) {
        yPosition = addHazardAnalysis(doc, reportData.hazards, yPosition);
      }
      
      // AI Analysis results
      if (reportData.aiAnalysis) {
        yPosition = addAIAnalysis(doc, reportData.aiAnalysis, yPosition);
      }
      
      // Action items and recommendations
      if (reportData.actionItems) {
        yPosition = addActionItems(doc, reportData.actionItems, yPosition);
      }
      
      // Attachments reference
      if (reportData.attachments) {
        yPosition = addAttachmentsReference(doc, reportData.attachments, yPosition);
      }
      
      addDocumentFooter(doc, reportData);
      
      return doc.output('blob');
      
    } catch (error) {
      console.error('❌ Failed to generate emergency report PDF:', error);
      throw new Error(`PDF generation failed: ${error.message}`);
    }
  }
  
  /**
   * Generate Patient Management Summary PDF
   * @param {Array} patients - Array of patient data
   * @param {Object} options - PDF generation options
   * @returns {Promise<Blob>} PDF blob
   */
  export async function generatePatientSummaryPDF(patients, options = {}) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({
      orientation: 'landscape',
      unit: 'pt',
      format: options.pageSize || PDF_CONFIG.pageSize
    });
  
    try {
      let yPosition = setupDocumentHeader(doc, 'PATIENT MANAGEMENT SUMMARY', new Date().toISOString());
      
      // Summary statistics
      yPosition = addPatientStatistics(doc, patients, yPosition);
      
      // Patient table
      yPosition = addPatientTable(doc, patients, yPosition);
      
      addDocumentFooter(doc, { generated: new Date().toISOString() });
      
      return doc.output('blob');
      
    } catch (error) {
      console.error('❌ Failed to generate patient summary PDF:', error);
      throw new Error(`PDF generation failed: ${error.message}`);
    }
  }
  
  /**
   * Generate Custom Report PDF
   * @param {Object} reportConfig - Report configuration
   * @param {Object} data - Report data
   * @returns {Promise<Blob>} PDF blob
   */
  export async function generateCustomReportPDF(reportConfig, data) {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({
      orientation: reportConfig.orientation || 'portrait',
      unit: 'pt',
      format: reportConfig.pageSize || PDF_CONFIG.pageSize
    });
  
    try {
      let yPosition = setupDocumentHeader(doc, reportConfig.title, data.timestamp);
      
      // Process sections based on configuration
      for (const section of reportConfig.sections) {
        yPosition = await addCustomSection(doc, section, data, yPosition);
      }
      
      addDocumentFooter(doc, data);
      
      return doc.output('blob');
      
    } catch (error) {
      console.error('❌ Failed to generate custom report PDF:', error);
      throw new Error(`PDF generation failed: ${error.message}`);
    }
  }
  
  /**
   * Setup document header with logo and title
   */
  function setupDocumentHeader(doc, title, timestamp) {
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = PDF_CONFIG.margins.left;
    
    // Header background
    doc.setFillColor(PDF_CONFIG.colors.primary);
    doc.rect(0, 0, pageWidth, 100, 'F');
    
    // Title
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);
    doc.setFont(PDF_CONFIG.defaultFont, 'bold');
    doc.text(title, margin, 45);
    
    // Timestamp
    doc.setFontSize(12);
    doc.setFont(PDF_CONFIG.defaultFont, 'normal');
    const formattedDate = new Date(timestamp).toLocaleString();
    doc.text(`Generated: ${formattedDate}`, margin, 75);
    
    // Report ID
    const reportId = `RPT-${Date.now().toString(36).toUpperCase()}`;
    doc.text(`Report ID: ${reportId}`, pageWidth - margin - 150, 75);
    
    // Reset text color
    doc.setTextColor(PDF_CONFIG.colors.text);
    
    return 130; // Return next Y position
  }
  
  /**
   * Add patient information section
   */
  function addPatientInfo(doc, patient, yPosition) {
    yPosition = addSectionHeader(doc, 'PATIENT INFORMATION', yPosition);
    
    const info = [
      ['Name:', patient.name || 'Not provided'],
      ['Age:', patient.age ? `${patient.age} years` : 'Not provided'],
      ['Gender:', patient.gender || 'Not provided'],
      ['ID:', patient.id || 'Not assigned'],
      ['Emergency Contact:', patient.emergencyContact || 'Not provided']
    ];
    
    yPosition = addKeyValuePairs(doc, info, yPosition);
    
    return yPosition + 20;
  }
  
  /**
   * Add vital signs section
   */
  function addVitalSigns(doc, vitals, yPosition) {
    yPosition = addSectionHeader(doc, 'VITAL SIGNS', yPosition);
  
    const vitalData = [
      ['Heart Rate:', vitals.heartRate ? `${vitals.heartRate} bpm` : 'Not recorded'],
      ['Blood Pressure:', vitals.bloodPressure ? `${vitals.bloodPressure.systolic}/${vitals.bloodPressure.diastolic} mmHg` : 'Not recorded'],
      ['Respiratory Rate:', vitals.respiratoryRate ? `${vitals.respiratoryRate} breaths/min` : 'Not recorded'],
      ['Temperature:', vitals.temperature ? `${vitals.temperature}°F` : 'Not recorded'],
      ['Oxygen Saturation:', vitals.oxygenSat ? `${vitals.oxygenSat}%` : 'Not recorded'],
      ['Pain Level:', vitals.painLevel ? `${vitals.painLevel}/10` : 'Not assessed']
    ];
  
    yPosition = addKeyValuePairs(doc, vitalData, yPosition);
  
    // Add vital signs status indicators
    if (vitals.assessment) {
      yPosition += 10;
      doc.setFontSize(10);
      doc.setFont(PDF_CONFIG.defaultFont, 'italic');
    }
  
    return yPosition + 20;
  }  