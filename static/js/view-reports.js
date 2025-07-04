// Simple but enhanced reports viewer
const toneFilter = document.getElementById("tone-filter");
const escalationFilter = document.getElementById("escalation-filter");
const sortOrder = document.getElementById("sort-order");
const keywordInput = document.getElementById("keyword-search");
const reportCount = document.getElementById("report-count");
const reportsList = document.getElementById("reports-list");
const totalCount = document.getElementById("total-count");

let allReports = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
  // Store original reports
  allReports = Array.from(document.querySelectorAll(".report-card"));
  
  // Set up event listeners
  toneFilter.addEventListener("change", applyFilters);
  escalationFilter.addEventListener("change", applyFilters);
  sortOrder.addEventListener("change", applyFilters);
  keywordInput.addEventListener("input", debounce(applyFilters, 300));
  
  // Initial filter
  applyFilters();
  
  // Auto-refresh every 30 seconds if online
  setInterval(() => {
    if (navigator.onLine && document.visibilityState === 'visible') {
      refreshReports();
    }
  }, 30000);
});

function highlightText(text, keyword) {
  if (!keyword.trim()) return text;
  
  const regex = new RegExp(`(${escapeRegex(keyword.trim())})`, "gi");
  return text.replace(regex, '<span class="highlight">$1</span>');
}

function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function applyFilters() {
  const tone = toneFilter.value;
  const escalation = escalationFilter.value;
  const keyword = keywordInput.value.toLowerCase().trim();
  const order = sortOrder.value;

  // Filter reports
  const filtered = allReports.filter(card => {
    const matchesTone = !tone || card.dataset.tone === tone;
    const matchesEscalation = !escalation || card.dataset.escalation === escalation;
    const matchesKeyword = !keyword || card.dataset.message.includes(keyword);
    
    return matchesTone && matchesEscalation && matchesKeyword;
  });

  // Sort reports
  filtered.sort((a, b) => {
    const dateA = new Date(a.dataset.timestamp);
    const dateB = new Date(b.dataset.timestamp);
    return order === "asc" ? dateA - dateB : dateB - dateA;
  });

  // Clear and re-render
  reportsList.innerHTML = "";
  
  if (filtered.length === 0) {
    reportsList.innerHTML = `
      <div class="no-reports">
        <div class="no-reports-icon">🔍</div>
        <h3>No Matching Reports</h3>
        <p>Try adjusting your filters or search terms.</p>
      </div>
    `;
  } else {
    filtered.forEach(card => {
      // Highlight search terms
      const msgElem = card.querySelector(".message-text");
      const originalMessage = card.dataset.message;
      msgElem.innerHTML = highlightText(originalMessage, keyword);
      
      reportsList.appendChild(card);
    });
  }

  // Update count
  reportCount.textContent = filtered.length;
}

function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

async function refreshReports() {
  if (!navigator.onLine) {
    showMessage('Unable to refresh - you are offline', 'warning');
    return;
  }
  
  const refreshBtn = document.querySelector('.refresh-button');
  const originalText = refreshBtn.textContent;
  
  refreshBtn.textContent = '🔄 Refreshing...';
  refreshBtn.disabled = true;
  
  try {
    // Reload the page to get fresh data
    window.location.reload();
  } catch (error) {
    console.error('Refresh failed:', error);
    showMessage('Failed to refresh reports', 'error');
  } finally {
    refreshBtn.textContent = originalText;
    refreshBtn.disabled = false;
  }
}

function showMessage(text, type = 'info') {
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
    animation: slideIn 0.3s ease;
    ${type === 'error' ? 'background: #dc2626;' : 
      type === 'warning' ? 'background: #f59e0b;' : 
      'background: #16a34a;'}
  `;
  message.textContent = text;
  
  document.body.appendChild(message);
  
  setTimeout(() => {
    message.remove();
  }, 3000);
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  if (e.ctrlKey || e.metaKey) {
    switch(e.key) {
      case 'f':
        e.preventDefault();
        keywordInput.focus();
        break;
      case 'r':
        e.preventDefault();
        refreshReports();
        break;
    }
  }
});

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
  }
`;
document.head.appendChild(style);