async function fetchRiskPrediction() {
    // Step 1: Set default or dynamic location
    let lat = 34.05;
    let lon = -118.25;
  
    // Try to get user geolocation
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          lat = position.coords.latitude;
          lon = position.coords.longitude;
          console.log("📍 User location acquired:", lat, lon);
          sendPredictionRequest(lat, lon);
        },
        (error) => {
          console.warn("⚠️ Geolocation error:", error.message);
          sendPredictionRequest(lat, lon); // Fallback to default
        }
      );
    } else {
      sendPredictionRequest(lat, lon); // Geolocation not available
    }
  }
  
  async function sendPredictionRequest(lat, lon) {
    const weather = {
      temperature: 42,       // Replace with dynamic input later
      wind_speed: 60,
      rainfall: 20
    };
    const hazardType = "wildfire";
  
    try {
      const response = await fetch("/predict-risk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: { lat, lon },
          weather,
          hazard_type: hazardType
        })
      });
  
      const data = await response.json();
      updateRiskUI(data);
    } catch (error) {
      console.error("❌ Error fetching risk prediction:", error);
      document.getElementById("risk-score").textContent = "Risk Score: Error";
      document.getElementById("severity").textContent = "Severity: Error";
      document.getElementById("resources").textContent = "Resources: Failed to load.";
    }
  }
  
  function updateRiskUI(data) {
    document.getElementById("risk-score").innerHTML = `Risk Score: <strong>${data.risk_score}</strong>`;
    document.getElementById("severity").innerHTML = `Severity: <strong>${data.severity}</strong>`;
    document.getElementById("resources").textContent = JSON.stringify(data.suggested_resources, null, 2);
  
    // 🚨 Trigger emergency broadcast if risk is high
    if (data.risk_score >= 0.8) {
      triggerBroadcast(data);
    }
  }
  
  async function triggerBroadcast(data) {
    const payload = {
      message: `🚨 Emergency Risk Detected (${data.severity})`,
      location: data.location,
      severity: data.severity
    };
  
    try {
      const response = await fetch("/broadcast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
  
      const result = await response.json();
      console.log("📢 Broadcast triggered:", result);
  
      // Show alert banner
      const alertBanner = document.getElementById("broadcast-alert");
      if (alertBanner) {
        alertBanner.style.display = "block";
        alertBanner.textContent = `🚨 Broadcast triggered: ${data.severity} risk near your location.`;
      }
    } catch (error) {
      console.error("❌ Failed to trigger broadcast:", error);
    }
  }
  
  // Start on page load
  document.addEventListener("DOMContentLoaded", fetchRiskPrediction);  