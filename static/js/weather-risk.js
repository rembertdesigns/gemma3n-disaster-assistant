async function fetchRiskPrediction() {
    const lat = 34.05;  // Replace with actual geolocation or user input later
    const lon = -118.25;
    const weather = {
      temperature: 42,
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
  }
  
  document.addEventListener("DOMContentLoaded", fetchRiskPrediction);  