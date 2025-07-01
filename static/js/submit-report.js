document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("report-form");
    const statusBox = document.getElementById("submit-status");
  
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      statusBox.textContent = "⏳ Submitting...";
  
      const message = document.getElementById("message").value;
      let location = { lat: 0, lon: 0 };
  
      if ("geolocation" in navigator) {
        try {
          const pos = await new Promise((res, rej) =>
            navigator.geolocation.getCurrentPosition(res, rej)
          );
          location = {
            lat: pos.coords.latitude,
            lon: pos.coords.longitude
          };
        } catch (err) {
          console.warn("⚠️ Geolocation failed:", err.message);
        }
      }
  
      try {
        const response = await fetch("/api/submit-report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            location,
            user: "anonymous"
          })
        });
  
        const result = await response.json();
        if (result.success) {
          statusBox.textContent = `✅ Report submitted! Escalation: ${result.report.escalation}`;
          form.reset();
        } else {
          statusBox.textContent = "❌ Submission failed.";
        }
      } catch (error) {
        console.error("❌ Error:", error);
        statusBox.textContent = "❌ An error occurred.";
      }
    });
  });  