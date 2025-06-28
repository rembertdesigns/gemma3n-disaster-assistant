document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("hazardForm");
    const input = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const resultSection = document.getElementById("resultSection");
    const canvas = document.getElementById("resultCanvas");
    const hazardList = document.getElementById("hazardList");
    const toast = document.getElementById("toast");
    const testBtn = document.getElementById("runTestHazardBtn");
    const progressBar = document.getElementById("progressBar");
    const toggleBoxes = document.getElementById("toggleBoxes");
    const downloadBtn = document.getElementById("downloadCanvasBtn");
  
    const openDrawerBtn = document.getElementById("openDrawerBtn");
    const closeDrawerBtn = document.getElementById("closeDrawerBtn");
    const settingsDrawer = document.getElementById("settingsDrawer");
    const drawerToggleTheme = document.getElementById("drawerToggleTheme");
    const drawerToggleBoxes = document.getElementById("drawerToggleBoxes");
    const drawerSoundAlerts = document.getElementById("drawerSoundAlerts");
  
    let imageElement = null;
    let lastDetections = [];
  
    function showToast(message, duration = 3000) {
      toast.textContent = message;
      toast.style.display = "block";
      toast.style.opacity = "1";
      setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => (toast.style.display = "none"), 500);
      }, duration);
    }
  
    function setProgress(percent) {
      if (progressBar) {
        progressBar.style.width = percent + "%";
        progressBar.parentElement.style.display = percent < 100 ? "block" : "none";
      }
    }
  
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => {
          previewContainer.innerHTML = "";
          imageElement = new Image();
          imageElement.src = e.target.result;
          imageElement.onload = () => {
            previewContainer.appendChild(imageElement);
          };
        };
        reader.readAsDataURL(file);
      }
    });
  
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!input.files[0]) {
        showToast("Please select an image to detect hazards.");
        return;
      }
  
      setProgress(10);
      const formData = new FormData();
      formData.append("file", input.files[0]);
  
      showToast("🔍 Analyzing image for hazards...");
  
      try {
        const res = await fetch("/detect-hazards", {
          method: "POST",
          body: formData,
        });
  
        setProgress(70);
        const data = await res.json();
  
        if (data.error) {
          showToast(data.error);
          setProgress(0);
          return;
        }
  
        lastDetections = data.predictions || [];
        drawResults(data);
        setProgress(100);
      } catch (err) {
        console.error(err);
        showToast("❌ Something went wrong during detection.");
        setProgress(0);
      }
    });
  
    function drawResults(data) {
      if (!imageElement) {
        showToast("No image loaded.");
        return;
      }
  
      resultSection.style.display = "block";
      canvas.width = imageElement.width;
      canvas.height = imageElement.height;
  
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  
      hazardList.innerHTML = "";
  
      if (Array.isArray(data.predictions)) {
        data.predictions.forEach(({ label, score, box }) => {
          const [x1, y1, x2, y2] = box;
  
          if (toggleBoxes?.checked) {
            ctx.strokeStyle = "#dc2626";
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
            ctx.fillStyle = "#dc2626";
            ctx.font = "bold 14px sans-serif";
            ctx.fillText(`${label} (${(score * 100).toFixed(1)}%)`, x1 + 4, y1 + 16);
          }
  
          const li = document.createElement("li");
          li.textContent = `🚨 ${label} (${(score * 100).toFixed(1)}%)`;
          hazardList.appendChild(li);
        });
      } else {
        showToast("✅ No hazards detected.");
      }
    }
  
    toggleBoxes?.addEventListener("change", () => {
      drawResults({ predictions: lastDetections });
    });
  
    const mockDetections = [
      { label: "Downed Power Line", confidence: 0.91, box: [50, 40, 300, 160] },
      { label: "Flooded Area", confidence: 0.87, box: [100, 200, 280, 320] }
    ];
  
    function drawMockDetections(image, detections) {
      resultSection.style.display = "block";
      canvas.width = image.width;
      canvas.height = image.height;
  
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  
      hazardList.innerHTML = "";
      lastDetections = detections;
  
      detections.forEach(({ label, confidence, box }) => {
        const [x1, y1, x2, y2] = box;
  
        if (toggleBoxes?.checked) {
          ctx.strokeStyle = "#dc2626";
          ctx.lineWidth = 3;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
          ctx.fillStyle = "rgba(220, 38, 38, 0.85)";
          ctx.font = "bold 14px sans-serif";
          ctx.fillText(`${label} (${(confidence * 100).toFixed(1)}%)`, x1 + 5, y1 - 8);
        }
  
        const li = document.createElement("li");
        li.textContent = `🚨 ${label} (${(confidence * 100).toFixed(1)}%)`;
        hazardList.appendChild(li);
      });
    }
  
    function runMockDetection() {
      const mockImage = new Image();
      mockImage.src = "/static/mock_hazard_image.jpg";
      mockImage.onload = () => {
        drawMockDetections(mockImage, mockDetections);
        showToast("🧠 Simulated hazard detection complete");
        setProgress(100);
      };
      mockImage.onerror = () => {
        showToast("⚠️ Failed to load mock image");
        setProgress(0);
      };
    }
  
    testBtn?.addEventListener("click", runMockDetection);
  
    if (downloadBtn && canvas) {
      downloadBtn.addEventListener("click", () => {
        const imageURI = canvas.toDataURL("image/png");
        const link = document.createElement("a");
        link.download = "hazard_detection_result.png";
        link.href = imageURI;
        link.click();
      });
    }
  
    // -------- Settings Drawer Logic --------
    openDrawerBtn?.addEventListener("click", () => {
      settingsDrawer?.classList.add("open");
    });
  
    closeDrawerBtn?.addEventListener("click", () => {
      settingsDrawer?.classList.remove("open");
    });
  
    drawerToggleTheme?.addEventListener("change", () => {
      document.documentElement.setAttribute("data-theme", drawerToggleTheme.checked ? "dark" : "light");
    });
  
    drawerToggleBoxes?.addEventListener("change", () => {
      if (toggleBoxes) {
        toggleBoxes.checked = drawerToggleBoxes.checked;
        drawResults({ predictions: lastDetections });
      }
    });
  
  });  

  // --- Service Worker Registration ---
if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/static/js/sw.js")
    .then(() => console.log("✅ Service Worker registered"))
    .catch((err) => console.error("❌ SW registration failed:", err));
}