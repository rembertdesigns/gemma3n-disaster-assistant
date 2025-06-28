document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("hazardForm");
    const input = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const resultSection = document.getElementById("resultSection");
    const canvas = document.getElementById("resultCanvas");
    const hazardList = document.getElementById("hazardList");
    const toast = document.getElementById("toast");
    const testBtn = document.getElementById("runTestHazardBtn");
  
    let imageElement = null;
  
    function showToast(message, duration = 3000) {
      toast.textContent = message;
      toast.style.display = "block";
      toast.style.opacity = "1";
      setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => (toast.style.display = "none"), 500);
      }, duration);
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
  
      const formData = new FormData();
      formData.append("file", input.files[0]);
  
      // Show loading message
      showToast("Analyzing image for hazards...");
  
      try {
        const res = await fetch("/detect-hazards", {
          method: "POST",
          body: formData,
        });
  
        const data = await res.json();
        if (data.error) {
          showToast(data.error);
          return;
        }
  
        // Draw result on canvas
        drawResults(data);
  
      } catch (err) {
        console.error(err);
        showToast("Something went wrong during detection.");
      }
    });
  
    function drawResults(data) {
      if (!imageElement) {
        showToast("No image loaded.");
        return;
      }
  
      // Show result section
      resultSection.style.display = "block";
  
      // Setup canvas size
      canvas.width = imageElement.width;
      canvas.height = imageElement.height;
  
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  
      // Draw boxes
      if (data.predictions && Array.isArray(data.predictions)) {
        hazardList.innerHTML = "";
        data.predictions.forEach((pred) => {
          const { label, score, box } = pred;
          const [x1, y1, x2, y2] = box;
  
          ctx.strokeStyle = "#dc2626";
          ctx.lineWidth = 2;
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
          ctx.fillStyle = "#dc2626";
          ctx.font = "bold 14px sans-serif";
          ctx.fillText(`${label} (${(score * 100).toFixed(1)}%)`, x1 + 4, y1 + 16);
  
          const li = document.createElement("li");
          li.textContent = `🚨 ${label} (${(score * 100).toFixed(1)}%)`;
          hazardList.appendChild(li);
        });
      } else {
        showToast("No hazards detected.");
      }
    }
  
    // --- MOCK TEST DETECTION LOGIC ---
    const mockDetections = [
      {
        label: "Downed Power Line",
        confidence: 0.91,
        box: [50, 40, 300, 160],
      },
      {
        label: "Flooded Area",
        confidence: 0.87,
        box: [100, 200, 280, 320],
      }
    ];
  
    function drawMockDetections(image, detections) {
      resultSection.style.display = "block";
      canvas.width = image.width;
      canvas.height = image.height;
  
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0);
  
      hazardList.innerHTML = "";
      detections.forEach(({ label, confidence, box }) => {
        const [x1, y1, x2, y2] = box;
  
        ctx.strokeStyle = "#dc2626";
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
        ctx.fillStyle = "rgba(220, 38, 38, 0.85)";
        ctx.font = "bold 14px sans-serif";
        ctx.fillText(`${label} (${(confidence * 100).toFixed(1)}%)`, x1 + 5, y1 - 8);
  
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
      };
      mockImage.onerror = () => {
        showToast("⚠️ Failed to load mock image");
      };
    }
  
    if (testBtn) {
      testBtn.addEventListener("click", runMockDetection);
    }
  });  