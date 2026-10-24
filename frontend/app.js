const API_URL = (window.location.origin && window.location.origin.startsWith("http")) 
  ? window.location.origin 
  : "http://127.0.0.1:8000";

// Botanical profiles for rich demo insights
const BOTANICAL_DATA = {
  kithul: {
    displayName: "Kithul Flower",
    scientific: "Caryota urens (Fish-tail Palm / Toddy Palm)",
    color: "#10b981",
    fillClass: "prob-fill-kithul",
    description: "Pendulous hanging inflorescence emerging from leaf axils with long cascade tassels. Traditionally tapped for sap in Sri Lanka to produce pure Kithul treacle and jaggery."
  },
  coconut: {
    displayName: "Coconut Flower",
    scientific: "Cocos nucifera (Coconut Palm)",
    color: "#06b6d4",
    fillClass: "prob-fill-coconut",
    description: "Branching spike inflorescence originally encased in a woody spathe. Features small male flowers towards the tip and spherical female flowers ('buttons') near the base."
  },
  date: {
    displayName: "Date Palm Flower",
    scientific: "Phoenix dactylifera (Date Palm)",
    color: "#f59e0b",
    fillClass: "prob-fill-date",
    description: "Dioecious spadix emerging from a fibrous sheath, containing dense clusters of small, fragrant waxy florets on pliable branchlets."
  }
};

// DOM Elements
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const dropzonePrompt = document.getElementById("dropzone-prompt");
const previewWrapper = document.getElementById("preview-wrapper");
const imagePreview = document.getElementById("image-preview");
const btnChangeImage = document.getElementById("btn-change-image");
const btnClassify = document.getElementById("btn-classify");
const btnSpinner = btnClassify.querySelector(".btn-spinner");
const btnText = btnClassify.querySelector(".btn-text");

const apiStatus = document.getElementById("api-status");
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsActive = document.getElementById("results-active");
const topResultBanner = document.getElementById("top-result-banner");
const resultStatusTag = document.getElementById("result-status-tag");
const topConfidenceVal = document.getElementById("top-confidence-val");
const predictedName = document.getElementById("predicted-name");
const predictedScientific = document.getElementById("predicted-scientific");
const botanicalInfo = document.getElementById("botanical-info");
const botanicalDesc = document.getElementById("botanical-desc");
const probHeading = document.getElementById("prob-heading");
const modelScopeNote = document.getElementById("model-scope-note");
const probabilitiesContainer = document.getElementById("probabilities-container");
const probBarsList = document.getElementById("prob-bars-list");
const uncertainDetails = document.getElementById("uncertain-details");
const uncertainDetailsText = document.getElementById("uncertain-details-text");

// Samples
const sampleButtons = document.querySelectorAll(".sample-btn");

let currentFile = null;

// Check Backend Health
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      apiStatus.className = "api-status status-online";
      apiStatus.innerHTML = `<span class="status-dot"></span><span>API Online (${data.classes.length} Classes)</span>`;
      return true;
    }
  } catch (err) {
    // API not responding
  }
  apiStatus.className = "api-status status-offline";
  apiStatus.innerHTML = `<span class="status-dot"></span><span>API Offline (Run Backend)</span>`;
  return false;
}

// Initial health check and interval polling
checkBackendHealth();
setInterval(checkBackendHealth, 5000);

// File Selection Handling
function handleFileSelected(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPG, PNG, or WEBP).");
    return;
  }

  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzonePrompt.classList.add("hidden");
    previewWrapper.classList.remove("hidden");
    btnClassify.removeAttribute("disabled");
  };
  reader.readAsDataURL(file);
}

// File input change
fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileSelected(e.target.files[0]);
  }
});

// Change image button
btnChangeImage.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.value = "";
  currentFile = null;
  imagePreview.src = "";
  previewWrapper.classList.add("hidden");
  dropzonePrompt.classList.remove("hidden");
  btnClassify.setAttribute("disabled", "true");
  showPlaceholder();
});

// Drag and drop events
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    handleFileSelected(e.dataTransfer.files[0]);
  }
});

// Quick Sample Buttons
sampleButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const sampleType = btn.getAttribute("data-sample");
    const samplePath = `samples/sample_${sampleType}.jpg`;

    try {
      btn.style.opacity = "0.7";
      const response = await fetch(samplePath);
      if (!response.ok) throw new Error("Could not load sample image.");
      const blob = await response.blob();
      const file = new File([blob], `sample_${sampleType}.jpg`, { type: "image/jpeg" });

      handleFileSelected(file);
      btn.style.opacity = "1";

      // Auto trigger classification for immediate demo delight
      setTimeout(() => {
        classifyCurrentFlower();
      }, 200);
    } catch (err) {
      console.error(err);
      alert(`Could not load sample image: ${samplePath}`);
      btn.style.opacity = "1";
    }
  });
});

// Classify Button Click
btnClassify.addEventListener("click", () => {
  classifyCurrentFlower();
});

// Classification Request
async function classifyCurrentFlower() {
  if (!currentFile) return;

  setLoading(true);

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    displayResults(data);
  } catch (error) {
    console.error(error);
    alert(`Classification Failed: ${error.message}\n\nPlease ensure your backend is running:\npython -m uvicorn backend.main:app --reload`);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  if (isLoading) {
    btnClassify.setAttribute("disabled", "true");
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Analyzing botanical features...";
  } else {
    btnClassify.removeAttribute("disabled");
    btnSpinner.classList.add("hidden");
    btnText.textContent = "Classify Flower";
  }
}

function showPlaceholder() {
  resultsActive.classList.add("hidden");
  resultsPlaceholder.classList.remove("hidden");
}

function displayResults(data) {
  if (!data) return;

  const status = data.status || (data.confident ? "ok" : "uncertain");
  const label = data.label || (data.prediction && data.prediction !== "uncertain" ? data.prediction : null);
  const probabilities = data.probabilities || {};
  const isOk = status === "ok";
  const activeLabel = (isOk && label) ? String(label) : "kithul";
  const activeKey = activeLabel.toLowerCase();

  const profile = BOTANICAL_DATA[activeKey] || {
    displayName: activeLabel ? activeLabel.toUpperCase() : "PALM SPECIMEN",
    scientific: "Palm Specimen",
    description: "Botanical specimen classification.",
    color: "#10b981",
    fillClass: "prob-fill-kithul"
  };

  if (isOk) {
    topResultBanner.classList.remove("banner-no-match");
    resultStatusTag.classList.remove("hidden");
    resultStatusTag.textContent = "Confident Match";
    resultStatusTag.className = "result-tag";
    predictedName.textContent = profile.displayName;
    predictedScientific.textContent = profile.scientific;
    predictedScientific.classList.remove("hidden");
    const topProb = (probabilities && label && probabilities[label] !== undefined) ? probabilities[label] : 0;
    topConfidenceVal.textContent = `${(topProb * 100).toFixed(1)}%`;
    topConfidenceVal.classList.remove("hidden");

    // Hide uncertain details toggle
    if (uncertainDetails) {
      uncertainDetails.classList.add("hidden");
      uncertainDetails.removeAttribute("open");
    }

    // Show botanical description
    botanicalDesc.textContent = profile.description;
    if (botanicalInfo) botanicalInfo.classList.remove("hidden");
    if (probHeading) probHeading.textContent = "Confidence Distribution";

    // Show and render confidence distribution bars
    if (probabilitiesContainer) probabilitiesContainer.classList.remove("hidden");
    probBarsList.innerHTML = "";
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sorted.forEach(([cls, prob]) => {
      const clsKey = cls ? String(cls).toLowerCase() : "";
      const clsProfile = BOTANICAL_DATA[clsKey] || {
        displayName: clsKey ? clsKey.charAt(0).toUpperCase() + clsKey.slice(1) : "Unknown",
        fillClass: "prob-fill-kithul"
      };

      const percentage = ((Number(prob) || 0) * 100).toFixed(1);

      const item = document.createElement("div");
      item.className = "prob-item";
      item.innerHTML = `
        <div class="prob-meta">
          <span class="prob-class-name">
            <span class="sample-dot dot-${clsKey}"></span>
            ${clsProfile.displayName}
          </span>
          <span class="prob-percentage">${percentage}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${clsProfile.fillClass}" style="width: 0%"></div>
        </div>
      `;

      probBarsList.appendChild(item);

      setTimeout(() => {
        const fillBar = item.querySelector(".prob-fill");
        if (fillBar) {
          fillBar.style.width = `${percentage}%`;
        }
      }, 50);
    });
  } else {
    // If "uncertain":
    // Show only the "Not recognised" card and the info line
    // Hide confidence distribution bars
    // Move Raw max logit / Margin into a collapsed "Show details" toggle
    topResultBanner.classList.add("banner-no-match");
    resultStatusTag.classList.add("hidden");
    topConfidenceVal.textContent = "";
    topConfidenceVal.classList.add("hidden");

    predictedName.textContent = "Not recognised as a Kithul, Coconut or Date palm flower";
    predictedScientific.textContent = "";
    predictedScientific.classList.add("hidden");

    const maxLogitStr = data.max_logit !== undefined ? Number(data.max_logit).toFixed(4) : "N/A";
    const marginStr = data.margin !== undefined ? (Number(data.margin) * 100).toFixed(1) + "%" : "N/A";

    if (uncertainDetailsText) {
      uncertainDetailsText.textContent = `Raw max logit: ${maxLogitStr} • Margin: ${marginStr}`;
    }
    if (uncertainDetails) {
      uncertainDetails.removeAttribute("open");
      uncertainDetails.classList.remove("hidden");
    }

    // Hide botanical info card
    if (botanicalInfo) botanicalInfo.classList.add("hidden");

    // Hide confidence distribution bars
    if (probabilitiesContainer) probabilitiesContainer.classList.add("hidden");
    probBarsList.innerHTML = "";
  }

  // Keep a visible line saying the model is trained on those three flowers only
  if (modelScopeNote) modelScopeNote.classList.remove("hidden");

  // Switch panels
  resultsPlaceholder.classList.add("hidden");
  resultsActive.classList.remove("hidden");
}
