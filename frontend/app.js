const API_URL = (window.location.origin && window.location.origin.startsWith("http")) 
  ? window.location.origin 
  : "http://127.0.0.1:8000";

// Botanical profiles
const BOTANICAL_DATA = {
  kithul: {
    displayName: "Kithul Palm",
    scientific: "Caryota urens (Fish-tail Palm / Toddy Palm)",
    fillClass: "prob-fill-kithul",
    description: "Pendulous hanging inflorescence emerging from leaf axils with long cascade tassels. Traditionally tapped for sap in Sri Lanka to produce pure Kithul treacle and jaggery."
  },
  coconut: {
    displayName: "Coconut Palm",
    scientific: "Cocos nucifera (Coconut Palm)",
    fillClass: "prob-fill-coconut",
    description: "Branching spike inflorescence originally encased in a woody spathe. Features small male flowers towards the tip and spherical female flowers ('buttons') near the base."
  },
  date: {
    displayName: "Date Palm",
    scientific: "Phoenix dactylifera (Date Palm)",
    fillClass: "prob-fill-date",
    description: "Dioecious spadix emerging from a fibrous sheath, containing dense clusters of small, fragrant florets on pliable branchlets."
  }
};

const MODEL_DISPLAY_INFO = {
  efficientnet_b0: {
    title: "EfficientNet-B0",
    badge: "Lightweight",
    isRecommended: false,
  },
  resnet18: {
    title: "ResNet-18",
    badge: "Recommended",
    isRecommended: true,
  },
  resnet50: {
    title: "ResNet-50",
    badge: "Deep Residual",
    isRecommended: false,
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
const multiCardsGrid = document.getElementById("multi-cards-grid");
const errorBanner = document.getElementById("error-banner");
const errorTitle = document.getElementById("error-title");
const errorDesc = document.getElementById("error-desc");

// Model Comparison Elements
const btnToggleComparison = document.getElementById("btn-toggle-comparison");
const toggleComparisonText = document.getElementById("toggle-comparison-text");
const comparisonCollapsible = document.getElementById("comparison-collapsible");
const bestModelHighlight = document.getElementById("best-model-highlight");
const bestModelTitle = document.getElementById("best-model-title");
const bestModelReason = document.getElementById("best-model-reason");
const benchmarkTbody = document.getElementById("benchmark-tbody");
const cmGalleryGrid = document.getElementById("cm-gallery-grid");
const btnRefreshResults = document.getElementById("btn-refresh-results");

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
      apiStatus.innerHTML = `<span class="status-dot"></span><span>API Online (${data.multi_models ? data.multi_models.length : data.classes.length} Models)</span>`;
      hideError();
      return true;
    }
  } catch (err) {
    // API offline
  }
  apiStatus.className = "api-status status-offline";
  apiStatus.innerHTML = `<span class="status-dot"></span><span>API Offline (Run Backend)</span>`;
  return false;
}

// Initial health check and interval polling
checkBackendHealth();
setInterval(checkBackendHealth, 8000);

// File Selection Handling
function handleFileSelected(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Invalid File", "Please select a valid image file (JPG, PNG, or WEBP).");
    return;
  }

  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzonePrompt.classList.add("hidden");
    previewWrapper.classList.remove("hidden");
    btnClassify.removeAttribute("disabled");
    hideError();
  };
  reader.readAsDataURL(file);
}

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) {
    handleFileSelected(e.target.files[0]);
  }
});

btnChangeImage.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.value = "";
  currentFile = null;
  imagePreview.src = "";
  previewWrapper.classList.add("hidden");
  dropzonePrompt.classList.remove("hidden");
  btnClassify.setAttribute("disabled", "true");
  showPlaceholder();
  hideError();
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
      if (!response.ok) throw new Error("Could not load sample image from frontend folder.");
      const blob = await response.blob();
      const file = new File([blob], `sample_${sampleType}.jpg`, { type: "image/jpeg" });

      handleFileSelected(file);
      btn.style.opacity = "1";

      setTimeout(() => {
        classifyCurrentFlower();
      }, 150);
    } catch (err) {
      console.error(err);
      showError("Sample Load Error", `Could not load sample ${sampleType}: ${err.message}`);
      btn.style.opacity = "1";
    }
  });
});

// Classify Button Click
btnClassify.addEventListener("click", () => {
  classifyCurrentFlower();
});

// Model Comparison Toggle
if (btnToggleComparison) {
  btnToggleComparison.addEventListener("click", () => {
    const isHidden = comparisonCollapsible.classList.contains("hidden");
    if (isHidden) {
      comparisonCollapsible.classList.remove("hidden");
      btnToggleComparison.setAttribute("aria-expanded", "true");
      btnToggleComparison.classList.add("expanded");
      toggleComparisonText.textContent = "Hide model comparison";
      // Load benchmark data if not yet populated
      loadModelComparison();
    } else {
      comparisonCollapsible.classList.add("hidden");
      btnToggleComparison.setAttribute("aria-expanded", "false");
      btnToggleComparison.classList.remove("expanded");
      toggleComparisonText.textContent = "Show model comparison";
    }
  });
}

if (btnRefreshResults) {
  btnRefreshResults.addEventListener("click", () => {
    loadModelComparison();
  });
}

// Classification Request: POST /predict_all
async function classifyCurrentFlower() {
  if (!currentFile) return;

  setLoading(true);
  hideError();

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const response = await fetch(`${API_URL}/predict_all`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned error (${response.status})`);
    }

    const data = await response.json();
    displayMultiModelResults(data.predictions || []);
  } catch (error) {
    console.error(error);
    showError(
      "Classification Failed",
      `Unable to reach backend: ${error.message}. Please verify the server is running:\npython -m uvicorn backend.main:app --port 8000`
    );
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  if (isLoading) {
    btnClassify.setAttribute("disabled", "true");
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Analyzing across 3 models...";
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

function showError(title, message) {
  if (errorBanner) {
    errorTitle.textContent = title;
    errorDesc.textContent = message;
    errorBanner.classList.remove("hidden");
  }
}

function hideError() {
  if (errorBanner) {
    errorBanner.classList.add("hidden");
  }
}

// Render 3 Side-by-Side Model Cards
function displayMultiModelResults(predictions) {
  if (!predictions || predictions.length === 0) return;

  multiCardsGrid.innerHTML = "";

  predictions.forEach((pred) => {
    const mid = pred.model;
    const meta = MODEL_DISPLAY_INFO[mid] || { title: mid, badge: "Model", isRecommended: false };
    const isOk = pred.status === "ok";
    const labelKey = pred.label ? String(pred.label).toLowerCase() : "";
    const profile = BOTANICAL_DATA[labelKey] || {
      displayName: pred.label ? pred.label.toUpperCase() : "UNKNOWN",
      scientific: "",
    };

    const card = document.createElement("div");
    card.className = `model-card ${meta.isRecommended ? "model-card-recommended" : ""}`;

    const recPillHtml = meta.isRecommended
      ? `<span class="badge-recommended">Recommended</span>`
      : "";

    let statusHeaderHtml = "";
    let bodyContentHtml = "";

    if (isOk) {
      statusHeaderHtml = `
        <div class="card-status-row">
          <span class="badge-recognised">Recognised</span>
          <span class="inference-time">${pred.inference_ms} ms</span>
        </div>
      `;

      bodyContentHtml = `
        <h4 class="pred-title">${profile.displayName}</h4>
        <p class="pred-sub">${profile.scientific}</p>
        
        <div class="card-prob-section">
          ${renderProbabilityBars(pred.probabilities || {})}
        </div>
      `;
    } else {
      // Uncertain / Rejected
      statusHeaderHtml = `
        <div class="card-status-row">
          <span class="badge-uncertain">Uncertain / Rejected</span>
          <span class="inference-time">${pred.inference_ms} ms</span>
        </div>
      `;

      bodyContentHtml = `
        <div class="rejection-msg-box">
          <p class="not-recognised-text">Not recognised as a Kithul, Coconut or Date palm flower</p>
        </div>
      `;
    }

    // Collapsed "Show details" toggle with Cosine Similarity and Threshold
    const detailsHtml = `
      <details class="card-details-toggle">
        <summary class="card-details-summary">Show details</summary>
        <div class="card-details-content">
          <div class="details-row">
            <span>Nearest Centroid Cosine:</span>
            <strong>${Number(pred.max_cosine).toFixed(4)}</strong>
          </div>
          <div class="details-row">
            <span>Threshold (tau_feat):</span>
            <strong>${Number(pred.tau_feat).toFixed(4)}</strong>
          </div>
          <div class="details-row">
            <span>Decision:</span>
            <span class="${isOk ? 'text-sage' : 'text-gold'}">${isOk ? 'Above threshold (Accepted)' : 'Below threshold (Rejected)'}</span>
          </div>
        </div>
      </details>
    `;

    card.innerHTML = `
      <div class="model-card-header">
        <div class="model-header-text">
          <h3 class="model-name-title">${meta.title}</h3>
          <span class="model-desc-tag">${meta.badge}</span>
        </div>
        ${recPillHtml}
      </div>
      ${statusHeaderHtml}
      <div class="model-card-body">
        ${bodyContentHtml}
      </div>
      ${detailsHtml}
    `;

    multiCardsGrid.appendChild(card);
  });

  // Trigger animation for probability fills on next frame
  requestAnimationFrame(() => {
    const fills = multiCardsGrid.querySelectorAll(".prob-fill");
    fills.forEach((el) => {
      const targetWidth = el.getAttribute("data-width");
      if (targetWidth) {
        el.style.width = targetWidth;
      }
    });
  });

  resultsPlaceholder.classList.add("hidden");
  resultsActive.classList.remove("hidden");
}

function renderProbabilityBars(probabilities) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return sorted.map(([cls, prob], index) => {
    const key = cls.toLowerCase();
    const prof = BOTANICAL_DATA[key] || { displayName: cls };
    const pct = (Number(prob) * 100).toFixed(1);
    const isTop = index === 0;
    const rankClass = isTop ? "rank-top" : "rank-other";

    return `
      <div class="prob-mini-item ${rankClass}">
        <div class="prob-mini-meta">
          <span class="prob-class-name">${prof.displayName}</span>
          <span class="prob-percentage">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${isTop ? 'fill-top' : 'fill-other'}" data-width="${pct}%" style="width: 0%;"></div>
        </div>
      </div>
    `;
  }).join("");
}

// 2. Load Model Comparison Benchmarks: GET /results
async function loadModelComparison() {
  if (!benchmarkTbody) return;

  try {
    const res = await fetch(`${API_URL}/results`);
    if (!res.ok) throw new Error(`Could not load comparison data (${res.status})`);
    const data = await res.json();

    const comp = data.comparison || {};
    const rej = data.rejection_scores || {};
    const models = comp.models || {};
    const rejModels = rej.models || {};

    const modelKeys = ["efficientnet_b0", "resnet18", "resnet50"];

    // Evaluate best model logic: highest macro-F1, tie-break by rejection AUROC
    let bestKey = "resnet18";
    let bestF1 = -1;
    let bestAuroc = -1;

    modelKeys.forEach((k) => {
      const mComp = models[k] || {};
      const mRej = rejModels[k] || {};
      const f1 = mComp.classification_report ? (mComp.classification_report["macro avg"] ? mComp.classification_report["macro avg"]["f1-score"] : 0) : 0;
      const cosAuroc = mRej.results && mRej.results.cosine_sim ? mRej.results.cosine_sim.auroc : 0;

      if (f1 > bestF1 || (Math.abs(f1 - bestF1) < 1e-4 && cosAuroc > bestAuroc)) {
        bestF1 = f1;
        bestAuroc = cosAuroc;
        bestKey = k;
      }
    });

    const bestMeta = MODEL_DISPLAY_INFO[bestKey] || { title: bestKey };
    if (bestModelHighlight) {
      bestModelTitle.textContent = `${bestMeta.title} is the Best Performing Classifier`;
      bestModelReason.innerHTML = `Highest Test Macro-F1 (<strong>${bestF1.toFixed(3)}</strong>) and Highest Out-of-Distribution Rejection AUROC (<strong>${bestAuroc.toFixed(4)}</strong>) with fast inference latency.`;
      bestModelHighlight.classList.remove("hidden");
    }

    // Build Table Rows (using plain wording without raw LaTeX)
    const rows = [
      {
        label: "Test Accuracy",
        format: (k) => models[k] ? `${(models[k].accuracy * 100).toFixed(2)}%` : "N/A",
        isBest: (k) => k === bestKey,
      },
      {
        label: "Macro-F1 Score",
        format: (k) => models[k] && models[k].classification_report && models[k].classification_report["macro avg"]
          ? models[k].classification_report["macro avg"]["f1-score"].toFixed(3)
          : "N/A",
        isBest: (k) => k === bestKey,
      },
      {
        label: "Coconut F1-Score",
        format: (k) => models[k] && models[k].classification_report && models[k].classification_report["coconut"]
          ? models[k].classification_report["coconut"]["f1-score"].toFixed(3)
          : "N/A",
      },
      {
        label: "Date F1-Score",
        format: (k) => models[k] && models[k].classification_report && models[k].classification_report["date"]
          ? models[k].classification_report["date"]["f1-score"].toFixed(3)
          : "N/A",
      },
      {
        label: "Kithul F1-Score",
        format: (k) => models[k] && models[k].classification_report && models[k].classification_report["kithul"]
          ? models[k].classification_report["kithul"]["f1-score"].toFixed(3)
          : "N/A",
      },
      {
        label: "Parameters",
        format: (k) => models[k] ? Number(models[k].parameter_count).toLocaleString() : "N/A",
      },
      {
        label: "Weight File Size",
        format: (k) => models[k] ? `${models[k].file_size_mb} MB` : "N/A",
      },
      {
        label: "Avg CPU Latency",
        format: (k) => models[k] ? `${models[k].avg_cpu_latency_ms} ms` : "N/A",
      },
      {
        label: "OOD Rejection AUROC (Cosine)",
        format: (k) => rejModels[k] && rejModels[k].results && rejModels[k].results.cosine_sim
          ? rejModels[k].results.cosine_sim.auroc.toFixed(4)
          : "N/A",
        isBest: (k) => k === bestKey,
      },
      {
        label: "Real Flowers Kept (at 95% threshold)",
        format: (k) => rejModels[k] && rejModels[k].results && rejModels[k].results.cosine_sim
          ? `${(rejModels[k].results.cosine_sim.real_kept_95 * 100).toFixed(1)}%`
          : "N/A",
      },
      {
        label: "Junk Rejected (at 95% threshold)",
        format: (k) => rejModels[k] && rejModels[k].results && rejModels[k].results.cosine_sim
          ? `${(rejModels[k].results.cosine_sim.junk_rej_95 * 100).toFixed(1)}%`
          : "N/A",
      },
    ];

    benchmarkTbody.innerHTML = rows.map((r) => {
      return `
        <tr>
          <td class="metric-name-col">${r.label}</td>
          ${modelKeys.map((k) => {
            const val = r.format(k);
            const isWinner = r.isBest && r.isBest(k);
            return `<td class="${isWinner ? 'metric-winner' : ''}">${val}</td>`;
          }).join("")}
        </tr>
      `;
    }).join("");

    // Render Confusion Matrix Images from /results/image/{name}
    if (cmGalleryGrid) {
      cmGalleryGrid.innerHTML = modelKeys.map((k) => {
        const meta = MODEL_DISPLAY_INFO[k] || { title: k };
        const imgSrc = `${API_URL}/results/image/cm_${k}.png`;
        return `
          <div class="cm-card ${k === bestKey ? 'cm-card-best' : ''}">
            <h4 class="cm-card-title">${meta.title}</h4>
            <div class="cm-image-container">
              <img src="${imgSrc}" alt="${meta.title} Confusion Matrix" class="cm-img" loading="lazy">
            </div>
          </div>
        `;
      }).join("");
    }
  } catch (err) {
    console.warn("Could not fetch benchmark comparison data:", err);
    if (benchmarkTbody) {
      benchmarkTbody.innerHTML = `
        <tr>
          <td colspan="4" class="table-error">
            Unable to load model comparison data from <code>${API_URL}/results</code>. Please verify the backend is running.
          </td>
        </tr>
      `;
    }
  }
}
