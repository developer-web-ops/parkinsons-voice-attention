const KEY_FEATURES = [
  ["gender", "0 = female, 1 = male"],
  ["PPE", "Pitch period entropy"],
  ["DFA", "Detrended fluctuation analysis"],
  ["RPDE", "Recurrence period density entropy"],
  ["numPulses", "Number of glottal pulses"],
  ["meanPeriodPulses", "Mean glottal period"],
  ["stdDevPeriodPulses", "Std dev of glottal period"],
  ["locPctJitter", "Local jitter (%)"],
  ["locAbsJitter", "Local absolute jitter"],
  ["rapJitter", "Relative average perturbation"],
  ["ppq5Jitter", "5-point period perturbation quotient"],
  ["locShimmer", "Local shimmer"],
  ["apq3Shimmer", "3-point amplitude perturbation"],
  ["apq11Shimmer", "11-point amplitude perturbation"],
  ["meanHarmToNoiseHarmonicity", "Harmonics-to-noise ratio"],
  ["meanNoiseToHarmHarmonicity", "Noise-to-harmonics ratio"],
  ["meanIntensity", "Mean intensity (dB)"],
];

const state = { defaults: {}, values: {}, examples: {}, extraFeatures: [] };

const $ = (sel) => document.querySelector(sel);

async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function setStatus(message, isError = false) {
  const el = $("#status");
  el.textContent = message;
  el.classList.toggle("error", isError);
}

function fmt(value) {
  if (Math.abs(value) >= 1000 || (Math.abs(value) < 0.001 && value !== 0)) {
    return value.toExponential(2);
  }
  return Number(value.toFixed(4)).toString();
}

function renderForm() {
  const rows = [...KEY_FEATURES, ...state.extraFeatures];
  $("#feature-form").innerHTML = rows
    .map(([name, desc]) => {
      const value = state.values[name] ?? state.defaults[name] ?? 0;
      return `<div class="feature">
        <span class="name">${name}</span>
        <span class="desc">${desc}</span>
        <input type="number" step="any" data-feature="${name}" value="${value}" />
      </div>`;
    })
    .join("");
}

function collectFeatures() {
  const features = { ...state.values };
  document.querySelectorAll("#feature-form input").forEach((input) => {
    const value = parseFloat(input.value);
    if (!Number.isNaN(value)) features[input.dataset.feature] = value;
  });
  return features;
}

function barRow(label, value, max, signed = false) {
  const width = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  const cls = signed ? (value >= 0 ? "pos" : "neg") : "";
  return `<div class="bar-row">
    <span title="${label}">${label}</span>
    <span class="bar-track"><span class="bar-fill ${cls}" style="width:${width}%"></span></span>
    <span class="bar-val">${signed ? value.toFixed(3) : (value * 100).toFixed(1) + "%"}</span>
  </div>`;
}

function renderResult(result) {
  const pd = result.prediction === 1;
  $("#result").classList.remove("empty");
  $("#result").innerHTML = `
    <div class="prob">${(result.probability * 100).toFixed(1)}%</div>
    <span class="badge ${pd ? "pd" : "healthy"}">${pd ? "Parkinson's indicated" : "Healthy"}</span>
    <div class="meter"><div style="width:${(result.probability * 100).toFixed(1)}%"></div></div>
    <div class="meta">Model: ${result.model} · decision threshold ${result.threshold.toFixed(2)}
      · research demo, not a diagnosis</div>`;

  if (result.attention) {
    const entries = Object.entries(result.attention).sort((a, b) => b[1] - a[1]);
    const max = entries[0][1];
    $("#attention-bars").classList.remove("empty-note");
    $("#attention-bars").innerHTML = entries.map(([g, w]) => barRow(g, w, max)).join("");
  } else {
    $("#attention-bars").classList.add("empty-note");
    $("#attention-bars").textContent = "The MLP baseline has no attention layer.";
  }

  if (result.shap && result.shap.length) {
    const max = Math.max(...result.shap.map((s) => Math.abs(s.shap_value)));
    $("#shap-bars").classList.remove("empty-note");
    $("#shap-bars").innerHTML = result.shap
      .map((s) => barRow(s.feature, s.shap_value, max, true))
      .join("");
  }
}

async function predict() {
  setStatus("Scoring…");
  try {
    const result = await getJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        features: collectFeatures(),
        model: $("#model-select").value,
        explain: true,
      }),
    });
    renderResult(result);
    setStatus("Done.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function uploadCSV(file) {
  setStatus(`Scoring ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await getJSON(`/api/predict/csv?model=${$("#model-select").value}`, {
      method: "POST",
      body: form,
    });
    $("#csv-card").hidden = false;
    $("#csv-summary").innerHTML =
      `<p class="hint">${data.count} rows scored — ${data.positive} predicted Parkinson's,
       ${data.negative} predicted healthy.</p>`;
    $("#csv-table").innerHTML =
      `<thead><tr><th>Row</th><th>Probability</th><th>Prediction</th></tr></thead><tbody>` +
      data.predictions
        .map(
          (p) =>
            `<tr><td>${p.row + 1}</td><td>${(p.probability * 100).toFixed(1)}%</td><td>${p.label}</td></tr>`
        )
        .join("") +
      `</tbody>`;
    setStatus("Batch scoring complete.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"];

async function loadMetrics() {
  try {
    const metrics = await getJSON("/api/metrics");
    const names = Object.keys(metrics);
    const bestF1 = Math.max(...names.map((n) => metrics[n].f1));
    $("#metrics-table").innerHTML =
      `<thead><tr><th>Model</th>${METRIC_KEYS.map((k) => `<th>${k}</th>`).join("")}
       <th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr></thead><tbody>` +
      names
        .map((n) => {
          const m = metrics[n];
          const cm = m.confusion_matrix;
          return `<tr class="${m.f1 === bestF1 ? "best" : ""}"><td>${n}</td>` +
            METRIC_KEYS.map((k) => `<td>${m[k].toFixed(3)}</td>`).join("") +
            `<td>${cm[0][0]}</td><td>${cm[0][1]}</td><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr>`;
        })
        .join("") +
      `</tbody>`;
  } catch (err) {
    $("#metrics-table").innerHTML = `<caption>${err.message}</caption>`;
  }
}

async function loadExplainability() {
  try {
    const data = await getJSON("/api/explainability");
    const entries = Object.entries(data.mean_attention).sort((a, b) => b[1] - a[1]);
    const max = entries[0][1];
    $("#global-attention").innerHTML = entries.map(([g, w]) => barRow(g, w, max)).join("");
    state.extraFeatures = data.top_features
      .slice(0, 6)
      .map((f) => [f.feature, "High-impact feature (global SHAP)"])
      .filter(([name]) => !KEY_FEATURES.some(([k]) => k === name));
    renderForm();
  } catch (err) {
    $("#global-attention").textContent = err.message;
  }
}

function applyExample(name) {
  const example = state.examples[name];
  if (!example) {
    setStatus("No example available.", true);
    return;
  }
  state.values = { ...example };
  renderForm();
  setStatus(`Loaded a held-out ${name} recording.`);
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#${tab.dataset.tab}`).classList.add("active");
    });
  });
}

async function init() {
  bindTabs();
  $("#predict-btn").addEventListener("click", predict);
  $("#load-healthy").addEventListener("click", () => applyExample("healthy"));
  $("#load-pd").addEventListener("click", () => applyExample("parkinsons"));
  $("#reset").addEventListener("click", () => {
    state.values = {};
    renderForm();
    setStatus("Reset to dataset medians.");
  });
  $("#csv-input").addEventListener("change", (e) => {
    if (e.target.files.length) uploadCSV(e.target.files[0]);
  });
  document.querySelector(".upload").addEventListener("click", () => $("#csv-input").click());

  try {
    const [features, examples] = await Promise.all([
      getJSON("/api/features"),
      getJSON("/api/examples"),
    ]);
    state.defaults = features.defaults;
    state.examples = examples;
    renderForm();
    setStatus("Ready.");
  } catch (err) {
    setStatus(`Could not load feature metadata: ${err.message}`, true);
  }

  loadMetrics();
  loadExplainability();
}

init();
