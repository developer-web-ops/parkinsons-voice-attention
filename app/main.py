"""FastAPI service exposing the Parkinson's voice-biomarker models."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.inference import MODEL_NAMES, get_bundle, load_report

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "app" / "static"
REPORTS_DIR = ROOT / "reports"

app = FastAPI(
    title="Parkinson's Disease Detection from Voice Biomarkers",
    description="Attention-based neural network over UCI PD speech features.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Feature name to value. Missing features fall back to the dataset median.",
    )
    model: str = Field(default="attention", description="'attention' or 'mlp'")
    explain: bool = Field(default=True, description="Include SHAP attributions")


class PredictResponse(BaseModel):
    model: str
    probability: float
    prediction: int
    label: str
    threshold: float
    attention: dict[str, float] | None = None
    shap: list[dict] = Field(default_factory=list)


def _validate_model(name: str) -> str:
    if name not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"model must be one of {list(MODEL_NAMES)}")
    return name


@app.get("/api/health")
def health() -> dict:
    bundle = get_bundle()
    return {"status": "ok", "models": list(MODEL_NAMES), "n_features": len(bundle.feature_names)}


@app.get("/api/features")
def features() -> dict:
    """Feature catalogue with medians and ranges, used to build the input form."""
    bundle = get_bundle()
    meta = bundle.metadata
    return {
        "groups": bundle.feature_groups,
        "defaults": bundle.defaults(),
        "min": {k: float(v) for k, v in meta["feature_min"].items()},
        "max": {k: float(v) for k, v in meta["feature_max"].items()},
    }


@app.get("/api/examples")
def examples() -> dict:
    """Real held-out recordings (one per class) for one-click demos."""
    return get_bundle().metadata.get("examples", {})


@app.get("/api/metrics")
def metrics() -> dict:
    data = load_report("metrics.json")
    if not data:
        raise HTTPException(status_code=404, detail="metrics.json not found; run src.train first")
    return data


@app.get("/api/explainability")
def explainability() -> dict:
    data = load_report("explainability.json")
    if not data:
        raise HTTPException(
            status_code=404, detail="explainability.json not found; run src.explain first"
        )
    return data


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    bundle = get_bundle()
    model = _validate_model(request.model)
    unknown = set(request.features) - set(bundle.feature_names)
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown features: {sorted(unknown)[:5]}")

    raw = bundle.vectorise(request.features)
    result = bundle.predict(raw, model=model)
    shap_values = bundle.shap_top_features(raw) if request.explain else []
    return PredictResponse(**result, shap=shap_values)


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...), model: str = "attention") -> dict:
    """Score every row of an uploaded CSV that uses the dataset's column names."""
    bundle = get_bundle()
    model = _validate_model(model)
    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise HTTPException(status_code=400, detail="CSV has no header row")

    known = set(bundle.feature_names)
    if not known & set(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail="No recognised feature columns; the header must use dataset feature names.",
        )

    predictions = []
    for i, row in enumerate(reader):
        if i >= 200:
            break
        values = {k: float(v) for k, v in row.items() if k in known and v not in (None, "")}
        result = bundle.predict(bundle.vectorise(values), model=model)
        result["row"] = i
        predictions.append(result)

    if not predictions:
        raise HTTPException(status_code=400, detail="CSV contained no data rows")
    positives = sum(p["prediction"] for p in predictions)
    return {
        "count": len(predictions),
        "positive": positives,
        "negative": len(predictions) - positives,
        "predictions": predictions,
    }


@app.get("/api/report/{name}")
def report_image(name: str) -> FileResponse:
    path = (REPORTS_DIR / name).resolve()
    if path.parent != REPORTS_DIR.resolve() or not path.is_file() or path.suffix != ".png":
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(path, media_type="image/png")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
