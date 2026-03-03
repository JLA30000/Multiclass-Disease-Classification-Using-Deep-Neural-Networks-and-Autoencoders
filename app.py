from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Literal, Set

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Must match your training/export classes
from neural_network_autoencoder import Autoencoder
from neural_network import NeuralNetwork
from logistic_regression_weighted import LogisticRegressionSoftmax
from fastapi.middleware.cors import CORSMiddleware

import xgboost as xgb


# ============================================================
# Bundle paths
# ============================================================
AE_BUNDLE_PATHS = {
    "ae_t100": r"export_model/ae_classifier.bundle.pt",
    "ae_t200": r"export_model_200/ae_classifier_200.bundle.pt",
    "ae_t300": r"export_model_300/ae_classifier_300.bundle.pt",
}

CW_BUNDLE_PATHS = {
    "cw_full": r"export_cw/cw_classifier.bundle.pt",
}

XGB_BUNDLE_PATHS = {
    "xgb_full": r"export_xgb/xgb_classifier.bundle.pt",
}

LR_BUNDLE_PATHS = {
    "lr_full": r"export_lr/lr_classifier.bundle.pt",
}

AE_CLF_FULL_BUNDLE_PATHS = {
    "ae_clf_full": r"export_ae_clf/ae_clf_full.bundle.pt",
}

ModelKey = Literal[
    "ae_t100", "ae_t200", "ae_t300",
    "cw_full",
    "xgb_full",
    "lr_full",
    "ae_clf_full",
]


# ============================================================
# API Schemas
# ============================================================
class PredictSymptomsRequest(BaseModel):
    """
    User provides a checklist of symptoms (feature names).
    Backend converts to binary vector: checked -> 1, unchecked -> 0.
    """
    symptoms: List[str] = Field(
        default_factory=list, description="List of symptom names to set to 1.")
    temperature: float = Field(default=1.0, ge=1e-6)
    strict: bool = Field(
        default=True,
        description="If true, unknown symptom names cause a 400 error. If false, unknown symptoms are ignored."
    )


class TopItem(BaseModel):
    disease: str
    prob: float


class ModelPrediction(BaseModel):
    model_key: str
    model_type: str  # "ae" or "cw"
    num_classes: int
    top1: TopItem
    top3: List[TopItem]
    top5: List[TopItem]


class PredictResponse(BaseModel):
    predictions: List[ModelPrediction]


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: Dict[str, List[str]]


class SchemaResponse(BaseModel):
    per_model_feature_cols: Dict[str, List[str]]


class SymptomsResponse(BaseModel):
    """
    For your dataset, 'symptoms' == 'feature_cols'.
    """
    symptoms: List[str]
    per_model_symptoms: Dict[str, List[str]]
    note: str


# ============================================================
# Helpers
# ============================================================
def require_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")


def softmax_topk(probs: torch.Tensor, label_names: List[str]) -> Dict[str, Any]:
    num_classes = probs.numel()
    k3 = min(3, num_classes)
    k5 = min(5, num_classes)

    top1_idx = int(torch.argmax(probs).item())
    top3_vals, top3_idx = torch.topk(probs, k3)
    top5_vals, top5_idx = torch.topk(probs, k5)

    def pack(idxs: torch.Tensor, vals: torch.Tensor) -> List[Dict[str, Any]]:
        out = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            out.append({"disease": label_names[int(i)], "prob": float(v)})
        return out

    return {
        "top1": {"disease": label_names[top1_idx], "prob": float(probs[top1_idx].item())},
        "top3": pack(top3_idx, top3_vals),
        "top5": pack(top5_idx, top5_vals),
    }


def normalize_symptom_names(symptoms: List[str]) -> List[str]:
    # Normalize incoming symptom names (trim spaces)
    return [s.strip() for s in symptoms if (s or "").strip()]


# ============================================================
# Bundle wrappers
# ============================================================
class BaseBundle:
    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        self.model_key = model_key
        self.bundle_path = bundle_path
        self.device = device

        require_exists(bundle_path)

        # PyTorch 2.6+ defaults weights_only=True; your bundle has metadata (numpy, lists, etc.)
        # You created these bundles -> trusted -> load with weights_only=False.
        bundle = torch.load(
            bundle_path, map_location=device, weights_only=False)

        try:
            # symptom names
            self.feature_cols: List[str] = bundle["feature_cols"]
            # disease names
            self.label_names: List[str] = bundle["label_names"]
            self.cfg: Dict[str, Any] = bundle["config"]
        except KeyError as e:
            raise KeyError(f"[{model_key}] bundle missing key {e}.") from e

        self._bundle: Optional[Dict[str, Any]] = bundle
        self.num_classes = int(self.cfg.get(
            "num_classes", len(self.label_names)))
        self.feature_set: Set[str] = set(self.feature_cols)

    def _release_bundle(self) -> None:
        # Drop the serialized payload once the runtime model is initialized.
        self._bundle = None

    def build_binary_feature_vector(self, symptoms_checked: List[str], strict: bool) -> np.ndarray:
        """
        Converts checklist to binary vector in the exact feature_cols order.
        """
        checked = normalize_symptom_names(symptoms_checked)

        unknown = [s for s in checked if s not in self.feature_set]
        if unknown and strict:
            raise HTTPException(
                status_code=400,
                detail=f"[{self.model_key}] Unknown symptom(s): {unknown[:20]} "
                f"(showing up to 20). Use GET /symptoms to see valid names."
            )

        # Build vector
        x = np.zeros(len(self.feature_cols), dtype=np.float32)
        idx_map = None  # lazily build if needed
        if checked:
            idx_map = {name: i for i, name in enumerate(self.feature_cols)}
            for s in checked:
                if s in idx_map:
                    x[idx_map[s]] = 1.0

        return x


class AEBundle(BaseBundle):
    model_type = "ae"

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)

        try:
            ae_sd = self._bundle["autoencoder_state_dict"]
            clf_sd = self._bundle["classifier_state_dict"]
        except KeyError as e:
            raise KeyError(f"[{model_key}] AE bundle missing key {e}.") from e
        self._release_bundle()

        raw_input_dim = int(self.cfg["raw_input_dim"])
        z_dim = int(self.cfg["z_dim"])
        hidden_dims = list(self.cfg["hidden_dims"])
        activation = str(self.cfg["activation"])
        num_classes = int(self.cfg["num_classes"])

        if len(self.feature_cols) != raw_input_dim:
            raise ValueError(
                f"[{model_key}] feature_cols ({len(self.feature_cols)}) != raw_input_dim ({raw_input_dim})")
        if len(self.label_names) != num_classes:
            raise ValueError(
                f"[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})")

        self.ae = Autoencoder(input_dim=raw_input_dim).to(device)
        self.ae.load_state_dict(ae_sd)
        self.ae.eval()

        self.clf = NeuralNetwork(
            input_dim=z_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            activation=activation,
        ).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(
            self.device).float().unsqueeze(0)  # [1, D]
        # [1, z_dim]
        z = self.ae.encoder(x)
        logits = self.clf(z).squeeze(0)                                  # [C]
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


class CWBundle(BaseBundle):
    model_type = "cw"

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)

        try:
            clf_sd = self._bundle["classifier_state_dict"]
        except KeyError as e:
            raise KeyError(f"[{model_key}] CW bundle missing key {e}.") from e
        self._release_bundle()

        input_dim = int(self.cfg["input_dim"])
        hidden_dims = list(self.cfg["hidden_dims"])
        activation = str(self.cfg["activation"])
        num_classes = int(self.cfg["num_classes"])

        if len(self.feature_cols) != input_dim:
            raise ValueError(
                f"[{model_key}] feature_cols ({len(self.feature_cols)}) != input_dim ({input_dim})")
        if len(self.label_names) != num_classes:
            raise ValueError(
                f"[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})")

        self.clf = NeuralNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            activation=activation,
        ).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(
            self.device).float().unsqueeze(0)  # [1, D]
        logits = self.clf(x).squeeze(0)                                  # [C]
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


class XGBBundle(BaseBundle):
    model_type = "xgb"

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)

        try:
            booster_raw = self._bundle["xgb_booster_raw"]
        except KeyError as e:
            raise KeyError(f"[{model_key}] XGB bundle missing key {e}.") from e
        self._release_bundle()

        self.bst = xgb.Booster()
        self.bst.load_model(
            booster_raw if isinstance(booster_raw, bytearray) else bytearray(booster_raw)
        )

    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        dm = xgb.DMatrix(x_np.reshape(1, -1))
        probs_np = self.bst.predict(dm)[0]  # [C] softprob output
        if temperature != 1.0:
            # apply temperature in log-space
            log_probs = np.log(np.clip(probs_np, 1e-30, None)) / float(temperature)
            log_probs -= log_probs.max()
            probs_np = np.exp(log_probs)
            probs_np /= probs_np.sum()
        probs = torch.from_numpy(probs_np).float()
        return softmax_topk(probs, self.label_names)


class LRBundle(BaseBundle):
    model_type = "lr"

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)

        try:
            clf_sd = self._bundle["classifier_state_dict"]
        except KeyError as e:
            raise KeyError(f"[{model_key}] LR bundle missing key {e}.") from e
        self._release_bundle()

        input_dim = int(self.cfg["input_dim"])
        num_classes = int(self.cfg["num_classes"])

        if len(self.feature_cols) != input_dim:
            raise ValueError(
                f"[{model_key}] feature_cols ({len(self.feature_cols)}) != input_dim ({input_dim})")
        if len(self.label_names) != num_classes:
            raise ValueError(
                f"[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})")

        self.clf = LogisticRegressionSoftmax(
            input_dim=input_dim, num_classes=num_classes,
        ).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(
            self.device).float().unsqueeze(0)  # [1, D]
        logits = self.clf(x).squeeze(0)                                  # [C]
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


class AEClfFullBundle(BaseBundle):
    model_type = "ae_clf_full"

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)

        try:
            enc_sd = self._bundle["encoder_state_dict"]
            clf_sd = self._bundle["classifier_state_dict"]
        except KeyError as e:
            raise KeyError(f"[{model_key}] AE-CLF-Full bundle missing key {e}.") from e
        self._release_bundle()

        raw_input_dim = int(self.cfg["raw_input_dim"])
        latent_dim = int(self.cfg["latent_dim"])
        ae_hidden_dims = tuple(self.cfg["ae_hidden_dims"])
        z_dim = int(self.cfg["z_dim"])
        hidden_dims = list(self.cfg["hidden_dims"])
        activation = str(self.cfg["activation"])
        num_classes = int(self.cfg["num_classes"])

        if len(self.feature_cols) != raw_input_dim:
            raise ValueError(
                f"[{model_key}] feature_cols ({len(self.feature_cols)}) != raw_input_dim ({raw_input_dim})")
        if len(self.label_names) != num_classes:
            raise ValueError(
                f"[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})")

        # Reconstruct encoder architecture via Autoencoder, then load fine-tuned encoder weights
        ae_shell = Autoencoder(
            input_dim=raw_input_dim,
            latent_dim=latent_dim,
            hidden_dims=ae_hidden_dims,
        )
        self.encoder = ae_shell.encoder.to(device)
        self.encoder.load_state_dict(enc_sd)
        self.encoder.eval()

        self.clf = NeuralNetwork(
            input_dim=z_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            activation=activation,
        ).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(
            self.device).float().unsqueeze(0)  # [1, D]
        z = self.encoder(x)                                              # [1, z_dim]
        logits = self.clf(z).squeeze(0)                                  # [C]
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Disease Prediction API", version="3.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AE_MODELS: Dict[str, AEBundle] = {}
CW_MODELS: Dict[str, CWBundle] = {}
XGB_MODELS: Dict[str, XGBBundle] = {}
LR_MODELS: Dict[str, LRBundle] = {}
AE_CLF_FULL_MODELS: Dict[str, AEClfFullBundle] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_load_models():
    for k, path in AE_BUNDLE_PATHS.items():
        AE_MODELS[k] = AEBundle(model_key=k, bundle_path=path, device=DEVICE)
    for k, path in CW_BUNDLE_PATHS.items():
        CW_MODELS[k] = CWBundle(model_key=k, bundle_path=path, device=DEVICE)
    for k, path in XGB_BUNDLE_PATHS.items():
        XGB_MODELS[k] = XGBBundle(model_key=k, bundle_path=path, device=DEVICE)
    for k, path in LR_BUNDLE_PATHS.items():
        LR_MODELS[k] = LRBundle(model_key=k, bundle_path=path, device=DEVICE)
    for k, path in AE_CLF_FULL_BUNDLE_PATHS.items():
        AE_CLF_FULL_MODELS[k] = AEClfFullBundle(model_key=k, bundle_path=path, device=DEVICE)


# ============================================================
# Utility endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models_loaded": {
            "ae": list(AE_MODELS.keys()),
            "cw": list(CW_MODELS.keys()),
            "xgb": list(XGB_MODELS.keys()),
            "lr": list(LR_MODELS.keys()),
            "ae_clf_full": list(AE_CLF_FULL_MODELS.keys()),
        },
    }


@app.get("/schema", response_model=SchemaResponse)
def schema():
    """
    Returns feature columns per model (these are your symptom labels).
    """
    all_models = {**AE_MODELS, **CW_MODELS, **XGB_MODELS, **LR_MODELS, **AE_CLF_FULL_MODELS}
    if not all_models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    per_model: Dict[str, List[str]] = {}
    for k, m in all_models.items():
        per_model[k] = m.feature_cols

    return {"per_model_feature_cols": per_model}


@app.get("/symptoms", response_model=SymptomsResponse)
def symptoms():
    """
    Product-facing: returns the symptom list for your checklist UI.

    In your dataset, symptoms == feature_cols.
    If the four models use the same feature_cols (likely), `symptoms` will be that list.
    If they differ, `symptoms` is the union, and `per_model_symptoms` tells you each model's list.
    """
    all_models = {**AE_MODELS, **CW_MODELS, **XGB_MODELS, **LR_MODELS, **AE_CLF_FULL_MODELS}
    if not all_models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    per_model: Dict[str, List[str]] = {}
    all_syms: Set[str] = set()

    for k, m in all_models.items():
        per_model[k] = m.feature_cols
        all_syms.update(m.feature_cols)

    return {
        "symptoms": sorted(all_syms),
        "per_model_symptoms": per_model,
        "note": "These symptom names come directly from model feature_cols (binary symptom columns). "
                "Send a checklist to POST /predict_symptoms and the backend will build a binary vector."
    }


# ============================================================
# Prediction endpoint (symptom checklist)
# ============================================================
@app.post("/predict_symptoms", response_model=PredictResponse)
def predict_symptoms(req: PredictSymptomsRequest):
    """
    User inputs a checklist of symptoms; backend converts to binary vector and runs all models.
    """
    all_models = {**AE_MODELS, **CW_MODELS, **XGB_MODELS, **LR_MODELS, **AE_CLF_FULL_MODELS}
    if not all_models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    preds: List[ModelPrediction] = []

    for model_key, model in all_models.items():
        x_np = model.build_binary_feature_vector(
            req.symptoms, strict=req.strict)
        out = model.predict(x_np, temperature=req.temperature)
        preds.append(ModelPrediction(
            model_key=model_key,
            model_type=model.model_type,
            num_classes=model.num_classes,
            top1=TopItem(**out["top1"]),
            top3=[TopItem(**d) for d in out["top3"]],
            top5=[TopItem(**d) for d in out["top5"]],
        ))

    return PredictResponse(predictions=preds)
