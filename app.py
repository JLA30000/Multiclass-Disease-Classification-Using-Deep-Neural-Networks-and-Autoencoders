from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Literal, Set
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from neural_network_autoencoder import Autoencoder
from neural_network import NeuralNetwork
from logistic_regression_weighted import LogisticRegressionSoftmax
from fastapi.middleware.cors import CORSMiddleware

AE_BUNDLE_PATHS = {'ae_t100': 'export_model/ae_classifier.bundle.pt',
                   'ae_t200': 'export_model_200/ae_classifier_200.bundle.pt', 'ae_t300': 'export_model_300/ae_classifier_300.bundle.pt'}
CW_BUNDLE_PATHS = {'cw_full': 'export_cw/cw_classifier.bundle.pt'}
LR_BUNDLE_PATHS = {'lr_full': 'export_lr/lr_classifier.bundle.pt'}
AE_CLF_FULL_BUNDLE_PATHS = {
    'ae_clf_full': 'export_ae_clf/ae_clf_full.bundle.pt'}
ModelKey = Literal['ae_t100', 'ae_t200',
                   'ae_t300', 'cw_full', 'lr_full', 'ae_clf_full']


class PredictSymptomsRequest(BaseModel):
    symptoms: List[str] = Field(
        default_factory=list, description='List of symptom names to set to 1.')
    temperature: float = Field(default=1.0, ge=1e-06)
    strict: bool = Field(
        default=True, description='If true, unknown symptom names cause a 400 error. If false, unknown symptoms are ignored.')


class TopItem(BaseModel):
    disease: str
    prob: float


class ModelPrediction(BaseModel):
    model_key: str
    model_type: str
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
    pmfc: Dict[str, List[str]]


class SymptomsResponse(BaseModel):
    symptoms: List[str]
    pms: Dict[str, List[str]]
    note: str


def require_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Bundle not found: {path}')


def sidecar_json_path(bundle_path: str) -> str:
    return f'{os.path.splitext(bundle_path)[0]}.json'


def parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


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
            out.append({'disease': label_names[int(i)], 'prob': float(v)})
        return out
    return {'top1': {'disease': label_names[top1_idx], 'prob': float(probs[top1_idx].item())}, 'top3': pack(top3_idx, top3_vals), 'top5': pack(top5_idx, top5_vals)}


def normalize_symptom_names(symptoms: List[str]) -> List[str]:
    return [s.strip() for s in symptoms if (s or '').strip()]


class BaseBundle:

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        self.model_key = model_key
        self.bundle_path = bundle_path
        self.device = device
        require_exists(bundle_path)
        bundle = torch.load(
            bundle_path, map_location=device, weights_only=False)
        try:
            self.feature_cols: List[str] = bundle['feature_cols']
            self.label_names: List[str] = bundle['label_names']
            self.cfg: Dict[str, Any] = bundle['config']
        except KeyError as e:
            raise KeyError(f'[{model_key}] bundle missing key {e}.') from e
        self._bundle: Optional[Dict[str, Any]] = bundle
        self.num_classes = int(self.cfg.get(
            'num_classes', len(self.label_names)))
        self.feature_set: Set[str] = set(self.feature_cols)

    def _release_bundle(self) -> None:
        self._bundle = None

    def build_binary_feature_vector(self, sc: List[str], strict: bool) -> np.ndarray:
        checked = normalize_symptom_names(sc)
        unknown = [s for s in checked if s not in self.feature_set]
        if unknown and strict:
            raise HTTPException(
                status_code=400, detail=f'[{self.model_key}] Unknown symptom(s): {unknown[:20]} (showing up to 20). Use GET /symptoms to see valid names.')
        x = np.zeros(len(self.feature_cols), dtype=np.float32)
        idx_map = None
        if checked:
            idx_map = {name: i for i, name in enumerate(self.feature_cols)}
            for s in checked:
                if s in idx_map:
                    x[idx_map[s]] = 1.0
        return x


class AEBundle(BaseBundle):
    model_type = 'ae'

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)
        try:
            ae_sd = self._bundle['autoencoder_state_dict']
            clf_sd = self._bundle['classifier_state_dict']
        except KeyError as e:
            raise KeyError(f'[{model_key}] AE bundle missing key {e}.') from e
        self._release_bundle()
        raw_input_dim = int(self.cfg['raw_input_dim'])
        z_dim = int(self.cfg['z_dim'])
        hidden_dims = list(self.cfg['hidden_dims'])
        activation = str(self.cfg['activation'])
        num_classes = int(self.cfg['num_classes'])
        if len(self.feature_cols) != raw_input_dim:
            raise ValueError(
                f'[{model_key}] feature_cols ({len(self.feature_cols)}) != raw_input_dim ({raw_input_dim})')
        if len(self.label_names) != num_classes:
            raise ValueError(
                f'[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})')
        self.ae = Autoencoder(input_dim=raw_input_dim).to(device)
        self.ae.load_state_dict(ae_sd)
        self.ae.eval()
        self.clf = NeuralNetwork(input_dim=z_dim, hidden_dims=hidden_dims,
                                 output_dim=num_classes, activation=activation).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(self.device).float().unsqueeze(0)
        z = self.ae.encoder(x)
        logits = self.clf(z).squeeze(0)
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


class CWBundle(BaseBundle):
    model_type = 'cw'

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)
        try:
            clf_sd = self._bundle['classifier_state_dict']
        except KeyError as e:
            raise KeyError(f'[{model_key}] CW bundle missing key {e}.') from e
        self._release_bundle()
        input_dim = int(self.cfg['input_dim'])
        hidden_dims = list(self.cfg['hidden_dims'])
        activation = str(self.cfg['activation'])
        num_classes = int(self.cfg['num_classes'])
        if len(self.feature_cols) != input_dim:
            raise ValueError(
                f'[{model_key}] feature_cols ({len(self.feature_cols)}) != input_dim ({input_dim})')
        if len(self.label_names) != num_classes:
            raise ValueError(
                f'[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})')
        self.clf = NeuralNetwork(input_dim=input_dim, hidden_dims=hidden_dims,
                                 output_dim=num_classes, activation=activation).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(self.device).float().unsqueeze(0)
        logits = self.clf(x).squeeze(0)
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


class LRBundle(BaseBundle):
    model_type = 'lr'

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)
        try:
            clf_sd = self._bundle['classifier_state_dict']
        except KeyError as e:
            raise KeyError(f'[{model_key}] LR bundle missing key {e}.') from e
        self._release_bundle()
        input_dim = int(self.cfg['input_dim'])
        num_classes = int(self.cfg['num_classes'])
        if len(self.feature_cols) != input_dim:
            raise ValueError(
                f'[{model_key}] feature_cols ({len(self.feature_cols)}) != input_dim ({input_dim})')
        if len(self.label_names) != num_classes:
            raise ValueError(
                f'[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})')
        self.clf = LogisticRegressionSoftmax(
            input_dim=input_dim, num_classes=num_classes).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(self.device).float().unsqueeze(0)
        logits = self.clf(x).squeeze(0)
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


class AEClfFullBundle(BaseBundle):
    model_type = 'ae_clf_full'

    def __init__(self, model_key: str, bundle_path: str, device: torch.device):
        super().__init__(model_key, bundle_path, device)
        try:
            enc_sd = self._bundle['encoder_state_dict']
            clf_sd = self._bundle['classifier_state_dict']
        except KeyError as e:
            raise KeyError(
                f'[{model_key}] AE-CLF-Full bundle missing key {e}.') from e
        self._release_bundle()
        raw_input_dim = int(self.cfg['raw_input_dim'])
        latent_dim = int(self.cfg['latent_dim'])
        ae_hidden_dims = tuple(self.cfg['ae_hidden_dims'])
        z_dim = int(self.cfg['z_dim'])
        hidden_dims = list(self.cfg['hidden_dims'])
        activation = str(self.cfg['activation'])
        num_classes = int(self.cfg['num_classes'])
        if len(self.feature_cols) != raw_input_dim:
            raise ValueError(
                f'[{model_key}] feature_cols ({len(self.feature_cols)}) != raw_input_dim ({raw_input_dim})')
        if len(self.label_names) != num_classes:
            raise ValueError(
                f'[{model_key}] label_names ({len(self.label_names)}) != num_classes ({num_classes})')
        ae_shell = Autoencoder(input_dim=raw_input_dim,
                               latent_dim=latent_dim, hidden_dims=ae_hidden_dims)
        self.encoder = ae_shell.encoder.to(device)
        self.encoder.load_state_dict(enc_sd)
        self.encoder.eval()
        self.clf = NeuralNetwork(input_dim=z_dim, hidden_dims=hidden_dims,
                                 output_dim=num_classes, activation=activation).to(device)
        self.clf.load_state_dict(clf_sd)
        self.clf.eval()

    @torch.no_grad()
    def predict(self, x_np: np.ndarray, temperature: float = 1.0) -> Dict[str, Any]:
        x = torch.from_numpy(x_np).to(self.device).float().unsqueeze(0)
        z = self.encoder(x)
        logits = self.clf(z).squeeze(0)
        if temperature != 1.0:
            logits = logits / float(temperature)
        probs = torch.softmax(logits, dim=0)
        return softmax_topk(probs, self.label_names)


app = FastAPI(title='Disease Prediction API', version='3.0')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_GROUP_NAMES = ('ae', 'cw', 'lr', 'ae_clf_full')
PRELOAD_MODELS = parse_bool_env('PRELOAD_MODELS', default=False)
MODEL_SPECS: Dict[str, Dict[str, Any]] = {}
MODEL_METADATA: Dict[str, Dict[str, Any]] = {}
LOADED_MODELS: Dict[str, BaseBundle] = {}
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


def load_bundle_metadata(model_key: str, model_type: str, bundle_path: str) -> Dict[str, Any]:
    json_path = sidecar_json_path(bundle_path)
    require_exists(json_path)
    with open(json_path, 'r', encoding='utf-8') as fh:
        meta = json.load(fh)
    cfg = meta.get('config')
    if not isinstance(cfg, dict):
        raise KeyError(f'[{model_key}] metadata missing config.')
    feature_cols = cfg.get('feature_cols')
    if not isinstance(feature_cols, list):
        raise KeyError(f'[{model_key}] metadata missing feature_cols.')
    num_classes = cfg.get('num_classes', meta.get('label_names_count'))
    if num_classes is None:
        raise KeyError(f'[{model_key}] metadata missing num_classes.')
    return {'model_key': model_key, 'model_type': model_type, 'bundle_path': bundle_path, 'feature_cols': feature_cols, 'num_classes': int(num_classes)}


def register_model_spec(model_type: str, bundle_paths: Dict[str, str], bundle_cls: Any) -> None:
    for model_key, bundle_path in bundle_paths.items():
        require_exists(bundle_path)
        MODEL_SPECS[model_key] = {
            'model_type': model_type, 'bundle_path': bundle_path, 'bundle_cls': bundle_cls}
        MODEL_METADATA[model_key] = load_bundle_metadata(
            model_key=model_key, model_type=model_type, bundle_path=bundle_path)


def initialize_model_registry() -> None:
    if MODEL_SPECS and MODEL_METADATA:
        return
    register_model_spec('ae', AE_BUNDLE_PATHS, AEBundle)
    register_model_spec('cw', CW_BUNDLE_PATHS, CWBundle)
    register_model_spec('lr', LR_BUNDLE_PATHS, LRBundle)
    register_model_spec(
        'ae_clf_full', AE_CLF_FULL_BUNDLE_PATHS, AEClfFullBundle)
    if PRELOAD_MODELS:
        for model_key in MODEL_SPECS:
            get_model(model_key)


def grouped_loaded_models() -> Dict[str, List[str]]:
    grouped = {group: [] for group in MODEL_GROUP_NAMES}
    for model_key in LOADED_MODELS:
        model_type = MODEL_METADATA[model_key]['model_type']
        grouped[model_type].append(model_key)
    return grouped


def get_model(model_key: str) -> BaseBundle:
    initialize_model_registry()
    if model_key in LOADED_MODELS:
        return LOADED_MODELS[model_key]
    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        raise KeyError(f'Unknown model key: {model_key}')
    model = spec['bundle_cls'](
        model_key=model_key, bundle_path=spec['bundle_path'], device=DEVICE)
    LOADED_MODELS[model_key] = model
    return model


@app.on_event('startup')
def startup_load_models():
    initialize_model_registry()


@app.get('/health', response_model=HealthResponse)
def health():
    initialize_model_registry()
    return {'status': 'ok', 'device': str(DEVICE), 'models_loaded': grouped_loaded_models()}


@app.get('/schema', response_model=SchemaResponse)
def schema():
    initialize_model_registry()
    if not MODEL_METADATA:
        raise HTTPException(status_code=503, detail='Models not loaded.')
    per_model: Dict[str, List[str]] = {}
    for model_key, meta in MODEL_METADATA.items():
        per_model[model_key] = meta['feature_cols']
    return {'per_model_feature_cols': per_model}


@app.get('/symptoms', response_model=SymptomsResponse)
def symptoms():
    initialize_model_registry()
    if not MODEL_METADATA:
        raise HTTPException(status_code=503, detail='Models not loaded.')
    per_model: Dict[str, List[str]] = {}
    all_syms: Set[str] = set()
    for model_key, meta in MODEL_METADATA.items():
        feature_cols = meta['feature_cols']
        per_model[model_key] = feature_cols
        all_syms.update(feature_cols)
    return {'symptoms': sorted(all_syms), 'per_model_symptoms': per_model, 'note': 'These symptom names come directly from model feature_cols (binary symptom columns). Send a checklist to POST /predict_symptoms and the backend will build a binary vector.'}


@app.post('/predict_symptoms', response_model=PredictResponse)
def predict_symptoms(req: PredictSymptomsRequest):
    initialize_model_registry()
    if not MODEL_SPECS:
        raise HTTPException(status_code=503, detail='Models not loaded.')
    preds: List[ModelPrediction] = []
    for model_key in MODEL_SPECS:
        model = get_model(model_key)
        x_np = model.build_binary_feature_vector(
            req.symptoms, strict=req.strict)
        out = model.predict(x_np, temperature=req.temperature)
        preds.append(ModelPrediction(model_key=model_key, model_type=model.model_type, num_classes=model.num_classes, top1=TopItem(
            **out['top1']), top3=[TopItem(**d) for d in out['top3']], top5=[TopItem(**d) for d in out['top5']]))
    return PredictResponse(predictions=preds)
