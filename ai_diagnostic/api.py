"""
REST API for the AI diagnostic framework.

Run with:
    uvicorn ai_diagnostic.api:app --host 0.0.0.0 --port 8000

Provides endpoints for Bayesian diagnosis, sensitivity analysis,
preset scenarios, and evidence parameter reference.
"""

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .evidence import EVIDENCE_DESCRIPTIONS, EVIDENCE_PARAMS, Evidence
from .bayesian import BayesianDiagnostic, DiagnosisResult
from .likelihoods import FAILURE_CLASS_LABELS, FAILURE_CLASSES, LikelihoodTable
from .sensitivity import SensitivityReport, compute_sensitivity
from .diagnose import PRESETS


# ── Pydantic models ───────────────────────────────────────────────────


class EvidenceBody(BaseModel):
    e1_hardware_anomalies: float = Field(0.0, ge=0.0, le=1.0)
    e2_data_distribution_shift: float = Field(0.0, ge=0.0, le=1.0)
    e3_model_architecture_irregularities: float = Field(0.0, ge=0.0, le=1.0)
    e4_human_interface_errors: float = Field(0.0, ge=0.0, le=1.0)
    e5_temporal_pattern_sudden: float = Field(0.0, ge=0.0, le=1.0)
    e6_output_coherence_adversarial: float = Field(0.0, ge=0.0, le=1.0)
    e7_system_log_errors: float = Field(0.0, ge=0.0, le=1.0)
    e8_external_environment_changes: float = Field(0.0, ge=0.0, le=1.0)
    e9_obfuscated_code_or_weights: float = Field(0.0, ge=0.0, le=1.0)
    e10_hidden_triggers_or_backdoors: float = Field(0.0, ge=0.0, le=1.0)
    e11_unexplained_communication_channels: float = Field(0.0, ge=0.0, le=1.0)
    e12_behavioral_context_inconsistency: float = Field(0.0, ge=0.0, le=1.0)
    e13_unusual_training_data: float = Field(0.0, ge=0.0, le=1.0)
    e14_designer_history_red_flags: float = Field(0.0, ge=0.0, le=1.0)
    e15_legal_or_contractual_violations: float = Field(0.0, ge=0.0, le=1.0)

    def to_evidence(self) -> Evidence:
        return Evidence(**self.model_dump())


class PriorsBody(BaseModel):
    entropy: Optional[float] = Field(None, ge=0.0, le=1.0)
    engineering_limits: Optional[float] = Field(None, ge=0.0, le=1.0)
    human_error: Optional[float] = Field(None, ge=0.0, le=1.0)
    human_bias: Optional[float] = Field(None, ge=0.0, le=1.0)
    human_malice: Optional[float] = Field(None, ge=0.0, le=1.0)

    def to_overrides(self) -> Optional[Dict[str, float]]:
        return {k: v for k, v in self.model_dump().items() if v is not None} or None


class DiagnoseRequest(BaseModel):
    evidence: EvidenceBody
    priors: Optional[PriorsBody] = None
    confidence_threshold: float = Field(0.6, gt=0.0, le=1.0)


class BatchCase(BaseModel):
    evidence: EvidenceBody
    priors: Optional[PriorsBody] = None
    confidence_threshold: float = Field(0.6, gt=0.0, le=1.0)


class BatchRequest(BaseModel):
    cases: List[BatchCase]


class SensitivityRequest(BaseModel):
    evidence: EvidenceBody
    priors: Optional[PriorsBody] = None
    confidence_threshold: float = Field(0.6, gt=0.0, le=1.0)


class CompareRequest(BaseModel):
    evidence_a: EvidenceBody
    evidence_b: EvidenceBody
    priors: Optional[PriorsBody] = None
    confidence_threshold: float = Field(0.6, gt=0.0, le=1.0)


class DiagnosisResponse(BaseModel):
    posteriors: Dict[str, float]
    diagnosis: str
    diagnosis_label: str
    confidence: float
    needs_investigation: bool
    log_likelihoods: Dict[str, float]


class SensitivityResponse(BaseModel):
    parameter_sensitivity: Dict[str, float]
    flip_analysis: List[Dict[str, object]]
    tipping_points: Dict[str, Optional[float]]
    prior_sensitivity: Dict[str, float]


class CompareResponse(BaseModel):
    case_a: DiagnosisResponse
    case_b: DiagnosisResponse


class PresetSummary(BaseModel):
    name: str
    description: str


class EvidenceParamInfo(BaseModel):
    name: str
    description: str


class HealthResponse(BaseModel):
    status: str = "ok"
    presets_loaded: int
    evidence_params: int


# ── Rate limiter ───────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, limit: int = 100, window: float = 60.0):
        self.limit = limit
        self.window = window
        self._hits: Dict[str, List[float]] = defaultdict(list)

    def check(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        self._hits[key] = [t for t in self._hits[key] if t > cutoff]
        if len(self._hits[key]) >= self.limit:
            return False
        self._hits[key].append(now)
        return True


rate_limiter = _RateLimiter()


# ── App factory ────────────────────────────────────────────────────────

_engine: Optional[BayesianDiagnostic] = None


def _get_engine() -> BayesianDiagnostic:
    global _engine
    if _engine is None:
        _engine = BayesianDiagnostic()
    return _engine


def _build_result_response(result: DiagnosisResult) -> DiagnosisResponse:
    return DiagnosisResponse(
        posteriors=result.posteriors,
        diagnosis=result.diagnosis,
        diagnosis_label=FAILURE_CLASS_LABELS.get(result.diagnosis, result.diagnosis),
        confidence=result.confidence,
        needs_investigation=result.needs_investigation,
        log_likelihoods=result.log_likelihoods,
    )


def _build_sensitivity_response(report: SensitivityReport) -> SensitivityResponse:
    return SensitivityResponse(
        parameter_sensitivity=report.parameter_sensitivity,
        flip_analysis=report.flip_analysis,
        tipping_points=report.tipping_points,
        prior_sensitivity=report.prior_sensitivity,
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _get_engine()
        yield

    app = FastAPI(
        title="AI Control Failure Diagnostic API",
        description=(
            "Bayesian diagnostic framework for AI control failures. "
            "Computes posterior probabilities over five failure classes "
            "given observed evidence."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _require_api_key(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ):
        pass  # Optional — no key required; placeholder for future enforcement

    async def _check_rate_limit(request: Request):
        client_ip = request.client.host if request.client else "unknown"
        if not rate_limiter.check(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Max 100 requests per minute.",
            )

    # ── Routes ─────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            presets_loaded=len(PRESETS),
            evidence_params=len(EVIDENCE_PARAMS),
        )

    @app.post(
        "/diagnose",
        response_model=DiagnosisResponse,
        dependencies=[Depends(_check_rate_limit)],
    )
    async def diagnose(req: DiagnoseRequest):
        evidence = req.evidence.to_evidence()
        overrides = req.priors.to_overrides() if req.priors else None
        engine = _get_engine()
        result = engine.diagnose(
            evidence,
            prior_overrides=overrides,
            confidence_threshold=req.confidence_threshold,
        )
        return _build_result_response(result)

    @app.post(
        "/diagnose/batch",
        response_model=List[DiagnosisResponse],
        dependencies=[Depends(_check_rate_limit)],
    )
    async def diagnose_batch(req: BatchRequest):
        engine = _get_engine()
        results = []
        for case in req.cases:
            evidence = case.evidence.to_evidence()
            overrides = case.priors.to_overrides() if case.priors else None
            result = engine.diagnose(
                evidence,
                prior_overrides=overrides,
                confidence_threshold=case.confidence_threshold,
            )
            results.append(_build_result_response(result))
        return results

    @app.post(
        "/diagnose/preset/{preset_name}",
        response_model=DiagnosisResponse,
        dependencies=[Depends(_check_rate_limit)],
    )
    async def diagnose_preset(preset_name: str):
        if preset_name not in PRESETS:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown preset: {preset_name}. "
                f"Available: {', '.join(PRESETS.keys())}",
            )
        preset = PRESETS[preset_name]
        evidence = Evidence(**preset["evidence"])
        engine = _get_engine()
        result = engine.diagnose(
            evidence,
            prior_overrides=preset.get("priors"),
        )
        return _build_result_response(result)

    @app.get("/presets", response_model=List[PresetSummary])
    async def list_presets():
        return [
            PresetSummary(name=name, description=preset["description"])
            for name, preset in PRESETS.items()
        ]

    @app.post(
        "/sensitivity",
        response_model=SensitivityResponse,
        dependencies=[Depends(_check_rate_limit)],
    )
    async def sensitivity(req: SensitivityRequest):
        evidence = req.evidence.to_evidence()
        overrides = req.priors.to_overrides() if req.priors else None
        engine = _get_engine()
        result = engine.diagnose(
            evidence,
            prior_overrides=overrides,
            confidence_threshold=req.confidence_threshold,
        )
        report = compute_sensitivity(result, engine)
        return _build_sensitivity_response(report)

    @app.post(
        "/compare",
        response_model=CompareResponse,
        dependencies=[Depends(_check_rate_limit)],
    )
    async def compare(req: CompareRequest):
        engine = _get_engine()
        overrides = req.priors.to_overrides() if req.priors else None
        result_a = engine.diagnose(
            req.evidence_a.to_evidence(),
            prior_overrides=overrides,
            confidence_threshold=req.confidence_threshold,
        )
        result_b = engine.diagnose(
            req.evidence_b.to_evidence(),
            prior_overrides=overrides,
            confidence_threshold=req.confidence_threshold,
        )
        return CompareResponse(
            case_a=_build_result_response(result_a),
            case_b=_build_result_response(result_b),
        )

    @app.get(
        "/evidence/params",
        response_model=List[EvidenceParamInfo],
    )
    async def evidence_params():
        return [
            EvidenceParamInfo(name=name, description=EVIDENCE_DESCRIPTIONS[name])
            for name in EVIDENCE_PARAMS
        ]

    return app


app = create_app()
