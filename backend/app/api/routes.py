from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.core.audit import file_fingerprint, recent_audit, write_audit
from app.ml.cad import MedXpertCAD
from app.ml.imaging import load_image_bytes
from app.schemas import AnalyzeResponse


router = APIRouter(prefix="/api/v1")


@lru_cache(maxsize=1)
def get_cad() -> MedXpertCAD:
    return MedXpertCAD()


@router.get("/health")
def health() -> dict:
    cad = get_cad()
    return {
        "status": "ok",
        "app": settings.app_name,
        "device": str(cad.device),
        "chest_mode": cad.chest_mode,
        "mri_mode": cad.mri_mode,
        "unet_loaded": cad.unet is not None,
    }


@router.get("/labels")
def labels() -> dict:
    from app.ml.imaging import CHEST_LABELS, MRI_LABELS

    return {"chest_xray": CHEST_LABELS, "brain_mri": MRI_LABELS}


@router.get("/audit")
def audit(limit: int = 50) -> dict:
    return {"events": recent_audit(limit)}


@router.post("/analyze/xray", response_model=AnalyzeResponse)
async def analyze_xray(file: UploadFile = File(...)) -> AnalyzeResponse:
    return await _analyze(file, "chest_xray")


@router.post("/analyze/mri", response_model=AnalyzeResponse)
async def analyze_mri(file: UploadFile = File(...)) -> AnalyzeResponse:
    return await _analyze(file, "brain_mri")


async def _analyze(file: UploadFile, modality: str) -> AnalyzeResponse:
    filename = file.filename or "upload"
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="File exceeds size limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = load_image_bytes(data, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {exc}") from exc

    cad = get_cad()
    result = cad.analyze_chest(image) if modality == "chest_xray" else cad.analyze_mri(image)
    write_audit(
        "analyze",
        {
            "request_id": result.request_id,
            "modality": modality,
            "file": file_fingerprint(data, filename),
            "model_mode": result.model_mode,
            "primary": result.primary_impression,
            "latency_ms": result.latency_ms,
        },
    )
    return result
