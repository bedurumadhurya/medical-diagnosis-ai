from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.core.audit import write_audit

app = FastAPI(
    title=settings.app_name,
    description=(
        "AI diagnostic assistant for chest X-rays and brain MRI. "
        "Educational prototype — not for clinical use."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    settings.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    write_audit("startup", {"env": settings.app_env})


@app.get("/")
def root() -> dict:
    return {"service": settings.app_name, "docs": "/docs"}
