from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    app_name: str = "MedVision CAD"
    app_env: str = "development"
    secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    device: str = "cpu"
    weights_dir: Path = ROOT / "weights"
    audit_log_path: Path = ROOT / "audit" / "access.jsonl"
    store_uploads: bool = False
    upload_dir: Path = ROOT / "uploads"
    max_upload_mb: int = 25
    chest_checkpoint: Path = ROOT / "weights" / "chest_densenet121.pt"
    mri_checkpoint: Path = ROOT / "weights" / "mri_resnet50.pt"
    seg_checkpoint: Path = ROOT / "weights" / "mri_unet.pt"
    report_checkpoint: Path = ROOT / "weights" / "report_decoder.pt"
    biowordvec_path: Path = ROOT / "weights" / "BioWordVec_PubMed_MIMICIII_d200.bin"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
