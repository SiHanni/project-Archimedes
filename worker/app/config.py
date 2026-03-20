from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "archimedes"
    mysql_password: str = "archimedes"
    mysql_database: str = "archimedes"

    redis_url: str = "redis://localhost:6379"
    queue_name: str = "archimedes:queue"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "archimedes-uploads"
    s3_region: str = "us-east-1"

    algorithm_version: str = "archimedes-g1-v0.1.0"
    debug_save_masks: bool = False
    worker_output_dir: str = "/output"

    min_short_edge_px: int = Field(default=1600, validation_alias="ARCHIMEDES_MIN_SHORT_EDGE")
    # 라플라시안 분산(대략적 선명도). 높을수록 엄격. v0 스캐폴드는 로컬·데모에서 거절이 잦아 기본 완화.
    # 운영·파일럿 전 `ARCHIMEDES_BLUR_THRESHOLD`로 재튜닝(스펙 §8).
    blur_laplacian_threshold: float = Field(
        default=6.0, validation_alias="ARCHIMEDES_BLUR_THRESHOLD"
    )
    scale_mismatch_ratio: float = Field(default=0.08, validation_alias="ARCHIMEDES_SCALE_MISMATCH")
    voxel_penalty_resolution_ratio: float = Field(
        default=0.12, validation_alias="ARCHIMEDES_VOXEL_PENALTY_RATIO"
    )
    use_voxel_carve: bool = Field(default=True, validation_alias="ARCHIMEDES_USE_VOXEL_CARVE")
    allow_card_fallback: bool = Field(
        default=True, validation_alias="ARCHIMEDES_ALLOW_CARD_FALLBACK"
    )
    # 무게가 이 값(g)을 넘으면 비현실적로 간주하고 UI에서 숫자 숨김(meta.sanity)
    sanity_max_mass_g: float = Field(
        default=2500.0, validation_alias="ARCHIMEDES_SANITY_MAX_MASS_G"
    )
    segmentation_backend: str = Field(
        default="heuristic", validation_alias="ARCHIMEDES_SEGMENTATION_BACKEND"
    )
    # True(기본): 카드 면 위(on_card_inner)로 잡힌 세그는 거절 — §4 동일 평면·옆 배치와 정합, 과대 부피 방지
    reject_jewel_on_card: bool = Field(
        default=True, validation_alias="ARCHIMEDES_REJECT_ON_CARD_INNER"
    )

    otel_exporter_otlp_endpoint: str = ""

    @field_validator("use_voxel_carve", "allow_card_fallback", "reject_jewel_on_card", mode="before")
    @classmethod
    def _coerce_voxel_carve(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("0", "false", "no", "off"):
                return False
            if s in ("1", "true", "yes", "on"):
                return True
        return bool(v)

    @property
    def mysql_dsn_kwargs(self) -> dict:
        return {
            "host": self.mysql_host,
            "port": self.mysql_port,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "database": self.mysql_database,
            "charset": "utf8mb4",
        }


def get_settings() -> Settings:
    return Settings()
