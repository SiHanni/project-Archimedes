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

    # 폰 사진은 늘 이보다 크다. 이 게이트의 목적은 썸네일·스크린샷 같은
    # **분석 불가능한 저해상도**를 거르는 것이지 화질 심사가 아니다.
    min_short_edge_px: int = Field(default=1000, validation_alias="ARCHIMEDES_MIN_SHORT_EDGE")
    # 라플라시안 분산(대략적 선명도). 높을수록 엄격. v0 스캐폴드는 로컬·데모에서 거절이 잦아 기본 완화.
    # 운영·파일럿 전 `ARCHIMEDES_BLUR_THRESHOLD`로 재튜닝(스펙 §8).
    blur_laplacian_threshold: float = Field(
        default=6.0, validation_alias="ARCHIMEDES_BLUR_THRESHOLD"
    )
    # 뷰별 σ 는 **각 컷의 카드 크기로 따로 계산**되므로, 가까이 찍은 컷의 σ 가
    # 큰 것은 오류가 아니라 정답이다. 거리 차이는 이미 보정된다.
    # 따라서 이 게이트는 "촬영 거리를 맞춰라"가 아니라 **카드 오검출 탐지**용이다.
    # (예: 카드가 아닌 사각형을 잡아 σ 가 2~3배 튀는 경우)
    #
    # ⚠️ 이전 기본값 0.08 은 σ 가 상수 0.1 로 고정돼 있던 시절의 값이라
    # **한 번도 발동한 적이 없었다**. σ 를 고치자 그 게이트가 깨어나 손각대
    # 5뷰 촬영을 전부 거절했다. 검증된 적 없는 임계값이었다.
    scale_mismatch_ratio: float = Field(default=0.5, validation_alias="ARCHIMEDES_SCALE_MISMATCH")
    voxel_penalty_resolution_ratio: float = Field(
        default=0.12, validation_alias="ARCHIMEDES_VOXEL_PENALTY_RATIO"
    )
    use_voxel_carve: bool = Field(default=True, validation_alias="ARCHIMEDES_USE_VOXEL_CARVE")
    # v2: 카드는 선택적 "메트릭 앵커". 미검출 시 프레임 중앙에 가짜 카드를 가정하면
    # 스케일이 날조되므로 기본 OFF — 단일사진 경로는 깊이추정 단독(+tier 캡)으로 폴백한다.
    allow_card_fallback: bool = Field(
        default=False, validation_alias="ARCHIMEDES_ALLOW_CARD_FALLBACK"
    )
    # 무게가 이 값(g)을 넘으면 비현실적로 간주하고 UI에서 숫자 숨김(meta.sanity)
    sanity_max_mass_g: float = Field(
        default=2500.0, validation_alias="ARCHIMEDES_SANITY_MAX_MASS_G"
    )
    # ── v2 백엔드 라우팅 (archimedes-v2-single-photo.mdc §2) ──
    segmentation_backend: str = Field(
        default="heuristic", validation_alias="ARCHIMEDES_SEGMENTATION_BACKEND"
    )
    detector_backend: str = Field(default="stub", validation_alias="ARCHIMEDES_DETECTOR_BACKEND")
    depth_backend: str = Field(default="stub", validation_alias="ARCHIMEDES_DEPTH_BACKEND")
    # 모델 가중치는 이미지에 굽지 않고 볼륨으로 주입한다
    onnx_model_dir: str = Field(default="/models", validation_alias="ARCHIMEDES_ONNX_MODEL_DIR")
    # 파일명 — HuggingFace 배포본은 외부 가중치(.onnx_data)를 참조하므로
    # **원본 파일명을 유지**해야 한다(이름을 바꾸면 데이터 파일을 못 찾는다).
    detector_model_file: str = Field(
        default="detector.onnx", validation_alias="ARCHIMEDES_DETECTOR_MODEL_FILE"
    )
    segmenter_model_file: str = Field(
        default="segmenter.onnx", validation_alias="ARCHIMEDES_SEGMENTER_MODEL_FILE"
    )
    depth_model_file: str = Field(
        default="depth.onnx", validation_alias="ARCHIMEDES_DEPTH_MODEL_FILE"
    )
    depth_input_size: int = Field(default=518, validation_alias="ARCHIMEDES_DEPTH_INPUT_SIZE")
    # 깊이 모델 출력 성격 — 틀리면 스케일 복원이 통째로 어긋난다
    depth_output_kind: str = Field(
        default="affine_invariant", validation_alias="ARCHIMEDES_DEPTH_OUTPUT_KIND"
    )
    depth_output_inverse: bool = Field(
        default=False, validation_alias="ARCHIMEDES_DEPTH_OUTPUT_INVERSE"
    )
    depth_output_scale_to_mm: float = Field(
        default=1.0, validation_alias="ARCHIMEDES_DEPTH_OUTPUT_SCALE_MM"
    )
    # 1이면 앵커(카드) 없는 job 을 ERR_SCALE_UNRESOLVED 로 거절
    require_anchor: bool = Field(default=False, validation_alias="ARCHIMEDES_REQUIRE_ANCHOR")
    # 촬영 규약: 귀금속을 카드의 어느 쪽에 두는가.
    # 배치를 고정하면 탐색 영역이 절반으로 줄어 오검출이 크게 준다(§4).
    # "any" 는 규약을 강제하지 않는 완화 모드.
    object_side: str = Field(default="left", validation_alias="ARCHIMEDES_OBJECT_SIDE")
    # 세그 산출물(오버레이·마스크·누끼)을 S3 에 저장할지 — 평가·오토라벨링용
    save_segmentation_assets: bool = Field(
        default=True, validation_alias="ARCHIMEDES_SAVE_SEG_ASSETS"
    )
    # 홀드아웃 depth RMSE / 카드 거리 가 이 비율을 넘으면 신뢰도 감점.
    # 길이 오차는 부피에서 3배가 되므로 1% 면 부피 3% — 이 정도를 경계로 둔다.
    depth_rmse_penalty_ratio: float = Field(
        default=0.01, validation_alias="ARCHIMEDES_DEPTH_RMSE_PENALTY_RATIO"
    )
    # True(기본): 카드 면 위(on_card_inner)로 잡힌 세그는 거절 — §4 동일 평면·옆 배치와 정합, 과대 부피 방지
    reject_jewel_on_card: bool = Field(
        default=True, validation_alias="ARCHIMEDES_REJECT_ON_CARD_INNER"
    )

    otel_exporter_otlp_endpoint: str = ""

    @field_validator(
        "use_voxel_carve",
        "allow_card_fallback",
        "reject_jewel_on_card",
        "depth_output_inverse",
        "require_anchor",
        "save_segmentation_assets",
        mode="before",
    )
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
