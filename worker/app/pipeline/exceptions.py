class PipelineError(Exception):
    """Controlled failure with API-facing codes (concept §8, §17)."""

    def __init__(
        self,
        code: str,
        message: str,
        retry_step: str | None = None,
        *,
        retry_views: list[str] | None = None,
        error_severity: str = "hard",
        suggested_action: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_step = retry_step
        self.retry_views = retry_views or ([retry_step] if retry_step else [])
        self.error_severity = error_severity  # "hard" | "soft"
        self.suggested_action = suggested_action
