class PipelineError(Exception):
    """Controlled failure with API-facing codes (concept §8, §17)."""

    def __init__(self, code: str, message: str, retry_step: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retry_step = retry_step
