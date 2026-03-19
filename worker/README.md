# Archimedes Worker

FastAPI health endpoint + Redis queue consumer running the CV pipeline (see repo root `README.md`).

## Dev

```bash
cd worker
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check app tests
```

Run slow OpenCV smoke: `pytest -m slow`.
