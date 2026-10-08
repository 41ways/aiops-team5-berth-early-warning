"""Day1: 헬스체크 엔드포인트."""
from fastapi import APIRouter

from serving_app import model_loader, state

router = APIRouter()


@router.get("/health")
def health():
    model_loaded = model_loader._model_cache is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "loading_mode": _current_loading_mode(),
        "fallback_to_terminal_plan": state.fallback,
        "fallback_target": state.fallback_target if state.fallback else None,
        "retrain_running": state.retrain_running,
    }


def _current_loading_mode() -> str:
    import os

    return os.getenv("LOADING_MODE", "lazy")
