"""v1.1: 비동기 재학습 상태 조회 (교수님 Wrap-Up 1번)."""
from fastapi import APIRouter

from serving_app import model_loader, state

router = APIRouter(prefix="/retrain")


@router.get("/status")
def status():
    return {"running": state.retrain_running, "last": state.last_retrain,
            "fallback_to_terminal_plan": state.fallback,
            "serving_version": getattr(model_loader._model_cache, "version", None)}
