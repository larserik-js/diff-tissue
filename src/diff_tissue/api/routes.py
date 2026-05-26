from fastapi import APIRouter

from .schemas import SimulationRequest, SimulationResponse
from ..app.sim_service import run_and_serialize


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/simulate", response_model=SimulationResponse)
def simulate(request: SimulationRequest):
    return run_and_serialize(request)
