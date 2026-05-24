from fastapi import APIRouter

from ..app.sim_service import run_and_serialize


router = APIRouter()


@router.post("/simulate")
def simulate():
    return run_and_serialize()
