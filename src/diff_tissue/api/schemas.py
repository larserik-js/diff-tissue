from typing import List

from pydantic import BaseModel


class SimulationRequest(BaseModel):
    system: str = "few"
    shape: str = "petal"
    knots: bool = False
    quiet: bool = False
    trapezoid_angle: float = 75.0
    n_morph_steps: int = 500
    areas_pot_weight: float = 5.0
    anisotropies_pot_weight: float = 50.0
    angles_pot_weight: float = 13.0
    init_lr: float = 0.01
    n_shape_steps: int = 1000
    shape_loss_weight: float = 1.0
    var_loss_weight: float = 0.0
    poly_id_cfg: int = 0
    seed: int = 0


class SimulationResponse(BaseModel):
    loss_vals: List[float]
    shape_loss_vals: List[float]
    var_loss_vals: List[float]
    poly_id_loss_vals: List[float]
    valid: List[bool]
    final_vertices: List[List[List[float]]]
    goal_areas: List[List[float]]
    goal_anisotropies: List[List[float]]
    final_areas: List[List[float]]
    final_anisotropies: List[List[float]]
    n_edge_crossings: List[int]
