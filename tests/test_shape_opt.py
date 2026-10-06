import numpy as np
import pytest

from diff_tissue.core import shape_opt


def _make_sim_states(loss_vals, valid, n_edge_crossings):
    n_states = len(loss_vals)
    ones = np.ones(n_states)
    mock_floating_array = np.ones(5)

    return shape_opt.SimStates(
        loss_vals=np.array(loss_vals),
        shape_loss_vals=ones,
        var_loss_vals=ones,
        poly_id_loss_vals=ones,
        valid=np.array(valid),
        final_vertices=mock_floating_array,
        goal_areas=mock_floating_array,
        goal_anisotropies=mock_floating_array,
        final_areas=mock_floating_array,
        final_anisotropies=mock_floating_array,
        n_edge_crossings=np.array(n_edge_crossings),
    )


@pytest.mark.parametrize(
    ("loss_vals", "valid", "n_edge_crossings", "expected"),
    [
        pytest.param([1.0], [True], [0], 0, id="single_valid"),
        pytest.param([1.0], [False], [0], 0, id="single_invalid"),
        pytest.param([0.5, 0.1], [False, False], [0, 0], 1, id="two_invalid"),
        pytest.param([1.0, 1.0], [True, False], [0, 0], 0, id="mixed_valid"),
        pytest.param(
            [np.inf, np.inf],
            [False, False],
            [5, 5],
            0,
            id="all_inf_losses",
        ),
    ],
)
def test_get_valid_best_idx(loss_vals, valid, n_edge_crossings, expected):
    sim_states = _make_sim_states(loss_vals, valid, n_edge_crossings)
    assert shape_opt._get_valid_best_idx(sim_states) == expected
