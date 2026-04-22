import numpy as np

from diff_tissue.core import shape_opt


def _get_mock_sim_states():
    mock_floating_array = np.ones(5)
    mock_int_array = np.ones(5, dtype=int)
    mock_bool_array = np.ones(5, dtype=bool)

    mock_sim_states = shape_opt._SimStates(
        loss_vals=mock_floating_array,
        shape_loss_vals=mock_floating_array,
        var_loss_vals=mock_floating_array,
        poly_id_loss_vals=mock_floating_array,
        valid=mock_bool_array,
        final_vertices=mock_floating_array,
        goal_areas=mock_floating_array,
        goal_anisotropies=mock_floating_array,
        final_areas=mock_floating_array,
        final_anisotropies=mock_floating_array,
        n_edge_crossings=mock_int_array,
    )
    return mock_sim_states


def _test_single_valid_sim_state():
    mock_sim_states = _get_mock_sim_states()
    mock_single_val_array = np.array([1.0])

    mock_sim_states.loss_vals = mock_single_val_array
    mock_sim_states.shape_loss_vals = mock_single_val_array
    mock_sim_states.var_loss_vals = mock_single_val_array
    mock_sim_states.poly_id_loss_vals = mock_single_val_array
    mock_sim_states.valid = np.array([True])
    mock_sim_states.n_edge_crossings = np.array([0])

    valid_best_idx = shape_opt._get_valid_best_idx(mock_sim_states)
    assert valid_best_idx == 0


def _test_single_invalid_sim_state():
    mock_sim_states = _get_mock_sim_states()
    mock_single_val_array = np.array([1.0])

    mock_sim_states.loss_vals = mock_single_val_array
    mock_sim_states.shape_loss_vals = mock_single_val_array
    mock_sim_states.var_loss_vals = mock_single_val_array
    mock_sim_states.poly_id_loss_vals = mock_single_val_array
    mock_sim_states.valid = np.array([False])
    mock_sim_states.n_edge_crossings = np.array([0])

    valid_best_idx = shape_opt._get_valid_best_idx(mock_sim_states)
    assert valid_best_idx == 0


def _test_two_invalid_sim_states():
    mock_sim_states = _get_mock_sim_states()
    mock_two_val_array = np.array([1.0, 1.0])

    mock_sim_states.loss_vals = np.array([0.5, 0.1])
    mock_sim_states.shape_loss_vals = mock_two_val_array
    mock_sim_states.var_loss_vals = mock_two_val_array
    mock_sim_states.poly_id_loss_vals = mock_two_val_array
    mock_sim_states.valid = np.array([False, False])
    mock_sim_states.n_edge_crossings = np.array([0, 0])

    valid_best_idx = shape_opt._get_valid_best_idx(mock_sim_states)
    assert valid_best_idx == 1


def _test_mixed_valid_sim_states():
    mock_sim_states = _get_mock_sim_states()
    mock_two_val_array = np.array([1.0, 1.0])

    mock_sim_states.loss_vals = mock_two_val_array
    mock_sim_states.shape_loss_vals = mock_two_val_array
    mock_sim_states.var_loss_vals = mock_two_val_array
    mock_sim_states.poly_id_loss_vals = mock_two_val_array
    mock_sim_states.valid = np.array([True, False])
    mock_sim_states.n_edge_crossings = np.array([0, 0])

    valid_best_idx = shape_opt._get_valid_best_idx(mock_sim_states)
    assert valid_best_idx == 0


def _test_all_inf_losses_sim_state():
    mock_sim_states = _get_mock_sim_states()
    mock_two_val_array = np.array([1.0, 1.0])

    mock_sim_states.loss_vals = np.array([np.inf, np.inf])
    mock_sim_states.shape_loss_vals = mock_two_val_array
    mock_sim_states.var_loss_vals = mock_two_val_array
    mock_sim_states.poly_id_loss_vals = mock_two_val_array
    mock_sim_states.valid = np.array([False, False])
    mock_sim_states.n_edge_crossings = np.array([5, 5])

    valid_best_idx = shape_opt._get_valid_best_idx(mock_sim_states)
    assert valid_best_idx == 0


def test__get_valid_best_index():
    _test_single_valid_sim_state()
    _test_single_invalid_sim_state()
    _test_two_invalid_sim_states()
    _test_mixed_valid_sim_states()
    _test_all_inf_losses_sim_state()
