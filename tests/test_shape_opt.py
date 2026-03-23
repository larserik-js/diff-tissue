from diff_tissue.core.jax_bootstrap import jnp
from diff_tissue.core import shape_opt


def _test_single_valid_sim_state():
    mock_list_of_arrays = [jnp.ones(5)]

    sim_states = shape_opt._SimStates(
        loss_vals=[0.5],
        valid=[True],
        final_vertices=mock_list_of_arrays,
        goal_areas=mock_list_of_arrays,
        goal_anisotropies=mock_list_of_arrays,
        final_areas=mock_list_of_arrays,
        final_anisotropies=mock_list_of_arrays,
        n_edge_crossings=[0],
    )
    valid_best_idx = shape_opt._get_valid_best_idx(sim_states)
    assert valid_best_idx == 0


def _test_single_invalid_sim_state():
    mock_list_of_arrays = [jnp.ones(5)]

    sim_states = shape_opt._SimStates(
        loss_vals=[0.5],
        valid=[False],
        final_vertices=mock_list_of_arrays,
        goal_areas=mock_list_of_arrays,
        goal_anisotropies=mock_list_of_arrays,
        final_areas=mock_list_of_arrays,
        final_anisotropies=mock_list_of_arrays,
        n_edge_crossings=[0],
    )
    valid_best_idx = shape_opt._get_valid_best_idx(sim_states)
    assert valid_best_idx == 0


def _test_two_invalid_sim_states():
    mock_list_of_arrays = 2 * [jnp.ones(5)]

    sim_states = shape_opt._SimStates(
        loss_vals=[0.5, 0.1],
        valid=[False, False],
        final_vertices=mock_list_of_arrays,
        goal_areas=mock_list_of_arrays,
        goal_anisotropies=mock_list_of_arrays,
        final_areas=mock_list_of_arrays,
        final_anisotropies=mock_list_of_arrays,
        n_edge_crossings=[0, 0],
    )
    valid_best_idx = shape_opt._get_valid_best_idx(sim_states)
    assert valid_best_idx == 1


def _test_mixed_valid_sim_states():
    mock_list_of_arrays = 2 * [jnp.ones(5)]

    sim_states = shape_opt._SimStates(
        loss_vals=[0.5, 0.1],
        valid=[True, False],
        final_vertices=mock_list_of_arrays,
        goal_areas=mock_list_of_arrays,
        goal_anisotropies=mock_list_of_arrays,
        final_areas=mock_list_of_arrays,
        final_anisotropies=mock_list_of_arrays,
        n_edge_crossings=[0, 0],
    )
    valid_best_idx = shape_opt._get_valid_best_idx(sim_states)
    assert valid_best_idx == 0


def test__get_valid_best_index():
    _test_single_valid_sim_state()
    _test_single_invalid_sim_state()
    _test_two_invalid_sim_states()
    _test_mixed_valid_sim_states()
