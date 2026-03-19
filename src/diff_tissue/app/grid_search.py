import itertools
import sqlite3

import numpy as np

from . import io_utils, parameters
from ..core import init_systems, metrics, shape_opt


def _init_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS results (
            arpw REAL,
            aspw REAL,
            anpw REAL,
            loss REAL,
            PRIMARY KEY (arpw, aspw, anpw)
        )
    """)

    conn.commit()
    return conn


def _get_completed_count(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM results")
    return cur.fetchone()[0]


def _calc_n_total_runs(
    area_pot_loss_vals, anisotropy_pot_loss_vals, angle_pot_loss_vals
):
    n_total_runs = (
        len(area_pot_loss_vals)
        * len(anisotropy_pot_loss_vals)
        * len(angle_pot_loss_vals)
    )
    return n_total_runs


def _round(x, digits=3):
    return round(x, digits)


def _is_done(conn, arpw, aspw, anpw):
    arpw, aspw, anpw = map(_round, (arpw, aspw, anpw))

    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM results WHERE arpw=? AND aspw=? AND anpw=?",
        (arpw, aspw, anpw),
    )
    return cur.fetchone() is not None


def _simulate(areas_pot_w, anisotropies_pot_w, angles_pot_w):
    params = parameters.Params(
        areas_pot_weight=areas_pot_w,
        anisotropies_pot_weight=anisotropies_pot_w,
        angles_pot_weight=angles_pot_w,
    )
    sim_states = shape_opt.run(params)
    best_state = shape_opt.get_best_state(sim_states)

    polygon_inds = init_systems.get_system(params).indices
    n_edge_crossings = metrics.count_edge_crossings(
        best_state.final_vertices, polygon_inds
    )

    if n_edge_crossings > 0:
        loss = np.inf
    else:
        loss = best_state.loss
    return loss


def _save_result(conn, arpw, aspw, anpw, loss):
    arpw, aspw, anpw = map(_round, (arpw, aspw, anpw))

    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO results (arpw, aspw, anpw, loss)
        VALUES (?, ?, ?, ?)
    """,
        (arpw, aspw, anpw, loss),
    )
    conn.commit()


def run(areas_pot_ws, anisotropies_pot_ws, angles_pot_ws):
    output_manager = io_utils.OutputManager(None, base_dir="outputs")
    db_path = output_manager.file_path("grid_search.db")

    conn = _init_db(db_path)

    count = _get_completed_count(conn)
    n_total_runs = _calc_n_total_runs(
        areas_pot_ws, anisotropies_pot_ws, angles_pot_ws
    )

    param_combs = itertools.product(
        areas_pot_ws, anisotropies_pot_ws, angles_pot_ws
    )

    for arpw, aspw, anpw in param_combs:
        count += 1
        print(
            f"[{count}/{n_total_runs}] "
            f"Running (arpw={arpw}, "
            f"aspw={aspw}, "
            f"anpw={anpw})"
        )

        if _is_done(conn, arpw, aspw, anpw):
            print("  -> Skipping (already computed)")
            continue

        loss = _simulate(arpw, aspw, anpw)

        _save_result(conn, arpw, aspw, anpw, loss)

    conn.close()
