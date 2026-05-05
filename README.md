# diff-tissue

## Differentiable Tissue Morphogenesis

This repository provides code for simulating **plant tissue morphogenesis using differentiable programming**. It combines vertex-based tissue models with fully differentiable simulations.

Keeping the simulation differentiable allows model parameters and growth rules to be optimized or potentially learned from data, enabling new workflows in computational biology and developmental modeling.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/larserik-js/diff-tissue.git
cd diff-tissue
```

Create a virtual environment and install dependencies defined in `pyproject.toml` using your preferred tool. The examples below assume that `python` refers to the interpreter in this environment.

---

## Usage

Run the main optimization pipeline:

```bash
python scripts/run_shape_opt.py
```

This script performs shape-based optimization of tissue growth parameters. It produces:

* Final optimized tissues in `outputs/final_tissues/`
* Visualization of the morphogenesis process in `outputs/best_morph/`

You can also run the script with custom parameters:

```bash
python scripts/run_shape_opt.py --shape trapezoid --id 1 --seed 10
```

Parameters:

* `--shape`: target geometry
* `--id`: cell configuration
* `--seed`: random seed controlling mesh initialization

For a full list of parameters, see `src/diff_tissue/app/parameters.py`.

---

## Project Structure

```
diff_tissue/
├── scripts/          # Entry points and experiments
├── src/
│   ├── app/          # Application-level logic and configuration
│   └── core/         # Core simulation and modeling code
├── pyproject.toml    # Dependencies and project metadata
└── README.md
```

---

## License

This project is licensed under the MIT License.
