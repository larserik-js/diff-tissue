# diff-tissue

## Differentiable tissue growth

This repository contains code for simulating the **morphogenesis of plant tissues using differentiable programming**. The project explores how tissue development can be modeled with fully differentiable simulations in combination with vertex models.

By keeping the simulation differentiable, model parameters and growth rules can be optimized or potentially learned from data, enabling new workflows in computational biology and developmental modeling.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/larserik-js/diff-tissue.git
cd diff-tissue
```

Install dependencies (example shown using **uv**):

```bash
uv sync
```

---

## Usage

Run a basic tissue growth simulation:

```bash
cd scripts
python run_shape_opt.py
```

Typical workflow:

1. Optimize tissue growth
2. View output figures under ```outputs```


---

## Project Structure

```
diff_tissue/
├── scripts/
├── src/
│   ├── app/
│   └── core/
├── pyproject.toml
└── README.md
```

---

## License

This project is licensed under the MIT License.
