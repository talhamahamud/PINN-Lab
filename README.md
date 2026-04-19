# 🧠 Physics-Informed Neural Networks (PINNs) — Learning Repository

> A structured, hands-on journey through PINNs — from ODE/PDE foundations to solving real physics problems with neural networks.

---

## 📌 What are PINNs?

Traditional neural networks learn purely from **data**. **Physics-Informed Neural Networks (PINNs)** go further — they embed the governing **physical laws** (expressed as differential equations) directly into the training process.

Instead of needing thousands of labeled data points, a PINN learns by minimizing a loss function that penalizes violations of the underlying physics:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{physics}}}_{\text{PDE residual}} + \underbrace{\mathcal{L}_{\text{BC}}}_{\text{boundary conditions}} + \underbrace{\mathcal{L}_{\text{IC}}}_{\text{initial conditions}}$$

This makes PINNs especially powerful for:
- Solving forward problems (given a PDE, find the solution field)
- Solving inverse problems (given some observations, identify unknown parameters)
- Problems where data is scarce but physics is well-understood

> 📄 Original paper: [Raissi, Perdikaris & Karniadakis (2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125) — *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*

---

## 🎯 Learning Goals

This repository documents my personal journey learning PINNs from scratch. By the end, I aim to:

- [ ] Understand the mathematical foundations (ODEs, PDEs, boundary conditions)
- [ ] Implement PINNs from scratch using PyTorch
- [ ] Solve classic benchmark problems (SHO, Burgers, heat equation, Navier-Stokes)
- [ ] Tackle inverse problems using PINNs
- [ ] Use the DeepXDE library for more complex problems
- [ ] Understand the limitations and failure modes of PINNs

---

## 📁 Repository Structure

```
PINN-Learning/
│
├── 01_foundations/                  # Math prerequisites
│   ├── 01_ode_basics.ipynb          # ODE review + analytical solutions
│   ├── 02_pde_basics.ipynb          # PDE types, BCs, ICs
│   └── 03_autograd_intro.ipynb      # PyTorch autograd for derivatives
│
├── 02_pinn_theory/                  # How PINNs work
│   ├── 01_what_is_pinn.ipynb        # Architecture + loss function derivation
│   ├── 02_collocation_points.ipynb  # Domain sampling strategies
│   └── 03_loss_weighting.ipynb      # Balancing physics vs BC/IC loss
│
├── 03_pinn_basics/                  # First PINN implementations
│   ├── 01_simple_ode.ipynb          # Harmonic oscillator with PINN
│   ├── 02_heat_equation_1d.ipynb    # 1D diffusion/heat PDE
│   └── 03_poisson_2d.ipynb          # 2D Poisson equation
│
├── 04_intermediate/                 # Classic PINN benchmarks
│   ├── 01_burgers_equation.ipynb    # Nonlinear PDE benchmark
│   ├── 02_wave_equation.ipynb       # Time-dependent wave PDE
│   └── 03_inverse_problem.ipynb     # Inferring unknown PDE parameters
│
├── 05_advanced/                     # Advanced topics (in progress)
│   ├── 01_navier_stokes.ipynb       # Fluid dynamics with PINNs
│   └── 02_deepxde_library.ipynb     # Using DeepXDE for complex problems
│
├── utils/
│   ├── pinn_model.py                # Reusable PINN base class
│   ├── plotting.py                  # Common visualization functions
│   └── sampling.py                  # Collocation point samplers
│
├── assets/images/                   # Diagrams and result plots
├── requirements.txt
└── README.md
```

---

## 📚 Table of Contents

### 🔷 Part 1 — Foundations
| Notebook | Topics Covered | Status |
|---|---|---|
| [01 — ODE Basics](01_foundations/01_ode_basics.ipynb) | ODE types, IVP vs BVP, separable/linear/2nd-order methods, SHO, damping, numerical methods | ✅ Done |
| [02 — PDE Basics](01_foundations/02_pde_basics.ipynb) | PDE classification, heat/wave/Poisson equations, BCs and ICs | 🔄 In progress |
| [03 — Autograd Intro](01_foundations/03_autograd_intro.ipynb) | PyTorch autograd, computing derivatives of neural networks | 🔄 In progress |

### 🔷 Part 2 — PINN Theory
| Notebook | Topics Covered | Status |
|---|---|---|
| [01 — What is a PINN?](02_pinn_theory/01_what_is_pinn.ipynb) | Network architecture, loss function derivation, training loop | ⏳ Planned |
| [02 — Collocation Points](02_pinn_theory/02_collocation_points.ipynb) | Uniform, Latin hypercube, adaptive sampling | ⏳ Planned |
| [03 — Loss Weighting](02_pinn_theory/03_loss_weighting.ipynb) | Manual vs adaptive loss weights, gradient pathologies | ⏳ Planned |

### 🔷 Part 3 — PINN Basics
| Notebook | Topics Covered | Status |
|---|---|---|
| [01 — Simple ODE](03_pinn_basics/01_simple_ode.ipynb) | PINN for harmonic oscillator, residual loss, IC enforcement | ⏳ Planned |
| [02 — Heat Equation 1D](03_pinn_basics/02_heat_equation_1d.ipynb) | 1D diffusion, Dirichlet BCs, space-time collocation | ⏳ Planned |
| [03 — Poisson 2D](03_pinn_basics/03_poisson_2d.ipynb) | 2D spatial PDE, geometry handling, convergence study | ⏳ Planned |

### 🔷 Part 4 — Intermediate
| Notebook | Topics Covered | Status |
|---|---|---|
| [01 — Burgers Equation](04_intermediate/01_burgers_equation.ipynb) | Nonlinear PDE, shock formation, classic PINN benchmark | ⏳ Planned |
| [02 — Wave Equation](04_intermediate/02_wave_equation.ipynb) | Hyperbolic PDE, wave propagation | ⏳ Planned |
| [03 — Inverse Problem](04_intermediate/03_inverse_problem.ipynb) | Identifying unknown PDE parameters from sparse observations | ⏳ Planned |

### 🔷 Part 5 — Advanced
| Notebook | Topics Covered | Status |
|---|---|---|
| [01 — Navier-Stokes](05_advanced/01_navier_stokes.ipynb) | Incompressible fluid flow, pressure-velocity coupling | ⏳ Planned |
| [02 — DeepXDE Library](05_advanced/02_deepxde_library.ipynb) | High-level PINN framework, complex geometries | ⏳ Planned |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Install dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/PINN-Learning.git
cd PINN-Learning

# Create a virtual environment (recommended)
python -m venv pinn-env
source pinn-env/bin/activate        # Linux/macOS
# pinn-env\Scripts\activate         # Windows

# Install requirements
pip install -r requirements.txt
```

### Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

### `requirements.txt`

```
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
jupyter>=1.0.0
deepxde>=1.10.0
```

---

## 🧩 Each Notebook Follows This Structure

To keep learning consistent and reproducible, every notebook is organized as:

```
1. Problem Statement      — what equation are we solving and why it matters
2. Mathematical Background — write out the ODE/PDE, BCs, ICs clearly with LaTeX
3. Analytical Solution    — derive or state the exact solution (used for validation)
4. PINN / Method Setup    — architecture, loss function, collocation strategy
5. Implementation         — clean, well-commented PyTorch code
6. Results & Plots        — loss curves, predicted vs exact solution, error maps
7. Key Takeaways          — what worked, what didn't, and what comes next
```

---

## 🗺️ Learning Roadmap

```
ODEs & PDEs  ──►  Autograd  ──►  First PINN  ──►  Benchmarks  ──►  Inverse Problems
     │                │               │                  │                  │
 Understand       Derivatives     Harmonic           Burgers'          Identify
 the physics      from networks   Oscillator         Equation          parameters
```

---

## 📖 Key References

| Resource | Type | Link |
|---|---|---|
| Raissi et al. (2019) — Original PINN paper | Paper | [Link](https://www.sciencedirect.com/science/article/pii/S0021999118307125) |
| Cuomo et al. (2022) — PINN review | Review Paper | [Link](https://arxiv.org/abs/2201.05624) |
| DeepXDE Documentation | Library Docs | [Link](https://deepxde.readthedocs.io/) |
| PyTorch Autograd | Docs | [Link](https://pytorch.org/docs/stable/autograd.html) |
| Kreyszig — Advanced Engineering Mathematics | Textbook | ODEs & PDEs chapters |
| Strogatz — Nonlinear Dynamics and Chaos | Textbook | Phase portraits, stability |

---

## 📝 Notes & Acknowledgements

This repository is a personal learning project. All implementations are done from scratch (except where DeepXDE is explicitly used) to maximize understanding. Code prioritizes **clarity over performance**.

---

*Started: 2025 | Built with PyTorch, NumPy, Matplotlib*