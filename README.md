# PI-A-KMNet for Linearly Unstable Nonlinear Waves

Official code repository for the paper:

> **A physics-informed hybrid architecture of alternating KAN and MLP modules for solving linearly unstable nonlinear waves of Hadamard ill-posed equations**
>
> Jing-Jing Su, Shun Yi, Lei Liu, Sheng Zhang, Bo Ruan

This repository implements **PI-A-KMNet** (Physics-Informed **A**lternating **K**AN–**M**LP networks), a physics-based deep learning framework for solving **linearly unstable nonlinear waves** of **Hadamard ill-posed equations** — problems that are intractable for traditional grid-based numerical methods.

## Highlights

- Two hybrid architectures built by alternating **Kolmogorov–Arnold Network (KAN)** and **Multi-Layer Perceptron (MLP)** modules:
  - **PI-A-KMNet<sub>KAN</sub>** — KAN-based network
  - **PI-A-KMNet<sub>MLP</sub>** — MLP-based network
- Compared against the baselines **PINN** (pure MLP) and **PI-KAN** (pure KAN)
- Forward problems: evolution of initial **sech wave packets** and **Gaussian wave packets**
- Inverse problems: robust parameter inference of the `bad` JM equation, including data corrupted by **5% noise**
- Each experiment is repeated with **5 random seeds**; all results are fully reproducible via the archived notebooks

## Model Variants

The physics-informed framework embeds the PDE residual together with initial/boundary conditions into the loss function. Four network variants are compared in this repo:

| Folder | Model | Architecture |
|---|---|---|
| `DNN/` | PINN | Pure MLP |
| `KAN/` | PI-KAN | Pure Chebyshev–KAN |
| `DNN-KAN/` | **PI-A-KMNet<sub>MLP</sub>** | Alternating MLP → KAN modules (MLP-dominant) |
| `KAN-DNN/` | **PI-A-KMNet<sub>KAN</sub>** | Alternating KAN → MLP modules (KAN-dominant) |

The KAN module is a **Chebyshev–KAN layer** (Chebyshev polynomial basis, degree 4). Inputs $(x, t)$ are rescaled to $[-1, 1]$ before the first layer, and all models are trained in double precision (`float64`). Optimization follows the standard two-stage PINN scheme: **Adam** (lr = 1e-3) followed by **L-BFGS**.

## Equations

**`bad` Boussinesq equation**

$$\frac{\partial^2 h}{\partial t^2} + \alpha \frac{\partial^2 h}{\partial x^2} + \beta \frac{\partial^2 (h^2)}{\partial x^2} + \gamma \frac{\partial^4 h}{\partial x^4} = 0, \qquad \gamma < 0$$

**`bad` Jaulent–Miodek (JM) equation**

$$\begin{cases} r_t = 4 q_x + 6 r r_x \\ q_t = r_{xxx} + 2 r q_x + 4 q r_x \end{cases}$$

Both equations admit constant plane waves that are **linearly unstable** (the time frequency is purely imaginary for sufficiently large wavenumbers), which causes traditional discretization-based schemes to break down.

For the **inverse problem** of the `bad` JM equation, unknown equation parameters (e.g., $c_1, c_2, d_1, d_2, d_3$) are treated as trainable variables and identified from reference data.

## Repository Structure

```
├── Bad-Boussinesq-equation/
│   ├── gauss wave packet/              # Gaussian wave packets
│   │   ├── gauss-results/              #   single Gaussian packet
│   │   └── 2gauss-results/             #   two-Gaussian packet
│   └── solitary-wave/
│       ├── soliton-residualpoints/     # collocation (residual) point studies
│       └── soliton-results/
│           ├── 5000parameters/         # ~5k trainable parameters
│           ├── 10000parameters/        # ~10k trainable parameters
│           └── 30000parameters/        # ~30k trainable parameters
│               ├── DNN/  KAN/  DNN-KAN/  KAN-DNN/   (seed1–seed5 each)
├── Bad-JM-equation/
│   ├── solitary-wave/
│   │   ├── jmcase1/                    # analytical solutions + results
│   │   └── jmcase2/                    # analytical solutions + results
│   └── inverse problem/
│       ├── input-data/                 # reference data (.mat)
│       ├── inverse-results/            # noise-free parameter inference
│       └── inverse-noise5pct-results/  # parameter inference with 5% noise
```

Every experiment is a self-contained Jupyter notebook (`.ipynb`) containing the model definition, training loop, and result visualization.

## Requirements

- Python ≥ 3.8
- PyTorch (CUDA recommended)
- NumPy, SciPy, SymPy, Matplotlib, Seaborn, Pandas

```bash
pip install torch numpy scipy sympy matplotlib seaborn pandas
```

## Usage

Each notebook is an independent, ready-to-run experiment:

```bash
jupyter notebook
```

Open any notebook and run all cells. Notebook filenames encode the equation, setup, model variant, and random seed, e.g. `bq-sech-kan-dnn-seed3.ipynb` = bad Boussinesq / sech soliton / PI-A-KMNet<sub>KAN</sub> / seed 3.

## Citation

If you use this code in your research, please cite the paper above (citation details will be added upon publication).
