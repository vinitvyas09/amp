#!/usr/bin/env python3
"""Build the comprehensive AMP / Autocast notebook (amp.ipynb).

Run:  python _build_notebook.py
"""

import json, textwrap, pathlib

# ── helpers ──────────────────────────────────────────────────────────────────
cells = []

def md(src: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).strip().splitlines(keepends=True),
    })

def code(src: str):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": textwrap.dedent(src).strip().splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — INTRODUCTION & SETUP
# ═══════════════════════════════════════════════════════════════════════════════

md(r"""
# Autocast / AMP in PyTorch: A Deep Practical Reference

Every neural network is fundamentally a giant mathematical expression, and training is the process of tuning the constants (weights) so that expression approximates reality. We do this by computing a loss, backpropagating to get gradients, and nudging the weights a tiny bit in the right direction. Repeat billions of times.

But here's the thing: those weights and gradients are stored as **floating-point numbers**. And floating-point numbers are *not* real numbers. They are a finite approximation with a specific *resolution*. That resolution depends on the format — FP32, FP16, BF16, TF32, FP8 — and each format makes a different tradeoff between **range** (how large and small the numbers can get) and **precision** (how fine the steps between consecutive representable values are).

This matters because during training, we routinely deal with numbers spanning many orders of magnitude: weights around $10^0$, gradients as small as $10^{-8}$, attention logits that can spike to $10^3$, and accumulations over thousands of values. A format that can't represent tiny gradients kills learning (they become zero). A format that can't represent large attention logits kills stability (they become infinity). A format that rounds too aggressively corrupts the running statistics that normalization layers depend on.

The fundamental question this notebook answers is: **when can we get away with lower resolution, and when does it break training?**

Autocast (AMP) is PyTorch's answer: a per-operation precision policy that routes each computation to the right format. Some ops get the speed of 16-bit. Some ops get the safety of 32-bit. The combined effect is faster training with (nearly) no loss in quality. But to understand *when* and *why* AMP works — and to debug it when it doesn't — you need to understand what floating-point formats can and cannot represent.

That's what this notebook is for.

---

This notebook is organized into **three sections**:

| # | Section | What you get |
|---|---|---|
| **1** | **Theory** | Floating-point range vs precision, FP16/BF16/FP32/TF32 comparison tables, underflow/overflow, what AMP is really doing |
| **2** | **What the literature says** | Paper-driven mental models (Micikevicius et al., Kalamkar et al., ZeRO, NVIDIA guidance), written explanations — *no experiments* |
| **3** | **Practicalities** | Hands-on experiments + graphs: progressive mixed-precision implementation from scratch, operator policy probes, dtype flow through a transformer, loss/gradient/scale curves under different precision regimes |

---

**After completing this notebook you should be able to answer (and debug) questions like:**

- Why does FP16 often need **loss scaling**, but BF16 often does not?
- What does `autocast` *actually* do per operation (matmul vs softmax vs layernorm)?
- Why do people talk about **FP32 master weights** and **optimizer state precision**?
- Why does adding `1e-4` to `1.0` in FP16 produce exactly `1.0`? (with the exact bit-level explanation)
- How can you *see* autocast happening inside a transformer forward pass?
- What fails if you try to "just train in half precision everywhere"?
- Why does naive BF16 often work where naive FP16 fails?
- What is the "sum vs mean" mystery under autocast?
- How does the gradient distribution relate to FP16's representable range?
- How many bytes per parameter does mixed-precision Adam training actually use?

---

## How to use this notebook

- Read the markdown, then run the code cells.
- Most experiments are designed to run in a few minutes on a single GPU.
- CPU-only runs are supported for the *conceptual* demos, but some mixed-precision behaviors (and speedups) are fundamentally GPU-driven.

---

## Table of contents

**Section 1 — Theory**
- Quick reference cheat sheet (the table you should memorize)
- Floating-point: range vs precision, IEEE 754 anatomy
- Number line visualization: where representable floats live
- FP16 vs BF16 vs FP32 vs TF32 (tables you can trust)
- The bit-level addition trap (why `1 + 1e-4 = 1` in FP16) — with step-by-step binary alignment
- Underflow, overflow, accumulation error
- What AMP is (autocast + grad scaling)
- Master weights, optimizer state, and accumulation

**Section 2 — What the literature says**
- Micikevicius et al. — *Mixed Precision Training*
- Kalamkar et al. — *A Study of BFLOAT16 for Deep Learning Training*
- NVIDIA mixed precision guidance
- BF16 design intent
- PyTorch AMP operator policy
- Rajbhandari et al. — ZeRO and optimizer state precision
- LLM training stacks (FSDP/ZeRO) and where AMP fits
- FP8 and 8-bit optimizers

**Section 3 — Practicalities**
- Progressive mixed-precision implementation from scratch (FP32 → naive FP16 → **naive BF16** → master weights → loss scaling → PyTorch AMP)
- Build an operator policy table *from your local PyTorch*
- The "sum vs mean" mystery
- Visualize dtype flow through a transformer (4 configurations)
- Gradient underflow + the effect of loss scaling
- **Micikevicius-style gradient histogram analysis**
- Weight update stagnation
- Train a tiny causal LM under different precision regimes (FP32, FP16 naive, BF16 naive, AMP FP16, AMP BF16)
- Plot and interpret loss/time/scale/gradient curves + summary bar charts

---

## Quick glossary

| Term | Meaning |
|---|---|
| **AMP** | Automatic Mixed Precision (in PyTorch: `torch.amp`) |
| **autocast** | Context manager that applies a per-operation dtype policy |
| **GradScaler / loss scaling** | Rescales loss to avoid FP16 gradient underflow |
| **master weights** | Keep weights in FP32 for updates, cast for compute |
| **underflow** | Magnitude too small → becomes 0 (or subnormal/denormal) |
| **overflow** | Magnitude too large → becomes `inf` |
| **TF32** | TensorFloat-32 — an NVIDIA format with FP32 range but 10-bit mantissa, used transparently in Ampere+ matmuls |
| **ULP** | Unit in the Last Place — the spacing between adjacent representable floats |
| **epsilon** | Smallest number such that `1.0 + eps > 1.0` in a given format |
""")

md(r"""
## Prerequisites

You need:
- Python 3.10+
- PyTorch 2.x
- `matplotlib`, `numpy`, `pandas`

### Install (CPU-only quick start)
```bash
pip install torch numpy pandas matplotlib
```

### Install (CUDA)
Install the correct PyTorch + CUDA build from the [official PyTorch instructions](https://pytorch.org/get-started/locally/).

---

This notebook is written to *degrade gracefully*:
- If BF16 is not supported on your GPU, BF16 experiments will be skipped.
- If FP16 training without scaling explodes (often does), we record that as a result rather than pretending it "worked".

> **Note:** on Apple Silicon, this notebook defaults to **CPU** for maximum compatibility. If you want to try the Apple GPU backend, set `USE_MPS_IF_AVAILABLE = True` in the first code cell. AMP/autocast behavior is best-defined on CUDA; CPU/MPS support exists but has different operator coverage and performance characteristics.
""")

code(r"""
# Core imports + environment report
import os, math, time, random, struct, platform
from dataclasses import dataclass
from contextlib import nullcontext

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Missing dependency: numpy. Install with: pip install numpy") from e

try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Missing dependency: pandas. Install with: pip install pandas") from e

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Missing dependency: matplotlib. Install with: pip install matplotlib") from e

from IPython.display import display

plt.rcParams.update({
    "figure.figsize": (10, 4),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
})

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "PyTorch is required. CPU-only: pip install torch"
    ) from e

# Prefer torch.amp (newer API)
if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
    autocast = torch.amp.autocast
    GradScaler = torch.amp.GradScaler
else:
    autocast = torch.cuda.amp.autocast
    GradScaler = torch.cuda.amp.GradScaler

def amp_autocast(dev: torch.device, dtype: torch.dtype | None, enabled: bool = True, cache_enabled: bool = True):
    # Best-effort autocast context manager that degrades gracefully.
    if (not enabled) or (dtype is None):
        return nullcontext()
    try:
        return autocast(device_type=dev.type, dtype=dtype, enabled=True, cache_enabled=cache_enabled)
    except TypeError:
        # Older signatures may not support cache_enabled.
        try:
            return autocast(device_type=dev.type, dtype=dtype, enabled=True)
        except Exception as e:
            print(f"[warn] autocast unavailable for device_type={dev.type}: {e}. Running without autocast.")
            return nullcontext()
    except Exception as e:
        print(f"[warn] autocast unavailable for device_type={dev.type}: {e}. Running without autocast.")
        return nullcontext()

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# ---- User knobs -------------------------------------------------------------
# Leave as-is for "works everywhere" defaults; tweak if you have specific HW.
PREFERRED_DEVICE = None   # one of: "cuda", "mps", "cpu" (or None for auto)
USE_MPS_IF_AVAILABLE = False  # Apple Silicon: set True to try MPS when no CUDA

def choose_device():
    if PREFERRED_DEVICE is not None:
        pref = str(PREFERRED_DEVICE).lower()
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cpu":
            return torch.device("cpu")
        print(f"[warn] Requested device '{PREFERRED_DEVICE}' not available; falling back to auto.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if USE_MPS_IF_AVAILABLE and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = choose_device()

def supports_dtype_on_device(dtype: torch.dtype, dev: torch.device) -> bool:
    try:
        torch.tensor([0.0], device=dev, dtype=dtype)
        return True
    except Exception:
        return False

print(f"PyTorch {torch.__version__}")
print(f"Python  {platform.python_version()}")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"CUDA:   {torch.version.cuda}")
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"BF16:   {'supported' if torch.cuda.is_bf16_supported() else 'NOT supported'}")
elif device.type == "mps":
    print("MPS:    available (Apple Silicon)")
    print(f"FP16:   {'supported' if supports_dtype_on_device(torch.float16, device) else 'NOT supported'}")
    print(f"BF16:   {'supported' if supports_dtype_on_device(torch.bfloat16, device) else 'NOT supported'}")
""")

# ── Quick reference cheat sheet ──────────────────────────────────────────────

md(r"""
## Quick Reference: Floating-Point Formats for Deep Learning

This is the table you should memorize. Every design decision in AMP traces back to these numbers.
""")

code(r"""
# Generate the cheat sheet from your local PyTorch (so the numbers are trustworthy)

def _ulp_at_one(dtype):
    try:
        one = torch.tensor(1.0, dtype=dtype)
        return float(torch.nextafter(one, one + one) - one)
    except Exception:
        return None

def _smallest_subnormal(dtype):
    try:
        z = torch.tensor(0.0, dtype=dtype)
        o = torch.tensor(1.0, dtype=dtype)
        return float(torch.nextafter(z, o))
    except Exception:
        return None

rows = []
formats = [
    # FP8 is increasingly common for LLM training/inference, but support depends on HW + kernel stack.
    ("FP8 (E4M3)", getattr(torch, "float8_e4m3fn", None), 4, 3),
    ("FP8 (E5M2)", getattr(torch, "float8_e5m2", None), 5, 2),
    ("FP16 (IEEE half)", torch.float16, 5, 10),
    ("BF16 (brain float)", torch.bfloat16, 8, 7),
    ("FP32 (single)", torch.float32, 8, 23),
    ("FP64 (double)", torch.float64, 11, 52),
]

for name, dt, exp_b, mant_b in formats:
    if dt is None:
        continue
    try:
        fi = torch.finfo(dt)
    except Exception:
        continue

    ulp1 = _ulp_at_one(dt)
    sub = _smallest_subnormal(dt)

    rows.append({
        "Format": name,
        "Total bits": fi.bits,
        "Exponent bits": exp_b,
        "Mantissa bits": mant_b,
        "Precision bits (incl hidden 1)": mant_b + 1,
        "Approx decimal digits": round((mant_b + 1) * math.log10(2), 1),
        "Exponent range": f"[{-2**(exp_b-1)+2}, {2**(exp_b-1)-1}]",
        "epsilon (ULP at 1.0)": f"{fi.eps:.2e}",
        "ULP at 1.0": f"{ulp1:.2e}" if ulp1 is not None else "n/a",
        "Min normal": f"{fi.tiny:.2e}",
        "Min subnormal": f"{sub:.2e}" if sub is not None else "n/a",
        "Max finite": f"{fi.max:.2e}",
    })

# Add TF32 manually (not a storable dtype in PyTorch, but important to know)
tf32_row = {
    "Format": "TF32 (tensor float)",
    "Total bits": "19*",
    "Exponent bits": 8,
    "Mantissa bits": 10,
    "Precision bits (incl hidden 1)": 11,
    "Approx decimal digits": 3.3,
    "Exponent range": "[-126, 127]",
    "epsilon (ULP at 1.0)": "9.77e-04",
    "ULP at 1.0": "9.77e-04",
    "Min normal": "1.18e-38",
    "Min subnormal": "n/a (internal)",
    "Max finite": "3.40e+38",
}

# Insert TF32 right after FP32 if present; otherwise append.
insert_at = None
for i, r in enumerate(rows):
    if r["Format"].startswith("FP32"):
        insert_at = i + 1
        break
if insert_at is None:
    rows.append(tf32_row)
else:
    rows.insert(insert_at, tf32_row)

df_cheat = pd.DataFrame(rows).set_index("Format")
display(df_cheat)

print()
print("* TF32 is not a storage format. It is used internally by Tensor Cores on")
print("  Ampere+ GPUs for FP32 matmuls: FP32 range, but only 10 mantissa bits.")
print("  Your 'FP32 baseline' on Ampere+ may secretly be TF32 precision.")
""")

md(r"""
### How to read this table

Think of a floating-point format as a **measuring tape**:
- **Exponent bits** determine the **length** of the tape (range: how big/small magnitudes you can reach).
- **Mantissa bits** determine how **fine the tick marks** are (precision: how many significant digits you keep).

| Format | Range | Precision | Training implication |
|---|---|---|---|
| **FP8** | Very narrow | Very low | Needs careful scaling (per-tensor/per-channel), mostly used with specialized kernels/hardware |
| **FP16** | Narrow (5-bit exp) | Moderate (10-bit mantissa) | Underflow risk for gradients, overflow risk for activations → needs **loss scaling** |
| **BF16** | Wide (8-bit exp, same as FP32) | Low (7-bit mantissa) | Rarely underflows → usually **no loss scaling** needed, but coarse rounding in reductions |
| **FP32** | Wide | High | Stable baseline; slower and more memory |
| **TF32** | Wide | Moderate (10-bit mantissa) | Compute-only on Ampere+ GPUs: FP32 matmuls may silently use TF32 for speed |
| **FP64** | Very wide | Very high | Mostly for numeric reference/debugging (too slow for large training) |

**Two immediate consequences:**
1. FP16 has more mantissa bits than BF16 → **better precision per value**.
2. BF16 has the same exponent width as FP32 → **dramatically better range** than FP16.

So: FP16 fails first due to **range** (underflow/overflow). BF16 fails first due to **precision** (rounding/accumulation error). FP8 fails due to both unless extra care is taken.

Autocast exists to route computations so that you get the performance of 16-bit compute without the worst numeric failure modes.
""")

code(r"""
# Visual bit layout: sign / exponent / mantissa for each format
# This is the single most referenced diagram in floating-point explanations.

formats_vis = [
    ("FP64 (double)",     1, 11, 52, 64),
    ("FP32 (single)",     1,  8, 23, 32),
    ("TF32 (tensor)*",    1,  8, 10, 19),
    ("BF16 (brain float)",1,  8,  7, 16),
    ("FP16 (IEEE half)",  1,  5, 10, 16),
    ("FP8 (E5M2)",        1,  5,  2,  8),
    ("FP8 (E4M3)",        1,  4,  3,  8),
]

fig, ax = plt.subplots(figsize=(14, 5))

y_positions = list(range(len(formats_vis)))[::-1]
bar_height = 0.6
max_bits = max(f[4] for f in formats_vis)

for i, (name, s_bits, e_bits, m_bits, total) in enumerate(formats_vis):
    y = y_positions[i]
    # Scale bar width proportional to bit count (relative to max)
    scale = 0.85  # fraction of plot width for the widest format
    bit_width = scale / max_bits

    # Draw sign bits
    ax.barh(y, s_bits * bit_width, height=bar_height, left=0,
            color="#e74c3c", edgecolor="white", linewidth=1.5, zorder=3)
    # Draw exponent bits
    ax.barh(y, e_bits * bit_width, height=bar_height, left=s_bits * bit_width,
            color="#3498db", edgecolor="white", linewidth=1.5, zorder=3)
    # Draw mantissa bits
    ax.barh(y, m_bits * bit_width, height=bar_height, left=(s_bits + e_bits) * bit_width,
            color="#2ecc71", edgecolor="white", linewidth=1.5, zorder=3)

    # Labels inside bars
    mid_s = s_bits * bit_width / 2
    mid_e = (s_bits + e_bits / 2) * bit_width
    mid_m = (s_bits + e_bits + m_bits / 2) * bit_width

    fontsize = 8 if total >= 16 else 7
    if s_bits >= 1:
        ax.text(mid_s, y, f"S\n{s_bits}", ha="center", va="center", fontsize=6, fontweight="bold", color="white")
    ax.text(mid_e, y, f"Exp\n{e_bits}", ha="center", va="center", fontsize=fontsize, fontweight="bold", color="white")
    if m_bits >= 2:
        ax.text(mid_m, y, f"Mantissa\n{m_bits}", ha="center", va="center", fontsize=fontsize, fontweight="bold", color="white")
    else:
        ax.text(mid_m, y, f"M{m_bits}", ha="center", va="center", fontsize=6, fontweight="bold", color="white")

    # Format name and total bits on the left
    ax.text(-0.02, y, f"{name}  [{total} bits]", ha="right", va="center", fontsize=9, fontweight="bold")

ax.set_xlim(-0.55, max_bits * scale / max_bits + 0.05)
ax.set_ylim(-0.5, len(formats_vis) - 0.5)
ax.set_yticks([])
ax.set_xticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_title("Floating-Point Bit Layouts: where the bits go", fontsize=13, fontweight="bold", pad=15)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", edgecolor="white", label="Sign (1 bit)"),
    Patch(facecolor="#3498db", edgecolor="white", label="Exponent (range)"),
    Patch(facecolor="#2ecc71", edgecolor="white", label="Mantissa (precision)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

plt.tight_layout()

print("Key observations:")
print("  - BF16 has the SAME exponent width as FP32 (8 bits) → same range → no underflow issues")
print("  - FP16 has a NARROWER exponent (5 bits) but MORE mantissa than BF16 → better precision, worse range")
print("  - TF32 combines FP32's exponent with FP16's mantissa width (10 bits) → internal to Tensor Cores")
print("  - FP8 formats have tiny mantissa → need per-tensor scaling to be usable")
print()
print("* TF32 is 19 bits internally but is NOT a storage format. Tensor Cores use it transparently for FP32 matmuls.")
""")


md(r"""
## Mixed Precision / Autocast Regimes (PyTorch) — Cheat Sheet

This is the "optimizer table" equivalent for AMP: a small set of regimes that cover most real training setups.

**Rule of thumb (training):**
- If you have CUDA + BF16 support → use **BF16 autocast** (usually no GradScaler).
- Else if you have CUDA → use **FP16 autocast + GradScaler**.
- Avoid `model.half()` / "everything FP16" for training unless you're deliberately doing it (it removes autocast's safety policy and makes optimizer math low precision).

| Regime (common name) | What you set | What runs in 16-bit | What stays FP32 (by policy) | Loss scaling | Typical outcome |
|---|---|---|---|---|---|
| **FP32 baseline** | model params FP32, autocast OFF | nothing | everything | no | stable, slower |
| **FP32 + TF32 matmuls** | Ampere+ default unless disabled | matmuls use **TF32** internally | everything else FP32 | no | stable + faster, but matmul precision is ~FP16 mantissa |
| **AMP BF16 (recommended if supported)** | model params FP32, `autocast(dtype=bf16)` | matmuls / linears / convs | softmax / layernorm / losses / big reductions | no | stable + fast (range like FP32) |
| **AMP FP16 + GradScaler** | model params FP32, `autocast(dtype=fp16)` + `GradScaler` | matmuls / linears / convs | softmax / layernorm / losses / big reductions | **yes** | stable + fast (but FP16 gradients need scaling) |
| **Naive BF16** | model params BF16, autocast OFF | everything | almost nothing | no | often "works", but sensitive ops can drift (reductions/normalization) |
| **Naive FP16** | model params FP16, autocast OFF | everything | almost nothing | maybe (manual) | frequently unstable (underflow/overflow + update stagnation) |

**Key idea:** autocast is useful even if you *could* run everything in BF16/FP16 — it keeps the numerically sensitive operations in FP32.
""")

code(r"""
# Suggest an AMP recipe for THIS machine

def recommend_amp_recipe(dev: torch.device):
    rec = {"device": dev.type}
    if dev.type == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        rec["bf16_supported"] = bool(bf16_ok)
        if bf16_ok:
            rec["recommended_autocast_dtype"] = "bfloat16"
            rec["use_grad_scaler"] = False
            rec["why"] = "BF16 has FP32-like exponent range → underflow is rare."
        else:
            rec["recommended_autocast_dtype"] = "float16"
            rec["use_grad_scaler"] = True
            rec["why"] = "FP16 has narrow exponent range → GradScaler rescues gradients from underflow."
        rec["note"] = "Keep model parameters in FP32; let autocast choose per-op dtypes."
    elif dev.type == "cpu":
        rec["recommended_autocast_dtype"] = "bfloat16"
        rec["use_grad_scaler"] = False
        rec["why"] = "CPU autocast supports BF16; speedups vary by CPU/kernel support."
        rec["note"] = "CPU demos here are mostly about numerics, not performance."
    elif dev.type == "mps":
        rec["recommended_autocast_dtype"] = "float16"
        rec["use_grad_scaler"] = False
        rec["why"] = "MPS typically uses FP16 for reduced precision."
        rec["note"] = "Operator coverage differs from CUDA; verify dtype behavior with the probes in Section 3."
    else:
        rec["recommended_autocast_dtype"] = "float32"
        rec["use_grad_scaler"] = False
        rec["why"] = "Unknown device type."
        rec["note"] = ""
    return rec

display(pd.DataFrame([recommend_amp_recipe(device)]))
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — THEORY
# ═══════════════════════════════════════════════════════════════════════════════

md(r"""
# Section 1 — Theory

The core trick of autocast is simple to state:

> **Run the *right* operations in lower precision for speed/memory, while keeping *numerically sensitive* operations in FP32.**

But to understand *why* this works (and when it doesn't), we need to understand what floating-point formats can and cannot represent.
""")

md(r"""
## 1.0 Where floating-point lives during training (and why autocast exists)

A single training step can be decomposed into:

1. **Forward**: parameters + activations → logits
2. **Loss**: logits + targets → scalar loss
3. **Backward**: loss → gradients for parameters
4. **Optimizer update**: parameters + gradients (+ optimizer state) → new parameters

Different tensors have different numeric requirements:

| Tensor | Typical AMP dtype | Why |
|---|---|---|
| Activations / matmul results | FP16/BF16 (where safe) | Saves memory + uses Tensor Cores |
| Softmax / LayerNorm stats / reductions | FP32 | Protects against overflow + rounding accumulation |
| Gradients | Often FP32 *storage* (even if compute is mixed) | Stable updates + compatibility with optimizers |
| Parameters ("master weights") | FP32 | Prevents update stagnation |
| Optimizer state (Adam moments) | FP32 | Long-horizon accumulation is precision-sensitive |

**Autocast's job** is mostly about **(1) and (2)**: choose per-op dtypes during the forward pass.

**GradScaler's job** is mostly about **(3)** when FP16 is involved: keep gradients from underflowing to zero.
""")


md(r"""
## 1.1 Floating-point is "range + precision", not just "more bits = better"

A binary floating-point number is roughly:

$$(-1)^{\text{sign}} \times (1.\text{mantissa}) \times 2^{\text{exponent}}$$

The bit budget is split across:

- **Exponent bits** → *range* (how large/small magnitudes you can represent)
- **Mantissa (fraction) bits** → *precision* (how many significant bits you keep)

For deep learning training, the key question is not "can I store 3.14159?" but:

- Can I represent **tiny gradients** without them becoming 0 (underflow)?
- Can I represent **large activations** without them becoming `inf` (overflow)?
- Can I sum many numbers without destroying meaning via rounding?

These failure modes show up differently in FP16 and BF16.
""")


md(r"""
### 1.1.1 IEEE 754 anatomy (sign, exponent bias, hidden bit)

A *normalized* binary floating-point value is encoded as:
- **sign bit** $s$ (0 = positive, 1 = negative)
- **exponent field** $E$ (stored with a **bias** so we can represent negative exponents)
- **mantissa / fraction field** $m$

For normalized numbers:

$$\text{value} = (-1)^s \times (1 + m) \times 2^{(E - \text{bias})}$$

Key details:
- The leading `1.` is **implicit** (the "hidden bit"), giving you 1 extra bit of effective precision for free.
- Exponent all-zeros and all-ones are **reserved**: `E=0` → subnormals / zero; `E=all ones` → `inf` / `nan`.
- The **bias** is $2^{(\text{exp\_bits} - 1)} - 1$. For FP32: $127$. For FP16: $15$. For BF16: $127$.

Let's decode $\pi$ in all three formats to see this concretely.
""")

code(r"""
# Bit-level decoding of pi across FP32, FP16, BF16

def bits_f32(x: float) -> str:
    (u32,) = struct.unpack(">I", struct.pack(">f", float(x)))
    return f"{u32:032b}"

def bits_f16(x: float) -> str:
    u16 = np.frombuffer(np.float16(x).tobytes(), dtype=np.uint16)[0]
    return f"{int(u16):016b}"

def bits_bf16(x: float) -> str:
    t = torch.tensor(float(x), dtype=torch.bfloat16)
    i16 = int(t.view(torch.int16).item()) & 0xFFFF
    return f"{i16:016b}"

def decode_float(bits: str, exp_bits: int, mant_bits: int, bias: int):
    s = int(bits[0], 2)
    E = int(bits[1:1+exp_bits], 2)
    M_bits = bits[1+exp_bits:]
    assert len(M_bits) == mant_bits

    if E == 0:
        exp = 1 - bias
        mant = sum(int(b) * (2 ** (-(i+1))) for i, b in enumerate(M_bits))
        val = ((-1)**s) * mant * (2**exp)
        return "subnormal/zero", s, E, exp, mant, val

    if E == (2**exp_bits - 1):
        return "inf/nan", s, E, None, None, None

    exp = E - bias
    mant = sum(int(b) * (2 ** (-(i+1))) for i, b in enumerate(M_bits))
    val = ((-1)**s) * (1.0 + mant) * (2**exp)
    return "normal", s, E, exp, mant, val

x = math.pi
rows = []

for name, get_bits, eb, mb, bias in [
    ("float32", bits_f32, 8, 23, 127),
    ("float16", bits_f16, 5, 10, 15),
    ("bfloat16", bits_bf16, 8, 7, 127),
]:
    b = get_bits(x)
    kind, s, E, exp, mant, val = decode_float(b, eb, mb, bias)
    rows.append({
        "dtype": name,
        "bits": f"{b[:1]}|{b[1:1+eb]}|{b[1+eb:]}",
        "sign": s, "E(stored)": E, "exponent": exp,
        "1+mantissa": round(1 + mant, 8) if mant is not None else None,
        "decoded": round(val, 10) if val is not None else None,
        "error vs pi": f"{abs(val - math.pi):.2e}" if val is not None else None,
    })

display(pd.DataFrame(rows))
print(f"\nTrue pi = {math.pi}")
print("Notice: BF16 and FP16 both decode to 3.140625, but via different bit patterns.")
print("FP16 has more mantissa bits (10) giving finer precision; BF16 has fewer (7) but same exponent range as FP32.")
""")


md(r"""
### 1.1.2 Why FP32 → BF16 conversion is trivial (but FP32 → FP16 is not)

Since BF16 shares the same 8-bit exponent as FP32, converting FP32 to BF16 is just **truncating** (or rounding) the bottom 16 mantissa bits. The exponent field doesn't change, so no value can overflow or underflow during conversion.

Converting FP32 to FP16, on the other hand, requires **narrowing the exponent** from 8 bits to 5 bits. This means:
- FP32 values with exponents outside FP16's range (below $2^{-14}$ or above $2^{15}$) become 0 or `inf` during conversion.
- The conversion itself can destroy values even before you do any computation.

This is one reason BF16 is considered a "drop-in" replacement for FP32 in many training scenarios, while FP16 requires extra infrastructure (loss scaling, careful range management).
""")

code(r"""
# FP32 → BF16 vs FP32 → FP16 conversion: what survives?

test_values = [3.14159, 1e-6, 1e-10, 1e-30, 1e-38, 1e30, 65504.0, 65536.0, 1e38]

rows = []
for v in test_values:
    fp32 = torch.tensor(v, dtype=torch.float32)
    bf16 = fp32.to(torch.bfloat16)
    fp16 = fp32.to(torch.float16)
    rows.append({
        "FP32 value": f"{v:.2e}",
        "→ BF16": f"{float(bf16):.4e}",
        "BF16 survived?": "inf/0" if not torch.isfinite(bf16) or float(bf16) == 0 and v != 0 else "YES",
        "BF16 rel error": f"{abs(float(bf16) - v) / (abs(v) + 1e-45):.2e}" if torch.isfinite(bf16) and float(bf16) != 0 else "-",
        "→ FP16": f"{float(fp16):.4e}",
        "FP16 survived?": "inf" if torch.isinf(fp16) else ("0 (underflow)" if float(fp16) == 0 and v != 0 else "YES"),
        "FP16 rel error": f"{abs(float(fp16) - v) / (abs(v) + 1e-45):.2e}" if torch.isfinite(fp16) and float(fp16) != 0 else "-",
    })

display(pd.DataFrame(rows))

print("\nKey takeaway:")
print("  BF16 preserves ALL magnitudes (same exponent range as FP32) — just loses some decimal precision.")
print("  FP16 DESTROYS values outside its narrow range: 1e-10 underflows to 0, 65536 overflows to inf.")
print("  This is why BF16 conversion is 'just truncate the mantissa' — safe and trivial.")
""")


md(r"""
### 1.1.3 Normal vs subnormal numbers (and "flush-to-zero")

**Subnormals** (also called denormals) extend the representable range closer to 0 by giving up the implicit leading `1.`:

$$\text{subnormal value} = (-1)^s \times (0.\text{mantissa}) \times 2^{(1 - \text{bias})}$$

They matter because **gradients can be very small**. But subnormals can be slow on some hardware, so many compute paths enable **FTZ/DAZ** ("flush-to-zero" / "denormals-are-zero"), which means extremely small values become exactly 0.

**Practical lesson:** It is not enough to know the *spec* of a dtype. You also need to know what your hardware/kernel path does with subnormals.

Let's probe whether the smallest subnormal survives on your device.
""")

code(r"""
# Subnormal survival probe

def subnormal_survives(dtype, dev):
    z = torch.tensor(0.0, dtype=dtype, device=dev)
    o = torch.tensor(1.0, dtype=dtype, device=dev)
    sub = torch.nextafter(z, o)
    return {
        "dtype": str(dtype), "device": dev.type,
        "nextafter(0,1)": f"{float(sub):.6e}",
        "is_zero": bool((sub == 0).item()),
    }

rows = []
for dt in [torch.float16, torch.bfloat16, torch.float32]:
    try:
        rows.append(subnormal_survives(dt, device))
    except Exception as e:
        rows.append({"dtype": str(dt), "device": device.type, "error": type(e).__name__})

pd.DataFrame(rows)
""")


md(r"""
### 1.1.4 The measuring tape analogy

A good way to think about floating-point formats is as a **measuring tape**:
- The **length** of the tape is determined by the exponent bits (range: how big/small you can measure).
- The **fineness of the tick marks** is determined by the mantissa bits (precision: how closely you can read off a value).

FP32 is a long tape with fine tick marks. BF16 is equally long but with coarser tick marks. FP16 is a much shorter tape with tick marks finer than BF16 but coarser than FP32.

For training, **tape length (range) matters more than tick mark fineness (precision)** — because a gradient that falls *off the tape entirely* (underflow to zero) provides zero learning signal, while a gradient that lands *between tick marks* (rounding) still provides a useful approximate signal. SGD is inherently noisy; it tolerates imprecise gradients, but it cannot learn from absent ones.
""")

code(r"""
# The measuring tape: visualize range and precision as tape length vs tick density

fig, axes = plt.subplots(3, 1, figsize=(14, 5), gridspec_kw={"hspace": 0.6})

tape_configs = [
    ("FP32 (8-bit exp, 23-bit mantissa)", torch.float32, "#2ecc71", -126, 127, 23),
    ("BF16 (8-bit exp, 7-bit mantissa)",  torch.bfloat16, "#3498db", -126, 127, 7),
    ("FP16 (5-bit exp, 10-bit mantissa)", torch.float16, "#e74c3c", -14, 15, 10),
]

for ax, (label, dt, color, exp_min, exp_max, mant_bits) in zip(axes, tape_configs):
    # Draw the tape as a colored bar representing the exponent range
    tape_left = exp_min
    tape_right = exp_max
    full_range = 127 - (-126)  # FP32 range for normalization

    # Normalize to common axis
    ax.barh(0, tape_right - tape_left, left=tape_left, height=0.4,
            color=color, alpha=0.3, edgecolor=color, linewidth=2)

    # Draw tick marks proportional to mantissa precision
    # More mantissa bits = more ticks (denser)
    n_ticks = min(2 ** mant_bits, 200)  # cap for visualization
    tick_positions = np.linspace(tape_left, tape_right, n_ticks)
    for tp in tick_positions:
        ax.plot([tp, tp], [-0.15, 0.15], color=color, linewidth=0.3, alpha=0.6)

    # Labels
    ax.text(tape_left - 1, 0, label, ha="right", va="center", fontsize=9, fontweight="bold")
    ax.text((tape_left + tape_right) / 2, -0.35,
            f"Range: 2^{exp_min} to 2^{exp_max}  |  Precision: {2**mant_bits} levels per power-of-2",
            ha="center", va="top", fontsize=8, color="gray")

    ax.set_xlim(-140, 135)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("exponent (log2 scale)" if ax == axes[-1] else "")

fig.suptitle("The Measuring Tape: Range (tape length) vs Precision (tick density)", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()

print("Key insight:")
print("  FP32 and BF16 have the SAME tape length (range) — they can measure the same extremes.")
print("  FP16 has a MUCH SHORTER tape — values beyond 2^15 ≈ 65504 overflow to infinity,")
print("  and values below 2^-14 ≈ 6e-5 underflow to zero.")
print()
print("  But BF16's tick marks are 8x coarser than FP16's (128 vs 1024 levels per interval).")
print("  For training, this tradeoff overwhelmingly favors BF16: coarse ticks = noise, short tape = death.")
""")


md(r"""
### 1.1.5 ULP: spacing grows with magnitude

A float format has *roughly constant relative precision* but *variable absolute precision*.

- Near 1.0, FP16 spacing is ~$10^{-3}$.
- Near 1024, FP16 spacing is ~$1$.

This is the concrete reason "tiny updates disappear" when weights are stored in low precision.
""")

code(r"""
# ULP vs magnitude for each dtype

def ulp(x: torch.Tensor, dtype: torch.dtype):
    x = x.to(dtype)
    return (torch.nextafter(x, x * 2) - x).abs().to(torch.float32)

ks = torch.arange(-10, 21, device=device)
x = (2.0 ** ks).to(torch.float32)

plt.figure(figsize=(10, 4))
for dt, color in [(torch.float16, "C0"), (torch.bfloat16, "C1"), (torch.float32, "C2")]:
    if device.type == "cpu" and dt is torch.float16:
        continue
    u = ulp(x, dt).cpu().numpy()
    plt.plot(ks.cpu().numpy(), np.log2(u + 1e-45), marker="o", markersize=4, label=str(dt), color=color)

plt.title("ULP (spacing between adjacent floats) vs magnitude")
plt.xlabel("log2(|x|)")
plt.ylabel("log2(ULP(x))")
plt.legend()
plt.tight_layout();
""")


md(r"""
### 1.1.6 Number line: where representable floats actually live

The spacing between representable numbers is *not uniform* — it depends on the magnitude. Near zero, floats are dense; as magnitude grows, they spread apart. And crucially, **different formats have different densities at every scale**.

This visualization plots the actual representable numbers in each format within a small interval. Think of it as zooming into the "measuring tape" to see the tick marks.
""")

code(r"""
# Number line: representable floats in [1.0, 2.0) for each dtype
# This interval is illuminating because ULP is constant within a power-of-2 interval

fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True)

for ax, (dt, label, color) in zip(axes, [
    (torch.float32, "FP32 (23-bit mantissa: 8,388,608 values in [1,2))", "C2"),
    (torch.bfloat16, "BF16 (7-bit mantissa: 128 values in [1,2))", "C1"),
    (torch.float16, "FP16 (10-bit mantissa: 1,024 values in [1,2))", "C0"),
    (None, "Comparison overlay", "k"),
]):
    if dt is not None:
        one = torch.tensor(1.0, dtype=dt)
        two = torch.tensor(2.0, dtype=dt)
        vals = [float(one)]
        cur = one
        while True:
            cur = torch.nextafter(cur, two)
            if float(cur) >= 2.0:
                break
            vals.append(float(cur))
            if len(vals) > 2000:
                break
        vals = np.array(vals)
        ax.eventplot([vals], lineoffsets=0, linelengths=0.6, colors=color, linewidths=0.5)
        ax.set_ylabel(label, fontsize=8)
        ax.set_yticks([])
        ax.text(1.0, 0.35, f"{len(vals)} representable values", fontsize=8, color=color)
    else:
        # Overlay: show a narrow window [1.0, 1.02] with all three
        for dt2, c2, yoff in [(torch.float16, "C0", 0.3), (torch.bfloat16, "C1", 0.0), (torch.float32, "C2", -0.3)]:
            one2 = torch.tensor(1.0, dtype=dt2)
            limit = torch.tensor(1.02, dtype=dt2)
            vs = [float(one2)]
            cur2 = one2
            for _ in range(200):
                cur2 = torch.nextafter(cur2, limit)
                if float(cur2) >= 1.02:
                    break
                vs.append(float(cur2))
            vs = np.array(vs)
            ax.eventplot([vs], lineoffsets=yoff, linelengths=0.25, colors=c2, linewidths=1.0)
        ax.set_ylabel("Zoomed [1.0, 1.02]", fontsize=8)
        ax.set_yticks([])
        ax.set_xlim(1.0, 1.02)
        ax.legend(["FP16", "BF16", "FP32"], fontsize=7, loc="upper right")

axes[0].set_xlim(1.0, 2.0)
for ax in axes[:3]:
    ax.set_xlim(1.0, 2.0)
axes[-1].set_xlim(1.0, 1.02)
axes[-1].set_xlabel("value")
fig.suptitle("Representable floats in [1.0, 2.0): density depends on mantissa bits", fontsize=11, y=1.01)
plt.tight_layout();
print("Key insight: BF16 has ~8x fewer representable values than FP16 in [1,2),")
print("but FP16 has ~8,192x fewer than FP32. This is the precision-range tradeoff made visible.")
""")


md(r"""
### 1.1.7 The bit-level addition trap (why `1 + 1e-4 = 1` in FP16)

This is the single most important numeric fact for understanding **weight update stagnation**.

When adding two floating-point numbers, the hardware must **align exponents** by shifting the smaller number's mantissa to the right. If the shift pushes all significant bits past the mantissa width, the smaller number is effectively lost.

**Concrete example (from the FP16 bit-level):**

- `1.0` in FP16: exponent = $2^0$, mantissa = all zeros.
- `1e-4` in FP16: exponent = $2^{-14}$, mantissa encodes ~1.639.
- To add them, we must shift `1e-4`'s mantissa by **14 positions** to align with `1.0`'s exponent.
- FP16 has only **10 mantissa bits**. After shifting 14 positions right, *all* significant bits fall off the edge.
- Result: `1.0 + 1e-4 = 1.0` exactly.

This is exactly what happens during training: if `learning_rate * gradient` is smaller than the ULP at the weight's magnitude, the weight **never changes**.

FP16's epsilon is ~$9.77 \times 10^{-4}$. Any update smaller than this relative to the weight magnitude is silently dropped.
""")

code(r"""
# Demonstrate the addition trap across dtypes

print("Does 1.0 + delta produce a value > 1.0?\n")

deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
rows = []
for delta in deltas:
    row = {"delta": f"{delta:.0e}"}
    for name, dt in [("FP16", torch.float16), ("BF16", torch.bfloat16), ("FP32", torch.float32)]:
        one = torch.tensor(1.0, dtype=dt)
        d = torch.tensor(delta, dtype=dt)
        result = one + d
        changed = float(result) != float(one)
        row[name] = "YES" if changed else "no (lost!)"
    rows.append(row)

df_add = pd.DataFrame(rows)
display(df_add)

print("\nKey insight:")
print("- FP16 loses updates smaller than ~1e-3 relative to weight magnitude.")
print("- BF16 loses updates smaller than ~8e-3 (even coarser!).")
print("- FP32 loses updates smaller than ~1e-7.")
print("\nThis is why optimizers need FP32 master weights: typical lr*grad products")
print("are often 1e-5 to 1e-7, which FP16 and BF16 both silently discard.")
""")


md(r"""
### The bit-level mechanics: *why* `1.0 + 1e-4 = 1.0` in FP16

The table above shows *that* small updates get lost. Let's see *why* at the bit level.

When hardware adds two floats, it must **align their exponents** by right-shifting the smaller number's mantissa. If the shift exceeds the mantissa width, the smaller number's bits fall off completely.

This is the exact same mechanism that causes **weight update stagnation** during training: `weight += lr * gradient` produces the same weight if `lr * gradient` is too small relative to the weight's magnitude.
""")

code(r"""
# Step-by-step bit alignment: why 1.0 + 1e-4 = 1.0 in FP16

print("=== Bit-level addition in FP16: 1.0 + 1e-4 ===\n")

# 1.0 in FP16
one_f16 = np.float16(1.0)
one_bits = f"{int(np.frombuffer(one_f16.tobytes(), dtype=np.uint16)[0]):016b}"
print(f"1.0 in FP16:   {one_bits[0]}|{one_bits[1:6]}|{one_bits[6:]}")
print(f"               s| exp  | mantissa")
print(f"               = (-1)^0 × 1.0000000000 × 2^(15 - 15) = 1.0")

print()

# 1e-4 in FP16
small_f16 = np.float16(1e-4)
small_bits = f"{int(np.frombuffer(small_f16.tobytes(), dtype=np.uint16)[0]):016b}"
E_small = int(small_bits[1:6], 2)
print(f"1e-4 in FP16:  {small_bits[0]}|{small_bits[1:6]}|{small_bits[6:]}")
print(f"               = (-1)^0 × 1.{small_bits[6:]} × 2^({E_small} - 15) = 2^{{{E_small - 15}}}")
print(f"               ≈ {float(small_f16):.6e}")

print()
print("--- Addition step: align exponents ---")
print()
shift = 15 - E_small
print(f"To add these, we align the smaller exponent ({E_small - 15}) to the larger (0).")
print(f"This means shifting 1e-4's mantissa RIGHT by {shift} positions.")
print()
print(f"  1.0:     1.{'0' * 10}         (exponent = 0)")
print(f"+ 1e-4:    0.{'0' * (shift - 1)}1{'?' * max(0, 10 - shift)}   (shifted {shift} positions right)")
print()
print(f"FP16 mantissa is only 10 bits wide.")
print(f"After shifting right by {shift}, ALL significant bits of 1e-4 are")
print(f"beyond the 10-bit mantissa boundary → they are discarded.")
print()
print(f"Result: 1.0 + 1e-4 = 1.0  (the small value vanished completely)")
print()

# Verify
result = np.float16(1.0) + np.float16(1e-4)
eps_f16 = np.finfo(np.float16).eps
print(f"Verification:  np.float16(1.0) + np.float16(1e-4) = {result}")
print(f"FP16 epsilon:  {eps_f16:.4e}")
print(f"1e-4 < eps?    {1e-4 < eps_f16}  → update is below FP16's resolution at magnitude 1.0")
print()
print("Training implication: if weight ≈ 1.0 and lr × grad ≈ 1e-4,")
print("the weight NEVER changes in FP16. This is weight update stagnation.")
""")


md(r"""
### 1.1.8 Epsilon meets training: where do real weight updates fall?

The addition trap is not theoretical — it directly determines whether training succeeds. Let's connect the abstract epsilon values to concrete training scenarios.

For a weight $w$ stored in a given dtype, the **minimum detectable update** is approximately $|w| \times \epsilon$. If $\text{lr} \times |\text{grad}|$ is smaller than this, the weight never changes.

Typical training setups use learning rates of $10^{-3}$ to $10^{-5}$, and gradient magnitudes often range from $10^{-2}$ to $10^{-6}$. Their product ($\text{lr} \times |\text{grad}|$) is what the format must be able to represent *relative to the weight magnitude*.
""")

code(r"""
# Where do typical lr × grad products fall relative to epsilon boundaries?

fig, ax = plt.subplots(figsize=(14, 5))

# Epsilon values (minimum detectable relative update)
epsilons = {
    "FP16": float(torch.finfo(torch.float16).eps),
    "BF16": float(torch.finfo(torch.bfloat16).eps),
    "FP32": float(torch.finfo(torch.float32).eps),
}

# Typical lr × |grad| magnitudes in real training
typical_updates = {
    "lr=1e-3 × |g|=1e-1\n(early training, large grads)": 1e-4,
    "lr=1e-3 × |g|=1e-3\n(mid training)": 1e-6,
    "lr=1e-4 × |g|=1e-3\n(fine-tuning)": 1e-7,
    "lr=1e-4 × |g|=1e-5\n(late training, small grads)": 1e-9,
    "lr=1e-5 × |g|=1e-5\n(LLM fine-tune)": 1e-10,
}

y_pos = 0
colors_eps = {"FP16": "#e74c3c", "BF16": "#3498db", "FP32": "#2ecc71"}

# Draw epsilon boundaries as vertical lines
for name, eps in epsilons.items():
    ax.axvline(np.log10(eps), color=colors_eps[name], linewidth=3, alpha=0.7,
               label=f"{name} epsilon = {eps:.1e}")

# Shade the "safe update" zone (above all epsilons)
ax.axvspan(np.log10(max(epsilons.values())), 0, alpha=0.05, color="green")

# Draw typical update magnitudes as horizontal markers
y_updates = np.linspace(0.9, 0.1, len(typical_updates))
for (label, mag), y in zip(typical_updates.items(), y_updates):
    marker_color = "green"
    if mag < epsilons["FP16"]:
        marker_color = "red"
    elif mag < epsilons["BF16"]:
        marker_color = "orange"
    ax.plot(np.log10(mag), y, "D", color=marker_color, markersize=10, zorder=5)
    ax.text(np.log10(mag) + 0.15, y, label, fontsize=7, va="center", color=marker_color)

# Annotations
ax.text(np.log10(epsilons["FP16"]) - 0.1, 0.95,
        "← Updates here are LOST\n    in FP16 (weight never changes)",
        fontsize=8, color="#e74c3c", ha="right", va="top", fontstyle="italic")
ax.text(np.log10(epsilons["BF16"]) - 0.1, 0.5,
        "← Also lost in BF16",
        fontsize=8, color="#3498db", ha="right", va="center", fontstyle="italic")

ax.set_xlim(-12, 0.5)
ax.set_ylim(-0.05, 1.1)
ax.set_xlabel("log10(lr × |gradient|)  — relative to weight magnitude of ~1.0", fontsize=10)
ax.set_yticks([])
ax.set_title("Where do real weight updates fall relative to format epsilon?", fontsize=12, fontweight="bold")
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()

print("Interpretation:")
print("  - Green diamonds: update is large enough for ALL formats")
print("  - Orange diamonds: update works in FP32 and FP16, but NOT in BF16")
print("  - Red diamonds: update only works in FP32")
print()
print("This is why FP32 master weights are essential: the optimizer applies updates in FP32")
print("(where they're captured), then casts back to 16-bit for the next forward pass.")
""")


md(r"""
## 1.2 FP16 vs BF16 vs FP32: the complete numeric comparison

We generated the cheat sheet above. Here we dig deeper into what those numbers mean for training.
""")

code(r"""
# Detailed format facts table

def _ulp_at_one_v2(dtype):
    try:
        one = torch.tensor(1.0, dtype=dtype)
        return float(torch.nextafter(one, one + one) - one)
    except Exception:
        return None

def _smallest_subnormal_v2(dtype):
    try:
        z = torch.tensor(0.0, dtype=dtype)
        o = torch.tensor(1.0, dtype=dtype)
        return float(torch.nextafter(z, o))
    except Exception:
        return None

def dtype_row(name, dtype, exp_bits, mant_bits, exp_min, exp_max):
    fi = torch.finfo(dtype)
    ulp1 = _ulp_at_one_v2(dtype)
    sub = _smallest_subnormal_v2(dtype)
    return {
        "dtype": name,
        "bits": fi.bits,
        "exp_bits": exp_bits,
        "mant_bits": mant_bits,
        "precision_bits": mant_bits + 1,
        "decimal_digits": round((mant_bits + 1) * math.log10(2), 2),
        "exp_range": f"[{exp_min}, {exp_max}]",
        "epsilon": f"{fi.eps:.2e}",
        "ulp(1.0)": f"{ulp1:.2e}" if ulp1 is not None else "n/a",
        "min_normal": f"{fi.tiny:.2e}",
        "min_subnormal": f"{sub:.2e}" if sub is not None else "n/a",
        "max_finite": f"{fi.max:.2e}",
    }

dtype_info = pd.DataFrame([
    *( [dtype_row("float8_e4m3fn", torch.float8_e4m3fn, 4, 3, -6, 7)] if hasattr(torch, "float8_e4m3fn") else [] ),
    *( [dtype_row("float8_e5m2", torch.float8_e5m2, 5, 2, -14, 15)] if hasattr(torch, "float8_e5m2") else [] ),
    dtype_row("float16", torch.float16, 5, 10, -14, 15),
    dtype_row("bfloat16", torch.bfloat16, 8, 7, -126, 127),
    dtype_row("float32", torch.float32, 8, 23, -126, 127),
    dtype_row("float64", torch.float64, 11, 52, -1022, 1023),
]).set_index("dtype")

display(dtype_info)
""")


md(r"""
### Interpreting the table

**Precision** (how fine the tick marks are):
- FP64: ULP at 1.0 is ~2e-16. This is essentially "reference precision" for most deep learning numerics.
- FP32: ULP at 1.0 is ~$1.2 \times 10^{-7}$. Updates as small as $10^{-7}$ are captured.
- FP16: ULP at 1.0 is ~$9.8 \times 10^{-4}$. Updates smaller than $10^{-3}$ are lost.
- BF16: ULP at 1.0 is ~$7.8 \times 10^{-3}$. Updates smaller than $10^{-2}$ are lost. Even coarser than FP16!
- FP8: ULP at 1.0 is huge. FP8 is *not* a drop-in training dtype without extra scaling strategies and specialized kernels.

**Range** (how long the measuring tape is):
- FP16: smallest normal is ~$6 \times 10^{-5}$. Gradients below this become zero.
- BF16: smallest normal is ~$1.2 \times 10^{-38}$, same as FP32. Gradients essentially never underflow.
- FP8: range depends on format (E4M3 vs E5M2), but is far smaller than FP16/BF16/FP32.
- This is why **FP16 needs loss scaling** but **BF16 usually does not**.

### 1.2.1 TF32 — the hidden precision mode

On NVIDIA Ampere+ GPUs, FP32 matmuls can automatically use **TF32** internally:
- Same 8-bit exponent as FP32 (full range)
- But only 10 bits of mantissa (same as FP16 precision)
- Transparent: your code says `float32`, but Tensor Cores use TF32 for speed

This means your "FP32 baseline" on modern GPUs may actually be **TF32 precision** for matmuls. When comparing FP32 vs AMP, be aware of this hidden variable. You can control it with `torch.backends.cuda.matmul.allow_tf32`.
""")

md(r"""
### 1.2.2 Why autocast targets matmul/linear first: FP32 accumulation

Deep learning is dominated by large matrix multiplications (GEMMs): `Linear`, attention projections, and MLPs.

On modern accelerators, these kernels typically:
- **multiply** in FP16/BF16 (or TF32/FP8, depending on mode)
- **accumulate** partial sums in **FP32**

This is a sweet spot:
- massive speedup (Tensor Cores / specialized units)
- much better numeric behavior than "pure FP16 accumulation"

Autocast heavily leans on this: it prefers to cast matmul-like ops down because they are both *fast* and comparatively *stable* (relative to softmax, layernorm, exp/log, large reductions).
""")

code(r"""
# Matmul accuracy across dtypes (vs FP64 reference)

M = 256 if device.type != "cuda" else 512
a = torch.randn(M, M, device=device, dtype=torch.float32)
b = torch.randn(M, M, device=device, dtype=torch.float32)

ref = (a.double() @ b.double()).float()

rows = []
for dt in [torch.float16, torch.bfloat16, torch.float32]:
    if not supports_dtype_on_device(dt, device):
        continue
    try:
        c_dt = a.to(dt) @ b.to(dt)
    except Exception as e:
        rows.append({
            "dtype": str(dt).replace("torch.", ""),
            "matmul_output_dtype": "-",
            "max_abs_err": "-",
            "mean_abs_err": "-",
            "max_rel_err": "-",
            "mean_rel_err": "-",
            "note": f"matmul failed ({type(e).__name__})",
        })
        continue
    c = c_dt.float()
    abs_err = (c - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-6)
    rows.append({
        "dtype": str(dt).replace("torch.", ""),
        "matmul_output_dtype": str(c_dt.dtype).replace("torch.", ""),
        "max_abs_err": f"{float(abs_err.max()):.2e}",
        "mean_abs_err": f"{float(abs_err.mean()):.2e}",
        "max_rel_err": f"{float(rel_err.max()):.2e}",
        "mean_rel_err": f"{float(rel_err.mean()):.2e}",
        "note": "",
    })

display(pd.DataFrame(rows))
print("\nNotes:")
print("- Errors come from (1) input rounding to dt and (2) output rounding back to dt.")
print("- On CUDA, FP16/BF16 matmuls usually accumulate in FP32 internally, which helps stability.")
""")


md(r"""
## 1.3 The three numeric disasters that show up during training

### (A) Underflow — values become 0

- Common in **gradients**, especially late in training or in deep nets with tiny signals.
- Most harmful in **FP16** due to narrow exponent range (5 bits → min normal $\approx 6 \times 10^{-5}$).
- BF16 has the same exponent range as FP32, so underflow is rare.

### (B) Overflow — values become `inf`

- Common in **activations** (exponentials, attention logits) or badly-initialized models.
- FP16 max is only ~65,504. Easy to exceed.
- BF16 max is ~$3.4 \times 10^{38}$, same as FP32.

### (C) Accumulation / cancellation error

Even when values are in range, precision limits corrupt sums and products:
- Adding many small numbers to a large accumulator can lose the small contributions (same mechanism as the `1 + 1e-4` trap).
- For reductions (layernorm statistics, softmax normalization, large sums), frameworks often keep accumulation in FP32.

**Autocast** is partly about preventing A/B (range disasters), and partly about routing sensitive reductions so C doesn't destroy training.
""")

md(r"""
### 1.3.0 Accumulation error, made concrete

**Accumulation error** is just "rounding happens at every add/multiply, and it compounds".

The simplest microscope is:

> start at a value (like 1.0), add a tiny increment many times, and see when the increment stops "counting".

This is the same mechanism behind:
- weight-update stagnation (updates below ULP get rounded away)
- instability in reductions (sums/means/variances done in low precision)
""")

code(r"""
# Accumulation microscope: 1.0 + N * delta, computed sequentially in different dtypes

N = 20000
base = 1.0
delta = 1e-3

steps_i = torch.arange(N + 1, device=device, dtype=torch.int64)
expected64 = base + delta * steps_i.double()

rows = []
plt.figure(figsize=(10, 4))

every = max(1, (N + 1) // 800)

for dt in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    if not supports_dtype_on_device(dt, device):
        continue
    deltas = torch.full((N,), delta, device=device, dtype=dt)
    start = torch.tensor([base], device=device, dtype=dt)
    cs = torch.cat([start, deltas]).cumsum(0)
    err = (cs.double() - expected64).cpu().numpy()

    label = str(dt).replace("torch.", "")
    plt.plot(steps_i.cpu().numpy()[::every], err[::every], label=label, alpha=0.85)

    final = float(cs[-1].cpu())
    rows.append({
        "dtype": label,
        "final": f"{final:.6f}",
        "expected": f"{float(expected64[-1].cpu()):.6f}",
        "abs_err": f"{abs(final - float(expected64[-1].cpu())):.3e}",
    })

plt.axhline(0.0, color="k", lw=1, alpha=0.3)
plt.title("Accumulation error: sequentially adding delta many times")
plt.xlabel("step")
plt.ylabel("cumsum(dtype) - reference (float64 arithmetic)")
plt.legend()
plt.tight_layout();

display(pd.DataFrame(rows))
print("\nInterpretation:")
print("- If the error curve flattens, your increments stopped affecting the accumulator.")
print("- This is why AMP promotes certain reductions (like sum/prod and norm stats) to FP32.")
""")

code(r"""
# LayerNorm-like statistics are especially sensitive:
# mean/var of (large offset + small noise) is a classic cancellation problem.

M = 200_000
x = (torch.randn(M, device=device, dtype=torch.float32) * 0.1) + 1000.0

ref_mu = float(x.double().mean())
ref_var = float(((x.double() - ref_mu) ** 2).mean())

rows = []
for dt in [torch.float16, torch.bfloat16, torch.float32]:
    if not supports_dtype_on_device(dt, device):
        continue
    xd = x.to(dt)
    mu = xd.mean()
    var = ((xd - mu) ** 2).mean()
    rows.append({
        "dtype": str(dt).replace("torch.", ""),
        "mean": f"{float(mu):.6f}",
        "var": f"{float(var):.6e}",
        "mean_abs_err": f"{abs(float(mu) - ref_mu):.3e}",
        "var_rel_err": f"{abs(float(var) - ref_var) / (ref_var + 1e-30):.3e}",
    })

display(pd.DataFrame(rows))
print("\nThis is a simplified version of why LayerNorm/Softmax reductions are often forced to FP32 under autocast.")
""")

code(r"""
# Where does exp() overflow by dtype?

x = torch.linspace(-20, 20, 400, device=device)

def safe_exp(x, dtype):
    return torch.exp(x.to(dtype)).to(torch.float32).cpu().numpy()

plt.figure(figsize=(10, 4))
for dt, label in [(torch.float32, "FP32"), (torch.bfloat16, "BF16")]:
    y = safe_exp(x, dt)
    plt.plot(x.cpu().numpy(), np.log10(np.clip(y, 1e-30, 1e30)), label=label)

if device.type == "cuda":
    y16 = safe_exp(x, torch.float16)
    plt.plot(x.cpu().numpy(), np.log10(np.clip(y16, 1e-30, 1e30)), label="FP16")

plt.axhline(np.log10(65504), color="r", linestyle="--", alpha=0.5, label="FP16 max (65504)")
plt.title("log10(exp(x)) computed in different dtypes — FP16 overflows early")
plt.xlabel("x")
plt.ylabel("log10(exp(x))")
plt.legend()
plt.tight_layout();
""")


md(r"""
### 1.3.1 Loss functions are "log-sum-exp machines" (and dtype matters)

Many deep learning losses contain exponentials and logs. A classic example:

$$\log(1 + e^x) \quad \text{(softplus)}$$

- The naive formula overflows quickly in FP16 (for $x > 11$, $e^x > 65504$).
- Stable implementations (e.g., `F.softplus`) avoid overflow by rewriting the expression.
""")

code(r"""
# Naive vs stable softplus across dtypes

def naive_softplus(x):
    return torch.log1p(torch.exp(x))

x = torch.linspace(-80, 80, 2000, device=device)
ref = F.softplus(x.double()).float()  # high-precision reference

plt.figure(figsize=(10, 4))
for dt in [torch.float16, torch.bfloat16, torch.float32]:
    if device.type == "cpu" and dt is torch.float16:
        continue
    y_naive = naive_softplus(x.to(dt)).float()
    y_stable = F.softplus(x.to(dt)).float()
    err_naive = (y_naive - ref).abs().cpu().numpy()
    err_stable = (y_stable - ref).abs().cpu().numpy()
    plt.plot(x.cpu().numpy(), np.log10(err_naive + 1e-12), label=f"naive {dt}")
    plt.plot(x.cpu().numpy(), np.log10(err_stable + 1e-12), ls="--", label=f"stable {dt}")

plt.title("Softplus error: naive vs stable implementation")
plt.xlabel("x")
plt.ylabel("log10(|error|) vs FP64 reference")
plt.legend(ncols=2)
plt.tight_layout();
""")


md(r"""
### 1.3.2 Softmax: the overflow trap and the stability rewrite

Softmax is everywhere in transformers (attention). Naive softmax:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

This can overflow in low precision because $e^x$ explodes quickly. The stable rewrite subtracts the max:

$$\text{softmax}(x) = \text{softmax}(x - \max(x))$$

PyTorch's `F.softmax` uses a stable implementation. Let's verify.
""")

code(r"""
# Naive vs stable softmax

def naive_softmax(x, dim=-1):
    ex = torch.exp(x)
    return ex / ex.sum(dim=dim, keepdim=True)

logits = torch.tensor([0.0, 20.0, 40.0, 80.0], device=device)
ref = F.softmax(logits.double(), dim=0).float()

rows = []
for dt in [torch.float16, torch.bfloat16, torch.float32]:
    if device.type == "cpu" and dt is torch.float16:
        continue
    x = logits.to(dt)
    try:
        naive = naive_softmax(x, dim=0).float()
        naive_ok = bool(torch.isfinite(naive).all().item())
    except Exception:
        naive = torch.full_like(ref, float("nan"))
        naive_ok = False
    stable = F.softmax(x, dim=0).float()
    stable_ok = bool(torch.isfinite(stable).all().item())
    rows.append({
        "dtype": str(dt),
        "naive_finite": naive_ok,
        "stable_finite": stable_ok,
        "max_abs_err(stable vs ref)": f"{float((stable - ref).abs().max()):.2e}",
    })

pd.DataFrame(rows)
""")


md(r"""
## 1.4 What AMP actually is

In PyTorch, AMP is two complementary tools:

1. **`autocast`** (forward + loss)
   - A context manager that applies a **per-operation dtype policy**.
   - It *temporarily* casts inputs/weights for each operation. It does **not** permanently change model parameters.
   - Matmuls/linear → lower precision. Softmax/layernorm/losses → FP32.

2. **`GradScaler`** (backward + optimizer step)
   - Primarily for **FP16 training** (BF16 usually doesn't need it).
   - Multiplies loss by a scale factor $S$ before backward → gradients are $S\times$ larger → fewer underflow to zero.
   - Before optimizer step, divides gradients by $S$. If overflow is detected (`inf`/`nan`), skips the step and reduces $S$.

**Clean mental model:**
- `autocast` protects you from **bad forward dtypes**.
- `GradScaler` protects you from **bad backward magnitudes** (FP16 gradient underflow).
""")


md(r"""
## 1.5 The canonical AMP training loop (four conceptual changes)

Start with FP32 training:
```python
optimizer.zero_grad(set_to_none=True)
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

AMP adds four things:
```python
scaler = GradScaler()

optimizer.zero_grad(set_to_none=True)
with autocast(device_type="cuda", dtype=torch.float16):   # 1. wrap forward + loss
    logits = model(x)
    loss = loss_fn(logits, y)

scaler.scale(loss).backward()    # 2. scale loss before backward
scaler.step(optimizer)           # 3. unscale + check for inf/nan + step
scaler.update()                  # 4. adjust scale factor
```

That's the "small code change" people talk about. But the *reason* it works is the theory above.
""")


md(r"""
## 1.6 Master weights and optimizer state (why "just casting the model" is not the same)

There are **three** numeric objects in training that matter:

1. **Parameters** (weights) — used in forward/backward
2. **Gradients** — produced by backward
3. **Optimizer state** (e.g., Adam's first and second moment estimates $m_t, v_t$) — long-lived accumulators

A classic mixed precision recipe:
- Keep a **master copy of weights in FP32**.
- Do forward/backward in FP16/BF16 where safe.
- Maintain optimizer state (Adam moments) in FP32.

**Why FP32 master weights?**

Because 16-bit formats have coarse spacing. A small update $\Delta w = \text{lr} \times \text{grad}$ can be *below the ULP* at the magnitude of $w$, so the weight never changes. We showed this in the `1 + 1e-4` demo above.

Over many steps, these tiny updates *accumulate* in FP32 and eventually become large enough to appear in the 16-bit copy. This is the key insight from the Micikevicius et al. (2017) paper.

**Why FP32 optimizer state?**

Adam's moments are exponential moving averages. They accumulate information over the entire training run. Even small rounding errors compound over thousands of steps. Keeping moments in FP32 prevents this drift.

**Memory implication:**

Mixed precision doesn't eliminate FP32 — it just limits FP32 to parameters + optimizer state (which is fixed-size), while saving on activations (which scale with batch size and sequence length).
""")


md(r"""
## 1.7 Autocast is an operator policy (not a global cast)

Autocast does **not** "turn the whole model into FP16". Instead it applies a per-operation policy:

| Policy | Operations | Rationale |
|---|---|---|
| **Lower precision** (FP16/BF16) | `linear`, `matmul`, `mm`, `bmm`, convolutions | Compute-bound → Tensor Core speedup |
| **Force FP32** | `softmax`, `layer_norm`, `log_softmax`, `mse_loss`, `cross_entropy`, `sum`, `prod`, `exp`, `log`, ... | Numerically sensitive: overflow/underflow/accumulation risk |
| **Promote to widest** | Binary ops when inputs differ | If one input is FP32, the op runs in FP32 |
| **Pass-through** | `relu`, `dropout`, `max`, `min`, `mean`, ... | Element-wise, no numeric risk; output matches input dtype |

The exact policy is PyTorch-version-dependent. In Section 3 we will probe it empirically on your machine.
""")


md(r"""
## Section 1 summary

| Concept | Key takeaway |
|---|---|
| **FP16** | Better precision than BF16, but narrow range → needs loss scaling and careful op policies |
| **BF16** | FP32-like range → often trains without scaling, but coarse precision → reductions need care |
| **TF32** | Your "FP32 baseline" on Ampere+ GPUs may secretly be TF32 (10-bit mantissa) for matmuls |
| **AMP** | `autocast` (forward op policy) + `GradScaler` (backward magnitude control) |
| **Master weights** | Keep FP32 copy for updates; prevents stagnation from ULP rounding |
| **Optimizer state** | Keep in FP32; long-horizon accumulation is precision-sensitive |
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — WHAT THE LITERATURE SAYS
# ═══════════════════════════════════════════════════════════════════════════════

md(r"""
# Section 2 — What the literature says

This section is intentionally *written*: the point is to build a paper-and-doc-driven mental model that you can carry into real training code.

No experiments here — only explanations. Think of this section as: "what each source contributes, and how it maps to PyTorch AMP."
""")


md(r"""
## 2.0 A reading map

| Source | What it gives you | Maps to |
|---|---|---|
| Gupta et al. (2015) | Why limited precision can still train; stochastic rounding + scaling intuition | Section 1.3 (rounding/accumulation), Section 3 debugging |
| Micikevicius et al. (2017) | The canonical mixed precision recipe (master weights + loss scaling) | Sections 1.4–1.6, scaling experiments |
| Kalamkar et al. (2019) | BF16 bit-level analysis, proof that BF16 avoids FP16's underflow | Section 1.2, BF16 training runs |
| NVIDIA mixed precision guidance | Engineering intuition + failure modes | Underflow/overflow + "sensitive ops in FP32" |
| PyTorch `torch.amp` docs | The actual API + gotchas | `autocast` + `GradScaler` loops in Section 3 |
| PyTorch autocast op reference | *The* per-op policy | Probed empirically in Section 3.2 |
| Rajbhandari et al. (2020) — ZeRO | Memory breakdown of mixed-precision training at scale | Optimizer state precision, Section 2.6 |
| Distributed training docs (FSDP/ZeRO/DeepSpeed) | Where dtypes live in large systems | Section 2.7 |
| FP8 literature (e.g., NVIDIA) | Why FP8 needs scaling (E4M3 vs E5M2) + where it fits vs AMP | Section 1 tables, Section 3 practical guidance |
| Dettmers et al. (8-bit optimizers) | Optimizer-state precision as the next bottleneck after AMP | Section 2.6 memory discussion |

If you only read one thing: read the Micikevicius paper, then read the PyTorch autocast op reference.
""")


md(r"""
## 2.1 Micikevicius et al. (2017): *Mixed Precision Training*

> arXiv:1710.03740 — ICLR 2018

This is the foundational paper for everything in this notebook. It introduced the three-part recipe that all modern AMP implementations are based on.

**The problem:** FP16 arithmetic is fast (2–8$\times$ throughput on Tensor Cores) and memory-efficient (half the bytes), but naively training in FP16 breaks. Models either diverge, produce NaN losses, or silently stagnate. The paper identifies three distinct failure modes:

1. **Weight update stagnation.** When a weight $w$ is stored in FP16, the update $\Delta w = \eta \cdot g$ can be smaller than the ULP (unit in the last place) at $|w|$. The update is rounded away and the weight never changes. The paper's solution: maintain an FP32 "master copy" of all parameters. Forward and backward compute use the FP16 cast, but the actual parameter update happens in FP32 where the precision is sufficient to register small changes. The updated FP32 value is then cast back to FP16 for the next forward pass.

2. **Gradient underflow.** Even with FP32 master weights, gradients themselves are computed in FP16 during the backward pass. FP16's smallest normalized number is approximately $6 \times 10^{-5}$. The paper's analysis of gradient histograms across several production models (image classifiers, speech models, generative models) shows that a significant fraction of gradient values fall below this threshold. Once they underflow to zero, the update signal is lost. The solution: **loss scaling**. Multiply the scalar loss by a large constant $S$ before calling `backward()`. Since backpropagation is a chain of multiplications, every gradient in the network is scaled up by $S$. After backward, divide gradients by $S$ before the optimizer step. If $S$ is chosen well, gradients that would have underflowed are now safely within FP16's representable range, and the division by $S$ restores correct magnitudes. The paper demonstrates that a fixed $S$ works for many models, but notes that $S$ too large can cause gradient *overflow*. This motivates **dynamic loss scaling**: start with a large $S$, and if `inf`/`nan` gradients are detected, skip the update and reduce $S$. If training is stable, periodically increase $S$ to maximize headroom.

3. **Accumulation error in reductions.** Large dot products and reductions (e.g., summing thousands of values) accumulate rounding errors that compound in FP16. The paper recommends performing these accumulations in FP32, even when the operands are FP16. Modern Tensor Cores implement this: they accept FP16 inputs but accumulate in FP32.

**What to remember:** AMP is not a single trick. It is the *combination* of (a) per-op precision policies, (b) loss scaling, and (c) FP32 master weights/optimizer state. Removing any one of these can cause training to fail.
""")


md(r"""
## 2.2 Kalamkar et al. (2019): *A Study of BFLOAT16 for Deep Learning Training*

> arXiv:1905.12322

This paper provides the empirical and theoretical justification for BF16 as a training format.

**The key insight** is architectural: BF16 uses **8 exponent bits** (same as FP32), giving it the same dynamic range — roughly $10^{-38}$ to $10^{38}$. This means BF16 can represent the same extreme magnitudes as FP32. The gradient underflow problem that plagues FP16 (5-bit exponent → range only $10^{-5}$ to $6.5 \times 10^4$) essentially does not exist for BF16.

**The tradeoff:** BF16 sacrifices mantissa bits (7 vs FP16's 10 vs FP32's 23). This means worse *precision* per individual value. But the paper demonstrates empirically that deep neural networks are remarkably robust to precision loss — they are much more sensitive to *range* loss. Training with BF16 across a variety of models (ResNets, Transformers, LSTMs) converges to the same final accuracy as FP32, often without any loss scaling at all.

**Conversion simplicity:** Since BF16 and FP32 share the same exponent field, conversion is trivial: truncate (or round) the bottom 16 mantissa bits. FP32 → FP16 conversion is harder because the exponent field must also be narrowed, risking overflow or underflow in the conversion itself.

**Practical implication for AMP:** BF16 autocast is often a "drop-in" replacement that doesn't require a `GradScaler`. This is explicitly confirmed by modern frameworks: DeepSpeed documentation states, "Training with bfloat16 does not require loss scaling."

**When BF16 can still struggle:** Precision-sensitive accumulations (layernorm statistics, attention score normalization, large batch reductions) can still degrade with BF16's 7-bit mantissa. This is why autocast routes these operations to FP32 even in BF16 mode.
""")


md(r"""
## 2.3 NVIDIA mixed precision guidance (the engineering perspective)

NVIDIA's developer blog and documentation provide the practitioner's view of mixed precision. The key engineering decomposition:

1. **Tensor Core compute** wants FP16/BF16 inputs. Modern GPUs (Volta+) have dedicated matrix-multiply-accumulate units that operate on 16-bit inputs with FP32 accumulation. These deliver 2–8$\times$ the throughput of FP32 CUDA cores for matmuls, which dominate deep learning compute.

2. **Dimension alignment matters.** Tensor Cores process tiles of fixed size (typically 8 or 16 elements). If your tensor dimensions aren't multiples of 8, you either waste hardware or fall back to slower code paths. This is a practical detail that affects whether you actually see the theoretical speedup.

3. **Memory bandwidth matters as much as compute.** 16-bit formats halve the bytes transferred between GPU memory (HBM) and compute cores (SMs). For memory-bandwidth-bound models (which many transformer models are at inference time), this alone can yield significant speedup even without Tensor Core benefits.

4. **Some ops are numerically sensitive.** NVIDIA's guidance identifies the same ops that PyTorch's autocast routes to FP32: exponentials, logs, softmax, normalization statistics, and large reductions. The rationale is the same: overflow/underflow risk and accumulation error.

The practical outcome is the "few lines of code" AMP recipe. But the engineering reason it works is the numeric analysis in the theory section above.
""")


md(r"""
## 2.4 BF16: "FP32 range, fewer bits of precision"

BF16 was originally designed for Google's TPUs. The design goal was explicit: make it possible to train neural networks in 16-bit formats *without* the underflow problems of FP16.

**The design decision:** Given 16 bits, how should you split them between exponent and mantissa?

- FP16 (IEEE): 5 exponent + 10 mantissa → good precision, but range only $[6 \times 10^{-5}, 6.5 \times 10^4]$.
- BF16 (Google/Intel): 8 exponent + 7 mantissa → FP32-like range $[10^{-38}, 3.4 \times 10^{38}]$, coarser precision.

For training, range wins. Gradients span many orders of magnitude, and losing any of them to underflow is worse than representing them imprecisely. Networks are fundamentally robust to noise (they're trained with stochastic gradient descent after all), but they cannot learn from zero gradients.

**In practice, for many transformer trainings on modern GPUs:**
- BF16 autocast is "drop-in" without loss scaling.
- FP16 autocast typically needs a `GradScaler`.
- Both still benefit from FP32 master weights and FP32 optimizer state.
""")


md(r"""
## 2.5 PyTorch AMP docs: the operator policy is the decoder ring

A common mistake is to treat autocast like a global switch ("my model is now FP16"). But autocast is really a **per-operation dispatch table**:

- Some ops are eligible for lower precision (FP16/BF16). These are the compute-heavy ops where Tensor Cores give speedup.
- Some ops are forced to FP32. These are the numerically sensitive ops where lower precision would cause training failures.
- Some ops promote to the widest input type. These are binary operations where mixed-dtype inputs would be ambiguous.
- Unlisted ops run in whatever dtype they receive (pass-through).

**The key documentation page** is the [PyTorch Autocast Op Reference](https://pytorch.org/docs/stable/amp.html#autocast-op-reference). It lists every CUDA op and its autocast behavior. This is the authoritative answer to "why did this op run in FP32?" — faster and more reliable than any blog post.

**A common debugging pattern:** when AMP training fails, the first thing to check is which ops are running in which dtype. The dtype hooks we build in Section 3 make this visible.
""")


md(r"""
## 2.6 Rajbhandari et al. (2020): ZeRO and optimizer state precision

> arXiv:1910.02054 — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*

While ZeRO is primarily about distributed training, its memory analysis (Section 2) is essential for understanding where mixed precision fits.

**The memory breakdown for a model with $\Psi$ parameters trained with Adam in mixed precision:**

| Component | Dtype | Bytes per parameter |
|---|---|---|
| FP16 parameters (forward/backward) | FP16 | 2 |
| FP16 gradients | FP16 | 2 |
| FP32 master weights | FP32 | 4 |
| FP32 Adam first moment ($m$) | FP32 | 4 |
| FP32 Adam second moment ($v$) | FP32 | 4 |
| **Total** | | **16 bytes/param** |

For a 7B parameter model: $7 \times 10^9 \times 16 = 112$ GB just for parameters + optimizer state — before any activations.

**Key insight:** Mixed precision doesn't eliminate FP32 from the system. It moves FP32 to where it's needed (optimizer state, master weights) and uses 16-bit where it's safe (activations, compute). The memory savings come primarily from activations (which scale with batch size and sequence length), not from the fixed-size parameter/optimizer memory.

**Practical takeaway:** "AMP" in a distributed stack is not only about autocast during forward/backward. It is about **where each tensor lives and in what dtype** across the entire training loop: parameters, gradients, optimizer state, communication buffers, and activation checkpoints.
""")


md(r"""
## 2.7 LLM training stacks (FSDP / ZeRO / DeepSpeed): where precision choices multiply

At LLM scale, training systems shard parameters, gradients, and optimizer state across multiple GPUs. Mixed precision gets more complicated because dtype decisions affect:

- **Parameter storage:** "working" params in BF16/FP16 for forward/backward; FP32 master copy for updates.
- **Gradient communication:** All-reduce operations can accumulate rounding errors. Some stacks reduce in FP32 for stability, others use BF16 + gradient compression.
- **Optimizer state:** Typically FP32 (Adam moments need high precision over long training). Some frameworks offer FP16 optimizer states as a memory optimization, but this trades stability for memory.
- **Activation checkpointing:** Recomputed activations use the same autocast policy as the original forward pass, but the checkpointing mechanism itself must preserve dtypes correctly.

**FSDP mixed precision** (PyTorch): lets you separately control `param_dtype` (for sharded parameters), `reduce_dtype` (for gradient all-reduce), and `buffer_dtype`. FP16 + scaler or BF16 without scaler.

**DeepSpeed ZeRO:** explicit config knobs. `fp16.loss_scale = 0` enables dynamic loss scaling; `bf16.enabled = true` with no loss scaling.

**The practical lesson:** When debugging AMP issues in distributed training, the dtype of every tensor transfer (parameter broadcast, gradient reduce, optimizer state scatter/gather) is a potential source of numeric problems. This is well beyond the scope of a single `autocast` context manager.
""")


md(r"""
## 2.8 How to think about the autocast policy (conceptual categories)

Autocast decisions fall into a few conceptual buckets. These aren't arbitrary — each traces back to a specific numeric risk:

**1. Lower precision eligible (matmul-like ops)**

These are the ops where Tensor Cores provide speedup and numeric risk is low. The key property: these operations multiply pairs of values and accumulate in FP32 (on Tensor Cores), so the inputs can be 16-bit without precision loss in the output.

**2. Force FP32 (numerically sensitive ops)**

Two sub-categories:
- **Overflow/underflow risk:** `exp`, `log`, `softmax`, `log_softmax`. These can produce extreme magnitudes.
- **Accumulation risk:** `sum`, `prod`, `layer_norm`, `batch_norm`, `cross_entropy`, `mse_loss`. These reduce many values and small rounding errors compound.

**3. Promote to widest (binary ops with mixed inputs)**

If you add a BF16 tensor to an FP32 tensor, the result is FP32. This prevents silent precision loss when autocast and non-autocast regions interact.

**4. Pass-through (element-wise, no numeric risk)**

`relu`, `sigmoid`, `tanh`, `dropout`, `max`, `min`, `mean`. These just transform individual values without extreme magnitudes or accumulation. Whatever goes in comes out.

**Practitioner's two rules:**
- If an op creates very large/small magnitudes ($e^x$, $\log x$, softmax), it needs FP32.
- If an op reduces many values ($\sum$, variance, cross-entropy), it needs FP32 accumulation.
""")


md(r"""
## 2.9 Dynamic loss scaling: what `GradScaler` is doing

Loss scaling multiplies the scalar loss by a factor $S$ before calling `backward()`.

Since backpropagation is linear (gradients are proportional to the loss), every gradient in the network is multiplied by $S$. This shifts the entire gradient distribution toward larger magnitudes, rescuing values that would have underflowed to zero in FP16.

**The protocol:**

1. Compute loss normally under autocast.
2. Multiply loss by $S$ and call `backward()`.
3. Before the optimizer step, **unscale** all gradients by dividing by $S$.
4. Check for `inf`/`nan` in gradients:
   - If found: **skip** the optimizer step, reduce $S$ (usually halve it).
   - If not found: proceed with the step. Periodically increase $S$ (e.g., double it every $N$ successful steps) to maximize headroom.

**Why it's safe:** scaling and unscaling cancel out exactly if no overflow occurs. The optimizer sees the same gradients it would have without scaling.

**Critical detail:** gradient clipping must happen **after** unscaling. PyTorch's `scaler.unscale_(optimizer)` does the unscale explicitly; `scaler.step(optimizer)` then checks for inf/nan before calling `optimizer.step()`.

**Why BF16 doesn't need this:** BF16 has the same exponent range as FP32. Gradients that would be $10^{-10}$ in FP32 are also representable in BF16 (though with less precision). No underflow → no need for scaling.
""")

md(r"""
## 2.10 Gupta et al. (2015): *Deep Learning with Limited Numerical Precision*

This earlier line of work is worth knowing because it frames low-precision training as a **numerical analysis problem**, not a hardware trick.

**The core idea:** training can succeed even when weights/activations/gradients are represented in low precision, *if you control rounding and scaling*.

Key takeaways often cited from this family of results:
- **Rounding is the enemy.** Deterministic rounding can systematically bias updates. Stochastic rounding can remove that bias at the cost of noise.
- **Scaling matters.** Keeping values in a representable dynamic range is the difference between "noisy training" and "dead training" (underflow to zeros).
- **Noise tolerance is real.** SGD is already noisy; in many regimes, extra quantization noise is tolerable *as long as the signal is not destroyed*.

**How it maps to AMP/autocast:**
- Autocast is a modern, practical version of "do low precision where it's safe".
- Loss scaling is a specialized scaling strategy focused on preserving *gradient* signal in FP16.
- The accumulation demos in Section 1 (tiny increments and LayerNorm stats) are the same failure modes this literature is trying to avoid.
""")


md(r"""
## 2.11 FP8 for deep learning: why it exists and why it's not just "AMP but smaller"

FP8 is attractive because it can further reduce memory bandwidth and increase tensor-core throughput compared to FP16/BF16.

But FP8 is fundamentally different from FP16/BF16:
- the representable grid is *much* coarser
- range depends strongly on the FP8 variant (commonly summarized as **E4M3** vs **E5M2**)
- most practical FP8 training systems rely on **explicit scaling** (per-tensor/per-channel) and carefully chosen accumulation dtypes

**The mental model:** FP8 compute works when you pair it with an explicit scale factor that keeps values near the "sweet spot" of the format. This makes FP8 closer to "learned quantization with dynamic scaling" than to the drop-in nature of BF16 autocast.

**How it maps to AMP/autocast:**
- AMP is primarily about FP16/BF16 (and TF32) policies inside a framework like PyTorch.
- FP8 usually requires a specialized kernel stack (often beyond the default `torch.amp.autocast`) that manages scaling metadata alongside tensors.
""")


md(r"""
## 2.12 8-bit optimizer-state work: the next bottleneck after AMP

Classic AMP reduces activation memory and speeds up matmuls, but for large models the **optimizer state** becomes a dominant fixed cost (ZeRO's 16 bytes/parameter for Adam mixed precision).

This motivates a separate line of work: compressing the **optimizer state** (and sometimes gradients) to 8-bit representations while preserving training quality.

**How it relates to autocast:**
- Autocast is about *compute dtype choice per op* during forward/loss (and indirectly backward).
- 8-bit optimizers are about *storage precision* and *update math* for long-lived optimizer tensors.

They are complementary: you can use AMP for activations/compute and (in some stacks) use 8-bit optimizer states to reduce memory.
""")


md(r"""
## Section 2 summary

| Paper / Source | Key contribution |
|---|---|
| **Gupta et al.** | Limited precision can work when rounding/scaling are designed, not ignored |
| **Micikevicius et al.** | The three-part recipe: per-op policies + loss scaling + FP32 master weights |
| **Kalamkar et al.** | BF16's 8-bit exponent matches FP32 → no underflow → no loss scaling needed |
| **NVIDIA guidance** | Engineering view: Tensor Cores, dimension alignment, bandwidth savings |
| **PyTorch AMP docs** | The per-op dispatch table (the "decoder ring" for debugging) |
| **ZeRO (Rajbhandari et al.)** | Memory breakdown: 16 bytes/param with Adam mixed precision |
| **Distributed stacks** | Precision applies to params, grads, optimizer state, and communication separately |
| **FP8 literature** | FP8 needs explicit scaling + specialized kernels; not a drop-in autocast dtype |
| **8-bit optimizers** | Optimizer-state precision is the next memory target after AMP |

**The single most useful reference for debugging:** the [PyTorch Autocast Op Reference](https://pytorch.org/docs/stable/amp.html#autocast-op-reference). Bookmark it.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PRACTICALITIES
# ═══════════════════════════════════════════════════════════════════════════════

md(r"""
# Section 3 — Practicalities (experiments + graphs)

This section is where we turn everything into measurements.

**Principles:**
- Prefer experiments that are **small, fast, and explain a single idea**.
- Log everything you might need for debugging (loss, grad norms, scaler scale, step time, NaN/inf).
- Make results comparable across dtypes.
""")


md(r"""
## 3.0 Controlling confounders (TF32, randomness, and fair comparisons)

When comparing FP32 vs AMP, you can accidentally compare the wrong thing:

1. **TF32 on Ampere+:** Many FP32 matmuls use TF32 internally (10-bit mantissa). Your "FP32 baseline" may not be strict FP32 precision.
2. **Randomness:** Dropout, data sampling, and nondeterministic kernels introduce run-to-run variance.
""")

code(r"""
if device.type == "cuda":
    print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)
    print("TF32 cuDNN:", torch.backends.cudnn.allow_tf32)

    # Uncomment to disable TF32 for strict FP32 comparisons:
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
else:
    print("CUDA not available; TF32 not applicable")
""")


# ── 3.1 Progressive mixed-precision implementation ──────────────────────────

md(r"""
## 3.1 Building mixed precision from scratch (the progressive implementation)

This is the most important experiment in the notebook. Instead of jumping straight to `torch.amp`, we will:

1. Train a simple model in **FP32** (baseline).
2. Try **naive FP16** (just cast everything to half) — watch it fail or stagnate.
3. Add **FP32 master weights** — fix the stagnation.
4. Add **manual loss scaling** — fix gradient underflow.
5. Replace everything with **PyTorch AMP** — the clean two-line version.

By building each piece manually, the "magic" of AMP becomes completely transparent.

We'll use a small MLP on a synthetic regression task to keep things fast and focused. The model is intentionally simple — the numeric effects are the same at any scale.
""")

code(r"""
# Progressive mixed-precision: shared setup

set_seed(42)

# Synthetic dataset: regression
N_SAMPLES = 4096
INPUT_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 1

X_data = torch.randn(N_SAMPLES, INPUT_DIM, device=device)
# Target: a noisy linear function (so the model CAN learn it)
true_w = torch.randn(INPUT_DIM, OUTPUT_DIM, device=device) * 0.01
Y_data = X_data @ true_w + torch.randn(N_SAMPLES, OUTPUT_DIM, device=device) * 0.01

BATCH_SIZE = 256
LR = 1e-3
PROG_STEPS = 300

def get_batches(steps):
    for _ in range(steps):
        idx = torch.randint(0, N_SAMPLES, (BATCH_SIZE,))
        yield X_data[idx], Y_data[idx]

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def grad_stats(model):
    # Return fraction of zero gradients and median |grad|.
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().float().flatten())
    if not grads:
        return 0.0, 0.0
    g = torch.cat(grads)
    zero_frac = float((g == 0).float().mean())
    median_abs = float(g.abs().median())
    return zero_frac, median_abs

print(f"Dataset: {N_SAMPLES} samples, input_dim={INPUT_DIM}")
print(f"Model: MLP {INPUT_DIM} -> {HIDDEN_DIM} -> {HIDDEN_DIM} -> {OUTPUT_DIM}")
print(f"Training: {PROG_STEPS} steps, batch_size={BATCH_SIZE}, lr={LR}")
""")


md(r"""
### 3.1.1 Step 0: FP32 baseline

Everything in FP32. This is our reference for correct training behavior.
""")

code(r"""
# Step 0: FP32 baseline
set_seed(42)
model_fp32 = SimpleMLP().to(device).float()
opt_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=LR)

log_fp32 = {"loss": [], "zero_grad_frac": [], "median_grad": []}

for xb, yb in get_batches(PROG_STEPS):
    opt_fp32.zero_grad(set_to_none=True)
    pred = model_fp32(xb)
    loss = F.mse_loss(pred, yb)
    loss.backward()
    zf, mg = grad_stats(model_fp32)
    opt_fp32.step()
    log_fp32["loss"].append(float(loss))
    log_fp32["zero_grad_frac"].append(zf)
    log_fp32["median_grad"].append(mg)

print(f"FP32 baseline: final loss = {log_fp32['loss'][-1]:.6f}")
print(f"  Zero-grad fraction: {log_fp32['zero_grad_frac'][-1]:.4f}")
print(f"  Median |grad|: {log_fp32['median_grad'][-1]:.2e}")
""")


md(r"""
### 3.1.2 Step 1: Naive FP16 (just cast everything to half)

The simplest approach: `model.half()` and cast inputs to FP16. No master weights, no scaling, no autocast.

This is what the Micikevicius paper warns against. Expect problems.
""")

code(r"""
# Step 1: Naive FP16
log_fp16_naive = {"loss": [], "zero_grad_frac": [], "median_grad": [], "status": "ok"}

if device.type != "cuda":
    log_fp16_naive["status"] = "skipped_no_cuda"
    print("Skipping naive FP16 demo: CPU/MPS fp16 matmuls are often unsupported or unrepresentative.")
else:
    set_seed(42)
    model_fp16_naive = SimpleMLP().to(device).half()
    opt_fp16_naive = torch.optim.Adam(model_fp16_naive.parameters(), lr=LR)

    for xb, yb in get_batches(PROG_STEPS):
        opt_fp16_naive.zero_grad(set_to_none=True)
        pred = model_fp16_naive(xb.half())
        loss = F.mse_loss(pred, yb.half())
        if not torch.isfinite(loss):
            log_fp16_naive["status"] = "non_finite_loss"
            break
        loss.backward()
        zf, mg = grad_stats(model_fp16_naive)
        opt_fp16_naive.step()
        log_fp16_naive["loss"].append(float(loss))
        log_fp16_naive["zero_grad_frac"].append(zf)
        log_fp16_naive["median_grad"].append(mg)

    print(f"Naive FP16: status = {log_fp16_naive['status']}")
    if log_fp16_naive["loss"]:
        print(f"  Final loss = {log_fp16_naive['loss'][-1]:.6f}")
        print(f"  Zero-grad fraction: {log_fp16_naive['zero_grad_frac'][-1]:.4f}")
        print(f"  Median |grad|: {log_fp16_naive['median_grad'][-1]:.2e}")
    else:
        print("  Training failed immediately.")

    print()
    if log_fp16_naive["status"] != "ok" or (log_fp16_naive["loss"] and log_fp16_naive["loss"][-1] > log_fp32["loss"][-1] * 5):
        print("As expected, naive FP16 is problematic. The combination of:")
        print("  1. Weight update stagnation (updates below FP16 ULP)")
        print("  2. Gradient underflow (small gradients become zero)")
        print("makes training unstable or ineffective.")
    else:
        print("Naive FP16 happened to work for this model (it sometimes does for simple/small models).")
        print("This does NOT mean it's safe in general.")
""")


md(r"""
### 3.1.2b Step 1b: Naive BF16 (the same cast-everything approach, but with BF16)

Now do the *exact same thing* but with BF16 instead of FP16. This is the key comparison that shows **why BF16's range matters more than FP16's precision** for training stability.

Same code, same model, same learning rate — just a different 16-bit format.
""")

code(r"""
# Step 1b: Naive BF16
log_bf16_naive = {"loss": [], "zero_grad_frac": [], "median_grad": [], "status": "ok"}

if not supports_dtype_on_device(torch.bfloat16, device):
    log_bf16_naive["status"] = "skipped_no_bf16"
    print("Skipping naive BF16 demo: BF16 not supported on this device.")
else:
    set_seed(42)
    model_bf16_naive = SimpleMLP().to(device).to(torch.bfloat16)
    opt_bf16_naive = torch.optim.Adam(model_bf16_naive.parameters(), lr=LR)

    for xb, yb in get_batches(PROG_STEPS):
        opt_bf16_naive.zero_grad(set_to_none=True)
        pred = model_bf16_naive(xb.to(torch.bfloat16))
        loss = F.mse_loss(pred, yb.to(torch.bfloat16))
        if not torch.isfinite(loss):
            log_bf16_naive["status"] = "non_finite_loss"
            break
        loss.backward()
        zf, mg = grad_stats(model_bf16_naive)
        opt_bf16_naive.step()
        log_bf16_naive["loss"].append(float(loss))
        log_bf16_naive["zero_grad_frac"].append(zf)
        log_bf16_naive["median_grad"].append(mg)

    print(f"Naive BF16: status = {log_bf16_naive['status']}")
    if log_bf16_naive["loss"]:
        print(f"  Final loss = {log_bf16_naive['loss'][-1]:.6f}")
        print(f"  Zero-grad fraction: {log_bf16_naive['zero_grad_frac'][-1]:.4f}")
        print(f"  Median |grad|: {log_bf16_naive['median_grad'][-1]:.2e}")

    print()
    if log_bf16_naive["status"] == "ok" and log_bf16_naive["loss"]:
        if log_bf16_naive["loss"][-1] < log_fp32["loss"][-1] * 2:
            print("BF16 naive training WORKS! This is the key insight:")
            print("  - BF16 has the same exponent range as FP32 (8-bit exponent)")
            print("  - Gradients don't underflow, so training progresses even without loss scaling")
            print("  - The coarser precision (7-bit mantissa) introduces noise, but SGD tolerates noise")
            print("  - Compare this to FP16 naive above: same approach, different outcome, entirely due to range.")
        else:
            print("BF16 naive converged but to a higher loss — precision may have limited final accuracy.")
""")


md(r"""
### 3.1.3 Step 2: Add FP32 master weights

The first fix from the Micikevicius paper: keep an FP32 copy of parameters for the optimizer update.

**The flow:**
1. Forward pass uses FP16 parameters (for speed/memory).
2. Backward produces FP16 gradients.
3. Copy FP16 gradients to FP32 master parameters.
4. Optimizer updates FP32 master weights.
5. Copy updated FP32 weights back to FP16 model.
""")

code(r"""
# Step 2: FP16 + FP32 master weights
log_master = {"loss": [], "zero_grad_frac": [], "median_grad": [], "status": "ok"}

if device.type != "cuda":
    log_master["status"] = "skipped_no_cuda"
    print("Skipping FP16 master-weights demo: intended for CUDA FP16 behavior.")
else:
    set_seed(42)
    model_master = SimpleMLP().to(device).half()  # FP16 for forward/backward

    # Create FP32 master copy
    master_params = [p.detach().clone().float().requires_grad_(True) for p in model_master.parameters()]
    opt_master = torch.optim.Adam(master_params, lr=LR)

    for xb, yb in get_batches(PROG_STEPS):
        opt_master.zero_grad(set_to_none=True)
        model_master.zero_grad(set_to_none=True)

        # Forward in FP16
        pred = model_master(xb.half())
        loss = F.mse_loss(pred, yb.half())
        if not torch.isfinite(loss):
            log_master["status"] = "non_finite_loss"
            break
        loss.backward()

        # Copy FP16 grads -> FP32 master params
        for mp, p16 in zip(master_params, model_master.parameters()):
            if p16.grad is not None:
                mp.grad = p16.grad.float()

        zf, mg = grad_stats(model_master)

        # Update FP32 master weights
        opt_master.step()

        # Copy FP32 master -> FP16 model
        for mp, p16 in zip(master_params, model_master.parameters()):
            p16.data.copy_(mp.data.half())

        log_master["loss"].append(float(loss))
        log_master["zero_grad_frac"].append(zf)
        log_master["median_grad"].append(mg)

    print(f"FP16 + master weights: status = {log_master['status']}")
    if log_master["loss"]:
        print(f"  Final loss = {log_master['loss'][-1]:.6f}")
        print(f"  Zero-grad fraction: {log_master['zero_grad_frac'][-1]:.4f}")
""")


md(r"""
### 3.1.4 Step 3: Add loss scaling

The second fix: multiply the loss by a scale factor $S$ before backward, then unscale gradients before the optimizer step. This prevents small gradients from underflowing to zero in FP16.
""")

code(r"""
# Step 3: FP16 + FP32 master weights + loss scaling
log_scaled = {"loss": [], "zero_grad_frac": [], "median_grad": [], "status": "ok"}

if device.type != "cuda":
    log_scaled["status"] = "skipped_no_cuda"
    print("Skipping FP16 loss-scaling demo: intended for CUDA FP16 behavior.")
else:
    set_seed(42)
    model_scaled = SimpleMLP().to(device).half()
    master_params_s = [p.detach().clone().float().requires_grad_(True) for p in model_scaled.parameters()]
    opt_scaled = torch.optim.Adam(master_params_s, lr=LR)

    LOSS_SCALE = 2**13  # 8192 — a common starting point

    for xb, yb in get_batches(PROG_STEPS):
        opt_scaled.zero_grad(set_to_none=True)
        model_scaled.zero_grad(set_to_none=True)

        pred = model_scaled(xb.half())
        loss = F.mse_loss(pred, yb.half())
        if not torch.isfinite(loss):
            log_scaled["status"] = "non_finite_loss"
            break

        # Scale loss before backward
        (loss * LOSS_SCALE).backward()

        # Copy FP16 grads -> FP32 master, then UNSCALE
        for mp, p16 in zip(master_params_s, model_scaled.parameters()):
            if p16.grad is not None:
                mp.grad = p16.grad.float() / LOSS_SCALE

        zf, mg = grad_stats(model_scaled)

        opt_scaled.step()
        for mp, p16 in zip(master_params_s, model_scaled.parameters()):
            p16.data.copy_(mp.data.half())

        log_scaled["loss"].append(float(loss))
        log_scaled["zero_grad_frac"].append(zf)
        log_scaled["median_grad"].append(mg)

    print(f"FP16 + master weights + scaling: status = {log_scaled['status']}")
    if log_scaled["loss"]:
        print(f"  Final loss = {log_scaled['loss'][-1]:.6f}")
        print(f"  Zero-grad fraction: {log_scaled['zero_grad_frac'][-1]:.4f}")
""")


md(r"""
### 3.1.5 Step 4: PyTorch AMP (the clean version)

Now replace all the manual work with PyTorch's `autocast` + `GradScaler`. Two extra lines of code.

Note: autocast handles per-op dtype policies automatically. The model stays in FP32, and autocast temporarily casts operations during the forward pass.
""")

code(r"""
# Step 4: PyTorch AMP
set_seed(42)
model_amp = SimpleMLP().to(device).float()  # FP32 — autocast handles casting
opt_amp = torch.optim.Adam(model_amp.parameters(), lr=LR)

if device.type == "cuda":
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
elif device.type == "cpu":
    amp_dtype = torch.bfloat16
    use_scaler = False
else:  # mps (experimental AMP coverage)
    amp_dtype = torch.float16
    use_scaler = False

scaler = GradScaler(enabled=(use_scaler and device.type == "cuda")) if use_scaler else None

log_amp = {"loss": [], "zero_grad_frac": [], "median_grad": [], "status": "ok", "dtype": str(amp_dtype)}

for xb, yb in get_batches(PROG_STEPS):
    opt_amp.zero_grad(set_to_none=True)

    with amp_autocast(device, amp_dtype, enabled=True):
        pred = model_amp(xb)
        loss = F.mse_loss(pred, yb)

    if not torch.isfinite(loss):
        log_amp["status"] = "non_finite_loss"
        break

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(opt_amp)
        scaler.update()
    else:
        loss.backward()
        opt_amp.step()

    zf, mg = grad_stats(model_amp)
    log_amp["loss"].append(float(loss))
    log_amp["zero_grad_frac"].append(zf)
    log_amp["median_grad"].append(mg)

print(f"PyTorch AMP ({log_amp['dtype']}): status = {log_amp['status']}")
if log_amp["loss"]:
    print(f"  Final loss = {log_amp['loss'][-1]:.6f}")
    print(f"  Zero-grad fraction: {log_amp['zero_grad_frac'][-1]:.4f}")
""")

code(r"""
# Compare all progressive approaches

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

configs = [
    ("FP32 baseline", log_fp32, "C0"),
    ("Naive FP16", log_fp16_naive, "C3"),
    ("Naive BF16", log_bf16_naive, "C5"),
    ("FP16 + master wts", log_master, "C1"),
    ("FP16 + master + scaling", log_scaled, "C4"),
    (f"PyTorch AMP ({log_amp['dtype']})", log_amp, "C2"),
]

# Loss curves
for name, log, color in configs:
    if log["loss"]:
        axes[0].plot(log["loss"], label=name, color=color, alpha=0.8)
axes[0].set_title("Training loss")
axes[0].set_xlabel("step")
axes[0].set_ylabel("MSE loss")
axes[0].set_yscale("log")
axes[0].legend(fontsize=7)

# Zero gradient fraction
for name, log, color in configs:
    if log["zero_grad_frac"]:
        axes[1].plot(log["zero_grad_frac"], label=name, color=color, alpha=0.8)
axes[1].set_title("Fraction of zero gradients")
axes[1].set_xlabel("step")
axes[1].set_ylabel("fraction")
axes[1].legend(fontsize=7)

# Median gradient magnitude
for name, log, color in configs:
    if log["median_grad"]:
        axes[2].plot(log["median_grad"], label=name, color=color, alpha=0.8)
axes[2].set_title("Median |gradient|")
axes[2].set_xlabel("step")
axes[2].set_ylabel("|grad|")
axes[2].set_yscale("log")
axes[2].legend(fontsize=7)

fig.suptitle("Progressive Mixed Precision: from naive FP16 to PyTorch AMP", fontsize=12, y=1.02)
plt.tight_layout();
""")


md(r"""
### 3.1.6 What to observe

**Loss curves:**
- FP32 baseline should decrease smoothly.
- Naive FP16 may diverge, stagnate, or produce NaN.
- **Naive BF16 often trains successfully** — this is the dramatic demonstration of range vs precision. Same "cast everything to 16-bit" approach, but BF16's 8-bit exponent prevents the gradient underflow that kills FP16.
- Adding master weights to FP16 typically helps convergence.
- Adding loss scaling to FP16 reduces the fraction of zero gradients.
- PyTorch AMP should match or beat the manual implementations.

**Zero gradient fraction:**
- In naive FP16, many gradients may be exactly zero (underflow).
- In naive BF16, almost no gradients are zero — the exponent range is wide enough.
- Loss scaling shifts the FP16 distribution, rescuing underflowed gradients.

**Median gradient magnitude:**
- Shows the "signal strength" available to the optimizer.
- If it drops to zero, the model stops learning.
- Compare FP16 vs BF16: even with coarser mantissa, BF16 preserves gradient *signal*.

**The key lesson from this progressive build-up:**
Naive FP16 fails. BF16 naive often works. But both benefit from FP32 master weights and proper AMP policies. The Micikevicius paper's three-part recipe (per-op policy + loss scaling + master weights) handles FP16 correctly. BF16 simplifies the picture by removing the need for loss scaling, but master weights and per-op policies still help.
""")


# ── 3.2 Operator policy probe ───────────────────────────────────────────────

md(r"""
## 3.2 Build an "autocast operator policy table" from your local PyTorch

Instead of trusting a static table from the internet, we can probe your exact PyTorch version:
1. Run each op with autocast **disabled** → observe output dtype.
2. Run each op with autocast **enabled** → observe output dtype.
3. Compare to see which ops autocast changes.
""")

code(r"""
# Enhanced operator policy probe

def probe_ops(dev, amp_dtype):
    a32 = torch.randn(128, 128, device=dev, dtype=torch.float32)
    b32 = torch.randn(128, 128, device=dev, dtype=torch.float32)
    x128 = torch.randn(2, 128, device=dev, dtype=torch.float32)
    x_big = torch.randn(2, 20000, device=dev, dtype=torch.float32)
    w32 = torch.randn(128, 128, device=dev, dtype=torch.float32)
    bias32 = torch.randn(128, device=dev, dtype=torch.float32)
    target32 = torch.randn(128, 128, device=dev, dtype=torch.float32)
    x16 = x_big.to(amp_dtype)

    ops = {
        # Matmul-like (expect lower precision)
        "matmul (a @ b)": lambda: a32 @ b32,
        "F.linear": lambda: F.linear(a32, w32, bias32),
        # Numerically sensitive (expect FP32)
        "F.softmax": lambda: F.softmax(x128, dim=-1),
        "F.layer_norm": lambda: F.layer_norm(x128, [x128.size(-1)]),
        "F.cross_entropy": lambda: F.cross_entropy(a32[:10], torch.randint(0, 128, (10,), device=dev)),
        "F.mse_loss": lambda: F.mse_loss(a32, target32),
        "torch.exp": lambda: torch.exp(x128),
        "torch.log": lambda: torch.log(x128.abs() + 1e-6),
        "torch.sum": lambda: x_big.sum(),
        "torch.prod": lambda: x_big[:, :10].prod(),
        # Pass-through (expect input dtype)
        "F.relu": lambda: F.relu(x_big),
        "torch.max": lambda: x_big.max(),
        "torch.min": lambda: x_big.min(),
        "torch.mean": lambda: x_big.mean(),
        "F.dropout": lambda: F.dropout(x_big, p=0.0, training=True),
    }

    rows = []
    for name, fn in ops.items():
        # Without autocast
        y_no = fn()
        dt_no = str(y_no.dtype) if isinstance(y_no, torch.Tensor) else "n/a"
        # With autocast (FP32 inputs)
        with amp_autocast(dev, amp_dtype, enabled=True):
            y_ac = fn()
        dt_ac = str(y_ac.dtype) if isinstance(y_ac, torch.Tensor) else "n/a"

        # With autocast (16-bit inputs, where applicable)
        dt_ac16 = ""
        try:
            # Replace x32 temporarily
            ops_16 = {
                "F.relu": lambda: F.relu(x16),
                "torch.max": lambda: x16.max(),
                "torch.min": lambda: x16.min(),
                "torch.mean": lambda: x16.mean(),
                "torch.sum": lambda: x16.sum(),
            }
            if name in ops_16:
                with amp_autocast(dev, amp_dtype, enabled=True):
                    y16 = ops_16[name]()
                dt_ac16 = str(y16.dtype)
        except Exception:
            pass

        changed = dt_no != dt_ac
        rows.append({
            "op": name,
            "no_autocast": dt_no,
            f"autocast({amp_dtype})_fp32_input": dt_ac,
            "16bit_input": dt_ac16 if dt_ac16 else "-",
            "policy": "LOWER PREC" if "float16" in dt_ac or "bfloat16" in dt_ac else ("FP32" if changed or dt_ac == "torch.float32" else "pass-through"),
        })

    return pd.DataFrame(rows)

for amp_dt in [torch.float16, torch.bfloat16]:
    if device.type == "cuda" and amp_dt is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        continue
    if device.type == "cpu" and amp_dt is torch.float16:
        continue
    if not supports_dtype_on_device(amp_dt, device):
        continue
    print(f"\n=== Autocast policy with amp_dtype={amp_dt} ===")
    display(probe_ops(device, amp_dt))
""")


md(r"""
### 3.2.1 The "sum vs mean" mystery

Under autocast, `sum` produces FP32 output but `mean` stays in the input dtype. Why?

**`sum`** is a reduction that accumulates values. Summing 20,000 BF16 values can easily exceed the representable range (BF16 max ~ $3.4 \times 10^{38}$, but even FP16 max is only ~65,504). More critically, the rounding errors from adding many small values compound. PyTorch hardcodes `sum` to promote to FP32.

**`mean`** inherently divides by $N$, keeping the output bounded between the min and max of the input. It cannot overflow by accumulation. PyTorch treats it as a pass-through — whatever dtype goes in, the same comes out.

**`prod`** is similar to `sum` but even more extreme: multiplying many values can grow (or shrink) astronomically. Also forced to FP32.

Let's verify this directly.
""")

code(r"""
# The sum vs mean mystery — direct probe

if device.type == "cuda":
    dtype_16bit = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    x16 = torch.randn(2, 20000, device=device, dtype=dtype_16bit)

    with torch.autocast(device_type="cuda", dtype=dtype_16bit):
        results = {
            "max": x16.max().dtype,
            "min": x16.min().dtype,
            "sum": x16.sum().dtype,
            "mean": x16.mean().dtype,
            "prod": x16.prod().dtype,
            "exp": x16.exp().dtype,
            "log": x16.abs().log().dtype,
        }

    rows = []
    for op, dt in results.items():
        rows.append({
            "op": op,
            "output_dtype": str(dt),
            "promoted_to_fp32": "YES" if dt == torch.float32 else "no",
            "reason": {
                "max": "element-wise selection, no accumulation",
                "min": "element-wise selection, no accumulation",
                "sum": "ACCUMULATION: rounding errors compound over N values",
                "mean": "bounded output (divides by N), no overflow risk",
                "prod": "ACCUMULATION: multiplicative explosion/collapse",
                "exp": "can produce extreme magnitudes (overflow risk)",
                "log": "can produce extreme magnitudes (underflow risk)",
            }.get(op, ""),
        })

    display(pd.DataFrame(rows))
else:
    print("Run on CUDA to see the sum vs mean mystery in action.")
""")


# ── 3.3 Dtype flow through a transformer ────────────────────────────────────

md(r"""
## 3.3 Watch dtype flow through a transformer (4 configurations)

This is the "visceral" version of the operator policy: instead of probing individual ops, we observe dtypes flowing through a real transformer model.

We'll test all four combinations from the toy example (source1):

| Config | Model params | Autocast | What to observe |
|---|---|---|---|
| A | FP32 | OFF | Everything FP32 (baseline) |
| B | FP16/BF16 | OFF | Everything 16-bit (no policy) |
| C | FP16/BF16 | ON | LayerNorm→FP32, Linear→16-bit, residuals→16-bit |
| D | FP32 | ON | Linear→16-bit even with FP32 weights! LayerNorm stays FP32, residuals→FP32 (promotion) |
""")

code(r"""
# Tiny transformer model for dtype tracing

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, dropout=0.0):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=2, n_embd=128, n_heads=4, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_heads, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

print("TinyGPT defined: 2-layer transformer for dtype tracing and training experiments.")
""")

code(r"""
# Dtype hooks + 4-configuration trace

def install_dtype_hooks(model, watch=(nn.Linear, nn.LayerNorm, nn.Embedding)):
    hooks, records = [], []
    def make_hook(name):
        def hook(m, inp, out):
            def dt(x):
                return str(x.dtype) if isinstance(x, torch.Tensor) else type(x).__name__
            indt = dt(inp[0]) if isinstance(inp, (tuple, list)) and inp else dt(inp)
            oudt = dt(out) if not isinstance(out, (tuple, list)) else dt(out[0])
            records.append({"module": name, "type": type(m).__name__, "in_dtype": indt, "out_dtype": oudt})
        return hook
    for name, m in model.named_modules():
        if isinstance(m, watch):
            hooks.append(m.register_forward_hook(make_hook(name)))
    return hooks, records

VOCAB = 128
BLOCK = 64
idx = torch.randint(0, VOCAB, (2, BLOCK), device=device)

dtype_16 = torch.bfloat16
if device.type == "cuda" and not torch.cuda.is_bf16_supported():
    dtype_16 = torch.float16
elif device.type == "mps":
    dtype_16 = torch.float16

configs = [
    ("A: params=FP32, autocast=OFF", torch.float32, False),
    ("B: params=16bit, autocast=OFF", dtype_16, False),
    ("C: params=16bit, autocast=ON",  dtype_16, True),
    ("D: params=FP32, autocast=ON",   torch.float32, True),
]

for title, param_dt, use_ac in configs:
    model = TinyGPT(VOCAB, BLOCK).to(device).to(param_dt)
    hooks, rec = install_dtype_hooks(model)
    ctx = amp_autocast(device, dtype_16, enabled=use_ac)
    with torch.inference_mode(), ctx:
        _ = model(idx)
    for h in hooks:
        h.remove()
    df = pd.DataFrame(rec)
    df["count"] = 1
    summary = df.groupby(["type", "in_dtype", "out_dtype"], as_index=False)["count"].sum()
    summary = summary.sort_values(["type", "in_dtype", "out_dtype"])
    print(f"\n--- {title} ---")
    display(summary)
""")


md(r"""
### 3.3.1 What the traces reveal

**Config A (FP32, no autocast):** Everything is FP32. Baseline.

**Config B (16-bit, no autocast):** Everything is 16-bit. No per-op policy. LayerNorm runs in 16-bit (risky for accumulation). Softmax runs in 16-bit (risky for overflow).

**Config C (16-bit params, autocast ON):**
- LayerNorm: input is 16-bit → **output is FP32** (autocast forces FP32 for normalization)
- Linear after LayerNorm: **input is FP32 → output is 16-bit** (autocast casts FP32 weights/inputs to 16-bit for the matmul)
- This is the key insight: autocast doesn't just "use 16-bit everywhere." It routes different ops to different precisions.

**Config D (FP32 params, autocast ON):**
- Linear: **FP32 input → 16-bit output** (autocast temporarily casts FP32 weights to 16-bit!)
- LayerNorm: FP32 → FP32 (stays in FP32, as it should)
- Residual adds: 16-bit + FP32 → **FP32** (dtype promotion)
- This is the "standard" AMP configuration: model stays in FP32, autocast handles per-op casting.

```
Config D data flow (FP32 params + autocast):

[int64 tokens]
       |
  +----+---------------------+
  |                          |
[embed_tok] int64->fp32    [embed_pos] int64->fp32
  |                          |
  +-------- sum (fp32+fp32->fp32) --------+
                                          |
                                        [LN1] fp32->fp32
                                          |
                      +---------+---------+---------+
                      |         |                   |
                   [q_proj]  [k_proj]           [v_proj]
                    fp32->bf16  fp32->bf16      fp32->bf16
                      \         |                 /
                       \        |                /
                        +----[attn + softmax]----+   (bf16 compute)
                                          |
                                    [out_proj] bf16->bf16
                                          |
                (residual add: bf16 + fp32 -> fp32)   <- promotion!
                                          |
                                        [LN2] fp32->fp32
                                          |
                                       [fc1] fp32->bf16
                                          |
                                       [fc2] bf16->bf16
                                          |
                (residual add: bf16 + fp32 -> fp32)   <- promotion!
```

The promotion at residual connections is a key feature: it prevents precision loss from accumulating through the network's residual stream.
""")


# ── 3.4 Gradient underflow ──────────────────────────────────────────────────

md(r"""
## 3.4 Gradient underflow and why loss scaling works

We'll do a controlled experiment:
1. Create a synthetic gradient distribution spanning many orders of magnitude.
2. Cast it to FP16 and count how many values become exactly 0.
3. Apply a scale factor $S$, cast, then unscale.

This shows the core mechanism of loss scaling without needing a full training run.
""")

code(r"""
# Gradient underflow + rescue via scaling

N = 200_000
log10_mag = torch.empty(N).uniform_(-12, 0)  # 1e-12 to 1
sign = torch.randint(0, 2, (N,)) * 2 - 1
synthetic = (10 ** log10_mag) * sign
synthetic = synthetic.to(torch.float32)

rows = []
for S_label, S in [("unscaled (S=1)", 1), ("S=2^10 (1024)", 2**10), ("S=2^13 (8192)", 2**13), ("S=2^16 (65536)", 2**16)]:
    scaled = synthetic * S
    for dt in [torch.float16, torch.bfloat16]:
        g = scaled.to(dt)
        zeros = float((g == 0).float().mean())
        infs = float(torch.isinf(g).float().mean())
        rows.append({
            "scaling": S_label,
            "dtype": str(dt),
            "zero_frac": f"{zeros:.3f}",
            "inf_frac": f"{infs:.4f}",
            "preserved_frac": f"{1 - zeros - infs:.3f}",
        })

pd.DataFrame(rows)
""")

code(r"""
# Visualize gradient distribution vs FP16 thresholds

fi16 = torch.finfo(torch.float16)
min_normal = float(fi16.tiny)
min_sub = float(torch.nextafter(torch.tensor(0.0, dtype=torch.float16), torch.tensor(1.0, dtype=torch.float16)))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

vals = synthetic.abs().cpu().numpy()
axes[0].hist(np.log10(vals + 1e-30), bins=200, alpha=0.7, color="steelblue")
axes[0].axvline(np.log10(min_normal), color="r", ls="--", label=f"FP16 min normal ({min_normal:.1e})")
axes[0].axvline(np.log10(min_sub), color="m", ls=":", label=f"FP16 min subnormal ({min_sub:.1e})")
axes[0].set_title("Unscaled |grad| distribution")
axes[0].set_xlabel("log10(|grad|)")
axes[0].legend(fontsize=8)

# After scaling by 2^13
scaled_vals = (synthetic * 2**13).abs().cpu().numpy()
axes[1].hist(np.log10(scaled_vals + 1e-30), bins=200, alpha=0.7, color="darkorange")
axes[1].axvline(np.log10(min_normal), color="r", ls="--", label="FP16 min normal")
axes[1].axvline(np.log10(float(fi16.max)), color="darkred", ls="-.", label=f"FP16 max ({fi16.max:.0f})")
axes[1].set_title("Scaled by 2^13: distribution shifts right")
axes[1].set_xlabel("log10(|grad| * S)")
axes[1].legend(fontsize=8)

plt.suptitle("Loss scaling shifts gradients into FP16's representable range", fontsize=11, y=1.02)
plt.tight_layout();
""")


md(r"""
### 3.4.1 Underflow in a real backward pass

Now we show the actual training failure mode with a tiny FP16 network.
""")

code(r"""
# Real backward underflow demo

def tiny_backward(use_scaling, scale=2**13):
    model = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(device).half()
    x = (torch.randn(256, 128, device=device) * 1e-3).half()
    y = (torch.randn(256, 1, device=device) * 1e-3).half()
    pred = model(x)
    loss = ((pred - y)**2).mean()
    if use_scaling:
        (loss * scale).backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(scale)
    else:
        loss.backward()
    grads = torch.cat([p.grad.flatten().abs().float() for p in model.parameters() if p.grad is not None])
    return float(loss), float((grads == 0).float().mean()), float(grads.median()), grads

if device.type == "cuda":
    loss0, z0, med0, g0 = tiny_backward(use_scaling=False)
    loss1, z1, med1, g1 = tiny_backward(use_scaling=True)
    display(pd.DataFrame([
        {"setting": "FP16, no scaling", "loss": f"{loss0:.6f}", "zero_grad_frac": f"{z0:.3f}", "median|grad|": f"{med0:.2e}"},
        {"setting": "FP16, scaled+unscaled", "loss": f"{loss1:.6f}", "zero_grad_frac": f"{z1:.3f}", "median|grad|": f"{med1:.2e}"},
    ]))
else:
    print("Run on CUDA for the FP16 backward demo.")
""")


md(r"""
### 3.4.2 The Micikevicius gradient histogram: where do real gradients live?

The Micikevicius paper's most famous figure shows a histogram of gradient magnitudes during FP32 training, overlaid with FP16's representable range. Let's reproduce this analysis with our model.

This visualization answers the question: *What fraction of the training signal would you lose by switching to FP16 or BF16?*
""")

code(r"""
# Gradient histogram analysis (Micikevicius-style)

def collect_gradient_histogram(model_class, device, dtype=torch.float32, steps=20):
    # Collect all gradient values from several training steps.
    set_seed(42)
    model = model_class().to(device).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    all_grads = []

    for xb, yb in get_batches(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(xb.to(dtype))
        loss = F.mse_loss(pred, yb.to(dtype))
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                all_grads.append(p.grad.detach().float().flatten().cpu())
        opt.step()

    return torch.cat(all_grads)

grads_fp32 = collect_gradient_histogram(SimpleMLP, device, torch.float32, steps=30)
nonzero_grads = grads_fp32[grads_fp32 != 0]
log_abs_grads = torch.log10(nonzero_grads.abs()).numpy()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Left: gradient histogram with FP16 thresholds
ax = axes[0]
ax.hist(log_abs_grads, bins=200, alpha=0.7, color="steelblue", edgecolor="none", density=True)

fi16 = torch.finfo(torch.float16)
min_normal_16 = float(fi16.tiny)
min_sub_16 = float(torch.nextafter(torch.tensor(0.0, dtype=torch.float16), torch.tensor(1.0, dtype=torch.float16)))

ax.axvline(np.log10(min_normal_16), color="red", ls="--", lw=2, label=f"FP16 min normal ({min_normal_16:.1e})")
ax.axvline(np.log10(min_sub_16), color="darkred", ls=":", lw=1.5, label=f"FP16 min subnormal ({min_sub_16:.1e})")
ax.axvline(np.log10(float(fi16.max)), color="orange", ls="-.", lw=1.5, label=f"FP16 max ({fi16.max:.0f})")

# Shade the underflow zone
ax.axvspan(ax.get_xlim()[0], np.log10(min_sub_16), alpha=0.15, color="red", label="FP16 underflow zone")

ax.set_title("FP32 gradient magnitudes vs FP16 representable range", fontsize=10)
ax.set_xlabel("log10(|gradient|)")
ax.set_ylabel("density")
ax.legend(fontsize=7, loc="upper left")

# Right: same but with BF16 thresholds
ax = axes[1]
ax.hist(log_abs_grads, bins=200, alpha=0.7, color="steelblue", edgecolor="none", density=True)

fi_bf16 = torch.finfo(torch.bfloat16)
min_normal_bf16 = float(fi_bf16.tiny)

ax.axvline(np.log10(min_normal_bf16), color="green", ls="--", lw=2, label=f"BF16 min normal ({min_normal_bf16:.1e})")
ax.axvline(np.log10(min_normal_16), color="red", ls="--", lw=1.5, alpha=0.5, label=f"FP16 min normal (for comparison)")

ax.set_title("Same gradients vs BF16 representable range", fontsize=10)
ax.set_xlabel("log10(|gradient|)")
ax.set_ylabel("density")
ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Micikevicius-style analysis: gradient magnitude distribution", fontsize=12, y=1.02)
plt.tight_layout()

# Quantify
fp16_underflow_frac = float((nonzero_grads.abs() < min_normal_16).float().mean())
bf16_underflow_frac = float((nonzero_grads.abs() < min_normal_bf16).float().mean())
print(f"\nGradients collected: {len(nonzero_grads):,}")
print(f"Fraction that would underflow in FP16: {fp16_underflow_frac:.4f} ({fp16_underflow_frac*100:.2f}%)")
print(f"Fraction that would underflow in BF16: {bf16_underflow_frac:.4f} ({bf16_underflow_frac*100:.2f}%)")
print(f"\nThis is why FP16 needs loss scaling and BF16 usually doesn't.")
""")

md(r"""
### 3.4.3 Dynamic loss scaling (GradScaler) in action

Loss scaling has two failure modes:

- **Scale too small** → doesn't rescue underflow (gradients still become 0 in FP16).
- **Scale too large** → causes overflow (gradients become `inf`/`nan`).

`GradScaler` automates this tradeoff: it tries to keep the scale as large as possible *without* overflow.

In this demo we intentionally start with an absurdly large scale to trigger overflows, and watch GradScaler back off and skip optimizer steps.
""")

code(r"""
# GradScaler overflow + step skipping demo (CUDA only)

if device.type != "cuda":
    print("Run on CUDA to see GradScaler dynamically adjust scale and skip steps.")
else:
    set_seed(0)
    torch.cuda.synchronize()

    model = nn.Linear(256, 256, bias=False).to(device).float()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Large inputs to create large (but finite) gradients.
    x = (torch.randn(256, 256, device=device) * 5000).float()

    try:
        scaler = GradScaler(
            init_scale=2**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2,
            enabled=True,
        )
    except TypeError:
        scaler = GradScaler(enabled=True)
        print("[warn] GradScaler init args not supported on this version; using defaults.")

    logs = []
    for step in range(12):
        opt.zero_grad(set_to_none=True)
        scale_before = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else float("nan")

        with amp_autocast(device, torch.float16, enabled=True):
            y = model(x)  # FP16 matmul compute under autocast
            # Force loss computation in FP32 to avoid forward overflow; we want overflow from scaling.
            loss = (y.float() ** 2).mean()

        scaler.scale(loss).backward()

        # Unscale so we can inspect gradients in their true magnitude (and clip here if desired).
        scaler.unscale_(opt)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        found_inf = bool(any((~torch.isfinite(g)).any().item() for g in grads))
        max_abs_grad = float(torch.stack([g.detach().abs().max() for g in grads]).max())

        w_before = model.weight.detach().float().clone()
        scaler.step(opt)     # skipped if found_inf=True
        scaler.update()
        w_after = model.weight.detach().float()

        scale_after = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else float("nan")
        step_ran = bool((w_after - w_before).abs().max().item() != 0.0)

        logs.append({
            "step": step,
            "loss(fp32)": float(loss.detach().cpu()),
            "scale_before": scale_before,
            "found_inf": found_inf,
            "optimizer_step_ran": step_ran,
            "max_abs_grad(after_unscale)": max_abs_grad,
            "scale_after": scale_after,
        })

    df = pd.DataFrame(logs)
    display(df)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df["step"], df["scale_before"], marker="o", label="scale_before")
    ax.plot(df["step"], df["scale_after"], marker="o", ls="--", label="scale_after")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("scale (log)")
    ax.set_title("GradScaler: backoff on overflow + step skipping")
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(df["step"], df["found_inf"].astype(int), color="red", alpha=0.3, lw=2, label="found_inf")
    ax2.set_ylabel("found_inf (0/1)")
    plt.tight_layout();
""")


# ── 3.5 Weight update stagnation ────────────────────────────────────────────

md(r"""
## 3.5 Weight update stagnation (why FP32 master weights matter)

Even if you avoid underflow, you can lose learning signal if weight updates are **below the ULP** of the weight's dtype.

If $w \approx 1$ in FP16, the ULP is ~$10^{-3}$. Any update $\Delta w < 10^{-3}$ is rounded away.
""")

code(r"""
# Weight stagnation demo

def apply_updates(dtype, w0=1.0, delta=1e-5, steps=2000):
    w = torch.tensor(w0, dtype=dtype)
    changed = 0
    for _ in range(steps):
        w_new = w - torch.tensor(delta, dtype=dtype)
        changed += int(w_new.item() != w.item())
        w = w_new
    return {
        "dtype": str(dtype),
        "w0": w0,
        "delta": f"{delta:.0e}",
        "steps": steps,
        "steps_where_w_changed": changed,
        "final_w": f"{float(w):.6f}",
        "expected_final": f"{w0 - delta * steps:.6f}",
        "ulp_at_1.0": f"{_ulp_at_one(dtype):.2e}",
    }

rows = [apply_updates(dt) for dt in [torch.float16, torch.bfloat16, torch.float32]]
display(pd.DataFrame(rows))

print("\nFP16: delta=1e-5 is below ULP at 1.0 (~1e-3). Weight NEVER changes.")
print("BF16: delta=1e-5 is below ULP at 1.0 (~8e-3). Weight NEVER changes.")
print("FP32: delta=1e-5 is above ULP at 1.0 (~1e-7). Weight changes every step.")
print("\nThis is why optimizers need FP32 master weights.")
""")


# ── 3.6 Main training experiment ────────────────────────────────────────────

md(r"""
## 3.6 The main event: train a tiny causal LM under different precision regimes

We'll train TinyGPT on a character-level next-token prediction task using an in-notebook corpus.

**Why character-level?** No external downloads, stable, deterministic, and still exercises the transformer mechanics that matter for autocast (attention, layernorm, softmax, embeddings).

**We will compare:**
- FP32 baseline
- Naive FP16 (params in FP16, no autocast, no scaler) — expected to be unstable
- AMP FP16 (autocast + GradScaler)
- AMP BF16 (autocast, no scaler)

**We will log:**
- Training loss
- Validation loss
- Gradient norm
- Step time
- CUDA memory
- GradScaler scale (for FP16 AMP)
""")

code(r"""
# Tiny corpus + character-level tokenizer

corpus_lines = [
    "Autocast is not a global cast. It is an operator policy that routes each operation to the right precision.",
    "Some ops run in lower precision for speed. Matmuls and linear layers are the primary targets.",
    "Some ops run in float32 for stability. Softmax, layer normalization, and cross entropy need full precision.",
    "Loss scaling rescues fp16 gradients from underflow by multiplying the loss before backward.",
    "Bfloat16 has the same exponent range as float32, which means gradients rarely underflow to zero.",
    "Transformers amplify numeric issues because attention involves softmax over large logits.",
    "Master weights in fp32 prevent update stagnation where small gradients get rounded away.",
    "Optimizer state like Adam moments should be kept in fp32 for long-horizon accumulation stability.",
    "The fundamental tradeoff in floating point is range versus precision and they cannot both be maximized.",
    "FP16 has ten mantissa bits and five exponent bits giving it better precision but much narrower range.",
    "BF16 has seven mantissa bits and eight exponent bits giving it the same range as FP32 but coarser precision.",
    "Modern GPUs have tensor cores that execute sixteen bit matrix multiplications with fp32 accumulation.",
    "A single training step involves forward pass, loss computation, backward pass, and optimizer update.",
    "Gradient accumulation with mixed precision requires careful ordering of scale, backward, unscale, and step.",
    "Dynamic loss scaling starts with a large scale factor and halves it whenever overflow is detected.",
    "When the gradient scaler detects infinity or NaN values it skips the optimizer step entirely.",
    "The ZeRO optimizer partitions parameters gradients and optimizer state across multiple GPUs.",
    "Mixed precision training saves memory primarily through reduced activation storage not parameter storage.",
    "For a model with one billion parameters Adam mixed precision training requires about sixteen gigabytes.",
    "TF32 is not a storage format but an internal compute mode used by tensor cores on Ampere GPUs.",
    "Epsilon is the smallest number that when added to one produces a result strictly greater than one.",
    "Neural networks are robust to gradient noise because stochastic gradient descent is inherently noisy.",
    "The residual connection in transformers promotes dtype from sixteen bit back to thirty two bit under autocast.",
    "FP8 training requires explicit per tensor scaling because the representable range is extremely narrow.",
    "Weight stagnation happens when the learning rate times gradient product is below the unit in last place.",
]

corpus = "\n".join(corpus_lines).strip()
corpus = corpus * 30  # repeat for more training data

chars = sorted(set(corpus))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

def encode(s):
    return [stoi[c] for c in s]

def decode_tokens(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(corpus), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"Vocab size: {vocab_size}")
print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")
print(f"Sample: {decode_tokens(train_data[:80].tolist())}")
""")

code(r"""
# Batch sampling + evaluation

def get_batch(split, batch_size, block_size):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([src[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, block_size, batch_size, iters=20, use_autocast=False, amp_dtype=None):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(iters):
            x, y = get_batch(split, batch_size, block_size)
            with amp_autocast(device, amp_dtype, enabled=use_autocast):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(float(loss))
        out[split] = float(np.mean(losses))
    model.train()
    return out
""")

code(r"""
# Training infrastructure

@dataclass
class TrainConfig:
    name: str
    steps: int = 200
    batch_size: int = 32
    block_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    use_autocast: bool = False
    amp_dtype: torch.dtype = None
    use_grad_scaler: bool = False
    param_dtype: torch.dtype = torch.float32
    eval_interval: int = 50
    eval_iters: int = 10

def global_grad_norm(model):
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().float().norm())**2
    return math.sqrt(total_sq)

def global_zero_grad_frac(model):
    zeros = 0
    total = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        zeros += int((g == 0).sum())
        total += g.numel()
    return zeros / max(total, 1)

def train_one(cfg):
    set_seed(42)
    model = TinyGPT(
        vocab_size=vocab_size, block_size=cfg.block_size,
        n_layer=2, n_embd=128, n_heads=4, dropout=0.0,
    ).to(device).to(cfg.param_dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=True) if (cfg.use_grad_scaler and device.type == "cuda") else None

    logs = {
        "step": [], "train_loss": [], "grad_norm": [], "zero_grad_frac": [],
        "step_time_ms": [], "tokens_per_s": [], "scale": [],
        "cuda_mem_mb": [], "val_step": [], "val_loss": [],
    }
    status = "ok"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokens_per_step = cfg.batch_size * cfg.block_size

    for step in range(cfg.steps):
        t0 = time.perf_counter()
        x, y = get_batch("train", cfg.batch_size, cfg.block_size)
        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device, cfg.amp_dtype, enabled=cfg.use_autocast):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if not torch.isfinite(loss):
            status = "non_finite_loss"
            break

        if scaler is None:
            loss.backward()
            grad_norm = global_grad_norm(model)
            zero_frac = global_zero_grad_frac(model)
            optimizer.step()
            scale_val = float("nan")
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = global_grad_norm(model)
            zero_frac = global_zero_grad_frac(model)
            scaler.step(optimizer)
            scaler.update()
            scale_val = float(scaler.get_scale())

        dt = max(time.perf_counter() - t0, 1e-12)
        logs["step"].append(step)
        logs["train_loss"].append(float(loss))
        logs["grad_norm"].append(float(grad_norm))
        logs["zero_grad_frac"].append(float(zero_frac))
        logs["step_time_ms"].append(dt * 1000)
        logs["tokens_per_s"].append(tokens_per_step / dt)
        logs["scale"].append(scale_val)
        logs["cuda_mem_mb"].append(torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else float("nan"))

        if cfg.eval_interval and (step % cfg.eval_interval == 0 or step == cfg.steps - 1):
            try:
                ev = estimate_loss(
                    model,
                    cfg.block_size,
                    cfg.batch_size,
                    cfg.eval_iters,
                    use_autocast=cfg.use_autocast,
                    amp_dtype=cfg.amp_dtype,
                )
                logs["val_step"].append(step)
                logs["val_loss"].append(ev["val"])
            except Exception:
                logs["val_step"].append(step)
                logs["val_loss"].append(float("nan"))

    return {"config": cfg, "status": status, "logs": logs}
""")

code(r"""
# Define experiment suite

FAST_DEV_RUN = (device.type != "cuda")
BASE_STEPS = 60 if FAST_DEV_RUN else 300

suite = [
    TrainConfig(name="fp32", steps=BASE_STEPS, param_dtype=torch.float32),
]

if device.type == "cuda":
    suite.append(TrainConfig(
        name="fp16_naive", steps=BASE_STEPS,
        param_dtype=torch.float16,
    ))
    if torch.cuda.is_bf16_supported():
        suite.append(TrainConfig(
            name="bf16_naive", steps=BASE_STEPS,
            param_dtype=torch.bfloat16,
        ))
    suite.append(TrainConfig(
        name="amp_fp16_no_scaler", steps=BASE_STEPS,
        use_autocast=True, amp_dtype=torch.float16,
        use_grad_scaler=False, param_dtype=torch.float32,
    ))
    suite.append(TrainConfig(
        name="amp_fp16", steps=BASE_STEPS,
        use_autocast=True, amp_dtype=torch.float16,
        use_grad_scaler=True, param_dtype=torch.float32,
    ))
    if torch.cuda.is_bf16_supported():
        suite.append(TrainConfig(
            name="amp_bf16", steps=BASE_STEPS,
            use_autocast=True, amp_dtype=torch.bfloat16,
            param_dtype=torch.float32,
        ))
elif device.type == "cpu":
    suite.append(TrainConfig(
        name="amp_bf16_cpu", steps=BASE_STEPS,
        use_autocast=True, amp_dtype=torch.bfloat16,
        param_dtype=torch.float32,
    ))

print("Planned experiments:")
for cfg in suite:
    print(f"  {cfg.name}: params={cfg.param_dtype}, autocast={cfg.use_autocast} {cfg.amp_dtype}, scaler={cfg.use_grad_scaler}")
""")

code(r"""
# Run all experiments

results = []
for cfg in suite:
    print(f"\n{'='*50}")
    print(f"Running: {cfg.name}")
    print(f"{'='*50}")
    res = train_one(cfg)
    print(f"  Status: {res['status']}, Steps: {len(res['logs']['step'])}")
    if res['logs']['train_loss']:
        print(f"  Final train loss: {res['logs']['train_loss'][-1]:.4f}")
    results.append(res)

# Summary table
summary_rows = []
for r in results:
    cfg, logs = r["config"], r["logs"]
    n = len(logs["step"])
    summary_rows.append({
        "name": cfg.name,
        "status": r["status"],
        "steps": n,
        "final_train_loss": f"{logs['train_loss'][-1]:.4f}" if n else "n/a",
        "final_val_loss": f"{logs['val_loss'][-1]:.4f}" if logs["val_loss"] else "n/a",
        "mean_step_ms": f"{np.mean(logs['step_time_ms']):.1f}" if n else "n/a",
        "mean_tok/s": f"{np.mean(logs['tokens_per_s']):.0f}" if n else "n/a",
        "peak_cuda_MB": f"{np.nanmax(logs['cuda_mem_mb']):.1f}" if device.type == "cuda" and n else "n/a",
    })

print("\n\n=== Summary ===")
display(pd.DataFrame(summary_rows))
""")

code(r"""
# Plot: Training + Validation loss curves (annotated)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

color_map = {
    "fp32": "C0", "fp16_naive": "C3", "bf16_naive": "C5",
    "amp_fp16_no_scaler": "C6", "amp_fp16": "C4", "amp_bf16": "C1",
    "amp_bf16_cpu": "C1",
}

for r in results:
    name = r["config"].name
    logs = r["logs"]
    suffix = f" ({r['status']})" if r["status"] != "ok" else ""
    color = color_map.get(name, "C7")
    ls = "--" if "naive" in name else "-"
    if logs["train_loss"]:
        axes[0].plot(logs["step"], logs["train_loss"], label=f"{name}{suffix}",
                     alpha=0.85, color=color, linestyle=ls, linewidth=1.5 if "naive" not in name else 1.0)
    if logs["val_loss"]:
        axes[1].plot(logs["val_step"], logs["val_loss"], marker="o", ms=5, label=f"{name}{suffix}",
                     alpha=0.85, color=color, linestyle=ls)

axes[0].set_title("Training loss vs step", fontsize=11)
axes[0].set_xlabel("step")
axes[0].set_ylabel("cross-entropy loss")
axes[0].legend(fontsize=8, loc="upper right")

axes[1].set_title("Validation loss (periodic eval)", fontsize=11)
axes[1].set_xlabel("step")
axes[1].set_ylabel("cross-entropy loss")
axes[1].legend(fontsize=8, loc="upper right")

fig.suptitle("TinyGPT Training: How precision regime affects convergence", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

print("What to look for:")
print("  - FP32 (solid blue): smooth reference curve")
print("  - Naive FP16 (dashed red): may stagnate, diverge, or NaN")
print("  - Naive BF16 (dashed olive): often converges — BF16 range prevents gradient death")
print("  - AMP variants (solid): should track FP32 closely, showing that AMP preserves quality")
""")

code(r"""
# Plot: Step time + throughput

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for r in results:
    name = r["config"].name
    logs = r["logs"]
    if not logs["step"]:
        continue
    axes[0].plot(logs["step"], logs["step_time_ms"], label=name, alpha=0.7)
    axes[1].plot(logs["step"], logs["tokens_per_s"], label=name, alpha=0.7)

axes[0].set_title("Step time (ms)")
axes[0].set_xlabel("step"); axes[0].set_ylabel("ms")
axes[0].legend()

axes[1].set_title("Throughput (tokens/s)")
axes[1].set_xlabel("step"); axes[1].set_ylabel("tokens/s")
axes[1].legend()

plt.tight_layout();
""")

code(r"""
# Plot: Gradient norms across precision regimes

plt.figure(figsize=(12, 4))
for r in results:
    name = r["config"].name
    logs = r["logs"]
    if not logs["step"]:
        continue
    color = color_map.get(name, "C7")
    ls = "--" if "naive" in name else "-"
    plt.plot(logs["step"], logs["grad_norm"], label=name, alpha=0.75, color=color, linestyle=ls)

# Add FP16 min normal threshold line
fi16 = torch.finfo(torch.float16)
plt.axhline(float(fi16.tiny), color="red", ls=":", alpha=0.4, label=f"FP16 min normal ({fi16.tiny:.1e})")

plt.title("Gradient L2 norm vs step (after unscaling)", fontsize=11)
plt.xlabel("step")
plt.ylabel("||grad||_2")
plt.yscale("log")
plt.legend(fontsize=8)
plt.tight_layout()

print("If gradients fall below the FP16 min normal line, they underflow to zero in FP16.")
print("BF16 gradients essentially never underflow (BF16 min normal ≈ 1.2e-38).")
""")

code(r"""
# Plot: Exact-zero gradient fraction (a proxy for underflow / dead signal)

plt.figure(figsize=(12, 3))
for r in results:
    name = r["config"].name
    logs = r["logs"]
    if not logs["step"]:
        continue
    if "zero_grad_frac" not in logs:
        continue
    plt.plot(logs["step"], logs["zero_grad_frac"], label=name, alpha=0.75)

plt.title("Fraction of gradients that are exactly 0")
plt.xlabel("step")
plt.ylabel("zero_grad_frac")
plt.yscale("log")
plt.legend()
plt.tight_layout();
""")

code(r"""
# Plot: GradScaler dynamic scale (FP16 AMP only)

plt.figure(figsize=(12, 3))
plotted = False
for r in results:
    if not r["config"].use_grad_scaler:
        continue
    logs = r["logs"]
    scale = np.array(logs["scale"], dtype=np.float64)
    if len(scale) == 0 or np.all(np.isnan(scale)):
        continue
    plt.plot(logs["step"], scale, label=r["config"].name)
    plotted = True

if plotted:
    plt.title("GradScaler dynamic scale factor")
    plt.xlabel("step")
    plt.ylabel("scale")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
else:
    print("No GradScaler data to plot (did amp_fp16 run?)")
""")

code(r"""
# Plot: CUDA peak memory

if device.type == "cuda":
    fig, ax = plt.subplots(figsize=(10, 3))
    for r in results:
        logs = r["logs"]
        if not logs["step"]:
            continue
        ax.plot(logs["step"], logs["cuda_mem_mb"], label=r["config"].name, alpha=0.7)
    ax.set_title("Peak CUDA memory allocated (MB)")
    ax.set_xlabel("step"); ax.set_ylabel("MB")
    ax.legend()
    plt.tight_layout()
else:
    print("CUDA not available; skipping memory plot.")
""")


code(r"""
# Summary bar charts: compare all experiments at a glance

completed = [r for r in results if r["status"] == "ok" and r["logs"]["train_loss"]]

if len(completed) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [r["config"].name for r in completed]
    x_pos = np.arange(len(names))

    # 1. Final training loss
    final_losses = [r["logs"]["train_loss"][-1] for r in completed]
    colors = [color_map.get(r["config"].name, "C7") for r in completed]

    axes[0].bar(x_pos, final_losses, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("Final train loss")
    axes[0].set_title("Final Training Loss (lower is better)")

    # 2. Mean step time
    mean_times = [np.mean(r["logs"]["step_time_ms"]) for r in completed]
    axes[1].bar(x_pos, mean_times, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean step time (ms)")
    axes[1].set_title("Training Speed (lower is better)")

    # 3. Peak CUDA memory
    if device.type == "cuda":
        peak_mem = [np.nanmax(r["logs"]["cuda_mem_mb"]) for r in completed]
        axes[2].bar(x_pos, peak_mem, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        axes[2].set_ylabel("Peak memory (MB)")
        axes[2].set_title("Peak GPU Memory (lower is better)")
    else:
        axes[2].text(0.5, 0.5, "CUDA memory\nnot available", transform=axes[2].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
        axes[2].set_title("Peak GPU Memory")

    fig.suptitle("Experiment Summary: Loss, Speed, and Memory across Precision Regimes", fontsize=12, y=1.02)
    plt.tight_layout()
else:
    print("Need at least 2 completed experiments for summary charts.")
""")


md(r"""
## 3.7 Memory breakdown comparison (bytes per parameter)

For LLM-scale training, it helps to separate memory into:

- **Fixed-size memory** (scales with number of parameters): parameters + gradients + optimizer state (+ optional master weights).
- **Activation memory** (scales with batch size × sequence length × layers): intermediate tensors saved for backward.

AMP/autocast primarily helps with **activation memory** (and speed). With AdamW, the fixed-size **bytes/parameter** are often **~16 bytes/param** in both FP32 and "standard AMP" because optimizer state dominates.

We'll compute the fixed-size accounting for a few common patterns and show what that looks like for our TinyGPT.
""")

code(r"""
# Fixed-size memory breakdown analysis (parameters + grads + optimizer state)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

model_tmp = TinyGPT(vocab_size=vocab_size, block_size=64, n_layer=2, n_embd=128, n_heads=4)
n_params = count_params(model_tmp)
del model_tmp

rows = []
configs = [
    # AdamW has 2 FP32 moment buffers by default: m and v (8 bytes/param).
    {
        "config": "FP32 AdamW (baseline)",
        "param_B": 4, "master_B": 0, "grad_B": 4, "opt_B": 8,
        "notes": "Params/grads/Adam moments all FP32",
    },
    {
        "config": "PyTorch AMP FP16 (compute)",
        "param_B": 4, "master_B": 0, "grad_B": 4, "opt_B": 8,
        "notes": "Typical AMP: params/grads/state FP32; matmuls run in FP16 under autocast",
    },
    {
        "config": "PyTorch AMP BF16 (compute)",
        "param_B": 4, "master_B": 0, "grad_B": 4, "opt_B": 8,
        "notes": "Typical AMP: params/grads/state FP32; matmuls run in BF16 under autocast",
    },
    {
        "config": "Naive FP16 AdamW (NOT recommended)",
        "param_B": 2, "master_B": 0, "grad_B": 2, "opt_B": 4,
        "notes": "Small fixed memory, but numerically fragile (underflow/overflow + low-precision optimizer state)",
    },
    {
        "config": "Naive BF16 AdamW (sometimes works)",
        "param_B": 2, "master_B": 0, "grad_B": 2, "opt_B": 4,
        "notes": "Often trains, but sensitive reductions/normalizations can drift without an FP32 policy",
    },
    {
        "config": "FP16 params + FP32 master + FP32 AdamW (classic mixed precision)",
        "param_B": 2, "master_B": 4, "grad_B": 2, "opt_B": 8,
        "notes": "Common in some stacks: FP16 model copy + FP32 master weights + FP32 optimizer moments",
    },
]

for cfg in configs:
    total_B = int(cfg["param_B"] + cfg["master_B"] + cfg["grad_B"] + cfg["opt_B"])
    param_mb = n_params * cfg["param_B"] / 1024**2
    master_mb = n_params * cfg["master_B"] / 1024**2
    grad_mb = n_params * cfg["grad_B"] / 1024**2
    opt_mb = n_params * cfg["opt_B"] / 1024**2
    total_mb = param_mb + master_mb + grad_mb + opt_mb
    rows.append({
        "config": cfg["config"],
        "bytes_per_param": total_B,
        "param_MB": float(f"{param_mb:.3f}"),
        "master_MB": float(f"{master_mb:.3f}"),
        "grad_MB": float(f"{grad_mb:.3f}"),
        "optimizer_MB": float(f"{opt_mb:.3f}"),
        "total_fixed_MB": float(f"{total_mb:.3f}"),
        "notes": cfg["notes"],
    })

print(f"TinyGPT parameters: {n_params:,}")
print("Fixed-size bytes/param = params + master + grads + optimizer state (activations NOT included).")
print("(Real LLM AMP savings come mostly from activations, which scale with batch×seq_len.)\n")

df_mem = pd.DataFrame(rows)
display(df_mem)

# Visualize fixed memory breakdown as stacked bar chart
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(rows))
w = 0.5
names_m = df_mem["config"].tolist()
param_mb = df_mem["param_MB"].tolist()
master_mb = df_mem["master_MB"].tolist()
grad_mb = df_mem["grad_MB"].tolist()
opt_mb = df_mem["optimizer_MB"].tolist()

ax.bar(x, param_mb, w, label="Model params", color="C0", alpha=0.8)
ax.bar(x, master_mb, w, bottom=param_mb, label="Master params", color="C4", alpha=0.8)
ax.bar(x, grad_mb, w, bottom=[p+m for p,m in zip(param_mb, master_mb)], label="Gradients", color="C1", alpha=0.8)
ax.bar(
    x, opt_mb, w,
    bottom=[p+m+g for p,m,g in zip(param_mb, master_mb, grad_mb)],
    label="Optimizer state",
    color="C2",
    alpha=0.8,
)
ax.set_xticks(x)
ax.set_xticklabels(names_m, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Memory (MB)")
ax.set_title(f"Fixed memory breakdown by precision config ({n_params:,} params)")
ax.legend()
plt.tight_layout()

print("\nNote: This shows only fixed-size memory (params + grads + optimizer).")
print("Activations (which scale with batch×seq_len) are the real memory win for AMP.")
print("For large models, activation memory often exceeds parameter memory by 5-10x.")
""")


md(r"""
## 3.8 Interpreting results

### Loss curves
- **FP32**: should decrease smoothly. This is your reference.
- **FP16 naive**: may diverge, stagnate, or produce NaN. This demonstrates why casting everything to half precision is not safe.
- **BF16 naive**: often converges to a similar loss as FP32. This is the dramatic proof that BF16's range (8-bit exponent) matters more than FP16's precision (10-bit mantissa) for training stability. The network tolerates noisy gradient values; it cannot tolerate zero gradient values.
- **AMP FP16 (no scaler)**: may work but may show more zero gradients than the scaled version. This shows that autocast alone helps (per-op policy keeps sensitive ops in FP32) but doesn't fully solve the backward-pass underflow problem.
- **AMP FP16 (with scaler)**: should be stable. The GradScaler rescues gradients from underflow.
- **AMP BF16**: typically matches FP32 quality without needing a scaler.

### Step time
- On modern GPUs, AMP often reduces step time because matmuls hit Tensor Cores.
- If you don't see speedup: model may be too small (overhead dominates), or you're CPU-bound.
- The smallest models sometimes see AMP *overhead* because the per-op dispatch cost exceeds the Tensor Core gains. This goes away at realistic model sizes.

### Gradient norms
- Should be comparable across regimes for a well-behaved model.
- If FP16 naive shows erratic norms, that's the underflow/overflow instability.
- BF16 naive norms should be close to FP32 (same gradient magnitudes, just represented with fewer mantissa bits).

### Zero gradients
- A high `zero_grad_frac` indicates underflow or dead signal (exact zeros).
- **FP16 naive** and **AMP FP16 no scaler** may show elevated zero-gradient fractions.
- **BF16 naive** should show low zero-gradient fractions (gradients don't underflow with 8-bit exponent).
- The difference between FP16+scaler and FP16 without scaler directly measures the benefit of loss scaling.

### GradScaler scale
- If scale drops repeatedly, the model is hitting overflow events and GradScaler is skipping steps.
- If scale grows steadily, training is stable and GradScaler is increasing headroom.
- A "spiky" pattern (rapid drops followed by slow climbs) is normal and expected.

### Memory
- AMP typically uses ~same or slightly more memory than FP32 for parameters+optimizer (due to master weights), but saves on activations.
- Naive 16-bit uses less total memory but trades stability.
- The savings become dramatic at larger batch sizes and sequence lengths.
- At LLM scale with Adam: 16 bytes/param (mixed precision) vs 16 bytes/param (FP32). The *activation* memory savings are where AMP really wins.
""")


md(r"""
## 3.9 Practical checklist

### Defaults that usually work
- Prefer **BF16 autocast** if your GPU supports it (Ampere+). No GradScaler needed.
- Otherwise use **FP16 autocast + GradScaler**.
- Keep optimizer state in FP32 (default for most PyTorch optimizers).

### When things go wrong
1. **Loss becomes `nan`/`inf`:** Check for overflow sources (attention logits, exp/log, unstable loss). Consider lowering learning rate or adding gradient clipping.
2. **Gradients mostly zero in FP16:** Use GradScaler with higher initial scale. Consider switching to BF16.
3. **Training "does nothing":** Check for weight update stagnation — are weights actually changing? Ensure you have FP32 master weights (standard when model is FP32 + autocast).
4. **Unexplained dtype behavior:** Use dtype hooks (Section 3.3) to confirm what's running in what dtype.

### Common gotchas
- Mixing manual `.half()` casts with autocast can lead to unexpected behavior.
- Gradient clipping should happen **after** `scaler.unscale_(optimizer)`.
- `autocast` should cover forward + loss, but **not** the optimizer step.
- On Ampere+ GPUs, your "FP32 baseline" may use TF32 for matmuls. Check `torch.backends.cuda.matmul.allow_tf32`.
""")

md(r"""
## 3.10 Real-world AMP patterns (copy/paste templates)

This section is intentionally practical: patterns that show up once you move from a notebook demo to a real training run.

### 3.10.1 Gradient accumulation (microbatches)

If you do gradient accumulation, the safest pattern is:

```python
scaler = GradScaler()
optimizer.zero_grad(set_to_none=True)

for micro in range(grad_accum_steps):
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = loss_fn(model(x_micro), y_micro)
        loss = loss / grad_accum_steps      # IMPORTANT: normalize
    scaler.scale(loss).backward()

scaler.unscale_(optimizer)                  # IMPORTANT: before clipping / inspecting grads
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

### 3.10.2 Gradient clipping with FP16 AMP

Clipping should happen **after** `scaler.unscale_(optimizer)`. If you clip *scaled* gradients, you're clipping the wrong numbers.

### 3.10.3 Multiple optimizers / parameter groups

Use **one scaler**. Call `scaler.step(optimizer_i)` for each optimizer, then `scaler.update()` once per iteration.

### 3.10.4 Custom `autograd.Function` / custom ops

If you write custom autograd functions (or CUDA extensions), make them autocast-safe. In PyTorch there are decorators for this (API varies by version):

```python
try:
    from torch.amp import custom_fwd, custom_bwd
except Exception:
    from torch.cuda.amp import custom_fwd, custom_bwd

class MyFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, w):
        ...

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        ...
```

If you skip this, autocast may feed your op tensors in unexpected dtypes, and you'll get silent accuracy bugs or runtime errors.

### 3.10.5 Inference is simpler than training

For inference you usually only need:
- `torch.inference_mode()` (or `no_grad()`)
- `autocast(...)` (no GradScaler)

### 3.10.6 Troubleshooting table

| Symptom | Likely cause | Quick check | Fix |
|---|---|---|---|
| Loss becomes `nan`/`inf` quickly | overflow in activations (attention logits, exp/log) or too-high LR | check `torch.isfinite(loss)`; watch GradScaler `found_inf`/scale drops | lower LR, add grad clipping, prefer BF16, check initialization |
| Gradients mostly 0 in FP16 | underflow | measure `zero_grad_frac` or histogram of `log10(|grad|)` | use GradScaler (bigger initial scale), or switch to BF16 |
| GradScaler scale keeps collapsing | frequent overflow events | scale drops + many skipped steps | lower LR, clip grads, check for unstable ops, switch to BF16 |
| Training "does nothing" | weight update below ULP (stagnation) | log `max(|w_{t+1}-w_t|)` in model dtype | keep FP32 master weights (standard in AMP), avoid FP16 optimizer state |
| No speedup from AMP | model too small, CPU-bound, or bad matmul shapes | compare tokens/s; check shapes are multiples of 8 | increase batch/seq, use Tensor Core-friendly dims, benchmark with realistic sizes |
""")


md(r"""
# Appendix — References

## Papers

| Paper | Year | Key contribution |
|---|---|---|
| Gupta et al., *Deep Learning with Limited Numerical Precision* | 2015 | Low-precision training needs deliberate rounding/scaling; stochastic rounding intuition |
| Micikevicius et al., *Mixed Precision Training* | 2017 | FP32 master weights + loss scaling + per-op policies |
| Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* | 2019 | BF16 empirical validation, no loss scaling needed |
| Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* | 2020 | Memory breakdown of mixed-precision training |
| NVIDIA (and others), *FP8 Formats for Deep Learning* | 2022 | FP8 formats (E4M3/E5M2), scaling metadata, and accumulation strategies |
| Dettmers et al., *8-bit Optimizers via Block-wise Quantization* | 2022 | Reducing optimizer-state memory beyond AMP |

## Documentation

- [PyTorch `torch.amp` docs](https://pytorch.org/docs/stable/amp.html) — autocast op reference, GradScaler API
- [PyTorch AMP examples](https://pytorch.org/docs/stable/notes/amp_examples.html) — canonical training loop
- [NVIDIA mixed precision blog](https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/)
- [Google BF16 blog](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
- [DeepSpeed config docs](https://www.deepspeed.ai/docs/config-json/) — FP16/BF16 training config
- [PyTorch FSDP mixed precision](https://pytorch.org/docs/stable/fsdp.html)

---

This notebook's experiments are intentionally small; the *mechanisms* are the same at scale.
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  WRITE THE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = pathlib.Path(__file__).with_name("amp.ipynb")
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells to {out}")
