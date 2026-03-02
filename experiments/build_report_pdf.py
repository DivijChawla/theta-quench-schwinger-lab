from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")

    y = 0.98
    ax.text(0.0, y, title, fontsize=20, fontweight="bold", va="top")
    y -= 0.06
    for line in lines:
        ax.text(0.0, y, line, fontsize=11.5, va="top", family="monospace" if line.startswith("  ") else None)
        y -= 0.035

    pdf.savefig(fig)
    plt.close(fig)


def _add_image_page(pdf: PdfPages, title: str, image_paths: list[Path], captions: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax_title = fig.add_axes([0.08, 0.93, 0.84, 0.05])
    ax_title.axis("off")
    ax_title.text(0, 0.9, title, fontsize=16, fontweight="bold", va="top")

    n = len(image_paths)
    top = 0.88
    height = 0.78 / max(n, 1)

    for i, (img_path, cap) in enumerate(zip(image_paths, captions)):
        y0 = top - (i + 1) * height
        ax = fig.add_axes([0.1, y0 + 0.05, 0.8, height - 0.08])
        ax.axis("off")
        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Missing image: {img_path}", ha="center", va="center")

        ax_cap = fig.add_axes([0.1, y0 + 0.01, 0.8, 0.03])
        ax_cap.axis("off")
        ax_cap.text(0.0, 0.9, cap, fontsize=10, va="top")

    pdf.savefig(fig)
    plt.close(fig)


def _pick_image(outputs: Path, name: str) -> Path:
    p0 = outputs / "figs" / name
    if p0.exists():
        return p0
    p1 = outputs / "stage2_prlx" / "universal_scan_v2" / "figs" / name
    if p1.exists():
        return p1
    return p0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build short PDF report for reviewers")
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--out", type=str, default="report/short_report.pdf")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    figs = outputs / "figs"
    validation = outputs / "validation.json"
    quench = outputs / "quench_summary.csv"
    nnqs = outputs / "nnqs" / "nnqs_snapshot_metrics.csv"

    summary_lines = [
        "Hypothesis:",
        "  Stronger theta quenches increase transition signatures and stabilizer magic;",
        "  higher-magic snapshots should be harder for NNQS training (scoped claim).",
        "",
        "Model:",
        "  1+1D lattice Schwinger model (open boundaries), staggered fermions,",
        "  Gauss-law-eliminated spin Hamiltonian H = H_pm + H_ZZ + H_Z + const.",
        "",
        "Numerics:",
        "  Small-N exact diagonalization + real-time evolution (dense/Krylov),",
        "  Pauli-batched stabilizer Renyi magic (M2), GRU/MADE/RBM/independent NNQS fitting.",
        "",
        f"Artifacts:",
        f"  validation: {validation}",
        f"  quench summary: {quench}",
        f"  nnqs metrics: {nnqs}",
        "  claim gate: report/publishability_status.md",
        "  protocol: report/analysis_protocol.md",
    ]

    if validation.exists():
        with validation.open("r", encoding="utf-8") as f:
            v = json.load(f)
        summary_lines.extend(
            [
                "",
                "Validation checks:",
                f"  Hermitian H0/H1: {v.get('hermitian_h0')} / {v.get('hermitian_h1')}",
                f"  Norm drift: {v.get('norm_drift'):.3e}",
                f"  Dense-vs-Krylov diff: {v.get('dense_krylov_error'):.3e}",
                f"  Magic sanity |T>^n: {v.get('magic_|T>^⊗n')}",
            ]
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        _add_text_page(pdf, "Theta-Quench Schwinger Lab: Short Report", summary_lines)
        _add_image_page(
            pdf,
            "Key Results I: Dynamics + Magic",
            [figs / "fig1_loschmidt_family.png", figs / "fig2_magic_lambda_overlay.png", figs / "fig3_heatmaps_lambda_magic.png"],
            [
                "Loschmidt rate family across quench endpoints.",
                "Overlay of transition diagnostic and stabilizer magic M2(t).",
                "Heatmap view over (time, theta1).",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results II: Learnability",
            [figs / "fig4_nnqs_loss_vs_magic.png", figs / "fig4b_nnqs_val_curves.png", figs / "fig5_entropy_vs_magic.png"],
            [
                "NNQS final validation loss vs snapshot magic.",
                "NNQS optimization trajectories over training epochs.",
                "Entanglement vs magic trajectory (complementary diagnostic).",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results III: Claim Boundary",
            [figs / "fig6_regime_boundary_magic_valnll.png"],
            [
                "Regime/size boundary map for magic vs validation NLL (conditional, non-universal).",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results IV: Controls and Calibration",
            [figs / "fig7_positive_control_scatter.png", figs / "fig8_magic_calibration.png"],
            [
                "Positive control: synthetic state family confirms increased learnability cost with magic.",
                "Approximate magic estimator calibration envelope vs exact M2.",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results V: Stage-2 Cross-Model Bounds",
            [
                _pick_image(outputs, "fig_stage2_crossmodel_pooled_r.png"),
                _pick_image(outputs, "fig_stage2_beta_bounds.png"),
                _pick_image(outputs, "fig_stage2_mechanism_arch_slope.png"),
                _pick_image(outputs, "fig_stage2_corridor_robustness_all_arch.png"),
            ],
            [
                "Cross-model pooled primary effects (magic -> validation NLL | entropy).",
                "Bootstrap CI bounds for entropy-controlled magic slope beta_magic.",
                "Cluster-robust mechanism regression: positive beta_magic lower bounds by architecture.",
                "All-architecture robustness: beta-law is stable, correlation-sign law is heterogeneous.",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results VI: Stage-3 Null-Disproof Stress",
            [
                _pick_image(outputs, "fig_stage3_beta_forest.png"),
                _pick_image(outputs, "fig_stage3_perm_qvalues.png"),
                _pick_image(outputs, "fig_stage3_partial_sign_heatmap.png"),
            ],
            [
                "Stage-3 forest plot: beta_magic confidence bounds across Schwinger/TFIM/XXZ cells.",
                "Stage-3 condition-wise permutation null tests with BH-FDR correction.",
                "Stage-3 partial-correlation lower-CI heatmap (sign-law heterogeneity map).",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results VII: Stage-4 Thermodynamic + Theory Checks",
            [
                _pick_image(outputs, "fig_stage4_beta_by_size.png"),
                _pick_image(outputs, "fig_stage4_beta_inf_forest.png"),
                _pick_image(outputs, "fig_stage4_quantile_bound.png"),
            ],
            [
                "Finite-size beta-law trend by model family and architecture.",
                "N→∞ extrapolation of beta-law cells with uncertainty.",
                "Quantile-regression lower-envelope positivity check for beta_magic.",
            ],
        )
        _add_image_page(
            pdf,
            "Key Results VIII: Stage-5 External Family + Regime Stress",
            [
                _pick_image(outputs, "fig_stage5_beta_forest.png"),
                _pick_image(outputs, "fig_stage5_regime_minimax.png"),
                _pick_image(outputs, "fig_stage5_regime_heatmap.png"),
            ],
            [
                "External-family extension (ANNNI added): cell-level beta-law bounds remain positive.",
                "Adversarial minimax regime bound shows where strict regime-universal positivity fails.",
                "Regime boundary heatmap: explicit support/failure map for beta_magic CI lower bounds.",
            ],
        )
        _add_text_page(
            pdf,
            "Roadmap",
            [
                "1) Increase system size via symmetry sectors and sparse block structure.",
                "2) Add sampled/approximate magic estimators for N > 12.",
                "3) Extend NNQS architectures (Transformer/MADE) and measurement bases.",
                "4) Build Trotterized circuit path for hardware-facing experiments.",
                "",
                "Caveat:",
                "  Current claims are corridor-scoped empirical laws; not model-independent theorems.",
            ],
        )

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
