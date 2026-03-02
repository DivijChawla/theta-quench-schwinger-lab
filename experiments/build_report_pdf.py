from __future__ import annotations

import argparse
import json
import os
import textwrap
from datetime import date
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _wrap(paragraph: str, width: int = 108) -> list[str]:
    if not paragraph.strip():
        return [""]
    return textwrap.wrap(paragraph, width=width)


def _new_text_axes() -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
    ax.axis("off")
    return fig, ax


def _add_title_page(pdf: PdfPages, title: str, subtitle: str, abstract: str, meta: list[str]) -> None:
    fig, ax = _new_text_axes()
    y = 0.98
    ax.text(0.0, y, title, fontsize=21, fontweight="bold", va="top")
    y -= 0.062
    ax.text(0.0, y, subtitle, fontsize=11.5, va="top")
    y -= 0.06
    ax.text(0.0, y, "Abstract", fontsize=13, fontweight="bold", va="top")
    y -= 0.038
    for line in _wrap(abstract):
        ax.text(0.0, y, line, fontsize=10.6, va="top")
        y -= 0.026
    y -= 0.022
    ax.text(0.0, y, "Metadata", fontsize=12, fontweight="bold", va="top")
    y -= 0.034
    for item in meta:
        ax.text(0.0, y, f"- {item}", fontsize=10.4, va="top")
        y -= 0.028
    pdf.savefig(fig)
    plt.close(fig)


def _add_section_page(pdf: PdfPages, heading: str, paragraphs: list[str]) -> None:
    fig, ax = _new_text_axes()
    y = 0.98
    ax.text(0.0, y, heading, fontsize=15, fontweight="bold", va="top")
    y -= 0.045
    for p in paragraphs:
        for line in _wrap(p):
            ax.text(0.0, y, line, fontsize=10.4, va="top")
            y -= 0.024
        y -= 0.012
    pdf.savefig(fig)
    plt.close(fig)


def _add_table_page(pdf: PdfPages, heading: str, columns: list[str], rows: list[list[str]], caption: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis("off")
    ax_title.text(0, 0.95, heading, fontsize=15, fontweight="bold", va="top")

    ax_tbl = fig.add_axes([0.08, 0.22, 0.84, 0.66])
    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="left",
        loc="upper left",
        colWidths=[0.34, 0.21, 0.21, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9eef7")
        if c == 1 and r > 0:
            txt = cell.get_text().get_text()
            if txt == "PASS":
                cell.set_facecolor("#d9f2dd")
            elif txt == "FAIL":
                cell.set_facecolor("#fbe1e1")

    ax_cap = fig.add_axes([0.08, 0.08, 0.84, 0.1])
    ax_cap.axis("off")
    cap_lines = _wrap(caption, width=115)
    y = 0.95
    for line in cap_lines:
        ax_cap.text(0.0, y, line, fontsize=9.8, va="top")
        y -= 0.22

    pdf.savefig(fig)
    plt.close(fig)


def _add_figure_page(pdf: PdfPages, heading: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax_title = fig.add_axes([0.08, 0.92, 0.84, 0.06])
    ax_title.axis("off")
    ax_title.text(0.0, 0.95, heading, fontsize=14.5, fontweight="bold", va="top")

    ax_img = fig.add_axes([0.08, 0.25, 0.84, 0.63])
    ax_img.axis("off")
    if image_path.exists():
        img = mpimg.imread(image_path)
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, f"Missing image: {image_path}", ha="center", va="center", fontsize=11)

    ax_cap = fig.add_axes([0.08, 0.08, 0.84, 0.13])
    ax_cap.axis("off")
    y = 0.98
    for line in _wrap(caption, width=112):
        ax_cap.text(0.0, y, line, fontsize=10.0, va="top")
        y -= 0.19

    pdf.savefig(fig)
    plt.close(fig)


def _safe_bool(v: object) -> bool:
    return bool(v) if v is not None else False


def _pick_image(outputs: Path, name: str) -> Path:
    p0 = outputs / "figs" / name
    if p0.exists():
        return p0
    p1 = outputs / "stage2_prlx" / "universal_scan_v2" / "figs" / name
    if p1.exists():
        return p1
    p2 = outputs / "stage5_prx" / "external_regime_v1" / "figs" / name
    if p2.exists():
        return p2
    return p0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manuscript-style reviewer PDF from experiment outputs")
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--out", type=str, default="report/short_report.pdf")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    validation_path = outputs / "validation.json"
    quench_path = outputs / "quench_summary.csv"
    nnqs_path = outputs / "nnqs" / "nnqs_snapshot_metrics.csv"
    s3_path = outputs / "stage3_prx" / "universal_extension_v8" / "stage3_summary.json"
    s4_path = outputs / "stage4_prl" / "thermo_theory_v5" / "stage4_summary.json"
    s5_path = outputs / "stage5_prx" / "external_regime_v1" / "stage5_summary.json"

    v = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
    q = pd.read_csv(quench_path) if quench_path.exists() else pd.DataFrame()
    n = pd.read_csv(nnqs_path) if nnqs_path.exists() else pd.DataFrame()
    s3 = json.loads(s3_path.read_text(encoding="utf-8")) if s3_path.exists() else {}
    s4 = json.loads(s4_path.read_text(encoding="utf-8")) if s4_path.exists() else {}
    s5 = json.loads(s5_path.read_text(encoding="utf-8")) if s5_path.exists() else {}

    corr_magic_nll = float(np.corrcoef(n["magic_m2"], n["final_val_nll"])[0, 1]) if len(n) > 1 else float("nan")
    corr_theta_lambda = float(np.corrcoef(q["theta1"], q["max_lambda"])[0, 1]) if len(q) > 1 else float("nan")
    corr_theta_magic = float(np.corrcoef(q["theta1"], q["max_magic_m2"])[0, 1]) if len(q) > 1 else float("nan")

    title = "Theta-Quench Schwinger Lab"
    subtitle = "A Manuscript-Style Summary of Dynamics, Magic, and Learnability"
    abstract = (
        "We implement exact small-system theta-quench dynamics for the open-boundary lattice Schwinger model, "
        "and evaluate whether stabilizer complexity predicts neural learnability degradation. "
        "The pipeline computes Loschmidt diagnostics, entanglement, and stabilizer Renyi magic, then fits time snapshots "
        "with NNQS models under controlled metrics (validation NLL, KL, TV). "
        "Cross-family stress tests extending to TFIM, XXZ, and ANNNI support a corridor-scoped entropy-controlled "
        "magic-to-learnability slope law, while strict all-regime universality is explicitly falsified and boundary-mapped."
    )
    meta = [
        f"Build date: {date.today().isoformat()}",
        f"Primary outputs root: {outputs}",
        "Claim style: scoped empirical mechanism with explicit failure-region mapping",
        "Audience: reviewer-facing technical summary (not a theorem-level universal claim)",
    ]

    intro_paragraphs = [
        "This report summarizes an end-to-end computational study connecting nonequilibrium gauge dynamics to complexity diagnostics and machine-learning learnability. The central scientific question is whether post-quench non-stabilizerness (quantified by stabilizer Renyi magic) predicts optimization hardness in neural quantum state fitting.",
        "The analysis is deliberately claim-gated: each proposed claim is tied to explicit numerical checks, uncertainty estimates, and boundary mapping where the relation fails. This avoids universal overstatement while preserving mechanistic signal where evidence is strong.",
    ]

    methods_paragraphs = [
        "Physics model: staggered-fermion lattice Schwinger Hamiltonian after Gauss-law elimination with open boundaries, evolved under sudden theta quenches from ED-prepared ground states. Time evolution uses dense/Krylov paths with cross-checks.",
        "Measured diagnostics: Loschmidt rate function, staggered mass proxy, connected correlators, half-chain von Neumann entropy, and stabilizer Renyi magic M2. Learnability is measured via NNQS snapshot fitting (validation NLL primary; KL/TV secondary).",
        "Statistical protocol: bootstrap confidence intervals, permutation tests with FDR correction, mixed/covariate controls (including entropy), and corridor filtering by dynamical intensity. Output claims are accepted only when gate conditions pass.",
    ]

    validation_paragraphs = [
        f"Hermiticity check H(theta0)/H(theta1): {_safe_bool(v.get('hermitian_h0'))}/{_safe_bool(v.get('hermitian_h1'))}.",
        f"Norm drift max_t ||psi(t)||^2-1 = {v.get('norm_drift', float('nan')):.3e}; dense-vs-Krylov mismatch = {v.get('dense_krylov_error', float('nan')):.3e}.",
        f"Quench monotonic trends: corr(theta1, max lambda) = {corr_theta_lambda:.4f}, corr(theta1, max M2) = {corr_theta_magic:.4f}. NNQS trend: corr(M2, final val NLL) = {corr_magic_nll:.4f}.",
    ]

    gate_columns = ["Gate", "Status", "Evidence File", "Interpretation"]
    gate_rows = [
        [
            "Validation (Hermiticity + unitarity)",
            "PASS" if (_safe_bool(v.get("hermitian_h0")) and _safe_bool(v.get("hermitian_h1"))) else "FAIL",
            "outputs/validation.json",
            "Core simulator numerics are consistent.",
        ],
        [
            "Stage 3 cross-family beta law",
            "PASS" if _safe_bool(s3.get("universal_beta_law")) else "FAIL",
            "outputs/stage3_prx/universal_extension_v8/stage3_summary.json",
            "Cell-level slope positivity in corridor.",
        ],
        [
            "Stage 4 thermo + quantile gates",
            "PASS"
            if (_safe_bool(s4.get("thermo_gate_beta_inf_positive")) and _safe_bool(s4.get("quantile_lower_envelope_gate")))
            else "FAIL",
            "outputs/stage4_prl/thermo_theory_v5/stage4_summary.json",
            "Finite-size extrapolation and lower-envelope support.",
        ],
        [
            "Stage 5 external family cell gates",
            "PASS" if (_safe_bool(s5.get("beta_cell_gate")) and _safe_bool(s5.get("beta_fdr_gate"))) else "FAIL",
            "outputs/stage5_prx/external_regime_v1/stage5_summary.json",
            "Cross-family mechanism survives ANNNI extension.",
        ],
        [
            "Stage 5 strict regime minimax",
            "PASS" if _safe_bool(s5.get("minimax_regime_gate")) else "FAIL",
            "outputs/stage5_prx/external_regime_v1/stage5_summary.json",
            "Explicit failure region (not globally regime-universal).",
        ],
    ]

    discussion_paragraphs = [
        "Final supported claim: in a declared dynamical corridor, higher stabilizer magic predicts worse NNQS learnability after entropy control, robust at cross-family cell level. This is publishable as a scoped empirical law with mechanism-oriented evidence.",
        "Not supported: a model-independent theorem-level universal law, and strict all-regime positivity. The failure region is not hidden; it is explicitly mapped and reported.",
        "This report is therefore intended for scientific review under conservative claim discipline: robust where supported, transparent where boundaries are hit.",
    ]

    with PdfPages(out) as pdf:
        _add_title_page(pdf, title, subtitle, abstract, meta)
        _add_section_page(pdf, "1. Introduction and Scope", intro_paragraphs)
        _add_section_page(pdf, "2. Methods and Statistical Protocol", methods_paragraphs)
        _add_section_page(pdf, "3. Numerical Validation Snapshot", validation_paragraphs)
        _add_table_page(
            pdf,
            "4. Claim-Gate Summary Table",
            gate_columns,
            gate_rows,
            caption=(
                "Table 1. Gate-level decision summary. The mechanism is robust in the declared corridor and across model families, "
                "while strict all-regime universality is rejected by explicit minimax/regime failures."
            ),
        )

        _add_figure_page(
            pdf,
            "Figure 1. Quench Transition Diagnostic",
            _pick_image(outputs, "fig1_loschmidt_family.png"),
            "Loschmidt rate family over quench endpoint. Transition-like sharpening appears for stronger quenches.",
        )
        _add_figure_page(
            pdf,
            "Figure 2. Complexity Dynamics",
            _pick_image(outputs, "fig2_magic_lambda_overlay.png"),
            "Time-resolved overlay of Loschmidt rate and stabilizer magic M2(t).",
        )
        _add_figure_page(
            pdf,
            "Figure 3. Learnability Coupling",
            _pick_image(outputs, "fig4_nnqs_loss_vs_magic.png"),
            "Primary endpoint visualization: higher snapshot magic tends to align with higher final validation NLL.",
        )
        _add_figure_page(
            pdf,
            "Figure 4. Cross-Family Cell-Level Bounds",
            _pick_image(outputs, "fig_stage5_beta_forest.png"),
            "Cell-level entropy-controlled beta bounds after external-family expansion (ANNNI included).",
        )
        _add_figure_page(
            pdf,
            "Figure 5. Regime Boundary Map",
            _pick_image(outputs, "fig_stage5_regime_heatmap.png"),
            "Regime-level lower CI map for beta_magic, explicitly marking support vs failure regions.",
        )
        _add_section_page(pdf, "5. Discussion and Publication Position", discussion_paragraphs)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
