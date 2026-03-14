from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else ROOT / p


def _copy_figures(outputs: Path, report_dir: Path) -> int:
    report_figs = report_dir / "figs"
    report_figs.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen: set[str] = set()

    figure_dirs = [
        outputs / "figs",
        outputs / "stage2_prlx" / "universal_scan_v2" / "figs",
        outputs / "stage3_prx" / "universal_extension_v8" / "figs",
        outputs / "stage4_prl" / "thermo_theory_v5" / "figs",
        outputs / "stage5_prx" / "external_regime_v1" / "figs",
    ]
    for fig_dir in figure_dirs:
        if not fig_dir.exists():
            continue
        for src in sorted(fig_dir.glob("*.png")):
            # Prefer the first occurrence to keep deterministic figure provenance.
            if src.name in seen:
                continue
            shutil.copy2(src, report_figs / src.name)
            copied += 1
            seen.add(src.name)
    return copied


def _run_tectonic(tex_file: Path, tectonic_cmd: str, keep_intermediates: bool) -> None:
    cmd = [tectonic_cmd, tex_file.name]
    if keep_intermediates:
        cmd.insert(1, "--keep-intermediates")
        cmd.insert(2, "--keep-logs")
    subprocess.run(cmd, cwd=tex_file.parent, check=True)


def _build_fallback_pdf(report_dir: Path, out_pdf: Path) -> None:
    figs_dir = report_dir / "figs"
    images = sorted(figs_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No figure assets found for fallback PDF build in {figs_dir}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.text(
            0.5,
            0.8,
            "Theta-Quench Schwinger Lab\nFallback Report Artifact",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.55,
            "Tectonic was unavailable, so this CI build generated\n"
            "a lightweight figure bundle PDF instead of the full LaTeX manuscript.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.3,
            f"Figures bundled: {len(images)}",
            ha="center",
            va="center",
            fontsize=11,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for img_path in images[:12]:
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path.name, fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile manuscript PDF from report/main.tex")
    parser.add_argument("--outputs", type=str, default="outputs", help="Outputs root used to source figure assets")
    parser.add_argument("--out", type=str, default="report/short_report.pdf", help="Final PDF path")
    parser.add_argument("--tex", type=str, default="report/main.tex", help="LaTeX manuscript file")
    parser.add_argument("--tectonic", type=str, default="tectonic", help="Path to the tectonic executable")
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep .aux/.log artifacts from LaTeX compilation",
    )
    args = parser.parse_args()

    outputs = _resolve_path(args.outputs)
    out_pdf = _resolve_path(args.out)
    tex_file = _resolve_path(args.tex)

    if not tex_file.exists():
        raise FileNotFoundError(f"Missing manuscript file: {tex_file}")

    copied = _copy_figures(outputs=outputs, report_dir=tex_file.parent)
    if shutil.which(args.tectonic) is None:
        _build_fallback_pdf(report_dir=tex_file.parent, out_pdf=out_pdf)
        print(
            f"Copied {copied} figure(s) into {tex_file.parent / 'figs'}"
        )
        print(
            "tectonic not found on PATH; wrote fallback figure-bundle PDF "
            f"to {out_pdf}"
        )
        return

    _run_tectonic(tex_file=tex_file, tectonic_cmd=args.tectonic, keep_intermediates=args.keep_intermediates)

    compiled_pdf = tex_file.with_suffix(".pdf")
    if not compiled_pdf.exists():
        raise FileNotFoundError(f"Tectonic finished but did not produce: {compiled_pdf}")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if compiled_pdf.resolve() != out_pdf.resolve():
        shutil.copy2(compiled_pdf, out_pdf)

    print(f"Copied {copied} figure(s) into {tex_file.parent / 'figs'}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
