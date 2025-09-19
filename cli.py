from __future__ import annotations
import argparse
from pathlib import Path
from .data_loader import load_data
from .profile import summarize_dataframe
from .viz import save_histograms, save_boxplots, save_barplots, save_corr_heatmap
from .narrative import generate_narrative
from .report import build_report

def main():
    parser = argparse.ArgumentParser(description="EDA Copilot - CLI")
    parser.add_argument("--input", required=True, help="Ruta al CSV/Excel")
    parser.add_argument("--outdir", default="reports", help="Carpeta de salida")
    parser.add_argument("--rows", type=int, default=None, help="Límite de filas a leer")
    parser.add_argument("--model", type=str, default=None, help="Modelo de Ollama (ej. llama3)")
    parser.add_argument("--no-llm", action="store_true", help="Desactiva LLM y usa plantilla")
    args = parser.parse_args()

    infile = Path(args.input)
    outdir = Path(args.outdir)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(infile, nrows=args.rows)
    summary = summarize_dataframe(df)

    numeric_cols = summary.get("numeric_cols", [])
    cat_cols = summary.get("cat_cols", [])

    figs = []
    figs += save_histograms(df, figures_dir, numeric_cols)
    figs += save_boxplots(df, figures_dir, numeric_cols)
    figs += save_barplots(df, figures_dir, cat_cols)
    figs += save_corr_heatmap(df, figures_dir, numeric_cols)

    narrative_text = generate_narrative(summary, model=args.model, use_llm=not args.no_llm)

    build_report(summary, figs, outdir, infile.name, narrative_text)

    print(f"✅ Reporte generado en: {outdir.resolve()}")

if __name__ == "__main__":
    main()
