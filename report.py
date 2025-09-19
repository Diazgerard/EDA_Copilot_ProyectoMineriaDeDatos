from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from jinja2 import Template
import markdown

REPORT_TEMPLATE = Template("""
# Reporte de Análisis Exploratorio de Datos

**Archivo analizado:** {{ filename }}
**Filas x Columnas:** {{ shape.rows }} x {{ shape.cols }}

## 1) Perfil del dataset
### Tipos de datos
{{ dtypes_md }}

### Valores únicos por columna
{{ nunique_md }}

### Valores faltantes por columna
{{ missing_md }}

## 2) Estadísticas descriptivas (numéricas)
{{ desc_md }}

## 3) Visualizaciones
{% for fig in figures %}
![Figura]({{ fig }})
{% endfor %}

## 4) Narrativa
{{ narrative }}

*Generado automáticamente por EDA Copilot.*
""")

def to_markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(no aplica)_"
    return df.to_markdown(index=True)

def build_report(
    summary: Dict[str, Any],
    figures: List[str],
    outdir: Path,
    filename: str,
    narrative_text: str
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)

    dtypes = pd.DataFrame({"dtype": pd.Series(summary.get("dtypes", {}))})
    nunique = pd.DataFrame({"n_unique": pd.Series(summary.get("nunique", {}))})
    missing = pd.DataFrame({"missing": pd.Series(summary.get("missing", {}))})
    desc = summary.get("desc_numeric")
    if isinstance(desc, pd.DataFrame):
        desc_md = to_markdown_table(desc)
    else:
        desc_md = "_(no hay columnas numéricas)_"

    content = REPORT_TEMPLATE.render(
        filename=filename,
        shape=summary.get("shape", {"rows": 0, "cols": 0}),
        dtypes_md=to_markdown_table(dtypes),
        nunique_md=to_markdown_table(nunique),
        missing_md=to_markdown_table(missing),
        desc_md=desc_md,
        figures=[str(Path(f).as_posix()) for f in figures],
        narrative=narrative_text
    )

    md_path = outdir / "report.md"
    md_path.write_text(content, encoding="utf-8")

    # HTML render
    html = markdown.markdown(content, extensions=["tables"])
    html_path = outdir / "report.html"
    html_path.write_text(html, encoding="utf-8")

    return md_path
