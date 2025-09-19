from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional

def load_data(path: str | Path, nrows: Optional[int] = None, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Carga CSV o Excel en un DataFrame.

    - Detecta por extensión.

    - Permite limitar filas con nrows.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    ext = path.suffix.lower()
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path, nrows=nrows)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name, nrows=nrows)
    else:
        raise ValueError(f"Extensión no soportada: {ext}")
