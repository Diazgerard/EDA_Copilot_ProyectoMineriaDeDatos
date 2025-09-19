# EDA Copilot (con Ollama)

Un asistente de **Análisis Exploratorio de Datos (EDA)** que:
- Carga datasets (.csv, .xlsx).
- Calcula estadísticas y perfiles de variables.
- Genera visualizaciones claves.
- Produce un **reporte narrado** con ayuda de un LLM de **Ollama**.

> Genera un `report.md` y `report.html` dentro de `reports/` y guarda figuras en `reports/figures/`.

## 🧰 Requisitos

- Python 3.9+
- [Ollama](https://ollama.com) instalado y ejecutándose (`ollama serve`) si quieres la narrativa automática.
- Modelo sugerido: `llama3` (puedes cambiarlo con la variable de entorno `OLLAMA_MODEL`).

```bash
# Instalar dependencias
pip install -r requirements.txt

# (opcional) Descargar un modelo en Ollama
# ollama pull llama3
```

## 🚀 Uso por CLI

```bash
python -m eda_copilot.cli --input examples/sample.csv --outdir reports --model llama3
```

Parámetros útiles:
- `--input`: ruta al CSV/Excel.
- `--outdir`: carpeta de salida (por defecto `reports`).
- `--rows`: número máximo de filas a leer (para pruebas).
- `--model`: nombre del modelo Ollama (por defecto `llama3`).
- `--no-llm`: desactiva la narrativa por LLM y usa plantilla.

## 🖥️ App (Streamlit)

```bash
streamlit run eda_copilot/app_streamlit.py
```

Sube un archivo y obtén el EDA + narrativa. Si Ollama no está disponible, verás un aviso y se usará un texto de plantilla.

## 🧪 Prueba rápida con dataset de ejemplo

```bash
python -m eda_copilot.cli --input examples/sample.csv
```

## 📁 Estructura

```
eda_copilot/
├─ eda_copilot/
│  ├─ __init__.py
│  ├─ data_loader.py
│  ├─ profile.py
│  ├─ viz.py
│  ├─ narrative.py
│  ├─ report.py
│  ├─ cli.py
│  └─ app_streamlit.py
├─ examples/
│  └─ sample.csv
├─ reports/
│  └─ figures/   # se crean automáticamente
├─ tests/
│  └─ test_smoke.py
├─ requirements.txt
└─ README.md
```

## ⚠️ Notas

- Este proyecto evita dependencias pesadas. Para PDF podrías usar `weasyprint` o `wkhtmltopdf` si quieres exportar desde `report.html`.
- Las visualizaciones usan **matplotlib**.
- Si el servidor de Ollama no está disponible o falla, se usa una **narrativa fallback**.

¡Felices análisis! 📊
