from __future__ import annotations
import os
import traceback
from pathlib import Path
import streamlit as st
import pandas as pd
from typing import Optional

# Imports locales
from eda_copilot.profile import summarize_dataframe
from eda_copilot.viz import (
    save_enhanced_histograms, save_enhanced_boxplots, save_enhanced_barplots,
    save_enhanced_correlation_heatmap, save_missing_data_analysis
)
from eda_copilot.narrative import generate_narrative

# Configuración de página
st.set_page_config(
    page_title="EDA Copilot Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# EDA Copilot Pro\nAnálisis exploratorio automático con IA"
    }
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #2d3748;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-card h4 {
        color: #ffffff !important;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #e2e8f0 !important;
        margin: 0.25rem 0;
    }
    .metric-card ul {
        color: #e2e8f0 !important;
        margin: 0.5rem 0;
    }
    .metric-card li {
        color: #e2e8f0 !important;
    }
    /* Forzar colores en TODOS los elementos dentro de metric-card */
    .metric-card * {
        color: #e2e8f0 !important;
    }
    .metric-card h1, .metric-card h2, .metric-card h3, .metric-card h4, .metric-card h5, .metric-card h6 {
        color: #ffffff !important;
    }
    .metric-card strong, .metric-card b {
        color: #ffffff !important;
    }
    .success-box {
        background: #2f855a;
        border: 1px solid #38a169;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box h4 {
        color: #ffffff !important;
    }
    .success-box p {
        color: #e6fffa !important;
    }
    /* Asegurar que los textos en markdown sean visibles */
    .stMarkdown h4 {
        color: #ffffff !important;
    }
    .stMarkdown p {
        color: #e2e8f0 !important;
    }
    .stMarkdown strong {
        color: #ffffff !important;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }
    .stMarkdown li {
        color: #e2e8f0 !important;
    }
    .stMarkdown ul, .stMarkdown ol {
        color: #e2e8f0 !important;
    }
    /* Forzar texto visible en TODOS los elementos de Streamlit */
    div[data-testid="stMarkdownContainer"] * {
        color: #e2e8f0 !important;
    }
    div[data-testid="stMarkdownContainer"] h1, 
    div[data-testid="stMarkdownContainer"] h2, 
    div[data-testid="stMarkdownContainer"] h3, 
    div[data-testid="stMarkdownContainer"] h4, 
    div[data-testid="stMarkdownContainer"] h5, 
    div[data-testid="stMarkdownContainer"] h6 {
        color: #ffffff !important;
    }
    div[data-testid="stMarkdownContainer"] strong, 
    div[data-testid="stMarkdownContainer"] b {
        color: #ffffff !important;
    }
    /* Estilos específicos para contenedor de narrativa */
    .narrative-container * {
        color: #e2e8f0 !important;
    }
    .narrative-container h1, .narrative-container h2, .narrative-container h3, 
    .narrative-container h4, .narrative-container h5, .narrative-container h6 {
        color: #ffffff !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        line-height: 1.4 !important;
    }
    .narrative-container h1 {
        margin-top: 0 !important;
    }
    .narrative-container p {
        margin-bottom: 1.5rem !important;
        line-height: 1.6 !important;
        text-align: justify !important;
    }
    .narrative-container ul, .narrative-container ol {
        margin-bottom: 1.5rem !important;
        margin-top: 1rem !important;
    }
    .narrative-container li {
        margin-bottom: 0.8rem !important;
        line-height: 1.5 !important;
        padding-left: 0.5rem !important;
    }
    .narrative-container strong, .narrative-container b {
        color: #ffffff !important;
    }
    .narrative-container blockquote {
        margin: 1.5rem 0 !important;
        padding-left: 1rem !important;
        border-left: 3px solid #667eea !important;
    }
    /* Tema oscuro para toda la aplicación */
    .stApp {
        background-color: #1a202c !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #2d3748 !important;
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Muestra el header principal de la aplicación."""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 EDA Copilot Pro</h1>
        <p>Análisis exploratorio automático con visualizaciones avanzadas y narrativa IA</p>
    </div>
    """, unsafe_allow_html=True)

def configure_sidebar() -> tuple[Optional[str], int, str, bool, dict]:
    """Configura la barra lateral con controles."""
    st.sidebar.header("⚙️ Configuración")
    
    # Sección de archivo
    st.sidebar.subheader("📁 Cargar Datos")
    uploaded_file = st.sidebar.file_uploader(
        "Selecciona tu archivo",
        type=["csv", "xlsx", "xls"],
        help="Formatos soportados: CSV, Excel (.xlsx, .xls)"
    )
    
    # Sección de configuración de datos
    st.sidebar.subheader("🔧 Opciones de Carga")
    rows_limit = st.sidebar.number_input(
        "Límite de filas (0 = todas)", 
        min_value=0, 
        value=0, 
        step=1000,
        help="Limita el número de filas para datasets grandes"
    )
    
    # Sección de IA
    st.sidebar.subheader("🤖 Configuración IA")
    model = st.sidebar.text_input(
        "Modelo Ollama", 
        value=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Nombre del modelo de Ollama instalado"
    )
    
    use_llm = st.sidebar.checkbox(
        "Usar IA para narrativa", 
        value=True,
        help="Generar narrativa automática con IA"
    )
    
    # Sección de visualizaciones
    st.sidebar.subheader("📊 Opciones de Gráficos")
    viz_options = {
        "histograms": st.sidebar.checkbox("Histogramas mejorados", True),
        "boxplots": st.sidebar.checkbox("Box & Violin plots", True),
        "barplots": st.sidebar.checkbox("Gráficos de barras", True),
        "correlation": st.sidebar.checkbox("Mapas de correlación", True),
        "missing_analysis": st.sidebar.checkbox("Análisis de faltantes", True)
    }
    
    return uploaded_file, rows_limit, model, use_llm, viz_options

def load_data(uploaded_file, rows_limit: int) -> Optional[pd.DataFrame]:
    """Carga los datos del archivo subido."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, nrows=None if rows_limit == 0 else rows_limit)
        else:
            df = pd.read_excel(uploaded_file, nrows=None if rows_limit == 0 else rows_limit)
        
        if df.empty:
            st.error("❌ El archivo está vacío")
            return None
            
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {str(e)}")
        return None

def display_data_overview(df: pd.DataFrame):
    """Muestra una vista general de los datos."""
    st.header("📋 Vista General de los Datos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", f"{len(df.columns):,}")
    with col3:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        st.metric("Valores faltantes", f"{df.isnull().sum().sum():,}")
    
    # Vista previa de datos
    st.subheader("🔍 Vista Previa")
    st.dataframe(df.head(10), width='stretch')
    
    # Información de tipos de datos
    with st.expander("📊 Información de Columnas"):
        info_df = pd.DataFrame({
            'Tipo': df.dtypes.astype(str),
            'Valores faltantes': df.isnull().sum(),
            '% Faltantes': (df.isnull().sum() / len(df) * 100).round(2),
            'Valores únicos': df.nunique()
        })
        st.dataframe(info_df, width='stretch')

def display_summary_metrics(summary: dict):
    """Muestra métricas resumidas del análisis."""
    st.header("📈 Resumen del Análisis")
    
    # Métricas básicas
    basic_info = summary.get("basic_info", {})
    data_types = summary.get("data_types", {})
    quality = summary.get("data_quality", {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #ffffff !important;">📊 Tipos de Variables</h4>
            <ul style="color: #e2e8f0 !important;">
        """, unsafe_allow_html=True)
        
        for type_name, cols in data_types.items():
            if cols:
                st.markdown(f"""<li style="color: #e2e8f0 !important;">
                    <strong style="color: #ffffff !important;">{type_name.title()}:</strong> 
                    <span style="color: #e2e8f0 !important;">{len(cols)}</span>
                </li>""", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    with col2:
        missing = quality.get("missing", {})
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffffff !important;">⚠️ Calidad de Datos</h4>
            <p style="color: #e2e8f0 !important;"><strong style="color: #ffffff !important;">Faltantes:</strong> {missing.get('total_missing', 0):,} 
               ({missing.get('missing_percentage', 0):.1f}%)</p>
            <p style="color: #e2e8f0 !important;"><strong style="color: #ffffff !important;">Duplicados:</strong> {quality.get('cardinality', {}).get('duplicate_rows', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        correlations = summary.get("statistical_analysis", {}).get("correlations", {})
        outliers = quality.get("outliers", {})
        vars_with_outliers = sum(1 for v in outliers.values() 
                               if isinstance(v, dict) and v.get('iqr', 0) > 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #ffffff !important;">🔍 Estadísticas</h4>
            <p style="color: #e2e8f0 !important;"><strong style="color: #ffffff !important;">Correlaciones fuertes:</strong> {len(correlations.get('strong_correlations', []))}</p>
            <p style="color: #e2e8f0 !important;"><strong style="color: #ffffff !important;">Variables con outliers:</strong> {vars_with_outliers}</p>
        </div>
        """, unsafe_allow_html=True)

def generate_visualizations(df: pd.DataFrame, summary: dict, 
                           viz_options: dict, output_dir: Path) -> list[Path]:
    """Genera todas las visualizaciones seleccionadas."""
    figs_dir = output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    all_figures = []
    data_types = summary.get("data_types", {})
    numeric_cols = data_types.get("numeric", [])
    categorical_cols = data_types.get("categorical", [])
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = sum(viz_options.values())
    current_step = 0
    
    try:
        if viz_options.get("histograms") and numeric_cols:
            status_text.text("Generando histogramas mejorados...")
            all_figures.extend(save_enhanced_histograms(df, figs_dir, numeric_cols))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        if viz_options.get("boxplots") and numeric_cols:
            status_text.text("Generando box & violin plots...")
            all_figures.extend(save_enhanced_boxplots(df, figs_dir, numeric_cols))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        if viz_options.get("barplots") and categorical_cols:
            status_text.text("Generando gráficos de barras...")
            all_figures.extend(save_enhanced_barplots(df, figs_dir, categorical_cols))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        if viz_options.get("correlation") and len(numeric_cols) >= 2:
            status_text.text("Generando mapas de correlación...")
            all_figures.extend(save_enhanced_correlation_heatmap(df, figs_dir, numeric_cols))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        if viz_options.get("missing_analysis"):
            status_text.text("Analizando datos faltantes...")
            all_figures.extend(save_missing_data_analysis(df, figs_dir))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Visualizaciones completadas")
        
    except Exception as e:
        st.error(f"❌ Error generando visualizaciones: {str(e)}")
        
    return all_figures

def display_visualizations(figures: list[Path]):
    """Muestra las visualizaciones generadas."""
    if not figures:
        st.warning("⚠️ No se generaron visualizaciones")
        return
    
    st.header("📊 Visualizaciones")
    
    # Organizar por tipo
    viz_types = {
        "Histogramas": [f for f in figures if "hist" in f.name],
        "Box & Violin Plots": [f for f in figures if "box" in f.name],
        "Gráficos de Barras": [f for f in figures if "bar" in f.name],
        "Correlación": [f for f in figures if "correlation" in f.name],
        "Análisis de Faltantes": [f for f in figures if "missing" in f.name]
    }
    
    for viz_type, viz_files in viz_types.items():
        if viz_files:
            with st.expander(f"📈 {viz_type} ({len(viz_files)})", expanded=True):
                # Mostrar en columnas para mejor organización
                cols = st.columns(min(2, len(viz_files)))
                for i, fig_path in enumerate(viz_files):
                    with cols[i % 2]:
                        st.image(str(fig_path), width="stretch")

def main():
    """Función principal de la aplicación."""
    display_header()
    
    # Configuración de la barra lateral
    uploaded_file, rows_limit, model, use_llm, viz_options = configure_sidebar()
    
    if uploaded_file is None:
        st.info("👈 Por favor, sube un archivo CSV o Excel para comenzar el análisis")
        st.markdown("""
        ### 🎯 Características de EDA Copilot Pro:
        
        - **📊 Análisis Estadístico Avanzado**: Detección automática de tipos de datos, outliers múltiples métodos
        - **🎨 Visualizaciones Mejoradas**: Histogramas con distribución normal, box & violin plots, mapas de correlación
        - **🤖 Narrativa con IA**: Resúmenes ejecutivos generados automáticamente
        - **🔍 Análisis de Calidad**: Detección de patrones en datos faltantes y duplicados
        - **⚡ Optimizado**: Manejo eficiente de datasets grandes
        
        ### 📋 Formatos Soportados:
        - CSV (comma-separated values)
        - Excel (.xlsx, .xls)
        """)
        return
    
    try:
        # Cargar datos
        with st.spinner("🔄 Cargando datos..."):
            df = load_data(uploaded_file, rows_limit)
        
        if df is None:
            return
        
        # Mostrar vista general
        display_data_overview(df)
        
        # Análisis del perfil
        with st.spinner("🔍 Analizando datos..."):
            summary = summarize_dataframe(df)
        
        # Mostrar métricas resumidas
        display_summary_metrics(summary)
        
        # Generar visualizaciones
        output_dir = Path("reports")
        with st.spinner("🎨 Generando visualizaciones..."):
            figures = generate_visualizations(df, summary, viz_options, output_dir)
        
        # Mostrar visualizaciones
        display_visualizations(figures)
        
        # Generar narrativa
        st.header("📝 Análisis Narrativo")
        
        # Crear tabs para diferentes tipos de narrativa
        tab1, tab2 = st.tabs(["🤖 Narrativa IA", "📋 Resumen Técnico"])
        
        with tab1:
            if use_llm:
                with st.spinner("🤖 Generando narrativa con IA..."):
                    narrative = generate_narrative(summary, model=model, use_llm=True)
            else:
                narrative = generate_narrative(summary, use_llm=False)
            
            # Mostrar la narrativa con mejor formato
            if narrative and narrative.strip():
                # Crear un contenedor con mejor estilo para la narrativa
                st.markdown("""
                <div class="narrative-container" style="background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%); 
                           padding: 2rem; border-radius: 15px; 
                           box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                           border-left: 5px solid #667eea;
                           color: #e2e8f0 !important;">
                """, unsafe_allow_html=True)
                
                # Envolver la narrativa en un div con estilos forzados
                st.markdown(f"""
                <div style="color: #e2e8f0 !important;">
                {narrative}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Agregar información sobre el modelo usado
                if use_llm:
                    st.info(f"📡 Generado con modelo: **{model}**")
                else:
                    st.info("🛠️ Generado con plantilla automática")
            else:
                st.error("❌ No se pudo generar la narrativa")
                
        with tab2:
            # Mostrar resumen técnico detallado
            st.subheader("📊 Métricas del Análisis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔢 Variables Numéricas:**", unsafe_allow_html=True)
                numeric_cols = summary.get("data_types", {}).get("numeric", [])
                if numeric_cols:
                    for col in numeric_cols[:5]:  # Mostrar solo las primeras 5
                        st.markdown(f"<span style='color: #495057;'>• {col}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: #495057;'>No hay variables numéricas</span>", unsafe_allow_html=True)
                    
                st.markdown("**📝 Variables Categóricas:**", unsafe_allow_html=True)
                cat_cols = summary.get("data_types", {}).get("categorical", [])
                if cat_cols:
                    for col in cat_cols[:5]:  # Mostrar solo las primeras 5
                        st.markdown(f"<span style='color: #495057;'>• {col}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: #495057;'>No hay variables categóricas</span>", unsafe_allow_html=True)
            
            with col2:
                quality = summary.get("data_quality", {})
                missing = quality.get("missing", {})
                
                st.markdown("**⚠️ Calidad de Datos:**", unsafe_allow_html=True)
                st.markdown(f"<span style='color: #495057;'>• Valores faltantes: {missing.get('total_missing', 0):,}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: #495057;'>• % Faltantes: {missing.get('missing_percentage', 0):.1f}%</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: #495057;'>• Filas duplicadas: {quality.get('cardinality', {}).get('duplicate_rows', 0):,}</span>", unsafe_allow_html=True)
                
                correlations = summary.get("statistical_analysis", {}).get("correlations", {})
                strong_corr = correlations.get("strong_correlations", [])
                st.markdown(f"<span style='color: #495057;'>• Correlaciones fuertes: {len(strong_corr)}</span>", unsafe_allow_html=True)
            
            # Información técnica en expander
            with st.expander("🔧 Detalles Técnicos Completos"):
                st.json({
                    "configuracion": {
                        "modelo_ia": model if use_llm else "Sin IA",
                        "usa_ia": use_llm,
                        "total_variables": len(numeric_cols) + len(cat_cols)
                    },
                    "calidad_datos": {
                        "valores_faltantes": missing.get("total_missing", 0),
                        "porcentaje_faltantes": missing.get("missing_percentage", 0),
                        "filas_duplicadas": quality.get("cardinality", {}).get("duplicate_rows", 0)
                    },
                    "estadisticas": {
                        "correlaciones_fuertes": len(strong_corr),
                        "variables_numericas": len(numeric_cols),
                        "variables_categoricas": len(cat_cols)
                    }
                })
        
        # Resumen final
        file_name = uploaded_file.name if hasattr(uploaded_file, 'name') else 'archivo_cargado'
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Análisis Completado</h4>
            <p><strong>Archivo:</strong> {file_name}</p>
            <p><strong>Dimensiones:</strong> {len(df):,} filas × {len(df.columns)} columnas</p>
            <p><strong>Visualizaciones:</strong> {len(figures)} gráficos generados</p>
            <p><strong>Reportes guardados en:</strong> {output_dir.resolve()}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
        if st.checkbox("🔧 Mostrar detalles técnicos"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()