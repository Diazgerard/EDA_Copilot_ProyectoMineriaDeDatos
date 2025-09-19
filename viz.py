from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Configuración global de matplotlib y seaborn
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
warnings.filterwarnings('ignore', category=FutureWarning)

def _ensure_dir(path: Path) -> None:
    """Asegurar que el directorio existe."""
    path.mkdir(parents=True, exist_ok=True)

def _clean_column_name(col_name: str) -> str:
    """Limpiar nombres de columnas para archivos."""
    return "".join(c for c in col_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

def _configure_plot(title: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None, 
                   rotation: int = 0, grid: bool = True) -> None:
    """Configurar elementos comunes de los gráficos."""
    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    if xlabel:
        plt.xlabel(xlabel, fontsize=11)
    if ylabel:
        plt.ylabel(ylabel, fontsize=11)
    if rotation:
        plt.xticks(rotation=rotation)
    if grid:
        plt.grid(True, alpha=0.3)
    try:
        plt.tight_layout()
    except:
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

def save_enhanced_histograms(df: pd.DataFrame, outdir: Path, numeric_cols: List[str], 
                           max_plots: int = 8) -> List[Path]:
    """Genera histogramas mejorados con estadísticas y distribución normal superpuesta."""
    _ensure_dir(outdir)
    paths = []
    
    for i, col in enumerate(numeric_cols[:max_plots]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Datos sin NaN
        data = df[col].dropna()
        if len(data) == 0:
            continue
            
        # Histograma
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, density=True, 
                                  edgecolor='black', linewidth=0.5)
        
        # Estadísticas
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        # Líneas de referencia
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f}')
        
        # Distribución normal superpuesta
        x_norm = np.linspace(data.min(), data.max(), 100)
        y_norm = ((1/(std_val * np.sqrt(2 * np.pi))) * 
                 np.exp(-0.5 * ((x_norm - mean_val) / std_val) ** 2))
        ax.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.8, label='Normal teórica')
        
        _configure_plot(
            title=f'Distribución de {col}\n'
                  f'Asimetría: {data.skew():.2f} | Curtosis: {data.kurtosis():.2f}',
            xlabel=col,
            ylabel='Densidad'
        )
        
        ax.legend()
        
        clean_name = _clean_column_name(col)
        p = outdir / f"hist_enhanced_{i+1}_{clean_name}.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(p)
    
    return paths

def save_enhanced_boxplots(df: pd.DataFrame, outdir: Path, numeric_cols: List[str], 
                          max_plots: int = 8) -> List[Path]:
    """Genera boxplots mejorados con violin plots y estadísticas."""
    _ensure_dir(outdir)
    paths = []
    
    for i, col in enumerate(numeric_cols[:max_plots]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        data = df[col].dropna()
        if len(data) == 0:
            continue
        
        # Boxplot tradicional
        box_plot = ax1.boxplot(data, patch_artist=True, notch=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        ax1.set_title(f'Boxplot - {col}')
        ax1.set_ylabel(col)
        ax1.grid(True, alpha=0.3)
        
        # Violin plot con distribución
        parts = ax2.violinplot(data, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightcoral')
            pc.set_alpha(0.7)
        
        ax2.set_title(f'Violin Plot - {col}')
        ax2.set_ylabel(col)
        ax2.grid(True, alpha=0.3)
        
        # Estadísticas como texto
        q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_whisker = q1 - 1.5 * iqr
        upper_whisker = q3 + 1.5 * iqr
        outliers = data[(data < lower_whisker) | (data > upper_whisker)]
        
        stats_text = f"""Estadísticas:
Q1: {q1:.2f}
Mediana: {median:.2f}
Q3: {q3:.2f}
IQR: {iqr:.2f}
Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)"""
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        
        try:
            plt.tight_layout()
        except:
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        clean_name = _clean_column_name(col)
        p = outdir / f"box_enhanced_{i+1}_{clean_name}.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(p)
    
    return paths

def save_enhanced_barplots(df: pd.DataFrame, outdir: Path, cat_cols: List[str], 
                          max_plots: int = 8) -> List[Path]:
    """Genera gráficos de barras mejorados para variables categóricas."""
    _ensure_dir(outdir)
    paths = []
    
    for i, col in enumerate(cat_cols[:max_plots]):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Contar valores y ordenar
        value_counts = df[col].value_counts().head(15)  # Top 15 categorías
        
        if len(value_counts) == 0:
            continue
        
        # Gráfico de barras con colores
        import matplotlib.cm as cm
        colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(value_counts)))
        bars = ax.bar(range(len(value_counts)), value_counts.values.tolist(), color=colors)
        
        # Personalización
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Añadir porcentajes en las barras
        total = df[col].notna().sum()
        for j, (bar, count) in enumerate(zip(bars, value_counts.values)):
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + count*0.01,
                   f'{count}\n({percentage:.1f}%)', 
                   ha='center', va='bottom', fontsize=9)
        
        _configure_plot(
            title=f'Distribución de {col}\n'
                  f'Categorías únicas: {df[col].nunique()} | Valores faltantes: {df[col].isna().sum()}',
            xlabel=col,
            ylabel='Frecuencia',
            rotation=45
        )
        
        clean_name = _clean_column_name(col)
        p = outdir / f"bar_enhanced_{i+1}_{clean_name}.png"
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        paths.append(p)
    
    return paths

def save_enhanced_correlation_heatmap(df: pd.DataFrame, outdir: Path, 
                                     numeric_cols: List[str]) -> List[Path]:
    """Genera mapas de calor de correlación mejorados."""
    _ensure_dir(outdir)
    paths = []
    
    if len(numeric_cols) < 2:
        return paths
    
    # Correlación de Pearson
    corr_pearson = df[numeric_cols].corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Mapa de calor de Pearson
    mask1 = np.triu(np.ones_like(corr_pearson, dtype=bool))
    sns.heatmap(corr_pearson, mask=mask1, annot=True, cmap='RdBu_r', center=0,
               square=True, linewidths=0.5, ax=ax1, fmt='.2f')
    ax1.set_title('Correlación de Pearson', fontsize=14, fontweight='bold')
    
    # Correlación de Spearman
    corr_spearman = df[numeric_cols].corr(method='spearman')
    mask2 = np.triu(np.ones_like(corr_spearman, dtype=bool))
    sns.heatmap(corr_spearman, mask=mask2, annot=True, cmap='RdBu_r', center=0,
               square=True, linewidths=0.5, ax=ax2, fmt='.2f')
    ax2.set_title('Correlación de Spearman', fontsize=14, fontweight='bold')
    
    try:
        plt.tight_layout()
    except:
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    p = outdir / "correlation_enhanced_heatmap.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    paths.append(p)
    
    return paths

def save_missing_data_analysis(df: pd.DataFrame, outdir: Path) -> List[Path]:
    """Genera visualizaciones de análisis de datos faltantes."""
    _ensure_dir(outdir)
    paths = []
    
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) == 0:
        return paths
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico de barras de datos faltantes
    missing_data.plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title('Valores Faltantes por Variable', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Cantidad de Valores Faltantes')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Mapa de calor de patrones de datos faltantes
    missing_matrix = df[missing_data.index].isnull().astype(int)
    sns.heatmap(missing_matrix.T, cbar=True, ax=ax2, cmap='viridis', 
               yticklabels=True, xticklabels=False)
    ax2.set_title('Patrones de Datos Faltantes', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Observaciones')
    
    try:
        plt.tight_layout()
    except:
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    p = outdir / "missing_data_analysis.png"
    plt.savefig(p, bbox_inches="tight", dpi=150)
    plt.close()
    paths.append(p)
    
    return paths

# Funciones de compatibilidad con las versiones anteriores
def save_histograms(df: pd.DataFrame, outdir: Path, numeric_cols: List[str], 
                   max_plots: int = 8) -> List[Path]:
    """Función de compatibilidad para histogramas."""
    return save_enhanced_histograms(df, outdir, numeric_cols, max_plots)

def save_boxplots(df: pd.DataFrame, outdir: Path, numeric_cols: List[str], 
                 max_plots: int = 8) -> List[Path]:
    """Función de compatibilidad para boxplots."""
    return save_enhanced_boxplots(df, outdir, numeric_cols, max_plots)

def save_barplots(df: pd.DataFrame, outdir: Path, cat_cols: List[str], 
                 max_plots: int = 8) -> List[Path]:
    """Función de compatibilidad para gráficos de barras."""
    return save_enhanced_barplots(df, outdir, cat_cols, max_plots)

def save_corr_heatmap(df: pd.DataFrame, outdir: Path, numeric_cols: List[str]) -> List[Path]:
    """Función de compatibilidad para mapas de correlación."""
    return save_enhanced_correlation_heatmap(df, outdir, numeric_cols)
