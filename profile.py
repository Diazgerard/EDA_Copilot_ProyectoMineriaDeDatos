from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings
from scipy import stats

def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detecta tipos de datos más precisos que pandas básico."""
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    text_cols = []
    
    for col in df.columns:
        series = df[col]
        
        # Skip if all NaN
        if series.isna().all():
            continue
            
        # Check for datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
        # Check for numeric
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        # Check if object could be categorical
        elif series.dtype == 'object':
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.5 and series.nunique() < 50:  # Heurística para categorías
                categorical_cols.append(col)
            else:
                text_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "text": text_cols
    }

def detect_outliers_multiple_methods(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """Detecta outliers usando múltiples métodos."""
    outlier_results = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        outliers = {}
        
        # Método IQR
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        outliers['iqr'] = int(((series < lower_iqr) | (series > upper_iqr)).sum())
        
        # Método Z-score (> 3 desviaciones estándar)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z_scores = np.abs(stats.zscore(series))
            outliers['zscore'] = int((z_scores > 3).sum())
        
        # Método percentil (< 1% o > 99%)
        p1, p99 = series.quantile([0.01, 0.99])
        outliers['percentile'] = int(((series < p1) | (series > p99)).sum())
        
        outlier_results[col] = outliers
    
    return outlier_results

def analyze_distributions(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Analiza la distribución de variables numéricas."""
    distributions = {}
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:  # Muy pocos datos para análisis estadístico
            continue
            
        stats_dict = {
            'skewness': float(series.skew()) if pd.notna(series.skew()) else 0.0,
            'kurtosis': float(series.kurtosis()) if pd.notna(series.kurtosis()) else 0.0,
            'normality_shapiro_p': None,
            'cv': float(series.std() / series.mean()) if series.mean() != 0 and pd.notna(series.mean()) else float('inf')
        }
        
        # Test de normalidad (solo para muestras pequeñas)
        if len(series) <= 5000:
            try:
                _, p_value = stats.shapiro(series.sample(min(5000, len(series))))
                stats_dict['normality_shapiro_p'] = float(p_value)
            except:
                pass
        
        distributions[col] = stats_dict
    
    return distributions

def analyze_correlations_advanced(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Análisis avanzado de correlaciones."""
    if len(numeric_cols) < 2:
        return {}
    
    # Correlación de Pearson
    corr_pearson = df[numeric_cols].corr()
    
    # Correlación de Spearman (para relaciones no lineales)
    corr_spearman = df[numeric_cols].corr(method='spearman')
    
    # Encontrar correlaciones fuertes
    strong_correlations = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if j <= i:
                continue
            try:
                pearson_val = float(corr_pearson.loc[col1, col2])
                spearman_val = float(corr_spearman.loc[col1, col2])
                
                if (pd.notna(pearson_val) and abs(pearson_val) >= 0.7) or (pd.notna(spearman_val) and abs(spearman_val) >= 0.7):
                    strong_correlations.append({
                        'var1': col1,
                        'var2': col2,
                        'pearson': pearson_val if pd.notna(pearson_val) else 0.0,
                        'spearman': spearman_val if pd.notna(spearman_val) else 0.0
                    })
            except (ValueError, TypeError):
                continue
    
    return {
        'correlation_matrix_pearson': corr_pearson,
        'correlation_matrix_spearman': corr_spearman,
        'strong_correlations': strong_correlations
    }

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Análisis exploratorio completo y mejorado de un DataFrame.
    """
    # Información básica
    n_rows, n_cols = df.shape
    memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
    
    # Detección mejorada de tipos de datos
    data_types = detect_data_types(df)
    
    # Análisis de valores faltantes
    missing_analysis = {
        'counts': df.isna().sum().to_dict(),
        'percentages': (df.isna().sum() / len(df) * 100).to_dict(),
        'total_missing': int(df.isna().sum().sum()),
        'missing_percentage': float(df.isna().sum().sum() / (n_rows * n_cols) * 100)
    }
    
    # Análisis de cardinalidad
    cardinality = {
        'nunique': df.nunique(dropna=True).to_dict(),
        'duplicate_rows': int(df.duplicated().sum())
    }
    
    # Estadísticas descriptivas mejoradas
    numeric_cols = data_types['numeric']
    desc_numeric = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T if numeric_cols else pd.DataFrame()
    
    # Análisis de outliers con múltiples métodos
    outliers = detect_outliers_multiple_methods(df, numeric_cols)
    
    # Análisis de distribuciones
    distributions = analyze_distributions(df, numeric_cols)
    
    # Análisis avanzado de correlaciones
    correlation_analysis = analyze_correlations_advanced(df, numeric_cols)
    
    # Análisis de variables categóricas mejorado
    categorical_analysis = {}
    for col in data_types['categorical']:
        value_counts = df[col].value_counts(dropna=False)
        categorical_analysis[col] = {
            'top_categories': value_counts.head(10).to_dict(),
            'entropy': float(stats.entropy(value_counts.values)),
            'mode': value_counts.index[0] if len(value_counts) > 0 else None,
            'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        }
    
    # Resumen ejecutivo
    summary = {
        # Información básica
        "basic_info": {
            "shape": {"rows": int(n_rows), "cols": int(n_cols)},
            "memory_usage_mb": float(memory_usage),
            "dtypes": df.dtypes.astype(str).to_dict()
        },
        
        # Tipos de datos detectados
        "data_types": data_types,
        
        # Análisis de calidad de datos
        "data_quality": {
            "missing": missing_analysis,
            "cardinality": cardinality,
            "outliers": outliers
        },
        
        # Análisis estadístico
        "statistical_analysis": {
            "descriptive_stats": desc_numeric,
            "distributions": distributions,
            "correlations": correlation_analysis
        },
        
        # Análisis categórico
        "categorical_analysis": categorical_analysis,
        
        # Campos heredados para compatibilidad
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "numeric_cols": numeric_cols,
        "cat_cols": data_types['categorical'],
        "missing": missing_analysis['counts'],
        "outliers": {col: data['iqr'] for col, data in outliers.items()},
        "corr": correlation_analysis.get('correlation_matrix_pearson', pd.DataFrame()),
        "top_categories": {col: data['top_categories'] for col, data in categorical_analysis.items()}
    }
    
    return summary
