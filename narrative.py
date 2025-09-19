from __future__ import annotations
import os
from typing import Dict, Any, List, Optional
import warnings

def _format_basic_info(summary: Dict[str, Any]) -> str:
    """Formatea información básica del dataset."""
    basic = summary.get("basic_info", {})
    shape = basic.get("shape", summary.get("shape", {}))
    
    sections = [
        f"## 📊 Información General",
        "", 
        f"- **Dimensiones**: {shape.get('rows', 0):,} filas × {shape.get('cols', 0)} columnas",
        "",  # Línea vacía
        f"- **Memoria utilizada**: {basic.get('memory_usage_mb', 0):.1f} MB"
    ]
    return "\n".join(sections)

def _format_data_types(summary: Dict[str, Any]) -> str:
    """Formatea información sobre tipos de datos."""
    data_types = summary.get("data_types", {})
    if not data_types:
        return ""
    
    sections = ["## 🔍 Tipos de Variables", ""] 
    
    for type_name, cols in data_types.items():
        if cols and type_name != "text": 
            sections.append(f"- **{type_name.title()}** ({len(cols)}): {', '.join(cols[:5])}" + 
                          ("..." if len(cols) > 5 else ""))
            sections.append("")  
    
    return "\n".join(sections)

def _format_data_quality(summary: Dict[str, Any]) -> str:
    """Formatea información sobre calidad de datos."""
    quality = summary.get("data_quality", {})
    missing = quality.get("missing", {})
    cardinality = quality.get("cardinality", {})
    
    sections = ["## ⚠️ Calidad de Datos", ""]  
    
    # Datos faltantes
    if missing.get("total_missing", 0) > 0:
        sections.append(f"- **Valores faltantes**: {missing.get('total_missing', 0):,} " +
                       f"({missing.get('missing_percentage', 0):.1f}% del total)")
        sections.append("")  
        
        
        missing_counts = missing.get("counts", {})
        top_missing = sorted([(k, v) for k, v in missing_counts.items() if v > 0], 
                           key=lambda x: x[1], reverse=True)[:3]
        if top_missing:
            sections.append("- **Variables más afectadas**: " + 
                          ", ".join([f"{k} ({v})" for k, v in top_missing]))
            sections.append("")  
    else:
        sections.append("- ✅ **Sin valores faltantes**")
    
    # Filas duplicadas
    duplicates = cardinality.get("duplicate_rows", 0)
    if duplicates > 0:
        sections.append(f"- **Filas duplicadas**: {duplicates:,}")
    else:
        sections.append("- ✅ **Sin filas duplicadas**")
    
    return "\n".join(sections)

def _format_statistical_insights(summary: Dict[str, Any]) -> str:
    """Formatea insights estadísticos."""
    stats = summary.get("statistical_analysis", {})
    correlations = stats.get("correlations", {})
    distributions = stats.get("distributions", {})
    outliers = summary.get("data_quality", {}).get("outliers", {})
    
    sections = ["\n## 📈 Análisis Estadístico"]
    
    # Correlaciones fuertes
    strong_corr = correlations.get("strong_correlations", [])
    if strong_corr:
        sections.append("- **Correlaciones destacadas**:")
        for corr in strong_corr[:3]:
            pearson = corr.get('pearson', 0)
            sections.append(f"  - {corr['var1']} ↔ {corr['var2']}: r={pearson:.2f}")
    
    # Distribuciones no normales
    non_normal = []
    for var, dist_info in distributions.items():
        skew = abs(dist_info.get('skewness', 0))
        if skew > 1: 
            non_normal.append((var, skew))
    
    if non_normal:
        sections.append("- **Variables asimétricas**: " + 
                       ", ".join([f"{v} ({s:.1f})" for v, s in non_normal[:3]]))
    
    
    high_outliers = [(k, v) for k, v in outliers.items() 
                    if isinstance(v, dict) and v.get('iqr', 0) > 0]
    if high_outliers:
        sections.append("- **Variables con outliers**: " + 
                       ", ".join([f"{k} ({v.get('iqr', 0)})" for k, v in high_outliers[:3]]))
    
    return "\n".join(sections)

def _format_categorical_insights(summary: Dict[str, Any]) -> str:
    """Formatea insights de variables categóricas."""
    cat_analysis = summary.get("categorical_analysis", {})
    if not cat_analysis:
        return ""
    
    sections = ["\n## 🏷️ Variables Categóricas"]
    
    
    high_entropy = []
    for var, info in cat_analysis.items():
        entropy = info.get('entropy', 0)
        if entropy > 2: 
            high_entropy.append((var, entropy))
    
    if high_entropy:
        sections.append("- **Variables más diversas**: " + 
                       ", ".join([f"{v}" for v, _ in high_entropy[:3]]))
    
    # Categorías dominantes
    dominant = []
    for var, info in cat_analysis.items():
        top_cats = info.get('top_categories', {})
        if top_cats:
            top_cat, count = list(top_cats.items())[0]
            if len(str(top_cat)) < 20:  
                dominant.append(f"{var}: '{top_cat}' ({count})")
    
    if dominant:
        sections.append("- **Categorías dominantes**: " + "; ".join(dominant[:3]))
    
    return "\n".join(sections)

def _format_recommendations(summary: Dict[str, Any]) -> str:
    """Genera recomendaciones basadas en el análisis."""
    sections = ["## 💡 Recomendaciones", ""]  
    
    # Basado en valores faltantes
    missing = summary.get("data_quality", {}).get("missing", {})
    if missing.get("total_missing", 0) > 0:
        sections.append("- **Datos faltantes**: Considerar imputación o eliminación según el contexto del negocio")
        sections.append("")  
    
    # Basado en outliers
    outliers = summary.get("data_quality", {}).get("outliers", {})
    if any(isinstance(v, dict) and v.get('iqr', 0) > 0 for v in outliers.values()):
        sections.append("- **Outliers**: Investigar si son errores o observaciones legítimas")
        sections.append("")  

    # Basado en correlaciones
    correlations = summary.get("statistical_analysis", {}).get("correlations", {})
    strong_corr = correlations.get("strong_correlations", [])
    if strong_corr:
        sections.append("- **Multicolinealidad**: Evaluar redundancia entre variables altamente correlacionadas")
        sections.append("") 
    
    # Basado en distribuciones
    distributions = summary.get("statistical_analysis", {}).get("distributions", {})
    if any(abs(d.get('skewness', 0)) > 2 for d in distributions.values()):
        sections.append("- **Transformaciones**: Considerar log, Box-Cox para variables muy asimétricas")
        sections.append("")  
    
    # Recomendación general
    sections.append("- **Próximos pasos**: Definir variable objetivo y estrategia de modelado según el problema de negocio")
    
    return "\n".join(sections)

def _fallback_narrative(summary: Dict[str, Any], reason: Optional[str] = None) -> str:
    """Genera narrativa estructurada cuando no hay LLM disponible."""
    note_section = ""
    if reason:
        note_section = f"\n> ℹ️ *Reporte generado automáticamente - {reason}*\n"
    
    sections = [
        note_section,
        _format_basic_info(summary),
        _format_data_types(summary), 
        _format_data_quality(summary),
        _analyze_visualizations(summary),  
        _format_statistical_insights(summary),
        _format_categorical_insights(summary),
        _format_recommendations(summary)
    ]
    
    # Filtrar secciones vacías y unir con espaciado amplio
    content_sections = [section for section in sections if section.strip()]
    return "\n\n\n".join(content_sections)  

def _analyze_visualizations(summary: Dict[str, Any], viz_paths: Optional[List] = None) -> str:
    """Analiza las visualizaciones generadas y describe lo que muestran."""
    sections = ["## 📊 Análisis de Visualizaciones", ""]
    
    # Análisis de histogramas
    numeric_vars = summary.get("data_types", {}).get("numeric", [])
    if numeric_vars:
        sections.append("### 📈 **Distribuciones (Histogramas)**")
        sections.append("")
        
        distributions = summary.get("statistical_analysis", {}).get("distributions", {})
        for var in numeric_vars[:3]:  
            dist_info = distributions.get(var, {})
            skewness = dist_info.get('skewness', 0)
            
            if abs(skewness) > 1:
                skew_desc = "**muy asimétrica**" if abs(skewness) > 2 else "**moderadamente asimétrica**"
                direction = "hacia la derecha" if skewness > 0 else "hacia la izquierda"
                sections.append(f"- **{var}**: La distribución es {skew_desc} {direction}, " +
                              f"indicando valores extremos en la cola.")
            else:
                sections.append(f"- **{var}**: Muestra una distribución **aproximadamente normal**, " +
                              "ideal para análisis estadísticos.")
            sections.append("")
    
    # Análisis de boxplots
    if numeric_vars:
        sections.append("### 📦 **Detección de Outliers (Boxplots)**")
        sections.append("")
        
        outliers = summary.get("data_quality", {}).get("outliers", {})
        outlier_vars = [k for k, v in outliers.items() 
                       if isinstance(v, dict) and v.get('iqr', 0) > 0]
        
        if outlier_vars:
            sections.append("**Variables con outliers visibles en los boxplots:**")
            sections.append("")
            for var in outlier_vars[:3]:
                outlier_info = outliers[var]
                count = outlier_info.get('iqr', 0)
                sections.append(f"- **{var}**: Se observan **{count} valores atípicos** que se extienden " +
                              "más allá de los bigotes del boxplot, requiriendo investigación.")
                sections.append("")
        else:
            sections.append("- **Excelente calidad**: Los boxplots muestran **datos bien distribuidos** " +
                          "sin outliers extremos visibles.")
            sections.append("")
    
    # Análisis de correlaciones
    correlations = summary.get("statistical_analysis", {}).get("correlations", {})
    strong_corr = correlations.get("strong_correlations", [])
    
    if strong_corr:
        sections.append("### 🔗 **Matriz de Correlación**")
        sections.append("")
        sections.append("**Relaciones fuertes identificadas en el heatmap:**")
        sections.append("")
        
        for corr in strong_corr[:3]:
            var1, var2 = corr['var1'], corr['var2']
            pearson = corr.get('pearson', 0)
            
            if pearson > 0.7:
                sections.append(f"- **{var1} ↔ {var2}**: Correlación **positiva fuerte** (r={pearson:.2f}), " +
                              "visible como colores calientes en la matriz.")
            elif pearson < -0.7:
                sections.append(f"- **{var1} ↔ {var2}**: Correlación **negativa fuerte** (r={pearson:.2f}), " +
                              "mostrada en colores fríos en el heatmap.")
            sections.append("")
    
    # Análisis de variables categóricas
    cat_vars = summary.get("data_types", {}).get("categorical", [])
    if cat_vars:
        sections.append("### 📊 **Gráficos de Barras (Variables Categóricas)**")
        sections.append("")
        
        cat_analysis = summary.get("categorical_analysis", {})
        for var in cat_vars[:2]:  
            var_info = cat_analysis.get(var, {})
            unique_count = var_info.get('unique_count', 0)
            dominant = var_info.get('dominant_category', {})
            
            if dominant:
                cat_name = dominant.get('category', 'N/A')
                percentage = dominant.get('percentage', 0)
                sections.append(f"- **{var}**: El gráfico de barras revela **{unique_count} categorías**, " +
                              f"con '{cat_name}' dominando ({percentage:.1f}% de los datos).")
            else:
                sections.append(f"- **{var}**: Distribución balanceada entre **{unique_count} categorías** " +
                              "visible en el gráfico de barras.")
            sections.append("")
    
    return "\n".join(sections)

def _enhance_llm_context(summary: Dict[str, Any]) -> str:
    """Crea contexto enriquecido para el LLM."""
    context_parts = []
    
    # Información básica
    basic = summary.get("basic_info", {})
    shape = basic.get("shape", summary.get("shape", {}))
    context_parts.append(f"Dataset: {shape.get('rows', 0)} filas, {shape.get('cols', 0)} columnas")
    
    # Tipos de variables con nombres específicos
    data_types = summary.get("data_types", {})
    for type_name, cols in data_types.items():
        if cols:
            # Mostrar nombres específicos de variables en lugar de solo contar
            if len(cols) <= 5:
                var_names = ", ".join(cols)
            else:
                var_names = ", ".join(cols[:5]) + f" (y {len(cols)-5} más)"
            context_parts.append(f"{type_name.title()}: {var_names}")
    
    # Calidad de datos
    quality = summary.get("data_quality", {})
    missing = quality.get("missing", {})
    if missing.get("total_missing", 0) > 0:
        context_parts.append(f"Datos faltantes: {missing.get('missing_percentage', 0):.1f}%")
    
    # Correlaciones destacadas con nombres específicos
    correlations = summary.get("statistical_analysis", {}).get("correlations", {})
    strong_corr = correlations.get("strong_correlations", [])
    if strong_corr:
        corr_pairs = []
        for pair in strong_corr[:3]:  # Top 3 correlaciones
            if isinstance(pair, (list, tuple)) and len(pair) >= 3:
                corr_pairs.append(f"{pair[0]} ↔ {pair[1]} ({pair[2]:.2f})")
        if corr_pairs:
            context_parts.append(f"Correlaciones fuertes: {', '.join(corr_pairs)}")
    
    # Outliers con nombres específicos
    outliers = quality.get("outliers", {})
    vars_with_outliers = [k for k, v in outliers.items() 
                         if isinstance(v, dict) and v.get('iqr', 0) > 0]
    if vars_with_outliers:
        if len(vars_with_outliers) <= 5:
            outlier_names = ", ".join(vars_with_outliers)
        else:
            outlier_names = ", ".join(vars_with_outliers[:5]) + f" (y {len(vars_with_outliers)-5} más)"
        context_parts.append(f"Variables con outliers: {outlier_names}")
    
    return ". ".join(context_parts) + "."

def generate_narrative(summary: Dict[str, Any], model: Optional[str] = None, use_llm: bool = True) -> str:
    """
    Genera narrativa del EDA usando LLM o plantilla estructurada.
    
    Args:
        summary: Resumen del análisis exploratorio
        model: Modelo de Ollama a usar
        use_llm: Si usar LLM o plantilla
    
    Returns:
        Narrativa en formato Markdown
    """
    if not use_llm:
        return _fallback_narrative(summary)
    
    model = model or os.getenv("OLLAMA_MODEL", "llama3")
    context = _enhance_llm_context(summary)
    
    try:
        import ollama
        
        prompt = f"""Como analista de datos senior, escribe un reporte ejecutivo del siguiente EDA en español.

INSTRUCCIONES:
- Usa formato Markdown con emojis para secciones
- Sé específico con números y métricas
- SIEMPRE usa los nombres REALES de las variables mencionadas en el contexto
- NO uses nombres genéricos como "variable x1", "variable1", etc.
- INCLUYE análisis de las visualizaciones (histogramas, boxplots, correlaciones)
- Describe qué se observa en las gráficas y qué significan los patrones visuales
- Incluye insights accionables y recomendaciones
- Máximo 12 puntos principales
- Evita repetir información obvia

CONTEXTO DEL ANÁLISIS:
{context}

ESTRUCTURA SUGERIDA:
1. Resumen ejecutivo
2. Hallazgos clave sobre calidad de datos  
3. Análisis de las visualizaciones generadas (histogramas, boxplots, correlaciones)
4. Insights estadísticos importantes
5. Recomendaciones específicas

NOTA IMPORTANTE: 
- Las gráficas han sido generadas automáticamente. Analiza las distribuciones, 
outliers y correlaciones basándote en los datos estadísticos proporcionados.
- Usa SIEMPRE los nombres reales de las variables del dataset, NO nombres genéricos.
- Los nombres de las variables están especificados en el contexto anterior.

Escribe un análisis profesional y conciso:"""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = ollama.generate(model=model, prompt=prompt)
        
        # Extraer solo el texto de la respuesta, ignorando metadatos
        if isinstance(response, dict):
            narrative = response.get("response", "")
        elif hasattr(response, 'response'):
            narrative = response.response
        else:
            narrative = str(response)
        
        # Limpiar la narrativa de metadatos y caracteres extraños
        narrative = _clean_narrative_text(narrative)
        
        if not narrative or len(narrative.strip()) < 50:
            return _fallback_narrative(summary, "respuesta del LLM demasiado corta")
        
        return narrative.strip()
        
    except ImportError:
        return _fallback_narrative(summary, "Ollama no disponible")
    except Exception as e:
        return _fallback_narrative(summary, f"Error del LLM: {str(e)[:50]}")

def _clean_narrative_text(text: str) -> str:
    """
    Limpia el texto narrativo de metadatos y caracteres no deseados.
    """
    if not text:
        return ""
    
    
    text = str(text)
    
    
    import re
    
    # Patrones a eliminar
    patterns_to_remove = [
        r"model='[^']*'",
        r"created_at='[^']*'",
        r"done=\w+",
        r"done_reason='[^']*'",
        r"total_duration=\d+",
        r"load_duration=\d+",
        r"prompt_eval_count=\d+",
        r"prompt_eval_duration=\d+", 
        r"eval_count=\d+",
        r"eval_duration=\d+",
        r"response=",
        r"thinking=\w+",
        r"context=\[.*?\]",
        r"\\n\\n\*",
        r"^\s*\*\s*$"
    ]
    
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL)
    
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        
        if not line or len(line) <= 2:
            continue
            
        
        if re.match(r'^[\s\*\-_=\\\d]+$', line):
            continue
            
        
        if any(keyword in line.lower() for keyword in [
            'model=', 'created_at=', 'done=', 'total_duration=', 
            'load_duration=', 'prompt_eval', 'eval_count=', 
            'eval_duration=', 'response=', 'thinking=', 'context='
        ]):
            continue
        
        
        line = re.sub(r'^[\s\*\-_=\\]+', '', line)
        line = re.sub(r'[\s\*\-_=\\]+$', '', line)
        
        if line:  
            cleaned_lines.append(line)
    
   
    cleaned_text = '\n'.join(cleaned_lines)
    
    
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    
    cleaned_text = cleaned_text.strip()
    
    
    if len(cleaned_text) < 50:
        return ""
    
    return cleaned_text

def _fallback_text(context: str, reason: str | None = None) -> str:
    """Función de compatibilidad con versión anterior."""
    note = f"(Nota: se usó narrativa de plantilla{' por: ' + reason if reason else ''}.)"
    formatted_context = context.replace('\n', '\n- ')
    return f"""## Resumen ejecutivo (plantilla)

- {note}
- {formatted_context}
- Recomendación: verificar variables con alta proporción de nulos y considerar imputación.
- Recomendación: revisar outliers con reglas de negocio y decidir winsorización o transformación.
- Recomendación: si hay correlaciones fuertes, evaluar multicolinealidad antes de modelar.
"""
