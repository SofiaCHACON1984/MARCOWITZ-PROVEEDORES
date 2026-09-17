import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import REQUIRED_COLUMNS, load_excel
from src.portfolio import (
    calculate_statistics,
    concentration_metrics,
    portfolio_metrics,
    current_weights_from_image,
)
from src.optimization import (
    optimize_min_variance,
    optimize_max_return,
    efficient_frontier,
    simulate_portfolios,
)
from src.charts import (
    concentration_chart,
    frontier_chart,
    correlation_heatmap,
    weights_comparison_chart,
)

st.set_page_config(
    page_title="Portafolio de Clientes | Markowitz",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Estilo
# -----------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f7f9fa; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border: 1px solid #dce5e8;
        border-radius: 10px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Portafolio de Clientes — Frontera Eficiente de Markowitz")
st.caption(
    "Aplicación para analizar la concentración y diversificación de ingresos "
    "de clientes usando la lógica cuantitativa de Markowitz, adaptada al contexto empresarial."
)

# -----------------------------
# Carga de datos
# -----------------------------
DEFAULT_FILE = Path(__file__).parent / "data" / "Base_Portafolio_Clientes_Markowitz.xlsx"

with st.sidebar:
    st.header("1. Carga de datos")
    uploaded = st.file_uploader(
        "Carga el Excel del portafolio",
        type=["xlsx"],
        help="Debe contener las columnas requeridas por la aplicación.",
    )

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        df, validation = load_excel(io.BytesIO(raw_bytes))
        source_name = uploaded.name
    elif DEFAULT_FILE.exists():
        df, validation = load_excel(DEFAULT_FILE)
        source_name = DEFAULT_FILE.name
    else:
        df, validation = None, ["No se encontró el archivo Excel."]
        source_name = ""

    if df is not None:
        st.success(f"Fuente: {source_name}")
        st.write(f"**Registros:** {len(df):,}")
        st.write(f"**Clientes:** {df['Cliente'].nunique()}")
        st.write(
            f"**Periodo:** {df['Fecha'].min():%Y-%m} a {df['Fecha'].max():%Y-%m}"
        )

if df is None:
    st.warning("Carga el archivo Excel para iniciar el análisis.")
    st.stop()

if validation:
    st.error("La base no cumple la estructura requerida:")
    for item in validation:
        st.write(f"- {item}")
    st.stop()

# -----------------------------
# Preparación
# -----------------------------
clients = sorted(df["Cliente"].dropna().unique().tolist())
stats, returns = calculate_statistics(df, clients)

if len(returns) < 2:
    st.error("Se requieren al menos dos observaciones temporales para calcular rendimientos y covarianzas.")
    st.stop()

missing_clients = [c for c in ["Cliente A", "Cliente B", "Cliente C", "Otros"] if c not in clients]
if missing_clients:
    st.warning(
        "No se encontraron todos los nombres de referencia de la imagen: "
        + ", ".join(missing_clients)
        + ". La aplicación continuará usando los clientes presentes en el Excel."
    )

current_weights = current_weights_from_image(clients)
current_metrics = portfolio_metrics(current_weights, stats["expected_return"].values, stats["covariance"].values)
current_conc = concentration_metrics(current_weights)

# -----------------------------
# A. CARGA DE DATOS
# -----------------------------
st.header("A. Carga y validación de datos")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{len(df):,}")
c2.metric("Clientes", f"{df['Cliente'].nunique()}")
c3.metric("Meses", f"{df['Fecha'].nunique()}")
c4.metric("Columnas", f"{df.shape[1]}")

with st.expander("Vista previa de la base"):
    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# B. PORTAFOLIO ACTUAL
# -----------------------------
st.header("B. Portafolio actual")

summary_current = stats[[
    "expected_return", "volatility", "avg_income", "avg_collection", "avg_dso"
]].copy()
summary_current["Peso actual"] = [current_weights.get(c, 0.0) for c in summary_current.index]
summary_current["Riesgo cobranza"] = 100 - summary_current["avg_collection"]
summary_current = summary_current.reset_index().rename(columns={"index": "Cliente"})
summary_current["Rendimiento esperado (%)"] = summary_current["expected_return"] * 100
summary_current["Volatilidad mensual (%)"] = summary_current["volatility"] * 100
summary_current["Peso actual (%)"] = summary_current["Peso actual"] * 100
summary_current["Ingreso mensual promedio (MXN)"] = summary_current["avg_income"]
summary_current["Cobranza (%)"] = summary_current["avg_collection"]
summary_current["DSO (días)"] = summary_current["avg_dso"]
summary_current["Riesgo cobranza (%)"] = summary_current["Riesgo cobranza"]
display_current = summary_current[[
    "Cliente", "Peso actual (%)", "Ingreso mensual promedio (MXN)",
    "Rendimiento esperado (%)", "Volatilidad mensual (%)",
    "Cobranza (%)", "DSO (días)", "Riesgo cobranza (%)"
]].copy()

st.dataframe(
    display_current.style.format({
        "Peso actual (%)": "{:.2f}",
        "Ingreso mensual promedio (MXN)": "${:,.2f}",
        "Rendimiento esperado (%)": "{:.2f}",
        "Volatilidad mensual (%)": "{:.2f}",
        "Cobranza (%)": "{:.2f}",
        "DSO (días)": "{:.2f}",
        "Riesgo cobranza (%)": "{:.2f}",
    }),
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(concentration_chart(current_weights), use_container_width=True)
with col2:
    st.metric("Rendimiento esperado mensual", f"{current_metrics['return']*100:.2f}%")
    st.metric("Volatilidad mensual", f"{current_metrics['risk']*100:.2f}%")
    st.metric("Concentración máxima", f"{current_conc['max_weight']*100:.2f}%")
    st.metric("HHI", f"{current_conc['hhi']:.4f}")

# -----------------------------
# C. ANÁLISIS ESTADÍSTICO
# -----------------------------
st.header("C. Análisis estadístico")

st.subheader("Estadísticos por cliente")
stats_display = stats.copy()
stats_display["expected_return"] *= 100
stats_display["volatility"] *= 100
stats_display["avg_collection"] = stats_display["avg_collection"]
stats_display["risk_collection"] = 100 - stats_display["avg_collection"]
stats_display = stats_display.rename(columns={
    "expected_return": "Rendimiento esperado (%)",
    "volatility": "Volatilidad mensual (%)",
    "avg_income": "Ingreso mensual promedio (MXN)",
    "avg_collection": "Cobranza promedio (%)",
    "avg_dso": "DSO promedio (días)",
    "risk_collection": "Riesgo cobranza (%)",
})
st.dataframe(
    stats_display[[
        "Rendimiento esperado (%)", "Volatilidad mensual (%)",
        "avg_income", "Cobranza promedio (%)", "DSO promedio (días)",
        "Riesgo cobranza (%)"
    ]].rename(columns={"avg_income": "Ingreso mensual promedio (MXN)"}).style.format({
        "Rendimiento esperado (%)": "{:.2f}",
        "Volatilidad mensual (%)": "{:.2f}",
        "Ingreso mensual promedio (MXN)": "${:,.2f}",
        "Cobranza promedio (%)": "{:.2f}",
        "DSO promedio (días)": "{:.2f}",
        "Riesgo cobranza (%)": "{:.2f}",
    }),
    use_container_width=True,
)

cc1, cc2 = st.columns(2)
with cc1:
    st.subheader("Matriz de covarianzas")
    cov_display = stats["covariance"] * 10000
    st.dataframe(cov_display.style.format("{:.4f}"), use_container_width=True)
    st.caption("Escala mostrada ×10,000 para facilitar lectura; los cálculos usan la matriz original.")
with cc2:
    st.subheader("Matriz de correlaciones")
    st.plotly_chart(correlation_heatmap(stats["correlation"]), use_container_width=True)

# -----------------------------
# F. SIMULADOR / RESTRICCIONES
# -----------------------------
st.header("F. Simulador de restricciones")

st.info(
    "Los límites se aplican a los pesos del portafolio. El peso mínimo es global y "
    "los pesos máximos son configurables por cliente."
)

min_weight_pct = st.slider(
    "Peso mínimo por cliente (%)",
    min_value=0.0, max_value=25.0, value=0.0, step=1.0
)

max_weights = {}
cols = st.columns(len(clients))
for i, client in enumerate(clients):
    default = int(round(current_weights.get(client, 0.25) * 100))
    default = min(max(default, 5), 100)
    with cols[i]:
        max_weights[client] = st.slider(
            f"Máximo {client} (%)",
            min_value=int(min_weight_pct),
            max_value=100,
            value=max(default, int(min_weight_pct)),
            step=1,
            key=f"max_{client}",
        ) / 100

# Validación simple de factibilidad por límites.
min_total = min_weight_pct / 100 * len(clients)
max_total = sum(max_weights.values())
if min_total > 1:
    st.error("La suma de pesos mínimos supera el 100%. Reduce el peso mínimo.")
    st.stop()
if max_total < 1:
    st.error("La suma de pesos máximos es menor al 100%. Aumenta algún máximo.")
    st.stop()


# Optimización base para conocer rango factible bajo restricciones.
minvar = optimize_min_variance(stats["expected_return"].values, stats["covariance"].values,
                               min_weight_pct/100, max_weights, clients)
maxret = optimize_max_return(stats["expected_return"].values,
                             min_weight_pct/100, max_weights, clients,
                             stats["covariance"].values)

if not minvar["success"] or not maxret["success"]:
    st.error(
        "No existe una solución factible con las restricciones seleccionadas. "
        "Reduce el peso mínimo o aumenta los pesos máximos."
    )
    st.stop()

min_feasible_return = minvar["return"]
max_feasible_return = maxret["return"]

target_pct = st.slider(
    "Rendimiento objetivo mensual (%)",
    min_value=float(min_feasible_return * 100),
    max_value=float(max_feasible_return * 100),
    value=float(min_feasible_return * 100),
    step=max(0.01, float((max_feasible_return - min_feasible_return) * 100 / 100)),
)

# -----------------------------
# D. FRONTERA EFICIENTE
# -----------------------------
st.header("D. Frontera eficiente")

sim = simulate_portfolios(
    expected_returns=stats["expected_return"].values,
    covariance=stats["covariance"].values,
    min_weight=min_weight_pct/100,
    max_weights=max_weights,
    clients=clients,
    n=12000,
    seed=42,
)

frontier = efficient_frontier(
    expected_returns=stats["expected_return"].values,
    covariance=stats["covariance"].values,
    min_weight=min_weight_pct/100,
    max_weights=max_weights,
    clients=clients,
    points=35,
)

if frontier.empty:
    st.warning("No fue posible construir la frontera con las restricciones actuales.")
else:
    fig = frontier_chart(
        sim,
        frontier,
        current_metrics["risk"],
        current_metrics["return"],
        minvar["risk"],
        minvar["return"],
        maxret["risk"],
        maxret["return"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "La nube representa combinaciones factibles simuladas. La curva representa "
        "portafolios de mínima volatilidad para distintos niveles de rendimiento objetivo."
    )

# -----------------------------
# E. PORTAFOLIO OPTIMIZADO
# -----------------------------
st.header("E. Portafolio optimizado")

from src.optimization import optimize_target_return
target = optimize_target_return(
    target_return=target_pct / 100,
    expected_returns=stats["expected_return"].values,
    covariance=stats["covariance"].values,
    min_weight=min_weight_pct/100,
    max_weights=max_weights,
    clients=clients,
)

if not target["success"]:
    st.warning(
        "No existe un portafolio factible para el rendimiento objetivo seleccionado. "
        "Mueve el objetivo dentro del rango factible."
    )
else:
    opt_weights = dict(zip(clients, target["weights"]))
    comparison = pd.DataFrame({
        "Cliente": clients,
        "Peso actual (%)": [current_weights.get(c, 0)*100 for c in clients],
        "Peso optimizado (%)": [opt_weights[c]*100 for c in clients],
        "Diferencia (pp)": [(opt_weights[c] - current_weights.get(c, 0))*100 for c in clients],
        "Rendimiento mensual (%)": stats.loc[clients, "expected_return"].values*100,
        "Volatilidad mensual (%)": stats.loc[clients, "volatility"].values*100,
    })

    st.dataframe(
        comparison.style.format({
            "Peso actual (%)": "{:.2f}",
            "Peso optimizado (%)": "{:.2f}",
            "Diferencia (pp)": "{:+.2f}",
            "Rendimiento mensual (%)": "{:.2f}",
            "Volatilidad mensual (%)": "{:.2f}",
        }),
        use_container_width=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    opt_conc = concentration_metrics(opt_weights)
    k1.metric("Rendimiento objetivo", f"{target['return']*100:.2f}%")
    k2.metric("Riesgo / volatilidad", f"{target['risk']*100:.2f}%")
    k3.metric("Concentración máxima", f"{opt_conc['max_weight']*100:.2f}%")
    k4.metric("HHI", f"{opt_conc['hhi']:.4f}")

    st.plotly_chart(
        weights_comparison_chart(current_weights, opt_weights),
        use_container_width=True,
    )

# -----------------------------
# G. CONCENTRACIÓN
# -----------------------------
st.header("G. Análisis de concentración")

if target["success"]:
    top_current = sorted(current_weights.values(), reverse=True)
    top_opt = sorted(opt_weights.values(), reverse=True)

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Portafolio actual")
        st.write(f"Concentración máxima: **{current_conc['max_weight']*100:.2f}%**")
        st.write(f"Top 2 clientes: **{sum(top_current[:2])*100:.2f}%**")
        st.write(f"Top 3 clientes: **{sum(top_current[:3])*100:.2f}%**")
        st.write(f"HHI: **{current_conc['hhi']:.4f}**")
    with g2:
        opt_conc = concentration_metrics(opt_weights)
        st.subheader("Portafolio calculado")
        st.write(f"Concentración máxima: **{opt_conc['max_weight']*100:.2f}%**")
        st.write(f"Top 2 clientes: **{sum(top_opt[:2])*100:.2f}%**")
        st.write(f"Top 3 clientes: **{sum(top_opt[:3])*100:.2f}%**")
        st.write(f"HHI: **{opt_conc['hhi']:.4f}**")

# -----------------------------
# H. INTERPRETACIÓN
# -----------------------------
st.header("H. Interpretación cuantitativa")

if target["success"]:
    st.write(
        f"El portafolio actual presenta un rendimiento esperado mensual de "
        f"**{current_metrics['return']*100:.2f}%** y una volatilidad mensual de "
        f"**{current_metrics['risk']*100:.2f}%**. Su concentración máxima es "
        f"**{current_conc['max_weight']*100:.2f}%**."
    )
    st.write(
        f"Bajo las restricciones seleccionadas, el portafolio calculado presenta un "
        f"rendimiento esperado de **{target['return']*100:.2f}%** y una volatilidad de "
        f"**{target['risk']*100:.2f}%**."
    )
    st.write(
        f"La concentración máxima del portafolio calculado es **{opt_conc['max_weight']*100:.2f}%** "
        f"y su HHI es **{opt_conc['hhi']:.4f}**."
    )
    st.caption(
        "Estas frases describen resultados cuantitativos; no constituyen una recomendación "
        "sobre qué cliente debe conservarse o reducirse."
    )

st.divider()
st.caption(
    "Modelo adaptado al contexto de ingresos empresariales. Los 'rendimientos' representan "
    "variaciones históricas mensuales de ingresos, no rendimientos bursátiles."
)
