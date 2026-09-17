import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Portafolio de Clientes | Markowitz",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #f7f9fa; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Portafolio de Clientes — Frontera Eficiente de Markowitz")
st.caption(
    "Análisis cuantitativo de concentración, riesgo, rendimiento y diversificación "
    "de ingresos por cliente."
)

# ============================================================
# FUNCIONES
# ============================================================
REQUIRED_COLUMNS = [
    "Fecha",
    "Cliente",
    "Ingresos_MXN",
    "Clientes_Activos",
    "Clientes_Nuevos",
    "Clientes_Perdidos",
    "Cobranza_Pct",
    "DSO_Dias",
    "Participacion_Ingresos_Pct",
    "Variacion_Ingresos_Mensual_Pct",
    "Rendimiento_Ingreso_Pct",
    "Concentracion_Cliente_Pct",
    "Riesgo_Cobranza_Pct",
    "Crecimiento_Neto_Clientes",
]

REFERENCE_WEIGHTS = {
    "Cliente A": 0.45,
    "Cliente B": 0.25,
    "Cliente C": 0.15,
    "Otros": 0.15,
}


def load_excel(source):
    """Carga y valida la hoja Base_Datos del Excel."""
    try:
        df = pd.read_excel(source, sheet_name="Base_Datos")
    except Exception as exc:
        return None, [f"No se pudo leer la hoja 'Base_Datos': {exc}"]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return df, ["Faltan columnas: " + ", ".join(missing)]

    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    numeric_cols = [
        c for c in REQUIRED_COLUMNS
        if c not in ["Fecha", "Cliente"]
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Fecha", "Cliente", "Ingresos_MXN"])
    df = df.sort_values(["Fecha", "Cliente"]).reset_index(drop=True)

    errors = []

    if df["Fecha"].isna().all():
        errors.append("La columna Fecha no contiene fechas válidas.")

    duplicated = df.duplicated(["Fecha", "Cliente"]).sum()
    if duplicated:
        errors.append(
            f"Hay {duplicated} combinaciones duplicadas de Fecha + Cliente."
        )

    return df, errors


def calculate_statistics(df, clients):
    """
    Construye la matriz temporal de rendimientos de ingresos.
    El rendimiento empresarial se representa mediante la variación
    mensual histórica de ingresos.
    """
    returns = (
        df.pivot_table(
            index="Fecha",
            columns="Cliente",
            values="Rendimiento_Ingreso_Pct",
            aggfunc="mean",
        )
        .reindex(columns=clients)
        .sort_index()
    )

    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="any") / 100.0

    expected_return = returns.mean()
    volatility = returns.std(ddof=1)
    covariance = returns.cov()
    correlation = returns.corr()

    avg_income = (
        df.groupby("Cliente")["Ingresos_MXN"]
        .mean()
        .reindex(clients)
    )

    avg_collection = (
        df.groupby("Cliente")["Cobranza_Pct"]
        .mean()
        .reindex(clients)
    )

    avg_dso = (
        df.groupby("Cliente")["DSO_Dias"]
        .mean()
        .reindex(clients)
    )

    stats = pd.DataFrame({
        "expected_return": expected_return,
        "volatility": volatility,
        "avg_income": avg_income,
        "avg_collection": avg_collection,
        "avg_dso": avg_dso,
    }).reindex(clients)

    return stats, returns


def portfolio_metrics(weights, expected_returns, covariance):
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    portfolio_return = float(w @ mu)
    variance = float(w @ cov @ w)
    risk = float(np.sqrt(max(variance, 0.0)))

    return {
        "return": portfolio_return,
        "risk": risk,
        "variance": variance,
    }


def concentration_metrics(weights):
    values = np.asarray(list(weights.values()), dtype=float)
    values = values[values >= 0]

    return {
        "max_weight": float(values.max()) if len(values) else 0,
        "top2": float(np.sort(values)[-2:].sum())
        if len(values) >= 2 else float(values.sum()),
        "top3": float(np.sort(values)[-3:].sum())
        if len(values) >= 3 else float(values.sum()),
        "hhi": float(np.sum(values ** 2)),
    }


def current_weights(clients):
    weights = {
        client: REFERENCE_WEIGHTS.get(client, 0)
        for client in clients
    }

    total = sum(weights.values())

    if total == 0:
        return {
            client: 1 / len(clients)
            for client in clients
        }

    return {
        client: weights[client] / total
        for client in clients
    }


def initial_weights(min_weight, max_weights, clients):
    """Genera un punto inicial factible para SLSQP."""
    n = len(clients)

    minimums = np.full(n, float(min_weight))
    remaining = 1.0 - minimums.sum()

    if remaining < -1e-10:
        return None

    capacity = np.array([
        max_weights[c] - min_weight
        for c in clients
    ])

    if capacity.sum() < remaining - 1e-10:
        return None

    weights = minimums.copy()

    if capacity.sum() > 0:
        weights += remaining * capacity / capacity.sum()

    return weights


def optimize_min_variance(mu, covariance, min_weight, max_weights, clients):
    cov = np.asarray(covariance, dtype=float)
    x0 = initial_weights(min_weight, max_weights, clients)

    if x0 is None:
        return {"success": False}

    def objective(w):
        return float(w @ cov @ w)

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        }
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[
            (min_weight, max_weights[c])
            for c in clients
        ],
        constraints=constraints,
        options={
            "maxiter": 1500,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        return {
            "success": False,
            "message": result.message,
        }

    risk = np.sqrt(max(objective(result.x), 0))

    return {
        "success": True,
        "weights": result.x,
        "risk": float(risk),
        "return": float(result.x @ mu),
    }


def optimize_max_return(mu, covariance, min_weight, max_weights, clients):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    x0 = initial_weights(min_weight, max_weights, clients)

    if x0 is None:
        return {"success": False}

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        }
    ]

    result = minimize(
        lambda w: -float(w @ mu),
        x0,
        method="SLSQP",
        bounds=[
            (min_weight, max_weights[c])
            for c in clients
        ],
        constraints=constraints,
        options={
            "maxiter": 1500,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        return {
            "success": False,
            "message": result.message,
        }

    risk = np.sqrt(max(result.x @ cov @ result.x, 0))

    return {
        "success": True,
        "weights": result.x,
        "risk": float(risk),
        "return": float(result.x @ mu),
    }


def optimize_target_return(
    target_return,
    mu,
    covariance,
    min_weight,
    max_weights,
    clients,
):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    x0 = initial_weights(min_weight, max_weights, clients)

    if x0 is None:
        return {"success": False}

    constraints = [
        {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1,
        },
        {
            "type": "eq",
            "fun": lambda w: float(w @ mu) - target_return,
        },
    ]

    result = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        method="SLSQP",
        bounds=[
            (min_weight, max_weights[c])
            for c in clients
        ],
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    if not result.success:
        return {
            "success": False,
            "message": result.message,
        }

    variance = float(result.x @ cov @ result.x)

    return {
        "success": True,
        "weights": result.x,
        "risk": float(np.sqrt(max(variance, 0))),
        "return": float(result.x @ mu),
    }


def simulate_portfolios(
    mu,
    covariance,
    min_weight,
    max_weights,
    clients,
    n=12000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(covariance, dtype=float)

    rows = []
    attempts = 0
    max_attempts = n * 40

    while len(rows) < n and attempts < max_attempts:
        attempts += 1

        weights = rng.dirichlet(
            np.ones(len(clients))
        )

        if np.any(weights < min_weight - 1e-12):
            continue

        if any(
            weights[i] > max_weights[c] + 1e-12
            for i, c in enumerate(clients)
        ):
            continue

        portfolio_return = float(weights @ mu)
        risk = float(
            np.sqrt(
                max(weights @ cov @ weights, 0)
            )
        )

        rows.append({
            "return": portfolio_return,
            "risk": risk,
        })

    return pd.DataFrame(rows)


def efficient_frontier(
    mu,
    covariance,
    min_weight,
    max_weights,
    clients,
    points=35,
):
    minvar = optimize_min_variance(
        mu, covariance, min_weight, max_weights, clients
    )

    maxret = optimize_max_return(
        mu, covariance, min_weight, max_weights, clients
    )

    if not minvar["success"] or not maxret["success"]:
        return pd.DataFrame()

    targets = np.linspace(
        minvar["return"],
        maxret["return"],
        points,
    )

    rows = []

    for target in targets:
        solution = optimize_target_return(
            target,
            mu,
            covariance,
            min_weight,
            max_weights,
            clients,
        )

        if solution["success"]:
            rows.append({
                "return": solution["return"],
                "risk": solution["risk"],
            })

    return pd.DataFrame(rows)


def concentration_chart(weights):
    data = pd.DataFrame({
        "Cliente": list(weights.keys()),
        "Participación (%)": [
            value * 100
            for value in weights.values()
        ],
    })

    fig = px.bar(
        data,
        x="Cliente",
        y="Participación (%)",
        text="Participación (%)",
        title="Concentración del portafolio actual",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_yaxes(
        range=[
            0,
            max(
                100,
                data["Participación (%)"].max() * 1.25
            ),
        ]
    )

    fig.update_layout(showlegend=False)

    return fig


def correlation_heatmap(correlation):
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Matriz de correlaciones",
    )

    fig.update_layout(height=430)

    return fig


def frontier_chart(
    simulation,
    frontier,
    current_risk,
    current_return,
    minvar_risk,
    minvar_return,
    maxret_risk,
    maxret_return,
):
    fig = go.Figure()

    if not simulation.empty:
        fig.add_trace(
            go.Scatter(
                x=simulation["risk"] * 100,
                y=simulation["return"] * 100,
                mode="markers",
                marker={
                    "size": 4,
                    "opacity": 0.30,
                },
                name="Portafolios simulados",
                hovertemplate=(
                    "Riesgo: %{x:.2f}%"
                    "<br>Rendimiento: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    if not frontier.empty:
        frontier = frontier.sort_values("risk")

        fig.add_trace(
            go.Scatter(
                x=frontier["risk"] * 100,
                y=frontier["return"] * 100,
                mode="lines+markers",
                line={"width": 3},
                name="Frontera eficiente",
                hovertemplate=(
                    "Riesgo: %{x:.2f}%"
                    "<br>Rendimiento: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    points = [
        (
            "Portafolio actual",
            current_risk,
            current_return,
        ),
        (
            "Mínima varianza",
            minvar_risk,
            minvar_return,
        ),
        (
            "Máximo rendimiento factible",
            maxret_risk,
            maxret_return,
        ),
    ]

    for name, risk, ret in points:
        fig.add_trace(
            go.Scatter(
                x=[risk * 100],
                y=[ret * 100],
                mode="markers+text",
                text=[name],
                textposition="top center",
                marker={"size": 11},
                name=name,
            )
        )

    fig.update_layout(
        title="Frontera eficiente: riesgo vs. rendimiento",
        xaxis_title="Riesgo / volatilidad mensual (%)",
        yaxis_title="Rendimiento esperado mensual (%)",
        height=620,
    )

    return fig


def comparison_chart(current, optimized):
    clients = list(current.keys())

    data = pd.DataFrame({
        "Cliente": clients * 2,
        "Peso (%)": (
            [current[c] * 100 for c in clients]
            + [optimized[c] * 100 for c in clients]
        ),
        "Portafolio": (
            ["Actual"] * len(clients)
            + ["Calculado"] * len(clients)
        ),
    })

    fig = px.bar(
        data,
        x="Cliente",
        y="Peso (%)",
        color="Portafolio",
        barmode="group",
        text="Peso (%)",
        title="Portafolio actual vs. portafolio calculado",
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    return fig


# ============================================================
# CARGA DEL EXCEL
# ============================================================
with st.sidebar:
    st.header("📁 Carga de datos")

    uploaded_file = st.file_uploader(
        "Selecciona Base_Portafolio_Clientes_Markowitz.xlsx",
        type=["xlsx"],
    )

    st.divider()

    st.header("⚙️ Configuración")

if uploaded_file is not None:
    df, errors = load_excel(
        io.BytesIO(uploaded_file.getvalue())
    )
else:
    default_path = (
        "Base_Portafolio_Clientes_Markowitz.xlsx"
    )

    try:
        df, errors = load_excel(default_path)
    except Exception:
        df, errors = None, [
            "No se encontró el Excel. "
            "Carga Base_Portafolio_Clientes_Markowitz.xlsx "
            "con el botón de la barra lateral."
        ]

if df is None:
    st.warning(
        "Carga el archivo Base_Portafolio_Clientes_Markowitz.xlsx "
        "para comenzar."
    )
    st.stop()

if errors:
    st.error("Se encontraron problemas en la base:")

    for error in errors:
        st.write(f"- {error}")

    st.stop()

# ============================================================
# PREPARACIÓN
# ============================================================
clients = sorted(
    df["Cliente"].dropna().unique().tolist()
)

stats, returns = calculate_statistics(
    df,
    clients,
)

if len(returns) < 2:
    st.error(
        "No hay suficientes observaciones mensuales para "
        "calcular rendimientos y covarianzas."
    )
    st.stop()

weights_current = current_weights(clients)

# Matriz de covarianzas y rendimientos esperados.
covariance = returns.cov().loc[clients, clients].values
mu = stats.loc[clients, "expected_return"].values

current_metrics = portfolio_metrics(
    [weights_current[c] for c in clients],
    mu,
    covariance,
)

# ============================================================
# A. CARGA DE DATOS
# ============================================================
st.header("A. Carga y validación de datos")

a1, a2, a3, a4 = st.columns(4)

a1.metric(
    "Registros",
    f"{len(df):,}",
)

a2.metric(
    "Clientes",
    f"{df['Cliente'].nunique()}",
)

a3.metric(
    "Meses",
    f"{df['Fecha'].nunique()}",
)

a4.metric(
    "Periodo",
    f"{df['Fecha'].min():%Y-%m} → {df['Fecha'].max():%Y-%m}",
)

with st.expander("Vista previa del Excel"):
    st.dataframe(
        df.head(20),
        use_container_width=True,
    )

# ============================================================
# B. PORTAFOLIO ACTUAL
# ============================================================
st.header("B. Portafolio actual")

table_current = stats.loc[clients].copy()

table_current["Peso actual (%)"] = [
    weights_current[c] * 100
    for c in clients
]

table_current["Riesgo cobranza (%)"] = (
    100 - table_current["avg_collection"]
)

table_current = table_current.reset_index()

table_current = table_current.rename(
    columns={
        "Cliente": "Cliente",
        "expected_return": "Rendimiento esperado (%)",
        "volatility": "Volatilidad mensual (%)",
        "avg_income": "Ingreso mensual promedio (MXN)",
        "avg_collection": "Cobranza promedio (%)",
        "avg_dso": "DSO promedio (días)",
    }
)

st.dataframe(
    table_current[
        [
            "Cliente",
            "Peso actual (%)",
            "Ingreso mensual promedio (MXN)",
            "Rendimiento esperado (%)",
            "Volatilidad mensual (%)",
            "Cobranza promedio (%)",
            "DSO promedio (días)",
            "Riesgo cobranza (%)",
        ]
    ].style.format({
        "Peso actual (%)": "{:.2f}",
        "Ingreso mensual promedio (MXN)": "${:,.2f}",
        "Rendimiento esperado (%)": "{:.2f}",
        "Volatilidad mensual (%)": "{:.2f}",
        "Cobranza promedio (%)": "{:.2f}",
        "DSO promedio (días)": "{:.2f}",
        "Riesgo cobranza (%)": "{:.2f}",
    }),
    use_container_width=True,
)

b1, b2 = st.columns(2)

with b1:
    st.plotly_chart(
        concentration_chart(weights_current),
        use_container_width=True,
    )

with b2:
    current_concentration = concentration_metrics(
        weights_current
    )

    st.metric(
        "Rendimiento esperado mensual",
        f"{current_metrics['return'] * 100:.2f}%",
    )

    st.metric(
        "Volatilidad mensual",
        f"{current_metrics['risk'] * 100:.2f}%",
    )

    st.metric(
        "Concentración máxima",
        f"{current_concentration['max_weight'] * 100:.2f}%",
    )

    st.metric(
        "HHI",
        f"{current_concentration['hhi']:.4f}",
    )

# ============================================================
# C. ANÁLISIS ESTADÍSTICO
# ============================================================
st.header("C. Análisis estadístico")

st.subheader("Rendimiento y riesgo por cliente")

stat_display = stats.loc[clients].copy()

stat_display["Rendimiento esperado (%)"] = (
    stat_display["expected_return"] * 100
)

stat_display["Volatilidad mensual (%)"] = (
    stat_display["volatility"] * 100
)

stat_display["Riesgo cobranza (%)"] = (
    100 - stat_display["avg_collection"]
)

stat_display = stat_display.rename(
    columns={
        "avg_income": "Ingreso mensual promedio (MXN)",
        "avg_collection": "Cobranza promedio (%)",
        "avg_dso": "DSO promedio (días)",
    }
)

st.dataframe(
    stat_display[
        [
            "Rendimiento esperado (%)",
            "Volatilidad mensual (%)",
            "Ingreso mensual promedio (MXN)",
            "Cobranza promedio (%)",
            "DSO promedio (días)",
            "Riesgo cobranza (%)",
        ]
    ].style.format({
        "Rendimiento esperado (%)": "{:.2f}",
        "Volatilidad mensual (%)": "{:.2f}",
        "Ingreso mensual promedio (MXN)": "${:,.2f}",
        "Cobranza promedio (%)": "{:.2f}",
        "DSO promedio (días)": "{:.2f}",
        "Riesgo cobranza (%)": "{:.2f}",
    }),
    use_container_width=True,
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("Matriz de covarianzas")

    st.dataframe(
        pd.DataFrame(
            covariance,
            index=clients,
            columns=clients,
        ).style.format("{:.6f}"),
        use_container_width=True,
    )

with c2:
    st.subheader("Matriz de correlaciones")

    st.plotly_chart(
        correlation_heatmap(
            returns.corr().loc[clients, clients]
        ),
        use_container_width=True,
    )

# ============================================================
# F. SIMULADOR
# ============================================================
st.header("F. Simulador de restricciones")

st.info(
    "Modifica los límites de participación. El algoritmo buscará "
    "portafolios que cumplan las restricciones seleccionadas."
)

min_weight_pct = st.slider(
    "Peso mínimo global por cliente (%)",
    0.0,
    25.0,
    0.0,
    1.0,
)

max_weights = {}

slider_columns = st.columns(
    min(len(clients), 4)
)

for i, client in enumerate(clients):
    default = int(
        round(
            weights_current.get(client, 0.25)
            * 100
        )
    )

    default = max(
        int(min_weight_pct),
        min(default, 100),
    )

    with slider_columns[
        i % len(slider_columns)
    ]:
        max_weights[client] = (
            st.slider(
                f"Máximo {client} (%)",
                min_value=int(min_weight_pct),
                max_value=100,
                value=default,
                step=1,
                key=f"max_weight_{client}",
            )
            / 100
        )

if (
    min_weight_pct / 100 * len(clients)
    > 1
):
    st.error(
        "La suma de los pesos mínimos supera 100%."
    )
    st.stop()

if sum(max_weights.values()) < 1:
    st.error(
        "La suma de los pesos máximos es menor a 100%."
    )
    st.stop()

minvar = optimize_min_variance(
    mu,
    covariance,
    min_weight_pct / 100,
    max_weights,
    clients,
)

maxret = optimize_max_return(
    mu,
    covariance,
    min_weight_pct / 100,
    max_weights,
    clients,
)

if not minvar["success"] or not maxret["success"]:
    st.error(
        "Las restricciones seleccionadas no producen "
        "una solución factible."
    )
    st.stop()

target_pct = st.slider(
    "Rendimiento objetivo mensual (%)",
    min_value=float(minvar["return"] * 100),
    max_value=float(maxret["return"] * 100),
    value=float(minvar["return"] * 100),
    step=max(
        0.01,
        float(
            (maxret["return"] - minvar["return"])
            * 100 / 100
        ),
    ),
)

# ============================================================
# D. FRONTERA EFICIENTE
# ============================================================
st.header("D. Frontera eficiente")

simulation = simulate_portfolios(
    mu,
    covariance,
    min_weight_pct / 100,
    max_weights,
    clients,
    n=12000,
    seed=42,
)

frontier = efficient_frontier(
    mu,
    covariance,
    min_weight_pct / 100,
    max_weights,
    clients,
    points=35,
)

st.plotly_chart(
    frontier_chart(
        simulation,
        frontier,
        current_metrics["risk"],
        current_metrics["return"],
        minvar["risk"],
        minvar["return"],
        maxret["risk"],
        maxret["return"],
    ),
    use_container_width=True,
)

# ============================================================
# E. PORTAFOLIO OPTIMIZADO
# ============================================================
st.header("E. Portafolio calculado")

target_solution = optimize_target_return(
    target_pct / 100,
    mu,
    covariance,
    min_weight_pct / 100,
    max_weights,
    clients,
)

if not target_solution["success"]:
    st.warning(
        "No existe una solución para el rendimiento objetivo "
        "seleccionado. Ajusta el objetivo o las restricciones."
    )
else:
    optimized_weights = {
        client: target_solution["weights"][i]
        for i, client in enumerate(clients)
    }

    comparison = pd.DataFrame({
        "Cliente": clients,
        "Peso actual (%)": [
            weights_current[c] * 100
            for c in clients
        ],
        "Peso calculado (%)": [
            optimized_weights[c] * 100
            for c in clients
        ],
        "Diferencia (pp)": [
            (
                optimized_weights[c]
                - weights_current[c]
            ) * 100
            for c in clients
        ],
        "Rendimiento (%)": mu * 100,
        "Volatilidad (%)": (
            stats.loc[clients, "volatility"].values
            * 100
        ),
    })

    st.dataframe(
        comparison.style.format({
            "Peso actual (%)": "{:.2f}",
            "Peso calculado (%)": "{:.2f}",
            "Diferencia (pp)": "{:+.2f}",
            "Rendimiento (%)": "{:.2f}",
            "Volatilidad (%)": "{:.2f}",
        }),
        use_container_width=True,
    )

    opt_concentration = concentration_metrics(
        optimized_weights
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Rendimiento esperado",
        f"{target_solution['return'] * 100:.2f}%",
    )

    e2.metric(
        "Riesgo / volatilidad",
        f"{target_solution['risk'] * 100:.2f}%",
    )

    e3.metric(
        "Concentración máxima",
        f"{opt_concentration['max_weight'] * 100:.2f}%",
    )

    e4.metric(
        "HHI",
        f"{opt_concentration['hhi']:.4f}",
    )

    st.plotly_chart(
        comparison_chart(
            weights_current,
            optimized_weights,
        ),
        use_container_width=True,
    )

# ============================================================
# G. CONCENTRACIÓN
# ============================================================
st.header("G. Análisis de concentración")

if target_solution["success"]:
    current_values = sorted(
        weights_current.values(),
        reverse=True,
    )

    optimized_values = sorted(
        optimized_weights.values(),
        reverse=True,
    )

    g1, g2 = st.columns(2)

    with g1:
        st.subheader("Portafolio actual")

        st.write(
            f"Concentración máxima: "
            f"**{current_concentration['max_weight'] * 100:.2f}%**"
        )

        st.write(
            f"Top 2 clientes: "
            f"**{sum(current_values[:2]) * 100:.2f}%**"
        )

        st.write(
            f"Top 3 clientes: "
            f"**{sum(current_values[:3]) * 100:.2f}%**"
        )

        st.write(
            f"HHI: "
            f"**{current_concentration['hhi']:.4f}**"
        )

    with g2:
        st.subheader("Portafolio calculado")

        st.write(
            f"Concentración máxima: "
            f"**{opt_concentration['max_weight'] * 100:.2f}%**"
        )

        st.write(
            f"Top 2 clientes: "
            f"**{sum(optimized_values[:2]) * 100:.2f}%**"
        )

        st.write(
            f"Top 3 clientes: "
            f"**{sum(optimized_values[:3]) * 100:.2f}%**"
        )

        st.write(
            f"HHI: "
            f"**{opt_concentration['hhi']:.4f}**"
        )

# ============================================================
# H. INTERPRETACIÓN
# ============================================================
st.header("H. Interpretación cuantitativa")

if target_solution["success"]:
    st.write(
        f"El portafolio actual presenta un rendimiento esperado "
        f"mensual de **{current_metrics['return'] * 100:.2f}%** "
        f"y una volatilidad mensual de "
        f"**{current_metrics['risk'] * 100:.2f}%**."
    )

    st.write(
        f"Bajo las restricciones seleccionadas, el portafolio "
        f"calculado presenta un rendimiento esperado de "
        f"**{target_solution['return'] * 100:.2f}%** y una "
        f"volatilidad de **{target_solution['risk'] * 100:.2f}%**."
    )

    st.write(
        f"La concentración máxima calculada es "
        f"**{opt_concentration['max_weight'] * 100:.2f}%** "
        f"y el HHI es **{opt_concentration['hhi']:.4f}**."
    )

    st.caption(
        "La interpretación se limita a resultados cuantitativos "
        "del modelo y no clasifica subjetivamente a los clientes."
    )

st.divider()

st.caption(
    "Markowitz se adapta aquí a un portafolio empresarial de clientes: "
    "el rendimiento corresponde a la variación histórica mensual de ingresos "
    "y el riesgo corresponde a su volatilidad/covarianza."
)
