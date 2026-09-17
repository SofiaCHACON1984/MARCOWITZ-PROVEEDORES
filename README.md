# Portafolio de Clientes — Markowitz + Streamlit

Aplicación web en Python/Streamlit para analizar la concentración y diversificación de un portafolio de ingresos de clientes mediante la lógica de la Frontera Eficiente de Markowitz.

## 1. Objetivo

La aplicación adapta Markowitz al contexto empresarial:

- Los "activos" son clientes o categorías de clientes.
- El rendimiento es la variación histórica mensual de ingresos.
- El riesgo se representa mediante la volatilidad de esas variaciones.
- La covarianza/correlación mide cómo se mueven conjuntamente los ingresos de los clientes.
- Los pesos representan la participación del portafolio de ingresos.

No se interpreta a los clientes como acciones bursátiles ni se emite una recomendación subjetiva sobre clientes.

## 2. Base de datos

Archivo:

`data/Base_Portafolio_Clientes_Markowitz.xlsx`

La estructura de referencia inicial es:

| Cliente | % ingresos |
|---|---:|
| Cliente A | 45% |
| Cliente B | 25% |
| Cliente C | 15% |
| Otros | 15% |

La hoja utilizada por la aplicación es `Base_Datos`.

## 3. Variables requeridas

- Fecha
- Cliente
- Ingresos_MXN
- Clientes_Activos
- Clientes_Nuevos
- Clientes_Perdidos
- Cobranza_Pct
- DSO_Dias
- Participacion_Ingresos_Pct
- Variacion_Ingresos_Mensual_Pct
- Rendimiento_Ingreso_Pct
- Concentracion_Cliente_Pct
- Riesgo_Cobranza_Pct
- Crecimiento_Neto_Clientes

## 4. Metodología

### Rendimiento

Se utiliza:

`Rendimiento_Ingreso_Pct / 100`

como serie histórica mensual por cliente.

### Rendimiento esperado

Es la media histórica mensual:

`E(Rp) = w' μ`

donde:

- `w` = vector de pesos
- `μ` = vector de rendimientos esperados

### Riesgo

Se calcula como:

`σp = sqrt(w' Σ w)`

donde `Σ` es la matriz de covarianzas de los rendimientos mensuales.

### Restricciones

Siempre se exige:

`Σ wi = 1`

Además, la aplicación permite definir:

- peso mínimo global
- peso máximo por cliente
- rendimiento objetivo

### Frontera eficiente

Para distintos niveles de rendimiento objetivo se resuelve el portafolio de mínima varianza sujeto a las restricciones.

## 5. Análisis de concentración

La aplicación muestra:

- concentración máxima
- concentración de los 2 principales clientes
- concentración de los 3 principales clientes
- HHI

El HHI utilizado es:

`HHI = Σ wi²`

con pesos expresados como proporciones entre 0 y 1.

## 6. Instalación local

Recomendado: Python 3.11 o superior.

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar:

```bash
streamlit run app.py
```

## 7. Uso

La aplicación incluye el Excel dentro de `data/`, por lo que puede abrirse directamente.

También se puede cargar otro Excel mediante el botón de carga.

La aplicación valida automáticamente las columnas requeridas.

## 8. GitHub

Subir al repositorio:

```text
app.py
requirements.txt
README.md
.gitignore
src/
    __init__.py
    data_loader.py
    portfolio.py
    optimization.py
    charts.py
data/
    Base_Portafolio_Clientes_Markowitz.xlsx
```

## 9. Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir todos los archivos.
3. Crear una aplicación en Streamlit Community Cloud.
4. Seleccionar el repositorio.
5. Seleccionar `app.py` como archivo principal.
6. Desplegar.

No es necesario configurar una ruta local absoluta para el Excel.

## 10. Consideraciones metodológicas

Este proyecto es un ejercicio de analítica empresarial. La interpretación de "rendimiento" corresponde a crecimiento/variación histórica de ingresos y no a rendimiento financiero de valores negociados.

El resultado de la optimización depende de:

- periodo histórico
- calidad de los datos
- definición del rendimiento
- matriz de covarianzas
- restricciones
- rendimiento objetivo

Por ello, la aplicación presenta resultados cuantitativos y evita etiquetar un cliente como "bueno" o "malo".
