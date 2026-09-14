# Predicción de Precios de Viviendas — Ames, Iowa

**Asignatura:** MLY1101 — Machine Learning
**Evaluación:** Parcial N°1 — Presentación y defensa técnica del proyecto
**Caso:** B — Predicción de precios de viviendas (Housing Prices)
**Equipo:** Javier Sagredo — Nicolás Osses
**Repositorio:** `Housing_Prices_ML_P1`
**Notebook:** `notebooks/EDA_Housing_v1_5.ipynb`

---

## Índice

1. [Descripción del problema de negocio](#1-descripción-del-problema-de-negocio)
2. [Objetivos del proyecto](#2-objetivos-del-proyecto)
3. [KPIs del proyecto y criterio de salida](#3-kpis-del-proyecto-y-criterio-de-salida)
4. [Metodología: CRISP-DM](#4-metodología-crisp-dm)
5. [Fuentes de datos y herramientas colaborativas](#5-fuentes-de-datos-y-herramientas-colaborativas)
6. [Análisis exploratorio de los datos (EDA)](#6-análisis-exploratorio-de-los-datos-eda)
7. [Preparación de los datos](#7-preparación-de-los-datos)
8. [Selección inicial de algoritmos](#8-selección-inicial-de-algoritmos)
9. [Métricas de evaluación](#9-métricas-de-evaluación)
10. [Análisis de sesgos](#10-análisis-de-sesgos)
11. [Consideraciones de ética y privacidad](#11-consideraciones-de-ética-y-privacidad)
12. [Reproducibilidad y estructura del proyecto](#12-reproducibilidad-y-estructura-del-proyecto)
13. [Próximos pasos](#13-próximos-pasos)

---

## 1. Descripción del problema de negocio

La tasación de viviendas depende habitualmente del juicio de un perito. Ese procedimiento presenta
tres limitaciones operativas. Es lento, porque cada tasación requiere una visita y un informe
manual. Es inconsistente, ya que dos peritos pueden valorar la misma propiedad con diferencias
relevantes. Y tiene poca trazabilidad, porque el criterio detrás de un valor rara vez queda
documentado de forma auditable.

Para corredoras, bancos hipotecarios y plataformas de compraventa esto implica costos concretos:
propiedades sobrevaloradas que permanecen meses sin venderse, propiedades subvaloradas que
representan una pérdida directa para el vendedor, y decisiones de crédito hipotecario apoyadas en
garantías mal estimadas.

El problema que aborda el proyecto es estimar el precio de venta de una vivienda a partir de sus
características físicas, de calidad y de ubicación, de forma automática, consistente y justificable.

Se trata de un problema de aprendizaje supervisado de regresión: la variable objetivo `SalePrice`
es continua y se dispone de datos históricos etiquetados con el precio efectivamente transado.

---

## 2. Objetivos del proyecto

### Objetivo general

Desarrollar un modelo de Machine Learning capaz de predecir el precio de venta de una vivienda en
Ames, Iowa, con un error suficientemente acotado para servir como herramienta de apoyo a la
tasación profesional.

### Objetivos específicos

1. Caracterizar el dataset mediante un EDA que identifique la calidad de los datos, la presencia de
   anomalías y las variables con mayor poder explicativo sobre el precio.
2. Construir un pipeline de preparación de datos reproducible que resuelva valores nulos, outliers,
   multicolinealidad y codificación de variables categóricas.
3. Entrenar y comparar un máximo de tres algoritmos de regresión, seleccionando el de mejor
   desempeño según las métricas definidas.
4. Validar que el modelo cumpla el criterio de salida establecido antes de considerarlo apto para
   uso.
5. Evaluar los sesgos del modelo, en particular los asociados a la ubicación geográfica, y
   documentar las implicancias éticas de su uso.

### Alcance

Esta entrega cubre la comprensión del negocio, el EDA, la preparación de datos, la selección
inicial de algoritmos y la definición de criterios de evaluación. Quedan fuera el entrenamiento
final, la optimización de hiperparámetros y el despliegue productivo.

---

## 3. KPIs del proyecto y criterio de salida

Los KPIs de este proyecto son indicadores del entrenamiento del modelo, no de rentabilidad del
negocio. El objetivo de esta entrega es que el modelo aprenda a predecir `SalePrice` dentro de un
error acotado y medible sobre datos que no vio en el entrenamiento; los KPIs son la forma concreta
de verificar si ese objetivo se cumplió. El impacto de negocio (tiempo de tasación, cobertura del
catálogo, etc.) es la motivación del proyecto, descrita en la sección 1, pero no es algo que se mida
en esta fase porque requiere un modelo desplegado, lo cual está fuera del alcance de esta entrega.

### 3.1 KPIs técnicos

| KPI | Qué mide | Umbral (criterio de salida) |
|---|---|---|
| R² (test) | Proporción de la varianza de `SalePrice` explicada por el modelo | ≥ 0,85 |
| RMSE (test) | Error promedio en dólares, penalizando cuadráticamente los errores grandes | ≤ 30.000 USD |
| MAE (test) | Error promedio en dólares, sin penalización adicional | ≤ 20.000 USD |
| MAPE (test) | Error relativo al precio de cada propiedad | ≤ 12% |
| Brecha train–test (R²) | Diferencia de R² entre entrenamiento y prueba | ≤ 5 puntos |
| Brecha de equidad territorial (MAPE) | Diferencia de MAPE entre el barrio mejor y peor predicho | ≤ 8 puntos porcentuales |

Estos seis indicadores son a la vez los KPIs del proyecto y el criterio de salida: el modelo se
considera válido, y apto para pasar a una eventual fase de despliegue, únicamente si cumple los seis
de forma simultánea sobre el conjunto de prueba. No hay un KPI "principal" con los demás como
apoyo; los seis miden aspectos distintos de la misma pregunta —¿aprendió el modelo algo que
generaliza y que generaliza igual de bien para todos los segmentos?— y ninguno por separado la
responde completa.

**Por qué cada uno se necesita:**

- **R² y RMSE/MAE** son complementarios y no redundantes. R² es adimensional y permite comparar
  entre modelos; RMSE y MAE están en dólares y permiten interpretar la magnitud del error. Que RMSE
  supere claramente a MAE es en sí mismo un diagnóstico: indica que el modelo comete errores graves
  en un subconjunto de casos, no que el error esté distribuido parejo.
- **MAPE** corrige un punto ciego de las métricas anteriores. Un error de 30.000 dólares es un 10%
  en una vivienda de 300.000 y un 34% en una de 88.000; un modelo puede tener buen RMSE y aun así
  fallar sistemáticamente en el segmento de menor precio. El umbral de 12% es el mismo que se usa
  para dimensionar la brecha de equidad (sección 10.7).
- **La brecha train–test** es el control de que el modelo generaliza y no memoriza. Un R² de 0,97 en
  entrenamiento y 0,84 en prueba no es un buen modelo con algo de ruido; es un modelo sobreajustado
  que no debe aprobarse aunque su R² de prueba en aislado se vea aceptable.
- **La brecha de equidad territorial** se incluyó como KPI técnico y no como un análisis aparte,
  porque el análisis de sesgos (sección 10.7) mostró que es alcanzable solo bajo ciertas decisiones
  de modelado (entrenar sobre `log1p(SalePrice)`) y no es automática. Un modelo que cumple los otros
  cinco KPIs y falla este entrena bien en promedio y mal para un segmento específico de la
  población, lo cual invalida el modelo para este proyecto aunque su desempeño agregado sea bueno.

### 3.2 Fundamento de cada umbral

| KPI | Umbral | Por qué ese número |
|---|---|---|
| R² | ≥ 0,85 | El modelo debe explicar al menos el 85% de la varianza del precio; es el nivel a partir del cual un RMSE ≤ 30.000 USD es matemáticamente consistente dada la desviación estándar de `SalePrice` (σ = 78.555) |
| RMSE | ≤ 30.000 USD | Equivale a 0,38 desviaciones estándar de `SalePrice` |
| MAE | ≤ 20.000 USD | Aproximadamente 12% del precio mediano (160.000 USD) |
| MAPE | ≤ 12% | Umbral de error relativo consistente con el MAE sobre el precio mediano |
| Brecha train–test | ≤ 5 puntos de R² | Diferencia a partir de la cual la brecha deja de explicarse por varianza de muestreo y empieza a indicar sobreajuste |
| Equidad territorial | ≤ 8 p.p. de MAPE entre barrios | Umbral definido de forma independiente al resultado, antes de correr la auditoría de la sección 10.7; sirve como prueba de que ningún segmento geográfico recibe un modelo sistemáticamente peor |

Si el modelo no cumple el criterio, la respuesta no es bajar el umbral sino volver a una fase
anterior de CRISP-DM: revisar la ingeniería de características, probar otra transformación del
objetivo o reconsiderar el algoritmo. El umbral de equidad se definió deliberadamente junto a los
umbrales de precisión y no como un chequeo posterior: un modelo que cumpla los cinco primeros y
falle el sexto funciona bien para unos usuarios y mal para otros, lo que en este dominio es un
problema de fondo y no un defecto menor.

### 3.3 Motivación de negocio (contexto, no medido en esta entrega)

Estos indicadores no se calculan en esta fase porque requieren un modelo desplegado y en
producción; se documentan aquí únicamente como la razón por la que los KPIs técnicos de la sección
3.1 importan.

| Indicador de negocio | Qué reflejaría | Depende de |
|---|---|---|
| Tiempo de tasación | Minutos desde el ingreso de datos hasta la entrega del valor estimado, frente a días en el proceso manual | Despliegue del modelo (fuera de alcance) |
| Cobertura del catálogo | % de propiedades tasables automáticamente sin intervención de un perito | Política de derivación (sección 10.1) y umbral de confianza |
| Tasa de derivación | % de casos que el modelo deriva a tasación manual por baja confianza | Política de derivación, aún no definida |

La consistencia territorial —el equivalente de negocio de la brecha de equidad territorial— no se
duplica aquí porque ya es, en sí misma, uno de los seis KPIs técnicos de la sección 3.1: un modelo
con buen error promedio pero que falla sistemáticamente en los barrios de menor precio produce un
daño real sobre los propietarios de esos sectores, como se detalla en la sección 10.

---

## 4. Metodología: CRISP-DM

El proyecto se organiza según CRISP-DM (*Cross-Industry Standard Process for Data Mining*), un
marco iterativo e independiente de la industria y la tecnología.

| Fase | Cómo se aplicó en este proyecto | Sección |
|---|---|---|
| 1. Comprensión del negocio | Definición del problema de tasación, objetivos, KPIs y criterio de salida antes de tocar los datos | 1, 2, 3 |
| 2. Comprensión de los datos | Diccionario de las 82 variables, correlaciones, nulos, outliers y multicolinealidad | 5, 6 |
| 3. Preparación de los datos | Imputación diferenciada, flags, poda de redundancia, encoding, split y escalado | 7 |
| 4. Modelado | Preselección de tres algoritmos y estrategia de comparación (entrenamiento fuera del alcance de esta entrega) | 8 |
| 5. Evaluación | Métricas, umbrales de aprobación y auditoría de sesgos | 3, 9, 10 |
| 6. Despliegue | Fuera del alcance | 13 |

El trabajo no avanzó de forma lineal. El diagnóstico de multicolinealidad, hecho en la fase 2,
obligó a volver a la fase 3 para eliminar variables redundantes. La detección de fuga de datos en
el escalado obligó a reordenar el pipeline completo (sección 7.6). Y el análisis de sesgos de la
fase 5 retroalimentó la preparación, al mostrar que conviene entrenar sobre `log1p(SalePrice)`
(sección 10.7).

---

## 5. Fuentes de datos y herramientas colaborativas

### 5.1 Fuente de datos

Se utiliza el *Ames, Iowa Housing Dataset*, compilado por Dean De Cock (Truman State University,
2011) a partir de los registros de la Oficina del Asesor de la Ciudad de Ames, Iowa.

| Atributo | Valor |
|---|---|
| Registros | 2.930 transacciones |
| Variables | 82 columnas |
| Período cubierto | 2006 – 2010 |
| Unidad de observación | Una venta residencial individual |
| Variable objetivo | `SalePrice` (dólares estadounidenses) |
| Origen | Registro administrativo oficial (no encuesta ni scraping) |
| Licencia | Uso público para fines educativos y de investigación |

La fuente es adecuada para el problema por dos razones. Al provenir de un registro catastral
oficial, las características físicas de las propiedades (superficies, año de construcción,
materiales) están verificadas y no son auto-reportadas. Y `SalePrice` corresponde al precio
efectivamente transado, no a un precio de lista, lo que evita el sesgo de expectativa del vendedor.

Composición de las variables:

| Tipo | Cantidad | Ejemplos |
|---|---|---|
| Cuantitativas continuas | 20 | `Lot Area`, `Gr Liv Area`, `Total Bsmt SF` |
| Cuantitativas discretas | 14 | `Full Bath`, `Bedroom AbvGr`, `Garage Cars` |
| Cualitativas ordinales | 23 | `Overall Qual`, `Kitchen Qual`, `Bsmt Exposure` |
| Cualitativas nominales | 23 | `Neighborhood`, `MS Zoning`, `Sale Type` |
| Identificadores | 2 | `Order`, `PID` |

El notebook incluye un diccionario con la definición y el tipo de cada una de las 82 columnas.

### 5.2 Herramientas colaborativas

| Herramienta | Uso en el proyecto | Justificación |
|---|---|---|
| GitHub | Repositorio central, con una rama por integrante y estructura `data/raw`, `notebooks/` | Permite trabajo paralelo con control de versiones y trazabilidad de cada cambio. El dataset se carga desde la URL `raw` del repositorio, de modo que el notebook es ejecutable por cualquier integrante sin rutas locales |
| Jupyter Notebook / Google Colab | Desarrollo del EDA y el pipeline de preparación | Formato reproducible que integra código, resultados y documentación en un solo entregable |
| Python y ecosistema de datos | `pandas`, `numpy` (manipulación), `matplotlib`, `seaborn` (visualización), `scikit-learn`, `category_encoders` (preparación y modelado) | Estándar de la industria para ciencia de datos, con documentación madura y amplia compatibilidad |
| Markdown | Documentación del informe y de las decisiones dentro del notebook | Texto plano versionable en Git, a diferencia de los formatos binarios de ofimática |

Se optó por cargar los datos desde la URL pública del repositorio en lugar de un archivo local. Así
ambos integrantes trabajan sobre la misma versión del dataset y el notebook se ejecuta sin
modificaciones en cualquier máquina.

---

## 6. Análisis exploratorio de los datos (EDA)

### 6.1 Variable objetivo: `SalePrice`

| Estadístico | Valor (USD) |
|---|---|
| Media | 180.412 |
| Mediana | 160.000 |
| Desviación estándar | 78.555 |
| Mínimo | 12.789 |
| Máximo | 625.000 |
| Rango intercuartílico | 129.500 – 213.500 |
| Asimetría (*skew*) | 1,59 |

La distribución tiene asimetría positiva marcada. La media supera a la mediana en más de 20.000
dólares, arrastrada por la cola derecha de propiedades de alto valor. Esto tuvo dos consecuencias
de diseño.

La primera es que los modelos lineales asumen residuos aproximadamente normales, por lo que se
implementó una transformación logarítmica (`log1p`) de la variable objetivo, aplicada después del
split (sección 7.6). El análisis de sesgos mostró más tarde que esa transformación cumple además
una función de equidad (sección 10.7).

La segunda es que las métricas sensibles a valores extremos (RMSE) y las insensibles (MAE) darán
lecturas distintas, por lo que se reportan ambas (sección 9).

### 6.2 Variables más correlacionadas con el precio

| Variable | Correlación con `SalePrice` | Interpretación |
|---|---|---|
| `Overall Qual` | 0,799 | Calidad general de materiales y terminaciones |
| `Gr Liv Area` | 0,707 | Superficie habitable sobre el nivel del suelo |
| `Garage Cars` | 0,648 | Capacidad del garage en número de autos |
| `Garage Area` | 0,640 | Superficie del garage |
| `Total Bsmt SF` | 0,632 | Superficie total del sótano |
| `1st Flr SF` | 0,622 | Superficie del primer piso |
| `Year Built` | 0,558 | Año de construcción |

El predictor más fuerte no es una medida física sino una evaluación de calidad. `Overall Qual` es
una escala de 1 a 10 asignada por el tasador municipal, de modo que el modelo hereda parcialmente
el criterio humano que pretende automatizar. El punto se retoma en el análisis de sesgos.

En el extremo opuesto, `BsmtFin SF 2` (0,006), `3Ssn Porch` (0,032) y `Mo Sold` (0,035)
prácticamente no explican el precio. El caso de `Mo Sold` sugiere que en este mercado no hay
estacionalidad de precios relevante, aunque sí pueda haberla en el volumen de transacciones.

### 6.3 Calidad de los datos: valores nulos

Se detectaron 27 columnas con valores nulos:

| Columna | Nulos | % del total |
|---|---|---|
| `Pool QC` | 2.917 | 99,6% |
| `Misc Feature` | 2.824 | 96,4% |
| `Alley` | 2.732 | 93,2% |
| `Fence` | 2.358 | 80,5% |
| `Mas Vnr Type` | 1.775 | 60,6% |
| `Fireplace Qu` | 1.422 | 48,5% |
| `Lot Frontage` | 490 | 16,7% |
| Grupo garage (4 columnas) | 157–159 | ~5,4% |
| Grupo sótano (5 columnas) | 80–83 | ~2,8% |

La mayoría de estos nulos no son datos perdidos. Son nulos estructurales que codifican la ausencia
de una característica. Se validó cruzando cada columna categórica con su columna numérica
compañera: cuando `Pool QC` es nulo, `Pool Area` vale 0 en el 100% de los casos, y lo mismo ocurre
con `Misc Feature` / `Misc Val` y con los grupos de garage y sótano.

Este diagnóstico cambió el tratamiento. Eliminar filas o columnas con muchos nulos habría destruido
información válida y reducido el dataset a una fracción de su tamaño. `Lot Frontage` resultó ser el
único caso de dato genuinamente faltante.

No se detectaron filas duplicadas ni `PID` repetidos, lo que confirma que cada registro corresponde
a una transacción única.

### 6.4 Detección y análisis de outliers

Se aplicó el criterio del rango intercuartílico (IQR) sobre las 39 variables numéricas. En lugar de
eliminar automáticamente todo lo que excediera 1,5·IQR, cada grupo se evaluó según su naturaleza:

| Grupo | Ejemplos | Diagnóstico | Acción |
|---|---|---|---|
| Dispersas (Q1 = Q3 = 0) | `Pool Area`, `Misc Val`, `3Ssn Porch`, `Screen Porch`, `Enclosed Porch` | El IQR marca como outlier el simple hecho de tener la característica. Son variables cuasi-binarias | Conservar; crear flags binarios |
| Discretas / ordinales | `Bedroom AbvGr`, `TotRms AbvGrd`, `Overall Cond` | Outliers técnicos que corresponden a conteos reales y válidos | Conservar sin modificar |
| Continuas | `SalePrice`, `Lot Area`, `Gr Liv Area` | Colas largas graduales, propias del mercado inmobiliario | Conservar; tratar con `RobustScaler` |
| Anomalías reales | 5 propiedades con `Gr Liv Area` > 4.000 pies² | Ventas no representativas del mercado | Eliminar |

Al inspeccionar las cinco propiedades eliminadas se observó que tres de ellas (índices 1498, 2180,
2181) tienen calidad 10, superficies muy grandes y precios anormalmente bajos (160.000, 183.850 y
184.750 dólares), con `Sale Condition = Partial` y `Sale Type = New`. Corresponden a ventas de
casas aún en construcción, donde el precio registrado no refleja el valor de la propiedad
terminada. Un cuarto caso es `Abnorml`. Incluirlas habría enseñado al modelo una relación falsa
entre superficie y precio.

De Cock, autor del dataset, recomienda esta exclusión en la documentación original, lo que aporta
respaldo externo a la decisión.

El dataset pasó de 2.930 a 2.925 registros, una pérdida del 0,17%.

### 6.5 Multicolinealidad entre predictores

Se calculó la matriz de correlación entre predictores y se aislaron los pares con |r| > 0,6:

| Par | Correlación | Interpretación |
|---|---|---|
| `Garage Cars` / `Garage Area` | 0,890 | Miden lo mismo en distinta unidad |
| `Year Built` / `Garage Yr Blt` | 0,835 | El garage se construye junto con la casa |
| `Gr Liv Area` / `TotRms AbvGrd` | 0,808 | Más superficie implica más habitaciones |
| `Total Bsmt SF` / `1st Flr SF` | 0,801 | Comparten la huella de planta del edificio |
| `Order` / `Yr Sold` | -0,976 | Correlación espuria: artefacto del orden de registro |

El par `Order` / `Yr Sold` requiere una aclaración. Una correlación de -0,976 parece una señal muy
fuerte, pero no tiene significado: los registros están ordenados cronológicamente, de modo que el
índice de fila predice el año de venta. Es una correlación puramente administrativa que un modelo
entrenado sin cuidado aprendería como si fuera una relación del mundo real.

---

## 7. Preparación de los datos

### 7.1 Tratamiento de valores nulos

Se aplicaron cinco criterios diferenciados, según el significado de cada nulo:

| Grupo | Columnas | Criterio | Justificación |
|---|---|---|---|
| Categóricas de ausencia | `Pool QC`, `Misc Feature`, `Alley`, `Fence`, `Fireplace Qu` + grupos garage y sótano (14 en total) | Categoría explícita `"None"` | El NaN es informativo. Convertirlo en categoría preserva la información sin eliminar registros |
| Dato faltante real | `Lot Frontage` (490) | Mediana por `Neighborhood`, con mediana global de respaldo | Toda propiedad tiene frente de lote; el dato no se registró. El frente depende del trazado urbano del barrio, por lo que la mediana local es menos sesgada que la global |
| Numéricas compañeras | `Garage Cars`, `Garage Area`, `Total Bsmt SF`, `BsmtFin SF 1/2`, `Bsmt Unf SF`, `Bsmt Full/Half Bath` | Valor `0` | Corresponden a casas sin garage o sin sótano. El 0 es el valor semánticamente correcto, no una estimación |
| Caso mixto | `Mas Vnr Type` (1.775), `Mas Vnr Area` (23) | Ver detalle abajo | Requirió diagnóstico previo por inconsistencia entre columnas |
| Casos aislados | `Garage Yr Blt` (159), `Electrical` (1) | `Year Built` / moda (`SBrkr`) | Impacto mínimo; `Garage Yr Blt` se elimina después por redundancia |

Las dos columnas de `Mas Vnr` describen lo mismo (revestimiento de fachada) pero tenían distinta
cantidad de nulos, lo que indicaba una inconsistencia. El diagnóstico arrojó tres situaciones:
1.744 filas con tipo nulo y área 0, coherente con no tener revestimiento; 23 con ambas nulas; y 7
filas con área declarada pero sin tipo, algunas de hasta 344 pies². Rellenar esas 7 con `"None"`
habría afirmado que no tienen revestimiento cuando su propia área dice lo contrario, así que se
imputaron con la moda (`BrkFace`).

En el caso de `Lot Frontage`, dos barrios (`GrnHill` y `Landmrk`, 3 propiedades entre ambos) no
tienen ningún registro de la variable, por lo que su mediana local es indefinida. Para esos casos
se aplicó la mediana global de 68 pies.

El resultado fue 0 valores nulos, sin pérdida de registros.

### 7.2 Creación de variables flag

Se crearon cuatro variables binarias:

| Flag | Columna base | Propiedades con la característica |
|---|---|---|
| `Tiene_Piscina` | `Pool Area` > 0 | 11 (0,4%) |
| `Tiene_Sotano` | `Total Bsmt SF` > 0 | 2.845 (97,3%) |
| `Tiene_Garage` | `Garage Area` > 0 | 2.767 (94,6%) |
| `Tiene_Chimenea` | `Fireplaces` > 0 | 1.503 (51,4%) |

En el encoding ordinal posterior la categoría `"None"` se mapea al valor 0, lo que haría
indistinguible "no tiene piscina" de "tiene una piscina de pésima calidad". Los flags separan la
existencia de la característica de su calidad, de modo que el modelo pueda aprender ambos efectos
por separado.

### 7.3 Eliminación de identificadores

Se eliminaron `Order` y `PID`. Son identificadores administrativos sin capacidad predictiva, y las
correlaciones que muestran (`PID` vs `SalePrice` = -0,247, `Order` vs `Yr Sold` = -0,976) son
artefactos del orden de registro. Mantenerlos arriesgaría que el modelo aprenda un patrón
burocrático en lugar de las características reales de la vivienda.

`PID` es además un identificador catastral real, por lo que su eliminación contribuye a la
anonimización del dataset (sección 11).

### 7.4 Tratamiento de multicolinealidad

Antes de podar se guardó una copia íntegra del dataset (`df_completo`, 2.925 × 84) para el análisis
de sesgos de la sección 10. K-Means agrupa por similitud global y no se ve perjudicado por la
redundancia entre variables, a diferencia de los modelos de regresión.

Sobre `df_clean` se eliminaron cuatro variables, conservando en cada par la de mayor correlación
con `SalePrice`:

| Eliminada | Conservada | Corr. entre ellas | Criterio |
|---|---|---|---|
| `Garage Area` | `Garage Cars` | 0,892 | Diferencia mínima con el objetivo (0,652 vs 0,648); desempata la interpretabilidad de negocio |
| `Garage Yr Blt` | `Year Built` | 0,853 | Menor correlación con el objetivo y 159 valores imputados |
| `TotRms AbvGrd` | `Gr Liv Area` | 0,809 | 0,498 vs 0,719 con el objetivo; la superficie continua es más informativa que el conteo |
| `1st Flr SF` | `Total Bsmt SF` | 0,784 | Solapada con el sótano y además componente aritmético de `Gr Liv Area` |

Los pares residuales entre 0,6 y 0,7 (`2nd Flr SF`/`Gr Liv Area` = 0,662, `Year Built`/`Year
Remod/Add` = 0,611, entre otros) se conservaron. La correlación es moderada y cada variable aporta
un matiz propio, de manera que seguir podando en ese rango eliminaría señal real a cambio de una
mejora marginal en independencia.

El resultado son dos datasets: `df_clean` (2.925 × 80) para modelado y `df_completo` (2.925 × 84)
para el análisis de sesgos.

### 7.5 Codificación de variables categóricas

Se aplicaron tres estrategias según la naturaleza de cada variable:

| Estrategia | Variables | Criterio |
|---|---|---|
| Ordinal (mapeo manual) | 20 columnas: `Overall Qual`, `Exter Qual`, `Kitchen Qual`, `Bsmt Exposure`, `Garage Finish`, `Functional`, `Fence`, `Lot Shape`, entre otras | Tienen un orden natural de calidad que debe preservarse (`Po` < `Fa` < `TA` < `Gd` < `Ex`). Un one-hot destruiría esa jerarquía |
| One-hot (`drop_first=True`) | 21 columnas de baja cardinalidad: `MS Zoning`, `Bldg Type`, `Roof Style`, `Foundation`, `Sale Condition`, entre otras | Categorías sin orden y con pocos niveles. `drop_first` evita la trampa de variables dummy |
| Target encoding | 3 columnas de alta cardinalidad: `Neighborhood` (28 niveles), `Exterior 1st`, `Exterior 2nd` | Un one-hot agregaría más de 70 columnas dispersas. El target encoding las resume en el precio medio de cada categoría |

`MS SubClass` requirió un tratamiento aparte. Aunque está almacenada como entero (20, 30, 60,
190…), es un código categórico del tipo de vivienda y no una cantidad: la clase 190 no vale más que
la 20. Se convirtió explícitamente a texto antes del one-hot, porque de lo contrario el modelo
habría interpretado un orden inexistente.

Tras el encoding el dataset quedó con 170 columnas y sin nulos.

### 7.6 División train/test y escalado

La primera versión del pipeline aplicaba el target encoding y el escalado sobre el dataset
completo, y recién después dividía en entrenamiento y prueba. Eso constituye fuga de datos (*data
leakage*). En el caso del target encoding el problema es grave, porque esa codificación se
construye usando `SalePrice`: el precio de las viviendas destinadas a prueba quedaba incorporado
dentro de sus propias variables predictoras, y el modelo habría reportado métricas infladas,
imposibles de replicar con datos nuevos.

La corrección consistió en reordenar el pipeline. El split se realiza antes de cualquier
transformación que aprenda parámetros de los datos, y tanto el `TargetEncoder` como el
`ColumnTransformer` se ajustan (`fit`) exclusivamente con el conjunto de entrenamiento y se aplican
(`transform`) al de prueba sin reajustar. Se documenta el error porque ilustra un punto
metodológico del proyecto: el orden de las operaciones en un pipeline de ML afecta la validez de
los resultados, no solo su presentación.

Orden final del pipeline:

```
1. MS SubClass → texto
2. Encoding ordinal (mapeo manual, no aprende de los datos)
3. One-hot (categorías fijas y conocidas)
4. Separación X / y
5. ► train_test_split (80/20, random_state=42)
6. log1p(y)           → y_train_log / y_test_log (el target en dólares se conserva)
7. TargetEncoder      → fit con X_train, transform en X_test
8. ColumnTransformer  → fit con X_train, transform en X_test
```

Los pasos 2 y 3 pueden anteceder al split sin riesgo porque no estiman ningún parámetro a partir de
la distribución de los datos: son mapeos deterministas definidos por el analista.

Escalado diferenciado:

| Escalador | Variables | Justificación |
|---|---|---|
| `RobustScaler` | 10 superficies con colas largas: `Lot Area`, `Gr Liv Area`, `Total Bsmt SF`, `Lot Frontage`, entre otras | Usa mediana e IQR, robustos frente a los valores extremos propios del mercado inmobiliario |
| `StandardScaler` | 24 columnas: numéricas discretas, variables casi-todo-cero y las 3 target-encoded | Ver detalle abajo |
| Sin escalar (*passthrough*) | Ordinales mapeadas, one-hot, flags | Ya se encuentran en rangos acotados y comparables |

La auditoría del escalado obligó a dos ajustes. Las columnas casi-todo-cero (`Pool Area`,
`Misc Val`, `Screen Porch`, `Enclosed Porch`, `BsmtFin SF 2`, `Low Qual Fin SF`, `3Ssn Porch`)
tienen IQR = 0, por lo que `RobustScaler` las dejaba sin escalar y `Misc Val` llegaba a 15.500 en
escala original. Se trasladaron a `StandardScaler`, que usa la desviación estándar y sí funciona
con mediana 0. Las variables target-encoded, por su parte, quedaban en escala de dólares
(`Neighborhood` entre 95.756 y 324.229) frente al resto entre -3 y 3, lo que las habría hecho
dominar cualquier modelo regularizado o basado en distancia, así que se incorporaron al escalado.

El `TargetEncoder` se configuró con suavizado (`min_samples_leaf=20`, `smoothing=10`) porque
existen barrios con muy pocas observaciones. Sin suavizado, la codificación de `Landmrk` (1 casa)
sería literalmente el precio de esa casa.

Respecto de la transformación logarítmica del objetivo, se crean `y_train_log` e `y_test_log`
mediante `log1p` sin reemplazar `y_train` e `y_test`, porque el target encoding del paso siguiente
debe seguir calculándose con el precio en dólares reales. La transformación se aplica después del
split por coherencia con el resto del pipeline, aunque `log1p` es determinista y no habría generado
fuga.

Resultado final: `X_train` (2.340 × 169), `X_test` (585 × 169), sin nulos y con todas las columnas
en tipo numérico.

---

## 8. Selección inicial de algoritmos

Se preseleccionaron tres algoritmos, ordenados de menor a mayor complejidad.

### 8.1 Regresión lineal regularizada (Ridge / Lasso)

Sirve como línea base interpretable contra la cual medir si la complejidad adicional de los otros
modelos se justifica. Los coeficientes son directamente legibles (cada pie² adicional de superficie
habitable suma X dólares), lo que responde a la necesidad de trazabilidad planteada en el problema
de negocio. La regularización es necesaria porque quedan 169 predictores y varios pares con
correlación moderada.

Su limitación es que asume relaciones lineales, y la relación entre superficie y precio muestra
curvatura en los extremos.

### 8.2 Random Forest Regressor

Captura relaciones no lineales e interacciones entre variables sin requerir especificación previa,
por ejemplo que el efecto de la superficie dependa del barrio. Es robusto frente a outliers e
insensible a la escala de las variables, y entrega importancia de variables, útil para validar los
hallazgos del EDA.

Sus limitaciones son la menor interpretabilidad individual y la tendencia a sobreajustar si no se
controla la profundidad.

### 8.3 Gradient Boosting (XGBoost / HistGradientBoostingRegressor)

Es el algoritmo con mejor desempeño documentado en problemas de regresión sobre datos tabulares, y
este dataset es un caso de uso habitual para evaluarlo. Construye árboles secuencialmente,
corrigiendo los errores de los anteriores.

Sus limitaciones son el mayor costo computacional, la mayor cantidad de hiperparámetros a ajustar y
la menor transparencia, lo que puede ser un problema si el modelo debe justificar valores ante un
cliente.

### 8.4 Estrategia de comparación

Los tres se evalúan con validación cruzada de 5 folds sobre el conjunto de entrenamiento,
reservando el conjunto de prueba exclusivamente para la evaluación final. El modelo se selecciona
por desempeño, pero ante diferencias menores a un punto porcentual de R² se privilegia el más
interpretable, dada la naturaleza del problema de negocio.

---

## 9. Métricas de evaluación

Se seleccionaron cuatro métricas complementarias, porque ninguna basta por sí sola. Estas son las
mismas cuatro métricas de error que forman parte de los KPIs técnicos de la sección 3.1; esta
sección detalla qué mide cada una y por qué se necesitan las cuatro en conjunto.

| Métrica | Qué mide | Por qué se incluye |
|---|---|---|
| RMSE | Error promedio en dólares, penalizando cuadráticamente los errores grandes | Métrica principal. En tasación, un error de 100.000 dólares es mucho más grave que diez errores de 10.000, porque puede hacer fracasar una operación de crédito completa. El RMSE castiga esos casos |
| MAE | Error promedio en dólares, sin penalización adicional | Lectura directa para el negocio: el modelo se equivoca en promedio en X dólares. Contrastado con el RMSE, permite detectar si el error está concentrado en pocos casos extremos |
| R² | Proporción de la varianza del precio explicada por el modelo | Métrica adimensional que permite comparar modelos entre sí de forma estandarizada |
| MAPE | Error relativo al precio de la propiedad | Métrica de equidad. Un error de 30.000 dólares es un 10% en una casa de 300.000 y un 34% en una de 88.000. El MAPE expone el error que las métricas absolutas ocultan en el segmento de menor precio |

La combinación de RMSE y MAE aporta un diagnóstico que ninguna de las dos entrega por separado: si
el RMSE resulta muy superior al MAE, el modelo es razonable en general pero comete errores graves
en un subconjunto de propiedades.

El MAPE se incluyó por la asimetría de `SalePrice` (skew = 1,59) y la dispersión de precios entre
barrios documentada en la sección 10. Una métrica en dólares absolutos favorece estructuralmente al
segmento alto del mercado, y el MAPE es el contrapeso. Es también la métrica que conecta con el KPI
de consistencia territorial.

---

## 10. Análisis de sesgos

El análisis se realiza sobre los datos y no sobre un modelo entrenado. El objetivo es identificar
qué sesgos ya están presentes en el dataset y serán heredados por cualquier algoritmo que se
entrene con él. Todo lo que sigue está calculado en el notebook y es reproducible.

### 10.1 Sesgo de representación

| Barrio | Propiedades |
|---|---|
| `Landmrk` | 1 |
| `GrnHill` | 2 |
| `Greens` | 8 |
| `Blueste` | 10 |
| `NPkVill` | 23 |
| `Veenker` | 24 |
| `Blmngtn` | 28 |

Siete de los 28 barrios tienen menos de 30 propiedades y en conjunto representan el 3,3% del
dataset. El barrio más representado (`NAmes`, 443 propiedades) tiene 443 veces más datos que el
menos representado. El aporte de estos barrios al entrenamiento es marginal, pero el error que
reciben puede ser alto, porque el modelo los predice extrapolando desde barrios que no
necesariamente se les parecen.

Como medida, el sistema debe señalar baja confianza y derivar a tasación manual cuando la propiedad
pertenezca a un barrio sub-representado, lo que se refleja en el KPI de tasa de derivación. No es
un problema que se resuelva con una mejor arquitectura de modelo, porque su origen es una carencia
de datos.

### 10.2 Sesgo histórico: dispersión geográfica del precio

| Barrio | N° de propiedades | Mediana de precio | Precio por pie² |
|---|---|---|---|
| Meadow Village (`MeadowV`) | 37 | 88.250 USD | 86 USD |
| Briardale (`BrDale`) | 30 | 106.000 USD | 95 USD |
| Iowa DOT and Rail Road (`IDOTRR`) | 93 | 106.500 USD | 87 USD |
| … | | | |
| Northridge (`NoRidge`) | 69 | 301.500 USD | 129 USD |
| Northridge Heights (`NridgHt`) | 166 | 317.750 USD | 160 USD |
| Stone Brook (`StoneBr`) | 51 | 319.000 USD | 158 USD |

La razón entre el barrio más caro y el más barato es de 3,6 a 1 en mediana de precio. Al controlar
por superficie mediante el precio por pie cuadrado la brecha baja a 2,35 a 1, pero no desaparece:
más de la mitad de la diferencia de precio entre barrios no se explica por el tamaño de las
viviendas, sino por la ubicación.

Esto constituye un sesgo y no solo una descripción del mercado por el contexto en que se generaron
los datos. En el ámbito urbano estadounidense, la ubicación residencial está históricamente
correlacionada con la composición socioeconómica y racial de la población, como resultado
documentado de prácticas de exclusión crediticia (*redlining*). Un modelo que aprende que las
viviendas de un sector valen poco reproduce esa desigualdad y le da apariencia de objetividad
técnica. Si una tasadora automatizada subvalora sistemáticamente un sector, sus propietarios
acceden a menos crédito con la misma garantía, lo que refuerza el deterioro del sector y realimenta
el patrón que el modelo aprendió.

El target encoding aplicado a `Neighborhood` agrava el problema, porque codifica literalmente el
precio medio histórico de cada barrio y lo convierte en la variable predictiva más directa del
modelo.

### 10.3 El barrio como variable latente: análisis con K-Means

La objeción evidente al punto anterior es que si `Neighborhood` genera el sesgo, basta con
eliminarla del modelo. Esta sección pone a prueba esa idea, y el resultado es el hallazgo principal
del análisis.

Se agruparon las 2.925 propiedades con K-Means (k = 4) usando exclusivamente 18 atributos físicos
objetivos: superficies, año de construcción, baños, dormitorios, lote y garage. Se excluyeron el
precio, el barrio y las calificaciones subjetivas de calidad. Luego se verificó cómo se distribuye
cada barrio entre los segmentos resultantes.

| Segmento | N° | Precio mediano | Superficie media | Año medio |
|---|---|---|---|---|
| 0 | 1.196 | 127.000 USD | 1.101 pies² | 1953 |
| 1 | 666 | 176.250 USD | 1.728 pies² | 1975 |
| 2 | 789 | 207.500 USD | 1.559 pies² | 1990 |
| 3 | 274 | 287.250 USD | 2.453 pies² | 1988 |

Distribución de barrios seleccionados entre los segmentos, como porcentaje de las viviendas del
barrio:

| Barrio | Mediana | Seg. 0 | Seg. 1 | Seg. 2 | Seg. 3 |
|---|---|---|---|---|---|
| `MeadowV` | 88.250 | 91,9 % | 5,4 % | 0 % | 2,7 % |
| `IDOTRR` | 106.500 | 84,9 % | 12,9 % | 1,1 % | 1,1 % |
| `BrkSide` | 126.750 | 82,4 % | 16,7 % | 0,9 % | 0 % |
| `Blmngtn` | 191.500 | 0 % | 0 % | 100 % | 0 % |
| `NoRidge` | 301.500 | 0 % | 7,2 % | 8,7 % | 84,1 % |
| `NridgHt` | 317.750 | 0 % | 2,4 % | 62,0 % | 35,5 % |

Los indicadores de asociación confirman el patrón. La V de Cramér entre barrio y segmento físico es
de 0,51, una asociación moderada-fuerte. En promedio, el 66,4% de las viviendas de un barrio cae en
un mismo segmento. Y la correlación entre el porcentaje del barrio en el segmento más barato y la
mediana de precio del barrio es de -0,71.

La conclusión es que el barrio funciona como variable latente del dataset. Está codificado de forma
redundante en la superficie, el año de construcción, el tamaño del lote y la cantidad de baños. Los
segmentos se construyeron sin usar el precio ni el barrio y aun así reconstruyen la geografía de
Ames.

Esto invalida la solución intuitiva. Eliminar la columna `Neighborhood` no elimina el sesgo
geográfico: elimina la capacidad de auditarlo. El modelo seguiría discriminando territorialmente y
ya no habría forma de medirlo.

Se adoptaron tres medidas. Primero, conservar `Neighborhood` en el modelo con target encoding
suavizado (`min_samples_leaf=20`, `smoothing=10`), para que los barrios con pocas observaciones no
queden codificados por el precio de una o dos casas. Segundo, conservar `df_completo` con
`Neighborhood` en su forma categórica original para poder auditar el sesgo. Tercero, tratar la
equidad territorial como condición de aprobación medida explícitamente (sección 3), en lugar de
buscar una ceguera geográfica que los datos muestran que es ilusoria.

### 10.4 Sesgo de medición: `Overall Qual` es un juicio, no una medición

`Overall Qual` es la variable más correlacionada con el precio (r = 0,799) y no es una medición
objetiva: es una calificación de 1 a 10 asignada por un tasador municipal. Si esos evaluadores
aplicaron criterios sistemáticamente distintos según el sector de la ciudad, ese sesgo entra
directamente al modelo y resulta invisible en cualquier auditoría que solo revise variables
físicas.

Para ponerlo a prueba se predijo `Overall Qual` a partir de 12 atributos físicos objetivos y se
analizó el residuo medio por barrio. Un residuo positivo significa que el barrio recibió una
calificación mejor que la que sus características físicas justifican.

| Barrio | Mediana | Residuo de calidad |
|---|---|---|
| `MeadowV` | 88.250 | -0,82 |
| `Mitchel` | 153.500 | -0,47 |
| `Sawyer` | 135.000 | -0,36 |
| … | | |
| `NridgHt` | 317.750 | +0,41 |
| `StoneBr` | 319.000 | +0,86 |

Los atributos físicos explican solo el 64,8% de la varianza de `Overall Qual`, de modo que el 35%
restante corresponde al criterio del evaluador. La correlación entre la mediana de precio del
barrio y el residuo de calidad, considerando los 21 barrios con n ≥ 30, es de +0,52.

El resultado admite dos explicaciones que los datos disponibles no permiten separar. Puede tratarse
de calidad real no observada, si las viviendas caras tienen mejores terminaciones que ninguna de
las 12 variables físicas captura. O puede tratarse de sesgo del evaluador, si el tasador anticipa
el valor del sector y ajusta la nota en esa dirección. El patrón tampoco es perfectamente monótono:
`IDOTRR` y `OldTown`, barrios baratos, tienen residuos positivos, lo que debilita la hipótesis de
un sesgo puramente territorial.

La conclusión que sostenemos es más débil: el sesgo no está demostrado, pero tampoco puede
descartarse, y la variable individualmente más predictiva del modelo descansa en un juicio humano
no auditable. La medida adoptada es documentar la limitación y, en la fase de modelado, entrenar
una variante sin las variables de calidad subjetiva para dimensionar cuánto del desempeño depende
de ellas.

### 10.5 Sesgo temporal

| Año | Transacciones | Mediana |
|---|---|---|
| 2006 | 625 | 159.500 USD |
| 2007 | 690 | 165.000 USD |
| 2008 | 621 | 161.000 USD |
| 2009 | 648 | 160.850 USD |
| 2010 | 341 | 155.000 USD |

Los datos cubren 2006–2010, período que incluye la crisis financiera *subprime*. La mediana anual
se mantuvo estable, con una variación de 6,5%, probablemente por tratarse de una ciudad
universitaria con demanda menos especulativa que las grandes áreas metropolitanas. El año 2010
aparece incompleto, con 341 registros frente a unos 630 de los años anteriores, por lo que su caída
no debe leerse como tendencia.

El riesgo relevante no es la volatilidad interna del período sino la antigüedad del conjunto. El
modelo aprende relaciones precio-característica de un mercado de hace más de quince años. Por eso
no debe usarse para estimar valores absolutos actuales sin recalibración. Su uso apropiado es la
valoración relativa, es decir qué propiedades valen más que otras y por qué, o la predicción sobre
datos actualizados del mismo mercado.

### 10.6 Sesgo de selección

| Condición de venta | N° | % | Mediana |
|---|---|---|---|
| `Normal` | 2.412 | 82,0 % | 158.750 USD |
| `Partial` (en construcción) | 242 | 8,0 % | 250.290 USD |
| `Abnorml` (remate, ejecución) | 189 | 6,0 % | 129.000 USD |
| `Family` (entre familiares) | 46 | 2,0 % | 144.400 USD |
| `Alloca` / `AdjLand` | 36 | 1,0 % | — |

El 18% de las transacciones no sigue la lógica de oferta y demanda. En la fase de preparación se
eliminaron 5 propiedades, tres de ellas `Partial` con superficies superiores a 4.000 pies² y
precios anormalmente bajos. La decisión es correcta, porque enseñaban al modelo una relación falsa
entre superficie y precio, pero tiene una consecuencia explícita: el modelo se especializa en
transacciones normales de mercado y no es aplicable a remates judiciales, ventas entre familiares
ni propiedades en construcción. Aplicarlo a un remate produciría una sobrevaloración sistemática.

### 10.7 Auditoría de equidad: dónde se concentra el error

Los sesgos anteriores son propiedades del dataset. Esta sección mide su consecuencia práctica.

Se entrenó un modelo diagnóstico, una regresión Ridge que no es el modelo final del proyecto, y se
descompuso su MAPE por cuartil de barrio. El objetivo era verificar si el criterio de equidad
territorial definido en la sección 3 (brecha ≤ 8 p.p.) es alcanzable o si ya se incumple desde el
inicio. Se compararon dos variantes: el mismo modelo sobre el precio en dólares y sobre el precio
en escala logarítmica.

| Variante | MAPE global | R² | Brecha Q1 − Q4 | Brecha entre barrios | Criterio ≤ 8 p.p. |
|---|---|---|---|---|---|
| Ridge sobre dólares | 9,61 % | 0,923 | 5,92 p.p. | 16,87 p.p. | No cumple |
| Ridge sobre `log1p` | 6,94 % | 0,952 | 1,21 p.p. | 8,43 p.p. | No cumple, al límite |

Desglose de la variante en dólares:

| Cuartil de barrio | N° en test | MAPE |
|---|---|---|
| Q1 (barrios baratos) | 167 | 14,02 % |
| Q2 | 127 | 8,61 % |
| Q3 | 150 | 6,98 % |
| Q4 (barrios caros) | 141 | 8,10 % |

El peor barrio es `IDOTRR`, con 22,0% de error, y el mejor `CollgCr`, con 5,1%.

El modelo entrenado sobre el precio en dólares alcanza un MAPE global razonable de 9,61% y aun así
falla el criterio de equidad territorial por más del doble. La causa es estructural y no un defecto
de este algoritmo en particular. Minimizar el error cuadrático en dólares hace que un error de
30.000 dólares pese lo mismo en una casa de 300.000 (10%) que en una de 88.000 (34%), de manera que
el optimizador tiene un incentivo matemático a acertarle a las casas caras.

Entrenar sobre `log1p(SalePrice)` cambia ese incentivo. Minimizar el error cuadrático en escala
logarítmica equivale aproximadamente a minimizar el error relativo, que es lo que mide el MAPE. La
brecha entre cuartiles baja de 5,92 a 1,21 puntos y la brecha entre barrios de 16,87 a 8,43.

Esto cambia la lectura de una decisión que hasta ese momento era puramente estadística. La
transformación logarítmica se había justificado por la asimetría de `SalePrice` (skew = 1,59) y el
supuesto de normalidad de los residuos. El análisis de sesgos muestra que además redistribuye el
error entre segmentos de precio, de modo que funciona a la vez como corrección estadística y como
intervención de equidad.

Cabe una advertencia. La variante logarítmica sigue por sobre el umbral, con 8,43 frente a 8 p.p.,
y estos son resultados de un modelo diagnóstico sin ajuste de hiperparámetros. La brecha debe
re-medirse sobre el modelo final. Si persiste, la respuesta correcta no es relajar el umbral sino
intervenir el modelo, por ejemplo mediante ponderación por segmento, modelos por rango de precio o
una política de derivación ampliada.

### 10.8 Síntesis

| Sesgo | Evidencia cuantitativa | Medida adoptada |
|---|---|---|
| Representación | 7 de 28 barrios con n < 30; `Landmrk` con 1 propiedad | Derivación automática a tasación manual en barrios sub-representados |
| Histórico / geográfico | Razón 3,6 : 1 entre barrios; 2,4 : 1 controlando por superficie | Suavizado en el target encoding; equidad territorial como criterio de salida |
| Proxy geográfico | V de Cramér 0,51 entre barrio y segmento físico; corr. -0,71 | Conservar `Neighborhood` para poder auditar, en vez de ocultarlo |
| De medición | Solo 64,8% de `Overall Qual` explicado por lo físico; corr. +0,52 con el precio del barrio | Documentar; entrenar variante sin variables subjetivas |
| Temporal | Datos 2006–2010, más de 15 años de antigüedad | Uso restringido a valoración relativa o recalibración |
| De selección | 18% de ventas no son de mercado libre; 5 registros eliminados | Alcance limitado a transacciones normales |
| Asimetría del error | Brecha de MAPE entre barrios: 16,87 p.p. en dólares vs 8,43 p.p. en escala log | Entrenar sobre `log1p(SalePrice)`; medir la brecha como condición de aprobación |

El sesgo de este proyecto proviene de los datos y no del algoritmo, y no se corrige eliminando
variables sino midiéndolo de forma explícita y convirtiéndolo en una condición de aprobación del
modelo. Por eso el criterio de salida incluye un umbral de equidad territorial junto a los de
precisión.

---

## 11. Consideraciones de ética y privacidad

### 11.1 Privacidad de los datos

El dataset no contiene identificadores personales directos: no hay nombres de propietarios,
RUT/SSN, ni direcciones exactas. Sí contenía `PID`, el identificador de parcela del catastro de
Ames, que es una clave de vinculación con registros públicos de propiedad. Con ese número es
posible obtener la dirección exacta y, a partir de ella, el nombre del propietario en registros
abiertos.

`PID` fue eliminado en la fase de preparación. La justificación técnica principal fue la ausencia
de valor predictivo, pero la eliminación cumple también una función de anonimización.

Queda un riesgo residual de reidentificación. Incluso sin `PID`, la combinación de barrio, año de
construcción, superficie y número de habitaciones puede ser única para una vivienda específica. En
los barrios con pocas propiedades ese riesgo es material: en `Landmrk` (1 propiedad) o `GrnHill`
(2), conocer el barrio equivale a identificar la casa. Anonimizar no consiste solo en borrar el
identificador, porque el resto de los atributos puede funcionar como cuasi-identificador.

El análisis de la sección 10.3 refuerza este punto desde el ángulo opuesto. Si los atributos
físicos permiten reconstruir el barrio con una V de Cramér de 0,51, entonces también operan como
cuasi-identificadores geográficos. El mismo mecanismo que impide una ceguera geográfica efectiva
impide una anonimización efectiva por simple supresión de columnas.

En cuanto al marco normativo, el dataset es de uso público y educativo, sin datos personales en el
sentido de la Ley 19.628 chilena o el RGPD europeo. En un despliegue real con datos actuales
aplicarían obligaciones de consentimiento informado, limitación de finalidad y derecho de acceso y
rectificación.

### 11.2 Consideraciones éticas del uso del modelo

**Transparencia y derecho a explicación.** Una persona cuya vivienda es tasada por un algoritmo
debería poder conocer qué factores determinaron ese valor. Esto pesa en la selección del algoritmo:
si la diferencia de desempeño entre un modelo lineal y uno de boosting es marginal, la
interpretabilidad del primero tiene valor ético además de comercial.

**El modelo como apoyo y no como decisión final.** El sistema debe informar tasaciones, no
reemplazar el criterio profesional, sobre todo en operaciones de crédito hipotecario donde el
resultado afecta el acceso al financiamiento de una familia. Automatizar por completo una decisión
con ese nivel de impacto no es defendible con un MAPE del 12%.

**Asimetría del error.** Una subvaloración perjudica al vendedor y una sobrevaloración perjudica al
comprador y al banco. Los dos errores no tienen el mismo costo social, y un modelo optimizado
únicamente por RMSE los trata como equivalentes. Esta asimetría debe considerarse al definir la
política de uso.

**Responsabilidad sobre el sesgo heredado.** Como se expuso en la sección 10, el modelo puede
reproducir patrones históricos de segregación residencial, y la auditoría de equidad muestra que un
modelo perfectamente estándar ya falla el umbral territorial, con 16,87 p.p. de brecha entre
barrios. Que el sesgo provenga de los datos y no del código no exime de responsabilidad: un modelo
desplegado sin auditoría de equidad traslada la desigualdad histórica al presente y la presenta
como un resultado técnico neutral. Tampoco exime la buena intención de eliminar la variable
sensible, porque la sección 10.3 muestra que quitar `Neighborhood` habría producido un modelo
igualmente sesgado y además no auditable.

**Uso indebido previsible.** Un modelo de valoración inmobiliaria puede emplearse para identificar
propiedades subvaloradas y adquirirlas de forma oportunista, o para fijar precios de forma
coordinada entre actores del mercado. El diseño del despliegue debe contemplar estos escenarios.

---

## 12. Reproducibilidad y estructura del proyecto

```
Housing_Prices_ML_P1/
├── data/
│   └── raw/
│       └── Ames_Iowa_Housing_Dataset.csv
├── notebooks/
│   └── EDA_Housing_v1_5.ipynb
├── images/
│   └── [gráficos exportados del EDA]
├── models/
│   └── [modelos entrenados — fase siguiente]
└── README.md
```

Condiciones de reproducibilidad:

- El notebook se ejecuta de principio a fin sin modificaciones, cargando los datos desde la URL
  pública del repositorio.
- `random_state=42` fijo en el `train_test_split`, en el K-Means y en todos los modelos.
- Cada decisión de preparación está documentada en celdas Markdown adyacentes al código que la
  implementa.
- Las celdas que eliminan columnas usan `errors='ignore'` para tolerar la re-ejecución parcial del
  notebook.

Dependencias: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `category_encoders`. El
análisis de sesgos de la sección 10 usa adicionalmente `KMeans`, `silhouette_score`,
`LinearRegression` y `Ridge`, todos incluidos en `scikit-learn`.

---

## 13. Próximos pasos

| Fase CRISP-DM | Actividad |
|---|---|
| Modelado | Entrenar los tres algoritmos preseleccionados con validación cruzada de 5 folds, sobre `y_train_log` según el hallazgo de equidad de la sección 10.7 |
| Modelado | Optimización de hiperparámetros del modelo con mejor desempeño |
| Modelado | Entrenar una variante sin las variables de calidad subjetiva (`Overall Qual`, `Exter Qual`, `Kitchen Qual`) para dimensionar la dependencia del juicio del evaluador (sección 10.4) |
| Evaluación | Verificar el cumplimiento del criterio de salida completo, incluido el umbral de equidad territorial de 8 p.p. |
| Evaluación | Re-ejecutar la auditoría de equidad de la sección 10.7 sobre el modelo final, no sobre el diagnóstico |
| Evaluación | Análisis de residuos: verificar si el error se concentra en algún rango de precio, más allá del corte por barrio |
| Evaluación | Si la brecha de equidad persiste, evaluar ponderación por segmento o modelos separados por rango de precio |
| Despliegue | Definir la política de derivación a tasación manual (umbral de n por barrio) y el mecanismo de explicación al usuario final |

---

## Referencias

- De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester
  Regression Project*. Journal of Statistics Education, 19(3).
- Documentación oficial del dataset: `DataDocumentation.txt` (descripción de las 82 variables y sus
  categorías).
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
