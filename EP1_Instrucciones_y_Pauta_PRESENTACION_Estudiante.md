# Evaluación Parcial N°1

**Presentación y defensa técnica del proyecto | Docente**

**Instrucciones y pauta de evaluación | Estudiante**

*Subdirección de Diseño Curricular e Instruccional — 2026*

---

## Datos de la asignatura

| Sigla | Nombre Asignatura | Tiempo Asignado | % Ponderación | Semana inicio EFT |
|---|---|---|---|---|
| MLY1101 | MACHINE LEARNING | 5 h | 30% | |

---

## 1. Instrucciones generales

### Descripción

- La evaluación consiste en una actividad **grupal con defensa individual**, en la que cada grupo debe presentar un informe ejecutivo que sintetice los aspectos más relevantes del proyecto, donde el estudiante debe desarrollar y defender una solución de Machine Learning aplicada a un problema real de negocio, utilizando uno de los casos propuestos por la asignatura:
  - **Caso A:** Predicción de abandono de clientes (Telco Customer Churn).
  - **Caso B:** Predicción de precios de viviendas (Housing Prices).
  - **Caso C:** Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks).

- Durante el desarrollo del proyecto, los estudiantes deberán aplicar las distintas etapas del proceso de ciencia de datos, considerando la comprensión del problema, preparación de datos, análisis exploratorio y evaluación del impacto ético.

- El tiempo asignado para desarrollar esta evaluación en la sala de proyectos es de **5 horas** y se realiza de manera **grupal**.

- Cada estudiante/equipo cuenta con **10 minutos** para su presentación.

- Si bien esta evaluación es posible desarrollarla y rendirla de manera grupal, la presentación será evaluada de manera **individual** y estará asociada al desempeño de cada estudiante, ya sea mediante la calidad de su exposición, así como también en las respuestas a las preguntas realizadas por su docente. Por lo anterior, cada estudiante debe estar preparado para responder **preguntas cruzadas**, es decir, no necesariamente se le realizarán preguntas específicamente de lo que haya presentado individualmente.

### Instrucciones específicas de la evaluación

La presentación debe incluir los siguientes apartados:

- Descripción del problema de negocio, definición de KPIs y objetivos del proyecto.
- Análisis exploratorio de los datos (EDA), incluyendo identificación de patrones, calidad de datos y variables relevantes.
- Preparación y transformación de datos para el proceso de modelamiento.

**Modalidad de la evaluación:** Presentación de evidencias y defensa técnica del proyecto de Machine Learning.

### Aspectos formales

Los materiales, herramientas o insumos requeridos para realizar esta evaluación corresponden a un **informe técnico en formato Markdown**, el cual deberá documentar de manera estructurada todo el desarrollo del proyecto de Machine Learning.

Los estudiantes deberán entregar:

- **Informe técnico en formato Markdown (.md)**, elaborado como un documento reproducible similar a un README de un proyecto de Machine Learning, que incluya como mínimo las siguientes secciones:
  - Descripción del problema de negocio.
  - Objetivos del proyecto.
  - Definición de KPIs que resolverán el problema de negocio.
  - Descripción de las fuentes de datos utilizadas.
  - Preparación y análisis exploratorio de los datos (EDA).
  - Metodología utilizada (CRISP-DM).
- **Notebook en Python (.ipynb)** debidamente documentado, organizado y completamente ejecutable, que permita reproducir cada una de las etapas desarrolladas en el informe.
- **Conjunto de datos y archivos complementarios** utilizados durante el desarrollo del proyecto, organizados de manera que permitan ejecutar nuevamente la solución sin modificaciones adicionales.
- **Carpeta del proyecto organizada** siguiendo una estructura profesional (por ejemplo: `data/`, `notebooks/`, `models/`, `images/` y `README.md`), de forma que cualquier usuario pueda comprender, ejecutar y reproducir la solución desarrollada.

---

## 2. Pauta de Evaluación (Rúbrica, Escala de Valoración, Lista de cotejo)

### Escala general de niveles de logro

| Categoría | % logro | Descripción niveles de logro |
|---|---|---|
| Muy buen desempeño | 100% | Demuestra un desempeño destacado, evidenciando el logro de todos los aspectos evaluados en el indicador. |
| Buen desempeño | 80% | Demuestra un alto desempeño del indicador, presentando pequeñas omisiones, dificultades y/o errores. |
| Desempeño aceptable | 60% | Demuestra un desempeño competente, evidenciando el logro de los elementos básicos del indicador, pero con omisiones, dificultades o errores. |
| Desempeño incipiente | 30% | Presenta importantes omisiones, dificultades o errores en el desempeño, que no permiten evidenciar los elementos básicos del logro del indicador, por lo que no puede ser considerado competente. |
| Desempeño no logrado | 0% | Presenta ausencia o incorrecto desempeño. |

### Rúbrica por indicador de evaluación

#### 1. IE1: Identificación de fuentes de datos y herramientas colaborativas — **Ponderación 10%**

| Nivel | Descripción |
|---|---|
| Muy buen desempeño (100%) | Identifica de manera completa las fuentes de datos requeridas y justifica el uso de herramientas colaborativas pertinentes para resolver el problema de negocio. |
| Buen desempeño (80%) | Identifica las principales fuentes de datos y herramientas colaborativas, con leves omisiones en su justificación. |
| Desempeño aceptable (60%) | Identifica parcialmente las fuentes de datos o las herramientas colaborativas, con una justificación limitada. |
| Desempeño incipiente (30%) | Identifica de forma incompleta las fuentes o herramientas, sin justificar adecuadamente su utilización. |
| Desempeño no logrado (0%) | No identifica fuentes de datos ni herramientas colaborativas pertinentes para el proyecto. |

#### 2. IE2: Manipulación y preparación de datos en Python — **Ponderación 30%**

| Nivel | Descripción |
|---|---|
| Muy buen desempeño (100%) | Manipula y prepara los datos utilizando estructuras de datos en Python de forma correcta, organizada y eficiente para el desarrollo del proyecto. |
| Buen desempeño (80%) | Manipula adecuadamente los datos, presentando pequeñas omisiones en la organización o eficiencia del proceso. |
| Desempeño aceptable (60%) | Manipula los datos de forma básica, con errores menores o preparación incompleta. |
| Desempeño incipiente (30%) | Presenta dificultades importantes en la manipulación y preparación de los datos. |
| Desempeño no logrado (0%) | No logra preparar los datos para el desarrollo del proyecto. |

#### 3. IE3: Análisis exploratorio y calidad de los datos — **Ponderación 40%**

| Nivel | Descripción |
|---|---|
| Muy buen desempeño (100%) | Realiza un análisis exploratorio completo, identificando anomalías, evaluando la calidad de los datos y justificando las acciones de preparación realizadas. |
| Buen desempeño (80%) | Realiza un análisis exploratorio adecuado, con leves omisiones en la identificación de anomalías o evaluación de calidad. |
| Desempeño aceptable (60%) | Realiza un análisis exploratorio básico, identificando solo algunos aspectos relevantes. |
| Desempeño incipiente (30%) | El análisis exploratorio es incompleto y presenta dificultades para evaluar la calidad de los datos. |
| Desempeño no logrado (0%) | No realiza un análisis exploratorio pertinente. |

#### 4. IE4: Evaluación de sesgos, ética y privacidad de los datos — **Ponderación 20%**

| Nivel | Descripción |
|---|---|
| Muy buen desempeño (100%) | Evalúa de manera completa los posibles sesgos, aspectos éticos y estándares de privacidad asociados al uso de los datos del proyecto. |
| Buen desempeño (80%) | Evalúa adecuadamente los principales aspectos éticos y de privacidad, con leves omisiones. |
| Desempeño aceptable (60%) | Reconoce parcialmente los sesgos o aspectos éticos del proyecto. |
| Desempeño incipiente (30%) | Presenta un análisis superficial o incompleto sobre ética y privacidad. |
| Desempeño no logrado (0%) | No identifica sesgos ni aspectos éticos relevantes. |
