# Formulación Matemática Detallada de Kistmat_AI

## 1. Generación de Datos y Preprocesamiento

Sea $\mathcal{P}$ el conjunto de problemas matemáticos generados. Cada problema $p \in \mathcal{P}$ se define como:

$$p = (t, s, d, c)$$

Donde:
- $t \in \mathcal{T}$: texto del problema (espacio de todos los textos posibles)
- $s \in \mathbb{C}$: solución (número complejo para abarcar soluciones reales e imaginarias)
- $d \in [1, 3]$: nivel de dificultad (número real entre 1 y 3)
- $c \in \mathcal{C}$: concepto matemático (conjunto finito de conceptos)

La función de generación de datos $G$ está parametrizada por la etapa de aprendizaje $e \in \mathcal{E}$ y el nivel de dificultad $d$:

$$G : \mathcal{E} \times [1, 3] \rightarrow \mathcal{P}^n$$

Donde $n$ es el número de problemas generados, típicamente entre 4000 y 5000.

Para cada etapa $e$, existe una función específica de generación $G_e$:

$$G_e(d) = \{p_i = (t_i, s_i, d, c_i) \mid i = 1, ..., n\}$$

Por ejemplo, para la etapa "elementary1":

$$G_\text{elementary1}(d) = \{(f(a_i, b_i), \text{eval}(f(a_i, b_i)), d, \text{op}_i) \mid i = 1, ..., n\}$$

Donde $f(a, b) = "a \text{ op } b"$, $a_i, b_i \sim \mathcal{U}(1, \lfloor 10d \rfloor)$, $\text{op}_i \in \{+, -\}$, y $\text{eval}$ es la función de evaluación de la expresión.

## 2. Tokenización

La función de tokenización $T_e$ para la etapa $e$ se define como:

$$T_e: \mathcal{T} \rightarrow \mathbb{N}^m$$

Donde $m$ es la longitud máxima de la secuencia (MAX_LENGTH en el código).

Para problemas básicos:

$$T_\text{basic}(t) = [\text{hash}(w_i) \mod V \mid w_i \in \text{split}(t)]$$

Para problemas avanzados:

$$T_\text{advanced}(t) = [f_\text{token}(w_i) \mid w_i \in \text{split}(t)]$$

Donde:

$$f_\text{token}(w) = \begin{cases}
    \text{hash}(w) \mod V & \text{si } w \text{ es alfabético} \\
    \text{ord}(w) & \text{si } w \text{ es dígito o símbolo} \\
    \lfloor \text{float}(w) \cdot 100 \rfloor & \text{en otro caso}
\end{cases}$$

Para problemas de cálculo:

$$T_\text{calculus}(t) = [\alpha_1, ..., \alpha_k, \beta_1, ..., \beta_k]$$

Donde $\alpha_i$ son coeficientes normalizados y $\beta_i$ son exponentes normalizados de los términos en la expresión de cálculo.

## 3. Modelo Principal (Kistmat_AI)

### 3.1 Capa de Embedding

$$\mathbf{E} = \text{Emb}(\mathbf{x}) \in \mathbb{R}^{m \times d_e}$$

Donde $\mathbf{x} \in \mathbb{N}^m$ es el vector de tokens de entrada, $d_e$ es la dimensión del embedding.

### 3.2 LSTMs Bidireccionales

Para cada capa LSTM bidireccional $l = 1, 2$:

$$\vec{\mathbf{h}}_{l,t} = \text{LSTM}_l(\vec{\mathbf{h}}_{l,t-1}, \mathbf{E}_t)$$
$$\cev{\mathbf{h}}_{l,t} = \text{LSTM}_l(\cev{\mathbf{h}}_{l,t+1}, \mathbf{E}_t)$$
$$\mathbf{H}_l = [\vec{\mathbf{h}}_{l,1}, ..., \vec{\mathbf{h}}_{l,m}] \oplus [\cev{\mathbf{h}}_{l,1}, ..., \cev{\mathbf{h}}_{l,m}]$$

Donde $\oplus$ denota la concatenación, $\mathbf{H}_0 = \mathbf{E}$, y $\mathbf{H}_l \in \mathbb{R}^{m \times 2d_h}$, siendo $d_h$ la dimensión oculta del LSTM.

### 3.3 Capa de Dropout

$$\mathbf{H}_\text{drop} = \text{Dropout}(\mathbf{H}_2, p = 0.5)$$

### 3.4 Atención Multi-Cabeza

Para cada cabeza de atención $i = 1, ..., h$:

$$\mathbf{Q}_i = \mathbf{H}_\text{drop}\mathbf{W}_i^Q, \quad \mathbf{K}_i = \mathbf{H}_\text{drop}\mathbf{W}_i^K, \quad \mathbf{V}_i = \mathbf{H}_\text{drop}\mathbf{W}_i^V$$

$$\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i\mathbf{K}_i^T}{\sqrt{d_k}}\right)\mathbf{V}_i$$

$$\text{MultiHead}(\mathbf{H}_\text{drop}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O$$

Donde $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{2d_h \times d_k}$ y $\mathbf{W}^O \in \mathbb{R}^{hd_k \times 2d_h}$.

### 3.5 Consulta de Memoria

$$\mathbf{q} = \text{MemoryQuery}(\text{MultiHead}(\mathbf{H}_\text{drop}))$$

Donde $\mathbf{q} \in \mathbb{R}^{d_q}$ y $d_q$ es la dimensión de la consulta de memoria.

### 3.6 Razonamiento y Salida

$$\mathbf{r} = \text{ReLU}(\mathbf{W}_r\text{MultiHead}(\mathbf{H}_\text{drop}) + \mathbf{b}_r)$$
$$\mathbf{y} = \mathbf{W}_o\mathbf{r} + \mathbf{b}_o$$

Donde $\mathbf{W}_r \in \mathbb{R}^{d_r \times 2d_h}$, $\mathbf{b}_r \in \mathbb{R}^{d_r}$, $\mathbf{W}_o \in \mathbb{R}^{2 \times d_r}$, $\mathbf{b}_o \in \mathbb{R}^2$, y $\mathbf{y} \in \mathbb{R}^2$ representa la parte real e imaginaria de la solución predicha.

## 4. Sistema de Memoria Integrado

### 4.1 Memoria Externa

Sea $\mathbf{M}_\text{ext} \in \mathbb{R}^{n_\text{ext} \times d_v}$ la matriz de embeddings de memoria externa, donde $n_\text{ext}$ es el número de entradas y $d_v$ es la dimensión del valor.

$$\text{sim}_\text{ext}(\mathbf{q}) = \mathbf{q}\mathbf{M}_\text{ext}^T$$
$$\mathbf{m}_\text{ext} = \text{TopK}(\text{sim}_\text{ext}(\mathbf{q}), k) \cdot \mathbf{M}_\text{ext}$$

### 4.2 Memoria Formulativa

Sea $\mathcal{F} = \{(f_i, \mathbf{e}_i, \mathcal{T}_i) \mid i = 1, ..., n_f\}$ el conjunto de fórmulas almacenadas, donde $f_i$ es la fórmula, $\mathbf{e}_i \in \mathbb{R}^{d_f}$ es su embedding, y $\mathcal{T}_i$ es el conjunto de términos asociados.

$$\text{sim}_\text{form}(\mathbf{q}, \mathcal{T}) = \mathbf{q}\mathbf{E}_\mathcal{T}^T$$
$$\mathbf{m}_\text{form} = \text{TopK}(\text{sim}_\text{form}(\mathbf{q}, \mathcal{T}), k) \cdot \mathbf{E}_\mathcal{T}$$

Donde $\mathbf{E}_\mathcal{T}$ es la matriz de embeddings de las fórmulas cuyos términos intersectan con los de la consulta.

### 4.3 Memoria Conceptual

Sea $\mathcal{C} = \{(c_i, \mathbf{e}_i) \mid i = 1, ..., n_c\}$ el conjunto de conceptos almacenados, donde $c_i$ es el concepto y $\mathbf{e}_i \in \mathbb{R}^{d_c}$ es su embedding.

$$\text{sim}_\text{conc}(\mathbf{q}) = \mathbf{q}\mathbf{E}_c^T$$
$$\mathbf{m}_\text{conc} = \text{TopK}(\text{sim}_\text{conc}(\mathbf{q}), k) \cdot \mathbf{E}_c$$

### 4.4 Memoria a Corto Plazo

Sea $\mathcal{S} = [\mathbf{s}_1, ..., \mathbf{s}_{n_s}]$ la lista de estados recientes, donde $\mathbf{s}_i \in \mathbb{R}^{d_s}$.

$$\text{sim}_\text{short}(\mathbf{q}) = \mathbf{q}\mathbf{S}^T$$
$$\mathbf{m}_\text{short} = \text{TopK}(\text{sim}_\text{short}(\mathbf{q}), k) \cdot \mathbf{S}$$

### 4.5 Memoria a Largo Plazo

Sea $\mathcal{L} = \{(\mathbf{l}_i, \alpha_i) \mid i = 1, ..., n_l\}$ el conjunto de memorias a largo plazo, donde $\mathbf{l}_i \in \mathbb{R}^{d_l}$ es la memoria y $\alpha_i$ es su importancia.

$$\text{sim}_\text{long}(\mathbf{q}) = \mathbf{q}\mathbf{L}^T \odot \boldsymbol{\alpha}$$
$$\mathbf{m}_\text{long} = \text{TopK}(\text{sim}_\text{long}(\mathbf{q}), k) \cdot \mathbf{L}$$

### 4.6 Memoria de Inferencia

Sea $\mathcal{I} = \{(\mathbf{i}_j, \beta_j) \mid j = 1, ..., n_i\}$ el conjunto de inferencias almacenadas, donde $\mathbf{i}_j \in \mathbb{R}^{d_i}$ es la inferencia y $\beta_j$ es su confianza.

$$\text{sim}_\text{inf}(\mathbf{q}) = \mathbf{q}\mathbf{I}^T \odot \boldsymbol{\beta}$$
$$\mathbf{m}_\text{inf} = \text{TopK}(\text{sim}_\text{inf}(\mathbf{q}), k) \cdot \mathbf{I}$$

### 4.7 Integración de Memorias

$$\mathbf{M} = [\mathbf{m}_\text{ext}; \mathbf{m}_\text{form}; \mathbf{m}_\text{conc}; \mathbf{m}_\text{short}; \mathbf{m}_\text{long}; \mathbf{m}_\text{inf}]$$
$$\mathbf{m}_\text{integrated} = \text{Attention}(\mathbf{q}, \mathbf{M}, \mathbf{M})$$

## 5. Razonamiento Simbólico

Sea $\mathcal{S}$ el espacio de expresiones simbólicas y $f_s: \mathcal{S} \rightarrow \mathbb{C}$ la función de evaluación simbólica.

Para una ecuación lineal $ax + b = c$:

$$f_s(ax + b = c) = \frac{c - b}{a}$$

Para una ecuación cuadrática $ax^2 + bx + c = 0$:

$$f_s(ax^2 + bx + c = 0) = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

Para límites:

$$f_s(\lim_{x \to a} g(x)) = \lim_{h \to 0} g(a + h)$$

Para derivadas:

$$f_s(\frac{d}{dx}(\sum_{i=0}^n a_ix^i)) = \sum_{i=1}^n ia_ix^{i-1}$$

## 6. Aprendizaje por Refuerzo (continuación)

Sea $s_t$ el estado en el tiempo $t$, $a_t$ la acción tomada, $r_t$ la recompensa recibida, y $\pi_\theta(a|s)$ la política parametrizada por $\theta$. El gradiente del objetivo $J(\theta)$ se define como:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \left(\sum_{t'=t}^T \gamma^{t'-t} r_{t'}\right)\right]$$

Donde $\gamma \in [0, 1]$ es el factor de descuento.

Para el caso específico de Kistmat_AI, definimos:

- Estado $s_t$: El problema matemático actual y el contexto del aprendizaje.
- Acción $a_t$: La predicción del modelo para la solución del problema.
- Recompensa $r_t$: Una función del error entre la predicción y la solución real:

$$r_t = f(y_t, \hat{y}_t) = \begin{cases}
    1 & \text{si } |y_t - \hat{y}_t| < \epsilon \\
    -\frac{|y_t - \hat{y}_t|}{\max(|y_t|, 1)} & \text{en otro caso}
\end{cases}$$

Donde $y_t$ es la solución real, $\hat{y}_t$ es la predicción del modelo, y $\epsilon$ es un umbral de tolerancia.

La actualización de los parámetros se realiza mediante:

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)$$

Donde $\alpha$ es la tasa de aprendizaje.

## 7. Entrenamiento Curricular

Sea $\mathcal{E} = \{e_1, ..., e_9\}$ el conjunto de etapas de aprendizaje, donde:

$$e_1 = \text{elementary1}, ..., e_9 = \text{university}$$

Para cada etapa $e_i$, definimos un umbral de preparación $\tau_i$:

$$\tau = [\tau_1, ..., \tau_9] = [0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.80]$$

La función de pérdida $L_{e_i}$ para la etapa $e_i$ se define como:

$$L_{e_i}(\theta) = \frac{1}{|\mathcal{P}_{e_i}|} \sum_{p \in \mathcal{P}_{e_i}} \|y_p - \hat{y}_p\|^2$$

Donde $\mathcal{P}_{e_i}$ es el conjunto de problemas para la etapa $e_i$, $y_p$ es la solución verdadera y $\hat{y}_p$ es la predicción del modelo.

El criterio de avance a la siguiente etapa se define como:

$$\text{Avanzar si: } R^2(\mathcal{P}_{e_i}^\text{val}) > \tau_i$$

Donde $R^2(\mathcal{P}_{e_i}^\text{val})$ es el coeficiente de determinación en el conjunto de validación de la etapa $e_i$.

## 8. Evaluación

### 8.1 Error Cuadrático Medio (MSE)

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n \|y_i - \hat{y}_i\|^2$$

Donde $n$ es el número de muestras, $y_i$ es la solución verdadera y $\hat{y}_i$ es la predicción del modelo.

### 8.2 Coeficiente de Determinación (R²)

$$R^2 = 1 - \frac{\sum_{i=1}^n \|y_i - \hat{y}_i\|^2}{\sum_{i=1}^n \|y_i - \bar{y}\|^2}$$

Donde $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$ es la media de los valores verdaderos.

### 8.3 Error Absoluto Medio (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

## 9. Optimización

El objetivo general de optimización se puede expresar como:

$$\min_\theta \sum_{e \in \mathcal{E}} \lambda_e L_e(\theta) + \lambda_R R(\theta)$$

Donde $\lambda_e$ son los pesos para cada etapa de aprendizaje, $L_e(\theta)$ es la función de pérdida para la etapa $e$, $\lambda_R$ es el factor de regularización, y $R(\theta)$ es el término de regularización (por ejemplo, L2: $R(\theta) = \|\theta\|^2$).

La actualización de los parámetros se realiza mediante el algoritmo Adam:

$$\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta L(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta L(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{aligned}$$

Donde $\beta_1, \beta_2$ son los factores de decaimiento para las estimaciones del primer y segundo momento, $\eta$ es la tasa de aprendizaje, y $\epsilon$ es un pequeño valor para evitar la división por cero.
