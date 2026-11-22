# Root-Cause Detection in SpiderSense Lite

In **SpiderSense Lite**, root-cause detection is modeled directly after how a spider locates prey caught in its web, translated into an **independent mathematical residual-minimization algorithm**. The engine **does not look at the injected fault label**. Instead, it infers the origin solely from the pattern of observed metrics across all services.

Here is the step-by-step breakdown of how the fault is detected:

---

## 1. The Biological Analogy

* **In Nature:** When prey struggles in a spiderweb, mechanical vibrations travel outward along intersecting silk threads. As vibrations travel, silk elasticity and air resistance cause them to **attenuate** (lose amplitude) and **delay** in time. The spider rests its legs on different sensory threads, reading the intensity differences and arrival timestamps to triangulate exactly which junction or sector the vibration came from.
* **In SpiderSense:** When a service degrades or fails, it emits an **anomaly signal**. This signal cascades downstream across dependent services, losing strength at each hop according to a transmission coefficient $T_{ij}$ and accumulating delay $d_{ij}$.

---

## 2. Step 1: Gathering Real-Time Observations

When an outage occurs, the system records two key observations ($O$) for every service $i$:

1. $O_i$ (**Observed Anomaly Intensity**): How severe the disruption is at service $i$ (a composite score based on latency spikes, error rates, and signal reception, scaled from `0` to `100`).
2. $TA_i$ (**Observed Arrival Time**): The simulated timestamp at which the disturbance reached service $i$ (or `None` if the service was untouched).

---

## 3. Step 2: Candidate Hypotheses & Forward Simulation

The algorithm iterates through every single service in the mesh as a potential root-cause candidate $c$:

```math
\(\text{Candidates } c \in \lbrace \text{Frontend, API Gateway, Orders, Payment, Inventory, Database} \rbrace\)
```

For each candidate $c$, the engine runs a clean mathematical forward-propagation simulation (`predict_propagation_from`):

* It assumes an initial disturbance originated at candidate $c$ at time $t = 0$.
* It computes the theoretical signal $P_i(c)$ and theoretical arrival time $TP_i(c)$ that *should* appear at every other service $i$ following the physical attenuation rules:

```math
\(P_{\text{next}} = P_{\text{current}} \times\) T_{ij}
```

```math
\(TP_{\text{next}} = TP_{\text{current}} +\) d_{ij}
```

> **Note:** Services that cannot be reached downstream from candidate $c$ receive $P_i(c) = 0$ and $TP_i(c) = \infty$.

---

## 4. Step 3: Computing the Weighted Residual Score

The engine compares the **observed reality** against the **theoretical prediction** for candidate $c$ using a weighted residual function $R(c)$:

```math
R(c) \(= \sum_{i} w_i \times \Big[ \underbrace{(O_i - P_i(c))^2}_{\text{Intensity Mismatch}} + \lambda \times \underbrace{(TA_i - TP_i(c))^2}_{\text{Timing Mismatch}} \Big]\)
```

Where:

* $(O_i - P_i(c))^2$: Measures how far the predicted anomaly is from the actual anomaly.
* $\lambda \times (TA_i - TP_i(c))^2$: Penalizes candidates that predict impossible arrival timings (or services that were impacted before the candidate could have transmitted to them).
* $w_i$: Importance weight of each service (services deeper in the infrastructure like `Database` are weighted slightly higher due to their widespread blast radius).
* $\lambda$ (`0.05`): Weighting factor balancing timing versus intensity.

---

## 5. Step 4: Ranking & Root-Cause Selection

1. **Lowest Residual Wins:** The candidate with the **smallest residual error** $R(c)$ represents the scenario that most closely explains the observed anomaly distribution across the entire network:

```math
\(\hat{c}_{\text{root}} = \arg\min_\){c} R(c)
```

2. **Confidence Metric:** The algorithm calculates the score gap between Rank 1 (best) and Rank 2 (runner-up):

```math
\(\text{Gap Ratio} = \frac\){R\((\text{second}) -\) R\((\text{best})\)}{R\((\text{second})\)}
```

* If the gap is wide (e.g., $R(\text{Database}) = 2,859$ vs. $R(\text{Payment}) = 16,193$), the confidence is close to **100%**.
* If two candidates have very close residuals ($\text{Gap} < 15\%$), the system flags an **Uncertainty Warning** indicating that either node could reasonably be the root cause.

---

## Why This Beats Conventional Monitoring

* **Threshold Monitoring:** Simply checks `if latency > 500ms`. During a Database outage, it sounds alarms for *Database, Payment, Inventory, Orders, API Gateway, and Frontend* simultaneously, leaving engineers with "alert fatigue" and no clear origin.
* **SpiderSense:** Understands the topology and physics of cascade attenuation. It recognizes that because Payment and Inventory degraded *after* Database with *attenuated* signals, they are merely **symptomatic victims**, pinpointing **Database** as the genuine origin.

