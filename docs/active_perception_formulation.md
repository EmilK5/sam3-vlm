# Active-Perception Formulation for Orchard Counting

Adapted from He et al., *Active Perception using Neural Radiance Fields* (arXiv:2310.09892v2), §II.
The problem structure, shorthand, predictive-information objective, and argmax are kept identical to
that paper. The single substantive change is that the SE(3) camera *viewpoint* is replaced by a
multi-dimensional **SAM3 query** (the sensor's controllable parameters), and the generative model /
planner is realized by Qwen-3-VL as an amortized policy.

---

## 1. Setup

Let $\xi$ be a fixed 2D scene — an image of a tree / trellis region of interest. We obtain a
sequence of observations by pointing a controllable sensor (**SAM3**) at the scene. As in He et al.,
we collect observations $y_1^{t} = (y_1, \dots, y_{t-1})$ from sensing configurations
$x_1^{t} = (x_1, \dots, x_{t-1})$, and write $y_t^{t+\Delta t}$ for the sequence of future
observations, $\Delta t \in \mathbb{N}^+$. We reuse the shorthand

$$
y_{\text{past}} \equiv y_1^{t}, \qquad
y_{\text{future}} \equiv y_t^{t+\Delta t}, \qquad
y_{\text{all}} \equiv y_1^{t+\Delta t},
\qquad
x_{\text{past}} \equiv x_1^{t}.
$$

At each step, $p(y_t \mid \xi, x_t)$ is the probability of obtaining observation $y_t$ from
sensing configuration $x_t$.

## 2. Sensing action: the SAM3 query (replaces the SE(3) viewpoint)

In He et al. a viewpoint is a camera pose $x_k \equiv (R_k, t_k) \in SE(3)$. Here the sensor is not a
camera moving through space but SAM3 attending to a region of a *fixed* image, so a "viewpoint"
generalizes to the full SAM3 **query** — the set of knobs that determine what the sensor returns:

$$
x_k \;\equiv\; \big(\, b_k,\; \pi_k,\; E_k,\; \tau_k,\; T_k \,\big) \;\in\; \mathcal{X},
$$

| symbol | meaning |
|---|---|
| $b_k = (x_{\min}, y_{\min}, x_{\max}, y_{\max}) \in \mathbb{R}^4$ | ROI bounding box — the image region the sensor attends to (global-frame xyxy pixels) |
| $\pi_k$ | text prompt (e.g. "green fruit") |
| $E_k$ | exemplar set (visual exemplars conditioning the query) |
| $\tau_k$ | detection / mask threshold(s) |
| $T_k$ | tiling configuration |

$\mathcal{X}$ is the query space; it replaces $SE(3)$. All downstream equations are unchanged — $x_k$
still denotes "where and how we sense" — but the informative axis is now scale / prompt / threshold
rather than 3D pose, which is the correct notion of a viewpoint for a static, fully-visible image.

**Observation.** $y_k$ is the response SAM3 returns for query $x_k$: a set of detected instances
(masks/boxes with confidences), which the FM+V-IP verifier subsequently labels
$\{\text{fruit}, \text{leaf}, \text{spurious}\}$. We treat $y_k \sim p(y \mid \xi, x_k)$ as a random
variable; the sensor is stochastic and resolution-limited, which is what makes different queries
carry different information.

## 3. Control / action model

Following He et al. we write the next configuration as a function of the history and a control $u_t$,

$$
x_{t+1} = f\big(x_{\text{past}},\, y_{\text{past}},\, u_t\big).
$$

Unlike a quadrotor, SAM3 has no kinematics: any region/prompt is reachable at any time, so $f$ is
unconstrained and the control simply *is* the choice of the next query,

$$
x_{t+1} = u_t \in \mathcal{X}.
$$

## 4. Predictive information

If we did not observe the past, the future is drawn from $p(y_{\text{future}})$; the past tells us it
is instead drawn from $p(y_{\text{future}} \mid y_{\text{past}})$. The predictive information is the
mutual information between the two — identical to Eq. (1) of He et al.:

$$
\begin{aligned}
I(y_{\text{future}}, y_{\text{past}})
&= \int \mathrm{d}p(y_{\text{past}})\;
   \mathrm{KL}\!\big( p(y_{\text{future}} \mid y_{\text{past}}) \,\big\|\, p(y_{\text{future}}) \big) \\
&= \int \mathrm{d}p(y_{\text{past}})\;
   \big( S(y_{\text{future}}) - S(y_{\text{future}} \mid y_{\text{past}}) \big) \\
&= S(y_{\text{past}}) + S(y_{\text{future}}) - S(y_{\text{all}}),
\end{aligned}
$$

where $S(y) = -\int \mathrm{d}p(y)\,\log p(y)$ is the Shannon entropy.

## 5. Optimization objective

Active perception selects the future controls $u_{\text{future}}$ (equivalently, the future queries)
that maximize predictive information over the horizon $\Delta t$ — identical to Eq. (2):

$$
\hat{u}_{\text{future}} \;\in\; \operatorname*{arg\,max}_{u_{\text{future}}}\;
I(y_{\text{future}}, y_{\text{past}}).
$$

For the greedy horizon $\Delta t = 1$, $S(y_{\text{past}})$ is constant given the observed past, so
this reduces to a **next-best-query** rule over the full SAM3 query:

$$
x_t^\star \;=\; \operatorname*{arg\,max}_{x \in \mathcal{X}}\;
\Big[\, S\big(y(x)\big) - S\big(y(x) \mid y_{\text{past}}\big) \,\Big],
\qquad x = (b, \pi, E, \tau, T).
$$

This is the direct analog of next-best-*view*, but the optimization is over where **and how** SAM3
looks, not merely over a pose.

## 6. The generative model, and Qwen as an amortized policy

Evaluating the objective requires $p(y_{\text{future}} \mid y_{\text{past}})$ — a prediction of what
SAM3 *would* return at a not-yet-queried region. In He et al. the NeRF is exactly this generative
model (it synthesizes novel views). We do **not** build an explicit generative model; instead
**Qwen-3-VL** plays this role as an *amortized* approximation to the argmax. Qwen observes the scene
$\xi$, the past queries $x_{\text{past}}$, and the past observations $y_{\text{past}}$, and emits the
next control:

$$
u_t \;\sim\; \pi_\theta\big(u_t \mid \xi,\, x_{\text{past}},\, y_{\text{past}}\big)
\;\approx\; \operatorname*{arg\,max}_{u}\; I(y_{\text{future}}, y_{\text{past}}).
$$

Concretely, Qwen predicts the ROI $b_t$ expected to reveal the largest number of previously-unseen
fruit. This grounds the entropy gap in a task-relevant surrogate: the greedy predictive-information
term is approximated by the **expected count of newly-revealed fruit** in a candidate region,

$$
S\big(y(x)\big) - S\big(y(x) \mid y_{\text{past}}\big)
\;\approx\;
\mathbb{E}\big[\, N_{\text{new}}(b \mid y_{\text{past}}) \,\big],
$$

so "look where the most new fruit are" is a concrete instantiation of "look where predictive
information is highest."

**Sensing/estimation separation (hard constraint).** $\pi_\theta$ selects only the query $u_t$ — where
and how SAM3 looks. Observations $y$, and therefore all candidates, originate **solely from SAM3**; no
Qwen-proposed box ever becomes a candidate. A Qwen ROI parameterizes $b_t$ and nothing else.

## 7. Remarks

- **Readout.** The reported estimand is the fruit count $\hat{N}$, obtained from the candidate-graph
  posterior after verification; the objective above exists to sharpen it. A fully task-directed
  variant would replace $y_{\text{future}}$ in $I(\cdot,\cdot)$ with the count $N$, i.e. maximize
  $I(N; y(x) \mid y_{\text{past}})$; the observation-space predictive information used here is the
  tractable surrogate, and the $\mathbb{E}[N_{\text{new}}]$ approximation in §6 is the bridge between
  the two.
- **Stopping.** Halt when the expected information gain of the best remaining query falls below a
  threshold or the sensing budget is exhausted — the counting analog of He et al.'s
  semantic-uncertainty completion check.
