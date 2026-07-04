# Motivation

The current system is a strong hard-coded SAM3 counting cascade: it
repeatedly queries SAM3, reuses accepted detections as pseudo-exemplars,
optionally applies tiling, and stops when no new candidates are
discovered. This produces strong empirical results, but the control
policy is still mostly fixed. The missing component is an orchestrator
that can adapt the sensing process to the current image, candidate
graph, and compute budget.

I propose to frame this as zero-shot active perception for promptable
foundation-model counting. In the classical active perception view,
perception is not a passive mapping from image to interpretation; an
agent controls its sensing process to reduce uncertainty about the world
[@bajcsy1988active; @bajcsy2018revisiting; @aloimonos1988activevision].
Here, the controllable sensor is a frozen promptable
segmentation/detection model such as SAM3 [@sam3]. Its control variables
are text prompts, pseudo-exemplar boxes, confidence thresholds, tiling,
image regions, verification calls, and stopping rules. The controller is
a VLM that chooses how SAM3 should sense next.

The VLM is allowed to inspect the image or selected crops to assess
scene-level properties such as target presence, crowding, occlusion,
scale, and whether a region should be queried at higher resolution.
However, the VLM is not allowed to add detections to the candidate
graph. All candidate tracks must originate from SAM3 outputs and pass
through the graph update procedure. This keeps the count grounded in
detector evidence while allowing the sensing policy to be visually
adaptive.

# Related Work and Positioning

This proposal sits at the intersection of active perception,
open-vocabulary counting, promptable segmentation, and LLM/VLM tool
orchestration. The active-perception lineage argues that perception
should be controlled according to task objectives and sensing costs
[@bajcsy1988active; @bajcsy2018revisiting; @aloimonos1988activevision].
The POMDP formulation provides the standard language for acting under
hidden state uncertainty [@kaelbling1998pomdp], while classical search
theory studies how to allocate limited search effort over space
[@koopman1957search]. This project applies those ideas to a modern
foundation-model sensor.

Recent counting work has moved from density-map prediction toward
open-vocabulary, exemplar-guided, and detection-based counting.
CLIP-Count and CounTX study text-specified zero-shot counting
[@clipcount2023; @countx2023]; CountGD introduces multimodal text and
visual-exemplar prompting for open-world counting [@countgd2024]; and
CountGD++ extends this direction with negative prompts,
pseudo-exemplars, external exemplars, and LLM-driven use of a counting
model as a vision expert [@countgdpp2025]. DAVE and GeCo show the
strength of detection-based low-shot counting, with DAVE using a
detect-and-verify pipeline and GeCo jointly modeling detection,
segmentation, and counts [@dave2024; @geco2024]. SAM-based and
training-free alternatives such as PseCo and OmniCount further show the
relevance of promptable segmentation for counting
[@pseco2024; @omnicount2025].

Promptable vision models provide the sensing substrate for this project.
SAM introduced interactive promptable segmentation [@kirillov2023sam],
while SAM3 extends this idea toward promptable concept segmentation with
text and image exemplars [@sam3]. Text-visual prompt synergy has also
been studied in open-set detection systems such as T-Rex2 [@trex2_2024].
Separately, systems such as ViperGPT, VisProg, HuggingGPT, and Chameleon
show that language models can orchestrate external vision tools
[@viperGPT2023; @visprog2023; @hugginggpt2023; @chameleon2023]. The
proposed method differs by making orchestration closed-loop,
uncertainty-aware, and explicitly tied to an active perception
objective.

For agricultural scenes, datasets such as the green citruses dataset
from UC Merced or MinneApple highlight the difficulty of instance-level
fruit counting under clutter and occlusion [@minneapple2020]. This is
precisely where active tiling, region selection, and repeated
pseudo-exemplar support are useful: the system can spend more sensing
effort where global one-shot prompting is likely to miss small or
crowded objects.

# Problem Setup

Let $$I \in \mathbb{R}^{H \times W \times 3}$$ be an input image and let
$t$ denote the target text concept, for example "green citrus", "apple",
or "car". The hidden ground-truth object set is
$$\mathcal{Y}^{\star} = \{y_1^{\star},\ldots,y_{N^{\star}}^{\star}\},$$
where $N^{\star}=|\mathcal{Y}^{\star}|$ is the true count. Each object
$y_j^{\star}$ has a bounding box, and in some datasets a mask,
$$y_j^{\star} = (b_j^{\star}, m_j^{\star}).$$ The goal is to estimate
the count $N^{\star}$, and when box-level ground truth is available, to
also recover detections that match the objects in $\mathcal{Y}^{\star}$.
Following promptable segmentation models such as SAM and SAM3
[@kirillov2023sam; @sam3], a single SAM3 query may be written abstractly
as $$o = f_{\theta}(I, P, r, \tau),$$ where $f_{\theta}$ is the SAM3
model, $P$ is the prompt, $r$ is the image region being queried, and
$\tau$ is the confidence threshold. The prompt may contain text and
visual exemplars,
$$P = \left(t, \mathcal{E}^{+}, \mathcal{E}^{-}\right),$$ where
$\mathcal{E}^{+}$ denotes positive exemplar boxes and $\mathcal{E}^{-}$
denotes negative exemplar boxes if available. In the current
implementation, previously accepted detections are reused as positive
pseudo-exemplars for later passes. The output of a query is a set of
candidate detections:
$$o_t = \{(b_{t,k}, m_{t,k}, s_{t,k})\}_{k=1}^{M_t},$$ where $b_{t,k}$
is a box, $m_{t,k}$ is a mask, and $s_{t,k} \in [0,1]$ is the model
score.

# Partially Observable Markov Decision Process Formulation

Let's formulate the inference process for a single image as a Partially
Observable Markov Decision Process (POMDP), following the standard
formulation for decision-making under partial observability
[@kaelbling1998pomdp] $$\mathcal{P} = (S, A, T, R, \Omega, O, \gamma).$$
Here $S$ is the state space, $A$ is the action space, $T$ is the
transition kernel, $R$ is the reward function, $\Omega$ is the
observation space, $O$ is the observation kernel, and $\gamma \in [0,1]$
is the discount factor.

In this formulation, the image itself is fixed, but the true object set
is hidden. Therefore, uncertainty is epistemic: the objects are present
in the image, but the agent has not fully discovered or verified them.
The role of the VLM orchestrator is to choose sensor-control actions for
SAM3 based on the current history or belief summary.

## State Space

The hidden state is the true object set:
$$s = \mathcal{Y}^{\star} \in S.$$ Equivalently, the state may be
written as $$s =
\left(
N^{\star},
\{(b_j^{\star},m_j^{\star})\}_{j=1}^{N^{\star}}
\right).$$ This state contains the information needed to determine the
correct count and, when annotations are available, the correct
detections. Although $s$ is fixed for a static image, it is not directly
observed by the agent. The agent only observes the outputs of SAM3
queries and verification tools.

## Action Space

The graph-relevant action space consists of the allowed SAM3 sensing,
region-control, verification, and stopping operations:
$$a_t \in A_{\mathrm{sense}}.$$ The preliminary sensing action set is
$$A_{\mathrm{sense}}
=
\{
\textsc{Query}(r,P,\tau),
\textsc{TileQuery}(P,\tau),
\textsc{Subdivide}(r),
\textsc{Verify}(C),
\textsc{Stop}(\hat N)
\}.$$

Each action is defined as follows:

- [Query]{.smallcaps}$(r,P,\tau)$ - run SAM3 on region $r$ with prompt
  $P$ and threshold $\tau$.

- [TileQuery]{.smallcaps}$(P,\tau)$ - run SAM3 over a tiled split of the
  image using prompt $P$ and threshold $\tau$.

- [Subdivide]{.smallcaps}$(r)$ - split region $r$ into smaller regions
  for more focused sensing.

- [Verify]{.smallcaps}$(C)$ - send a set of uncertain candidate
  detections $C$ to a verifier, such as a VLM verifier or SAM3.

- [Stop]{.smallcaps}$(\hat N)$ - terminate the sensing process and
  return a final count estimate.

The VLM may also inspect the image or selected crops to produce a
scene-level assessment. However, this inspection is treated as a
pre-decision information-gathering step rather than as a normal
graph-updating sensing action. Inspection can inform which SAM3 action
should be selected next, but it does not itself produce detections and
does not modify the candidate graph.

Thus, inspection produces a visual assessment $z_t$, while candidate
tracks are created only by SAM3 observations passed through
$\operatorname{UpdateGraph}$. This distinction is important: the VLM can
decide where and how SAM3 should act, but it cannot directly add boxes,
masks, or candidate tracks to $\mathcal{G}_t$.

The current implemented cascade is a restricted policy over this action
space. It uses [Query]{.smallcaps}, optionally uses
[TileQuery]{.smallcaps}, reuses previous detections as positive
pseudo-exemplars, and terminates when the number of newly discovered
candidates falls below a threshold. The proposed extension is to allow a
VLM orchestrator to select among these actions adaptively to improve the
accuracy-compute tradeoff.

## Transition Kernel

The transition kernel is
$$T(s' \mid s,a) = \Pr(s_{t+1}=s' \mid s_t=s, a_t=a).$$ Since the image
is static, the true object set does not change as a result of sensing -
the transition is an identity: $$T(s' \mid s,a) = \mathbf{1}[s'=s].$$
This means that the agent is not changing the world, but only what it
knows about the world.

## Observation Space

The observation space $\Omega$ contains all possible outputs produced by
the sensing actions. For a SAM3 query action, an observation is a finite
set of detections: $$o_t =
\{(b_{t,k},m_{t,k},s_{t,k})\}_{k=1}^{M_t}
\in \Omega.$$ For a verification action, the observation may be a set of
verifier outputs: $$o_t =
\{v_{t,k}\}_{k=1}^{|C|},$$ where $v_{t,k}$ may be a binary decision, a
text judgment, or an uncalibrated confidence score.

For a VLM image or crop inspection step, the output is a scene-level
assessment $$z_t(r)
=
\{\text{presence},\text{density},\text{scale},\text{occlusion},\text{recommended action}\},$$
or a textual/structured summary of the same information. This assessment
is used by the orchestrator when choosing later SAM3 sensing actions.
Importantly, $z_t(r)$ is not a set of detections and is not inserted
into the candidate graph.

For a [Stop]{.smallcaps} action, the observation may be treated as
empty: $o_t = \varnothing$

## Observation Kernel

The observation kernel is $$O(o \mid s',a)
=
\Pr(o_t=o \mid s_{t+1}=s', a_t=a).$$ It describes how likely the agent
is to receive observation $o$ after taking action $a$ when the true
state is $s'$. For example, if $$a_t = \textsc{Query}(r,P,\tau),$$ then
$O(o_t \mid s',a_t)$ describes the probability that SAM3 returns a
particular set of boxes, masks, and scores under prompt $P$, region $r$,
and threshold $\tau$, given the true object set $s'$.

In practice, this observation kernel is not known. Since the intended
method is zero-shot, we do not assume access to a learned,
dataset-specific observation model. Instead, the implemented system uses
observable statistics from the query history, such as:

- number of newly discovered candidates

- support of a candidate across query signatures

- spatial stability of matched detections

- discovery-curve saturation

These statistics form a practical belief summary for the VLM
orchestrator, even though they are not a fully calibrated Bayesian
posterior.

## Reward Function

The reward function should encourage accurate counting while penalizing
unnecessary sensing and orchestration cost. It is defined as
$$R:S \times A \rightarrow \mathbb{R}.$$ A natural terminal reward after
[Stop]{.smallcaps} is
$$R_{\mathrm{terminal}}\!\left(s,\textsc{Stop}(\hat N)\right)
=
-\left(\hat N - N^{\star}(s)\right)^2,$$ where $\hat N$ is the count
returned by the policy and $N^{\star}(s)$ is the true count in hidden
state $s$.

For non-terminal steps, the reward penalizes the cost of sensing and
orchestration. Since the proposed policy uses both SAM3 sensing actions
and VLM visual inspection, the step cost should include both components.
Let $I_t^{\mathrm{inspect}}\in\{0,1\}$ indicate whether a VLM image or
crop inspection was invoked before choosing the sensing action at time
$t$. Then a useful cost model is $$\operatorname{cost}_t
=
\operatorname{cost}_{\mathrm{sense}}(a_t)
+
c_{\mathrm{inspect}} I_t^{\mathrm{inspect}},$$ where
$$\operatorname{cost}_{\mathrm{sense}}(a_t)
=
c_{\mathrm{sam}}N_{\mathrm{sam}}(a_t)
+
c_{\mathrm{tile}}N_{\mathrm{tile}}(a_t)
+
c_{\mathrm{verify}}N_{\mathrm{verify}}(a_t)
+
c_{\mathrm{orch}}N_{\mathrm{orch}}(a_t).$$ Here $N_{\mathrm{sam}}(a_t)$
counts SAM3 model calls, $N_{\mathrm{tile}}(a_t)$ counts tile-level SAM3
calls, $N_{\mathrm{verify}}(a_t)$ counts verifier calls, and
$N_{\mathrm{orch}}(a_t)$ counts lightweight orchestration decisions. The
term $c_{\mathrm{inspect}}$ accounts for VLM image or crop inspection.
This term should be reported explicitly, since visual VLM inference may
not be negligible relative to SAM3 inference.

For non-terminal sensing actions, the reward may be written as
$$R(s,a_t)
=
-\lambda_{\mathrm{cost}}\operatorname{cost}_t,
\qquad
a_t \neq \textsc{Stop}.$$

Since the true state $s$ is hidden at test time, the agent cannot
directly optimize the state reward. Instead, it acts using a belief
state or tractable belief summary. The corresponding belief-state reward
is $$R(b_t,a_t)
=
\mathbb{E}_{s\sim b_t}
\left[
R(s,a_t)
\right].$$ For count estimation, a useful design objective is
$$R(b_t,a_t)
=
-
\mathbb{E}_{s\sim b_t}
\left[
\left(\hat N(b_t)-N^{\star}(s)\right)^2
\right]
-
\lambda_{\mathrm{cost}}\operatorname{cost}_t.$$ This expression
formalizes the desired tradeoff: reduce expected count error while
spending as few SAM3, tiling, verification, and VLM inspection calls as
possible.

In the zero-shot implementation, this exact reward is not computed at
test time because $N^{\star}(s)$ is unknown. Instead, it serves as the
formal objective motivating the approximate policy. The implemented
policy uses observable quantities such as candidate support, discovery
saturation, spatial stability, SAM3 score, tiling status, VLM visual
assessment, and remaining budget.

## Discount Factor

The discount factor is $\gamma \in [0,1]$. For a finite-horizon sensing
problem that terminates with [Stop]{.smallcaps}, we can set
$\gamma = 1$, because future rewards are not inherently less important
than immediate rewards. The cost term already discourages unnecessary
sensing.

## History and Policy

At time $t$, the agent has access to the action-observation history
$$h_t =
(a_0,o_1,a_1,o_2,\ldots,a_{t-1},o_t).$$ A general POMDP policy maps
histories to actions: $$\pi(a_t \mid h_t).$$ Equivalently, if the agent
maintains a belief state $b_t$, the policy can be written as
$$\pi(a_t \mid b_t).$$

In the proposed system, the VLM orchestrator implements an approximate
policy $$\pi_{\mathrm{VLM}}(a_t \mid \phi_t),$$ where $\phi_t$ is a
structured summary of the current counting process. Unlike a purely
symbolic controller, the VLM may condition on both graph-derived
statistics and visual context from the image or selected crops. This
visual context is represented by
$$z_t = \operatorname{VLMInspect}(I,\mathcal{R}_t,\mathcal{G}_t,t),$$
where $z_t$ may summarize object presence, density, scale, occlusion,
clutter, and whether tiling or zooming appears useful.

The inspection step is not selected by the same myopic
value-of-information rule used for SAM3 sensing actions. Instead, it is
invoked by a fixed inspection protocol. For example, inspection may be
performed once at the beginning of the episode, and again whenever a
tiling or subdivision decision is pending, or when the system is
uncertain which sensing action has the highest value. This avoids
treating inspection as a graph-updating action. Since inspection does
not add detections, it would have zero one-step value under an
uncertainty functional defined only on
$\tilde b_t=(\mathcal{G}_t,D_t,U_t)$.

The policy input is therefore $$\phi_t =
\left(
\tilde b_t,\;
z_t,\;
\text{tiling status},\;
\text{available actions},\;
\text{remaining budget}
\right).$$ Here $\tilde b_t$ contains the candidate graph, discovery
curve, and zero-shot uncertainty summary, while $z_t$ contains the VLM's
scene-level visual assessment. Importantly, $z_t$ is not a detection
set. It does not add boxes, masks, or candidate tracks to the graph. The
candidate graph is updated only from SAM3 outputs: $$\mathcal{G}_{t+1}
=
\operatorname{UpdateGraph}(\mathcal{G}_t,o_{t+1}).$$ Thus, the VLM acts
as an active sensing controller: it may inspect the image and the
current graph state in order to decide where and how SAM3 should act
next, but it cannot directly modify the object set used for counting.

## Bayesian Belief Update

The belief state is a probability distribution over hidden states:
$$b_t(s) = \Pr(s_t=s \mid h_t).$$ After taking action $a_t$ and
receiving observation $o_{t+1}$, the Bayesian belief update is
$$b_{t+1}(s')
=
\eta_b
\,
O(o_{t+1} \mid s',a_t)
\sum_{s \in S}
T(s' \mid s,a_t)b_t(s),$$ where $\eta_b$ is the normalizing constant:
$$\eta_b^{-1}
=
\Pr(o_{t+1}\mid b_t,a_t)
=
\sum_{s' \in S}
O(o_{t+1}\mid s',a_t)
\sum_{s \in S}
T(s'\mid s,a_t)b_t(s).$$

Since the transition kernel is the identity,
$$T(s' \mid s,a_t)=\mathbf{1}[s'=s],$$ the belief update simplifies to
$$b_{t+1}(s)
=
\frac{
O(o_{t+1}\mid s,a_t)b_t(s)
}{
\sum_{\tilde s \in S}
O(o_{t+1}\mid \tilde s,a_t)b_t(\tilde s)
}.$$ This is the standard Bayes's rule for new belief given likelihood
of new evidence times old belief, normalized.

## Why the Exact Belief Is Intractable

Although the exact belief update is mathematically sound, it is not
computationally possible in this problem. The hidden state $s$ is a
variable-size set of objects: $$s =
\{(b_1,m_1),\ldots,(b_N,m_N)\},$$ where both $N$ and the object
locations are unknown. Therefore, the state space $S$ contains an
enormous set of possible object configurations. In addition, the
observation kernel $O(o\mid s,a)$ for SAM3 is unknown and not
analytically available.

For this reason, the implemented system should be viewed as an
approximate POMDP solver. Instead of maintaining the full posterior
$b_t(s)$, it maintains a tractable belief summary
$$\tilde b_t = (\mathcal{G}_t, D_t, U_t),$$ where $\mathcal{G}_t$ is the
current candidate graph, $D_t = (n_1, n_2, \ldots, n_t)$ is the
discovery curve, and $U_t := U(\tilde b_t)$ is a scalar zero-shot
uncertainty summary derived from $\mathcal{G}_t$ and $D_t$ through
support, score, spatial stability, and remaining sensing budget. The
functional $U(\cdot)$ is made explicit in the VLM Orchestration Policy
section.

# Candidate Graph as an Approximate Belief State

The candidate graph is $$\mathcal{G}_t =
\{c_1,\ldots,c_{K_t}\}.$$ Each candidate is represented as $$c_i =
(\bar b_i,\bar m_i,\bar s_i,\mathcal{Q}_i,k_i,\Delta_i,A_i),$$ where:

- $\bar b_i$ is the representative box,

- $\bar m_i$ is the representative mask,

- $\bar s_i$ is an aggregate SAM 3 score,

- $\mathcal{Q}_i$ is the set of query signatures that detected the
  candidate,

- $k_i = |\mathcal{Q}_i|$ is the candidate's support count,

- $\Delta_i$ measures spatial instability across matched detections

- $A_i$ is the candidate box area.

A query signature is defined as $$q =
(r,P,\tau,\mathrm{mode}),$$ where $r$ is the region, $P$ is the prompt,
$\tau$ is the threshold, and $\mathrm{mode}$ indicates full-image,
tiled, or region-level inference. The approximate belief summary
therefore records not only which candidates have been found, but also
how stable and repeatedly supported they are across active sensing
actions.

# Association and Deduplication

When a new detection $b$ is returned, it must either be associated with
an existing track or registered as a new track. Two overlap measures are
used.

The intersection-over-union score is $$\operatorname{IoU}(b,b_i)
=
\frac{|b\cap b_i|}{|b\cup b_i|}.$$ The intersection-over-minimum score
is $$\operatorname{IoM}(b,b_i)
=
\frac{|b\cap b_i|}{\min(|b|,|b_i|)}.$$

IoU is useful for standard duplicate suppression. IoM is useful for
dense scenes and nested boxes, because it can identify cases where one
box is mostly contained inside another even if the union is large. A new
detection $b$ is matched to an existing candidate $c_i$ if
$$\operatorname{IoU}(b,\bar b_i) \ge \tau_{\mathrm{IoU}} \quad \text{or} \quad \operatorname{IoM}(b,\bar b_i) \ge \tau_{\mathrm{IoM}}.$$

If no existing track satisfies the association criterion, the detection
becomes a new candidate track. The graph update is therefore:
$$\mathcal{G}_{t+1}
=
\operatorname{UpdateGraph}(\mathcal{G}_t,o_{t+1}),$$ where
$\operatorname{UpdateGraph}$ includes NMS, IoU/IoM deduplication,
optional seam stitching for tiled detections, and track registration.

# Zero-Shot Belief Representation

The main method is designed to remain fully zero-shot. Rather than
requiring a labeled calibration split for every new object category or
dataset, the agent represents candidate reliability using quantities
produced by the active sensing process itself: SAM3 score, repeated
support across distinct query signatures, spatial stability, and simple
geometric plausibility.

Using the candidate representation defined above, we summarize candidate
reliability with a zero-shot support score
$$w_i = F(\bar s_i,k_i,\Delta_i,A_i).$$ For example, one deterministic
support score is $$w_i
=
\bar s_i
+
\lambda_k \log(1+k_i)
-
\lambda_{\Delta}\Delta_i
-
\lambda_A
\mathbf{1}[A_i \notin \mathcal{A}_{\mathrm{plausible}}].$$ This score is
a policy-facing support statistic rather than a calibrated probability.
It lets the orchestrator rank candidates, filter weak tracks, prioritize
verification, and decide whether additional sensing is necessary while
keeping the method independent of category-specific calibration.

# Discovery Curve

Rather than defining discovery as a net graph-size difference, we define
$n_t$ as the number of newly registered candidate tracks produced by
action $a_t$: $$n_t
=
\#\{c_i : \operatorname{birth}(c_i)=t\}.$$ Equivalently, if
$\mathcal{A}_t$ denotes the set of candidates added by
$\operatorname{UpdateGraph}$ at step $t$, then
$$n_t = |\mathcal{A}_t|.$$ The discovery curve is therefore
$$D_t = (n_1,\ldots,n_t).$$

In the current implementation, the candidate graph is monotone:
duplicate detections are rejected before registration, while accepted
tracks are added and not later removed. Under this monotone-graph
assumption: $$n_t = |\mathcal{G}_t|-|\mathcal{G}_{t-1}|.$$ The stopping
rule that uses this discovery curve is defined in the
Value-of-Information Objective section.

# Induced Belief MDP

A POMDP can be converted into a fully observable belief MDP whose states
are beliefs $b_t$ rather than hidden states $s_t$. The belief transition
is deterministic given the previous belief, action, and observation:
$$b_{t+1}
=
\mathrm{SE}(b_t,a_t,o_{t+1}),$$ where $\mathrm{SE}$ denotes the Bayesian
belief-update (state-estimator) operator defined in the Bayesian Belief
Update section.

The value function over beliefs is $$V^{\pi}(b)
=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{H}
\gamma^t R(b_t,a_t)
\mid b_0=b
\right],$$ and the optimal value satisfies the Bellman equation
$$V^{\star}(b)
=
\max_{a\in A}
\left[
R(b,a)
+
\gamma
\sum_{o\in \Omega}
\Pr(o\mid b,a)\,
V^{\star}\!\left(\mathrm{SE}(b,a,o)\right)
\right].$$

where $$\Pr(o\mid b,a)
=
\sum_{s'\in S}
O(o\mid s',a)
\sum_{s\in S}
T(s'\mid s,a)b(s).$$

Solving this belief MDP exactly is intractable for the proposed counting
task. The purpose of the formulation is not to compute an exact optimal
policy. It only provides a mathematical framework for designing and
evaluating approximate active-perception policies.

# Approximate VLM Orchestration Policy

The proposed VLM orchestrator approximates the belief-MDP policy using
both a structured belief summary and a visual scene assessment, in the
spirit of recent work on language models orchestrating vision tools
[@viperGPT2023; @visprog2023; @hugginggpt2023; @chameleon2023]. The
structured belief summary is $$\tilde b_t = (\mathcal{G}_t,D_t,U_t),$$
where $\mathcal{G}_t$ is the candidate graph, $D_t$ is the discovery
curve, and $U_t := U(\tilde b_t)$ is the zero-shot uncertainty summary.

The VLM may also inspect the image or selected crops to produce a visual
assessment
$$z_t = \operatorname{VLMInspect}(I,\mathcal{R}_t,\mathcal{G}_t,t).$$
The assessment $z_t$ may summarize scene-level properties such as target
presence, crowding, object scale, occlusion, clutter, and whether tiling
or zooming appears likely to improve recall.

Crucially, VLM inspection is not a graph-updating sensing action. It
does not add boxes, masks, or candidate tracks to the graph. Candidate
tracks are created only from SAM3 outputs processed by
$$\mathcal{G}_{t+1}
=
\operatorname{UpdateGraph}(\mathcal{G}_t,o_{t+1}).$$ Thus, the VLM is an
active sensing controller, not a detector. It may inspect the image and
the current graph state in order to decide where and how SAM3 should act
next, but it cannot directly modify the object set used for counting.

The policy input is $$\phi_t =
\left(
\begin{aligned}
&K_t,\; n_t,\; D_t,\;
\{w_i\}_{i=1}^{K_t},\; \{\bar s_i\}_{i=1}^{K_t},\\
&\{k_i\}_{i=1}^{K_t},\; \{\Delta_i\}_{i=1}^{K_t},\; z_t,\;
\text{tiling status},\\
&\text{available actions},\; \text{remaining budget}
\end{aligned}
\right),$$ where $K_t = |\mathcal{G}_t|$. The VLM policy is then written
as $$a_t \sim \pi_{\mathrm{VLM}}(\cdot \mid \phi_t).$$

The image-inspection step is invoked by a fixed protocol rather than by
the same value-of-information rule used for SAM3 sensing actions. This
is necessary because inspection alone does not change $\mathcal{G}_t$ or
$D_t$, and therefore would have zero one-step value under an uncertainty
functional defined only on $\tilde b_t$. In the initial implementation,
inspection may be invoked:

1.  once at the beginning of the episode;

2.  before deciding whether to use tiling or subdivision;

3.  when several sensing actions have similar estimated value;

4.  when the discovery curve has flattened but the image still appears
    visually crowded, small-object dominated, or under-resolved.

After $z_t$ has been produced, the VLM chooses among validated sensing
actions. The graph-relevant sensing action set is $$A_{\mathrm{sense}}
=
\{
\textsc{Query}(r,P,\tau),
\textsc{TileQuery}(P,\tau),
\textsc{Subdivide}(r),
\textsc{Verify}(C),
\textsc{Stop}(\hat N)
\}.$$ The action arguments are generated or checked by the system. For
example, regions must come from the current image partition, candidate
sets must come from the current graph, and prompts or pseudo-exemplars
must be derived from validated system state.

The VLM can therefore reason visually about where SAM3 should look next
while the candidate graph remains grounded in SAM3 detections and
deterministic association rules. For example, if $z_t$ indicates that
the target objects are small or densely clustered, the policy may prefer
[TileQuery]{.smallcaps} or [Subdivide]{.smallcaps}. If $z_t$ indicates
that a region is unlikely to contain the target concept, the policy may
avoid querying that region. If $z_t$ indicates that the scene appears
saturated and the discovery curve is flat, the policy may stop with a
final count estimate.

## Value-of-Information Objective

Given the current belief summary $\tilde b_t$ and the most recent VLM
visual assessment $z_t$, the orchestrator chooses the next sensing
action by estimating which action is expected to reduce the most
zero-shot uncertainty per unit cost. The [Stop]{.smallcaps} action is
handled separately by the stopping rule, so the value-of-information
objective is applied only to non-terminal sensing actions: $$a_t^{\star}
=
\arg\max_{a\in A_{\mathrm{sense}}\setminus\{\textsc{Stop}\}}
\frac{
\widehat{\operatorname{VoI}}(a;\tilde b_t,z_t)
}{
c_0+\operatorname{cost}_{\mathrm{sense}}(a)
}.$$ Here $c_0>0$ is a fixed base cost accounting for orchestration
overhead. This base cost keeps the ratio finite for actions with low or
zero marginal sensing cost, such as [Subdivide]{.smallcaps}. The cost
term $\operatorname{cost}_{\mathrm{sense}}(a)$ accounts for the expected
cost of executing the selected sensing action. The cost of obtaining
$z_t$ is accounted for separately through the inspection protocol.

The zero-shot uncertainty functional is $$U(\tilde b_t)
=
\lambda_D
\left(
\frac{1}{m}
\sum_{j=t-m+1}^{t}
n_j
\right)
+
\lambda_S
\sum_{i=1}^{K_t}
\left(
\frac{1}{1+k_i}
+
\alpha_{\Delta}\Delta_i
+
\alpha_s(1-\bar s_i)
\right).$$ The first term measures remaining coverage uncertainty
through the recent discovery curve. If recent actions are still
discovering new candidates, then the system may not have fully covered
the image. The second term measures candidate-level instability: a
candidate is more uncertain if it has low query support, unstable
localization, or low SAM3 score.

The approximate value of information is defined as the expected
reduction in uncertainty:
$$\widehat{\operatorname{VoI}}(a;\tilde b_t,z_t)
\approx
U(\tilde b_t)
-
\widehat{U}(\tilde b_{t+1}\mid a,z_t).$$ Here
$\widehat{U}(\tilde b_{t+1}\mid a,z_t)$ is the predicted uncertainty
after taking action $a$, conditioned on both the current belief summary
and the VLM's visual assessment. The role of $z_t$ is to help estimate
which actions are likely to reduce uncertainty. For example, a visual
assessment of small or crowded objects may increase the estimated value
of tiling, while an assessment that a region is visually irrelevant to
the target concept may decrease the estimated value of querying that
region.

Inspection itself is not included in the argmax above. Since inspection
does not produce SAM3 detections, it leaves $\mathcal{G}_t$ and $D_t$
unchanged and therefore does not directly reduce $U(\tilde b_t)$ in a
one-step objective. Instead, inspection is invoked by the fixed protocol
described above and affects the ranking of later sensing actions through
$z_t$.

The stopping decision is also treated separately. A simple stopping rule
is to terminate when the recent discovery curve is saturated and the
remaining zero-shot uncertainty is low: $$\frac{1}{m}
\sum_{j=t-m+1}^{t}
n_j
\le
\delta_{\mathrm{disc}}
\quad
\text{and}
\quad
U(\tilde b_t)
\le
\delta_U.$$ When this condition holds, the policy returns
$$a_t = \textsc{Stop}(\hat N),$$ where $\hat N$ is one of the zero-shot
count estimates, such as $\hat N_{\mathrm{obs}}$,
$\hat N_{\mathrm{supp}}$, or $\hat N_{\mathrm{cons}}$.

Thus, the VLM policy has two distinct roles. First, through the
inspection protocol, it produces visual context $z_t$ about the image or
selected regions. Second, conditioned on $(\tilde b_t,z_t)$, it chooses
among validated SAM3 sensing actions according to an approximate
value-of-information objective. This keeps the policy visually informed
while ensuring that the count remains grounded in SAM3 detections and
graph-based association.

# Interpretable Candidate Verification with FM+V-IP

The [Verify]{.smallcaps}$(C)$ action defined in the Action Space section
requires a procedure that decides, for each uncertain candidate, whether
it depicts the target concept or a distractor. The current
implementation audits each candidate with two additional SAM3 queries on
an upsampled crop, one with the target prompt and one with a distractor
prompt, and thresholds the resulting pair of presence scores. This gate
is cheap but opaque: the verdict is a comparison of two uncalibrated
scalars, its thresholds are hand-tuned per dataset, and a wrong verdict
can neither be explained nor corrected. We propose to replace it with an
interpretable-by-design verifier based on the Information Pursuit family
[@ip2022; @vip2023] in its foundation-model form, FM+V-IP [@fmvip2023].
In FM+V-IP, an LLM generates a task-relevant set of natural-language
concept queries, a multimodal model answers those queries for a given
image, and classification proceeds by sequentially selecting the most
informative queries until the class posterior is confident. The
prediction is therefore justified by a short, human-readable
query--answer chain rather than by an opaque score.

## Verification as Interpretable Classification

For a candidate $c_i$ with representative box $\bar b_i$, let $x_i$
denote the crop obtained by enlarging $\bar b_i$ by a factor
$\rho \ge 1$ around its center and resampling it to a fixed resolution.
Verification is a small classification problem over
$$v_i \in \mathcal{V} = \{\textit{target}, \textit{distractor}, \textit{spurious}\},$$
which in the orchard instantiation corresponds to fruit, leaf, and
clutter, matching the classification tags already used in the candidate
graph. Following FM+V-IP, the verifier is specified by a query set: an
LLM is prompted once per target concept $t$ to produce $M$ binary visual
queries
$$Q_{\mathrm{ver}} = \{q_1,\ldots,q_M\},
\qquad
q_m : \mathcal{X} \rightarrow \{+1,-1,0\},$$ where $+1$, $-1$, and $0$
encode *yes*, *no*, and *not decidable from this crop*. Typical queries
for a fruit concept are "is the region approximately round", "is a
smooth waxy surface visible", or "is the region a flat, thin structure
with visible veins". Because the query set depends only on the concept
name, this step uses no labeled data and preserves the zero-shot
character of the overall system. Following the concept-filtering
argument of FM+V-IP, no manual pruning of $Q_{\mathrm{ver}}$ is
required: redundant or uninformative queries are simply never selected
by the information-pursuit criterion below [@fmvip2023].

## Answering Queries Without a New Model

FM+V-IP answers queries with CLIP image--text dot products. Its authors
observe that such answers are noisy for fine-grained concepts and
explicitly identify more precise VQA systems as future work
[@fmvip2023]. Since the proposed system already deploys two models
capable of answering visual queries, we restrict the verifier to them
rather than introducing a third model:

- **VLM oracle (primary).** The VLM orchestrator (e.g. Qwen-3-VL
  [@qwen3vl]) answers the full query set for a crop in a single
  structured call, returning an answer vector
  $$\hat a(x_i) = \left(\hat a_1,\ldots,\hat a_M\right) \in \{+1,0,-1\}^{M}.$$
  This replaces $M$ CLIP similarity evaluations with one VQA call per
  candidate, and upgrades answer quality on attribute-style queries
  (texture, shape, occlusion, context) that similarity scores handle
  poorly.

- **SAM3 presence channel (optional).** For the subset of queries that
  are segmentable noun phrases (for example "stem" or "leaf blade"),
  the answer may instead be obtained by thresholding the SAM3 presence
  score on the crop, exactly as in the current audit. This channel is
  disabled by default and retained for ablation.

The VLM oracle is the same model instance used for scene inspection
$z_t$, so verification adds no new component to the model inventory; its
calls are metered by the $N_{\mathrm{verify}}$ term of the cost model.

## Training-Free Information Pursuit

V-IP replaces the mutual-information computation of generative IP with a
learned querier network trained on sampled query--answer chains
[@vip2023]. That machinery exists to scale sequential selection to tens
of thousands of queries and hundreds of classes. Our verification
problem has $|\mathcal{V}|=3$ classes and $M \approx 20$--$30$ queries,
so exact greedy IP is directly computable and no learned component is
needed, keeping the verifier training-free.

The class-conditional answer model is also produced by the LLM at query
generation time: for every class $k \in \mathcal{V}$ and query $q_m$ it
emits an expected answer $\tau_{k,m} \in \{+1,-1,0\}$, where $0$ marks
the query as uninformative for that class. With a fixed answer-noise
rate $\varepsilon \in (0,\tfrac12)$, the likelihood of an observed
answer is $$P(\hat a_m \mid v = k)
=
\begin{cases}
1-\varepsilon & \text{if } \hat a_m \,\tau_{k,m} > 0,\\[2pt]
\varepsilon & \text{if } \hat a_m \,\tau_{k,m} < 0,\\[2pt]
\tfrac{1}{2} & \text{if } \hat a_m = 0 \ \text{or}\ \tau_{k,m} = 0.
\end{cases}$$ Assuming conditional independence of answers given the
class, the posterior after observing the answers indexed by
$S \subseteq \{1,\ldots,M\}$ is
$$P(v = k \mid \hat a_S)
\;\propto\;
\pi_k
\prod_{m \in S}
P(\hat a_m \mid v = k),$$ where the prior $\pi_k$ may be uniform or
derived from the candidate's detection confidence. The next query is
chosen by the greedy information-pursuit rule of [@ip2022],
$$q_{(j+1)}
=
\arg\max_{m \notin S_j}
I\!\left(\hat a_m ;\, v \mid \hat a_{S_j}\right),$$ where the
conditional mutual information is available in closed form under the
factorized model:
$$I\!\left(\hat a_m ; v \mid \hat a_{S_j}\right)
=
H\!\left(\hat a_m \mid \hat a_{S_j}\right)
-
\sum_{k \in \mathcal{V}}
P(v = k \mid \hat a_{S_j})\,
H\!\left(\hat a_m \mid v = k\right).$$ Selection terminates at the
standard IP stopping criterion
$$\max_{k}\; P(v = k \mid \hat a_{S_j}) \;\ge\; 1-\epsilon_{\mathrm{stop}}
\qquad\text{or}\qquad
j = J_{\max},$$ and the verdict is the maximum-a-posteriori class
$\hat v_i = \arg\max_k P(v=k \mid \hat a_{S_j})$.

Because the batched oracle produces all $M$ answers in one call, the
greedy chain can be computed post hoc over the precomputed answer
vector: the selected chain, its order, and the resulting posterior are
identical to the sequential variant, while the oracle cost is exactly
one VLM call per candidate,
$$\operatorname{cost}\big(\textsc{Verify}(C)\big) = c_{\mathrm{verify}}\,|C|.$$
A truly sequential mode, issuing one oracle call per selected query, is
retained as an ablation for the case where per-query answering is
expensive.

## Integration with the Candidate Graph

The [Verify]{.smallcaps}$(C)$ action selects the verification set $C$
from candidates that are unresolved or have low support score $w_i$. Its
observation is
$$o_t = \left\{\left(\hat v_i,\; P(v \mid \hat a_{S_i}),\; \left(q_{(1)},\hat a_{(1)},\ldots\right)_i\right)\right\}_{c_i \in C},$$
and $\operatorname{UpdateGraph}$ writes, for each verified candidate,
the classification tag $\hat v_i$ and the calibrated verification score
$P(v=\textit{target} \mid \hat a_{S_i})$ in place of the current raw
SAM3 audit scores. Verified targets are routed to the positive exemplar
set $\mathcal{E}^{+}$, verified distractors to $\mathcal{E}^{-}$, and
spurious candidates are excluded from both exemplar sets and from all
count estimators. The query--answer chain is stored on the node, so
every accepted or rejected count is justified by a short
natural-language trace. This also enables test-time intervention in the
sense of FM+V-IP [@fmvip2023]: correcting a single recorded answer and
re-running the closed-form posterior changes the verdict without
retraining or re-querying any model, which is useful when a domain
expert audits the system's counts.

# Zero-Shot Count Estimates

The simplest zero-shot count is the observed candidate-track count:
$$\hat N_{\mathrm{obs}}
=
|\mathcal{G}_t|.$$ A stricter support-filtered count is
$$\hat N_{\mathrm{supp}}
=
\sum_{i=1}^{K_t}
\mathbf{1}[w_i \ge \tau_w].$$ Another useful self-consistency count is
$$\hat N_{\mathrm{cons}}
=
\sum_{i=1}^{K_t}
\mathbf{1}
[
k_i \ge k_{\min}
\ \text{or}\
\bar s_i \ge \tau_{\mathrm{high}}
].$$

# Tiling and Region-Level Active Sensing

In our active perception formulation, tiling is treated as an active
sensing action rather than a separate preprocessing step
[@bajcsy1988active; @bajcsy2018revisiting]. A global query asks SAM3 to
reason over the full image, while a tiled query changes the effective
field of view and allows the model to inspect smaller regions at higher
local resolution. This is especially important for dense scenes, small
objects, and partially occluded objects.

Let $\mathcal{R}_t$ be the set of image regions used at step $t$. These
regions may come from a fixed tiling pattern, overlapping crops, or a
region-selection policy. Each region $r \in \mathcal{R}_t$ has a local
candidate graph $$\mathcal{G}_t(r)
=
\{c_i \in \mathcal{G}_t : \operatorname{center}(\bar b_i)\in r\},$$ and
a local discovery curve $$D_t(r) = (n_1(r),\ldots,n_t(r)).$$ Here
$n_t(r)$ denotes the number of newly discovered candidates in region $r$
at step $t$.

A simple region query value is $$V_{\mathrm{query}}(r)
=
\lambda_D \,\bar n_t(r)
+
\lambda_C \,C_t(r)
+
\lambda_U \,U(\tilde b_t; r),$$ where $\bar n_t(r)$ is the recent local
discovery rate, $C_t(r)$ is a crowding or density score, and
$U(\tilde b_t; r)$ is the region-restricted uncertainty. The local
discovery rate mirrors the windowed average used in the stopping rule,
$$\bar n_t(r)
=
\frac{1}{m}
\sum_{j=t-m+1}^{t}
n_j(r),$$ where $n_j(r)$ is the number of candidates newly registered in
region $r$ at step $j$. The crowding score is $$C_t(r)
=
\frac{|\mathcal{G}_t(r)|}{A(r)},$$ where $A(r)$ is the area of region
$r$. The region-restricted uncertainty is the candidate-instability term
of $U(\tilde b_t)$ summed over the candidates whose representative
center lies in $r$, $$U(\tilde b_t; r)
=
\sum_{i \,:\, c_i \in \mathcal{G}_t(r)}
\left(
\frac{1}{1+k_i}
+
\alpha_{\Delta}\Delta_i
+
\alpha_s(1-\bar s_i)
\right),$$ so that $U(\tilde b_t; r)$ is exactly the instability term of
the global uncertainty $U(\tilde b_t)$ restricted to region $r$. The
coverage component is already captured separately by $\bar n_t(r)$,
which avoids double-counting discovery uncertainty.

The regional score is not a separate policy objective from the global
value-of-information rule. Instead, it proposes or parameterizes the
region argument for actions such as [Query]{.smallcaps},
[TileQuery]{.smallcaps}, or [Subdivide]{.smallcaps}. Once a region $r_t$
has been selected, the resulting concrete action is still ranked by the
same global objective
$\widehat{\operatorname{VoI}}(a;\tilde b_t,z_t)/(c_0+\operatorname{cost}_{\mathrm{sense}}(a))$.

# Evaluation Plan

Although the method is zero-shot, ground truth is still used for
evaluation. No labeled data is used at any time before evaluation. The
evaluation should answer five questions:

1.  Does active pseudo-exemplar querying improve proposal coverage over
    one-shot SAM3?

2.  Does tiling improve recall in dense or small-object scenes?

3.  Do repeated-support count estimates improve count accuracy over the
    raw candidate count?

4.  Does the active stopping rule reduce compute while preserving
    accuracy?

5.  Does the VLM orchestrator improve the accuracy-cost tradeoff over
    both the fixed cascade and a non-visual heuristic policy using the
    same action space?

## Detection-Level Evaluation

For datasets with bounding-box ground truth, candidates are matched to
ground-truth objects using Hungarian matching with an IoU threshold:
$$\operatorname{IoU}(b_i,b_j^{\star}) \ge \tau_{\mathrm{match}}.$$ This
yields
$$\operatorname{Precision} = \frac{\#\text{matched candidates}}{\#\text{candidate tracks}}$$
$$\operatorname{Recall} = \frac{\#\text{matched ground-truth objects}}{\#\text{ground-truth objects}}$$
and
$$F_1 = \frac{2\operatorname{Precision}\cdot\operatorname{Recall}}{\operatorname{Precision}+\operatorname{Recall}}$$
The most important diagnostic is pool recall:
$$R_{\mathrm{pool}} = \frac{\#\text{ground-truth objects matched by the final candidate pool}}{N^{\star}}$$
This measures whether the active query process is actually discovering
true objects, rather than only adding more proposals. Pool recall should
also be reported after each pass, so that the evaluation can show
whether additional pseudo-exemplar or tiled queries continue to recover
previously missed objects.

## Count-Level Evaluation

For each image, the predicted count $\hat N_j$ is compared with the
ground-truth count $N_j^{\star}$. The main count metrics are
$$\operatorname{MAE} = \frac{1}{M} \sum_{j=1}^{M} |\hat N_j - N_j^{\star}|,$$
$$\operatorname{RMSE} = \sqrt{\frac{1}{M} \sum_{j=1}^{M} (\hat N_j - N_j^{\star})^2},$$
and exact-match accuracy:
$$\operatorname{Exact} = \frac{1}{M} \sum_{j=1}^{M} \mathbf{1}[\hat N_j = N_j^{\star}].$$
The compared zero-shot count estimators are
$$\hat N_{\mathrm{obs}}, \quad \hat N_{\mathrm{supp}}, \quad \hat N_{\mathrm{cons}}.$$
These compare the raw candidate-track count against stricter estimates
that use repeated support, SAM3 score, and stability across active
queries.

## Compute Evaluation

Active perception should also be evaluated by sensing cost. The
evaluation cost is the same model as the reward cost, normalized
relative to one global SAM3 call. A simple normalized form is
$$\begin{aligned}
\operatorname{Cost}
={}&
\#\text{global SAM3 calls}
+
\lambda_T\,\#\text{tile-level SAM3 calls}\\
&+
\lambda_{\mathrm{verify}}\,\#\text{verifier calls}
+
\lambda_V\,\#\text{VLM image/crop inspections}\\
&+
\lambda_O\,\#\text{orchestration decisions}.
\end{aligned}$$ The normalized weights correspond to the reward-section
costs, e.g. $\lambda_T=c_{\mathrm{tile}}/c_{\mathrm{sam}}$,
$\lambda_{\mathrm{verify}}=c_{\mathrm{verify}}/c_{\mathrm{sam}}$,
$\lambda_V=c_{\mathrm{inspect}}/c_{\mathrm{sam}}$, and
$\lambda_O=c_{\mathrm{orch}}/c_{\mathrm{sam}}$. The VLM inspection and
verification costs should be reported rather than assumed negligible,
since visual VLM inference may be comparable to other model calls
depending on the model size and crop resolution.

The final comparison should report both accuracy and cost:
$$(\operatorname{MAE}, \operatorname{RMSE}, \operatorname{Exact}, R_{\mathrm{pool}}, \operatorname{Cost}).$$

A key ablation is to compare the VLM-orchestrated policy against a
non-visual heuristic policy over the same action space. The heuristic
policy uses the same candidate graph, discovery curve, support scores,
tiling status, and budget, but does not receive image or crop
observations $z_t$. This isolates the value of VLM visual inspection
from the value of the action space itself.

The final evaluation should therefore include a comparison of accuracy
versus cost against recent open-vocabulary and detection-based counting
baselines
[@countgd2024; @countgdpp2025; @dave2024; @geco2024; @omnicount2025]
across:

- one-shot SAM3;

- fixed cascade;

- tiled cascade;

- convergence-based stopping;

- non-visual VoI heuristic;

- VLM-orchestrated policy.

This comparison tests whether the VLM provides an actual benefit beyond
hand-designed rules and beyond the existing multi-pass SAM3 cascade.

# Expected Contribution

The expected contribution is a zero-shot active perception framework for
open-vocabulary counting with promptable foundation models. The system
treats SAM3 as a controllable sensor, interprets pseudo-exemplar
propagation and tiling as active sensing actions, represents belief
using candidate tracks and discovery statistics, and uses a VLM to
choose validated sensing actions without allowing the VLM to add
detections directly to the graph.

::: thebibliography
99

R. Bajcsy, "Active perception," *Proceedings of the IEEE*, vol. 76, no.
8, pp. 966--1005, 1988.

R. Bajcsy, Y. Aloimonos, and J. K. Tsotsos, "Revisiting active
perception," *Autonomous Robots*, vol. 42, no. 2, pp. 177--196, 2018.

J. Aloimonos, I. Weiss, and A. Bandyopadhyay, "Active vision,"
*International Journal of Computer Vision*, vol. 1, pp. 333--356, 1988.

L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, "Planning and
acting in partially observable stochastic domains," *Artificial
Intelligence*, vol. 101, no. 1--2, pp. 99--134, 1998.

B. O. Koopman, "The theory of search. III: The optimum distribution of
searching effort," *Operations Research*, vol. 5, no. 5, pp. 613--626,
1957.

A. Kirillov *et al.*, "Segment anything," in *Proc. IEEE/CVF Int. Conf.
Computer Vision (ICCV)*, 2023.

N. Carion *et al.*, "SAM 3: Segment anything with concepts,"
arXiv:2511.16719, 2025.

R. Jiang and S. Shen, "CLIP-Count: Towards text-guided zero-shot object
counting," in *Proc. ACM Int. Conf. Multimedia*, 2023.

N. Amini-Naieni, K. Amini-Naieni, T. Han, and A. Zisserman, "Open-world
text-specified object counting," in *Proc. British Machine Vision Conf.
(BMVC)*, 2023.

N. Amini-Naieni, T. Han, and A. Zisserman, "CountGD: Multi-modal
open-world counting," in *Advances in Neural Information Processing
Systems (NeurIPS)*, 2024.

N. Amini-Naieni and A. Zisserman, "CountGD++: Generalized prompting for
open-world counting," arXiv:2512.23351, 2025.

J. Pelhan, A. Lukežič, V. Zavrtanik, and M. Kristan, "DAVE: A
detect-and-verify paradigm for low-shot counting," in *Proc. IEEE/CVF
Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024.

J. Pelhan, A. Lukežič, V. Zavrtanik, and M. Kristan, "A novel unified
architecture for low-shot counting by detection and segmentation," in
*Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

H. Huang *et al.*, "Point, segment and count: A generalized framework
for object counting," in *Proc. IEEE/CVF Conf. Computer Vision and
Pattern Recognition (CVPR)*, 2024.

A. Mondal, S. Nag, X. Zhu, and A. Dutta, "OmniCount: Multi-label object
counting with semantic-geometric priors," in *Proc. AAAI Conf.
Artificial Intelligence*, 2025.

Q. Jiang, F. Li, Z. Zeng, T. Ren, S. Liu, and L. Zhang, "T-Rex2: Towards
generic object detection via text-visual prompt synergy," in *Proc.
European Conf. Computer Vision (ECCV)*, 2024.

D. Surís, S. Menon, and C. Vondrick, "ViperGPT: Visual inference via
Python execution for reasoning," in *Proc. IEEE/CVF Int. Conf. Computer
Vision (ICCV)*, 2023.

T. Gupta and A. Kembhavi, "Visual programming: Compositional visual
reasoning without training," in *Proc. IEEE/CVF Conf. Computer Vision
and Pattern Recognition (CVPR)*, 2023.

Y. Shen *et al.*, "HuggingGPT: Solving AI tasks with ChatGPT and its
friends in Hugging Face," in *Advances in Neural Information Processing
Systems (NeurIPS)*, 2023.

P. Lu *et al.*, "Chameleon: Plug-and-play compositional reasoning with
large language models," in *Advances in Neural Information Processing
Systems (NeurIPS)*, 2023.

N. Häni, P. Roy, and V. Isler, "MinneApple: A benchmark dataset for
apple detection and segmentation," *IEEE Robotics and Automation
Letters*, vol. 5, no. 2, pp. 852--858, 2020.

A. Chattopadhyay, S. Slocum, B. D. Haeffele, R. Vidal, and D. Geman,
"Interpretable by design: Learning predictors by composing interpretable
queries," *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, vol. 45, no. 6, pp. 7430--7443, 2023.

A. Chattopadhyay, K. H. R. Chan, B. D. Haeffele, D. Geman, and R. Vidal,
"Variational information pursuit for interpretable predictions," in
*Proc. Int. Conf. Learning Representations (ICLR)*, 2023.

K. H. R. Chan, A. Chattopadhyay, B. D. Haeffele, and R. Vidal,
"Variational information pursuit with large language and multimodal
models for interpretable predictions," arXiv:2308.12562, 2023.

Qwen Team, "Qwen3-VL," technical report, Alibaba Group, 2025.
:::
