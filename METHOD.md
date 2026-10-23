# Proposed training operator

This is a proposed categorical vision adaptation of marginal/joint credit assignment. It is not the human paper's fitted model, a demonstrated cognitive mechanism in CNNs, or an established novel algorithm. The source paper's exact excerpts and the distinction between its two adaptive models appear in [SOURCES.md](SOURCES.md).

## Area-preserving shared readout

For a backbone feature map $H\in\mathbb R^{B\times D\times h\times w}$, partition its spatial positions into $K=g^2$ disjoint, nonempty grid regions $R_k$. Define

$$v_k=\frac{1}{|R_k|}\sum_{r\in R_k} H_r,\qquad a_k=\frac{|R_k|}{hw},\qquad u_k=Wv_k+b.$$

One classifier $(W,b)$ is shared across all cues and all comparison arms. The joint logits and categorical predictions are

$$z=\sum_k a_k u_k=W\operatorname{GAP}(H)+b,\qquad p^J=\operatorname{softmax}(z),\qquad p_k^M=\operatorname{softmax}(u_k).$$

With 224px ResNet-18 and a 2×2 grid, the final map is 7×7. Areas are 9/49, 12/49, 12/49, and 16/49. Equal 1/4 weights would change the original global-average inference rule, so the implementation preserves actual areas. The module adds no trainable parameters and inference collapses to the original global-mean linear head. This does not mean the training computation or learned weights are unchanged.

## Detached strategy arbitration

For a one-hot label $e_y$, define unweighted categorical prediction-error vectors

$$\delta_k^M=p_k^M-e_y,\qquad \delta^J=p^J-e_y.$$

Their difference does not require labels:

$$D_k=\frac{\|\delta_k^M-\delta^J\|_2}{\sqrt2}=\frac{\|p_k^M-p^J\|_2}{\sqrt2}\in[0,1].$$

The proposed joint fraction is

$$\gamma_k=\operatorname{sg}\left[\gamma_{\min}+(1-\gamma_{\min})\sigma\left(\frac{D_k-\tau}{T}\right)\right].$$

Defaults are $g=2$, $\tau=0.15$, $T=0.05$, $\gamma_{\min}=0.5$. These are fixed design choices, not parameters estimated from the human experiment or tuned on PACS targets. Large discrepancy favors joint updating because independent local supervision can assign misleading credit. The joint floor limits pressure to make every region independently predict the image label. A constant joint floor is an engineering modification to the cognitive analogy. Probabilities make the discrepancy invariant to independent scalar shifts of each cue's logits. Raw class-logit standard deviation lacks this invariance.

The expected update at the cue-logit interface is

$$\frac{\partial\mathcal L}{\partial u_k}=\frac{a_k}{B}\left[\gamma_k(p^J-e_y)+(1-\gamma_k)(p_k^M-e_y)\right].$$

The factor $1/B$ applies to each sample in a batch-mean loss. Both label errors and both strategies still use supervised training labels; the gate alone is label independent.

## Exact autograd realization

Use stop-gradient $\operatorname{sg}$ and define

$$\widetilde z_k=\operatorname{sg}(z)+u_k-\operatorname{sg}(u_k),$$

$$\mathcal L_{\mathrm{route}}=\frac1B\sum_{n=1}^B\sum_k a_k\left[\gamma_{nk}\operatorname{CE}(\widetilde z_{nk},y_n)+(1-\gamma_{nk})\operatorname{CE}(u_{nk},y_n)\right].$$

Numerically $\widetilde z_k=z$, but its gradient reaches only $u_k$. This realizes the stated per-cue update exactly. A single weighted global CE would distribute joint gradients with an averaged gate and implement a different operator. The gate, comparison probabilities, and reference joint logits are detached; otherwise additional derivatives would change the update. Loss values are useful for diagnostics but do not define an ordinary smooth global objective under re-evaluation of all detached quantities. A naive black-box finite-difference check of that re-evaluated scalar is not the relevant gradient test. Tests compare autograd to the analytical operator and verify the all-joint reduction through shared features and parameters.

There is no guarantee that the method monotonically reduces joint CE, particularly with a shared trainable backbone and classifier. The proposed rule controls local update signals rather than proving a descent theorem.

## Classification implementation

The classification runner uses an equivalent gradient implementation:

$$\mathcal L_{\mathrm{class}}=\operatorname{CE}(z,y)+\frac1B\sum_{n,k}a_k(1-\gamma_{nk})\operatorname{KL}(\operatorname{sg}(p_n^J)\Vert p_{nk}^M).$$

Both the teacher and gate are detached. Its derivative at each cue logit is

$$\frac{a_k}{B}\left[(p^J-e_y)+(1-\gamma_k)(p_k^M-p^J)\right]=\frac{a_k}{B}\left[\gamma_k(p^J-e_y)+(1-\gamma_k)(p_k^M-e_y)\right].$$

Thus it realizes the proposed cue-specific update, including shared backbone and classifier gradients. Its scalar loss differs from the surrogate above. Compare ERM and CUE using the same evaluation CE/accuracy; comparing their total training loss values is not a method comparison. Independent NumPy tests calculate probabilities, gates, cue gradients, and shared feature/head derivatives without calling the routing reference implementation.

## Relation to established regularization

Let $\mathcal L_M=\sum_k a_k\operatorname{CE}(u_k,y)$ and $\mathcal L_J=\operatorname{CE}(z,y)$. Exactly,

$$\mathcal L_M-\mathcal L_J=\sum_k a_k\operatorname{KL}(p^J\|p_k^M).$$

For an image-common detached gate $\gamma$, the gradient matches that of $\mathcal L_J+(1-\gamma)\sum_k a_k\operatorname{KL}(p^J\|p_k^M)$ with the gate held fixed and the probabilities differentiated normally. Consequently, marginal/global mixing alone is insufficient to establish novelty. Cue-specific routing is the operator being investigated, and still needs a broader functional prior-art audit before a novelty claim. The current package establishes neither novelty nor efficacy.

## CNN and ViT integration

The readout accepts BCHW feature maps or BND patch tokens plus an explicit `(height, width)` token grid. Remove CLS and other special tokens explicitly. For a ViT, the proposed joint rule is mean patch pooling followed by a shared linear classifier. It does not preserve an existing CLS-head model's inference function. Replacing CLS with mean pooling must be applied to ERM and RSC controls too; a comparison that changes pooling only for cue routing is confounded.

```python
from cue_routing.models import CueReadout
from cue_routing.routing import routed_loss

head = CueReadout(feature_dim=768, num_classes=7, grid_size=2)
cue_logits, areas = head.cue_logits(patch_tokens, token_grid=(14, 14))
loss, joint_logits, diagnostics = routed_loss(cue_logits, labels, areas)
loss.backward()
```

Place the head on the same device as the backbone. Use the same initialization and optimizer for all controls. This example assumes patch tokens already exclude special tokens. It specifies a generic interface, not a completed pretrained ViT recipe.

## Falsifiable expectations and limits

The primary question is whether cue routing improves held-out-domain accuracy over RSC under the matched recipe. ERM checks whether either training intervention improves on ordinary training. Failure to beat ERM is a negative result even if cue routing beats RSC. A gain over RSC alone cannot establish state-of-the-art domain generalization.

The feature regions are overlapping receptive-field representations of an image, not experimentally independent cues. Final ViT tokens mix global context even more strongly. Background-only regions can receive incorrect local supervision, spatial grids can break object parts, and shallow probability disagreement can reflect calibration or classifier scale rather than causal evidence conflict. A learned backbone may increase gate values to evade local pressure even with detached gates. Log joint fraction, disagreement, and both losses; a saturated gate makes the adaptive-routing interpretation weak.

Optional source-configured routing modes are `image` (area-weighted RMS disagreement shared across regions), `fixed` (constant joint fraction), and `shuffled` (a one-position cyclic permutation of region gates). The last mode is deterministic permutation, not a random shuffle. Set `fixed_joint=1` to recover ERM gradients; set 0 for pure marginal training. These are ablations within the proposal, not extra named competitors. They are not part of the default 36-run matrix, and their target results must not be used to retroactively select the primary method.
