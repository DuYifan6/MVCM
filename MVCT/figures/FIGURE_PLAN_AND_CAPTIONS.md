# IJML figure plan and visual specification

## Figure sequence

The recommended main-text sequence is:

1. **Fig. 1 — Overall MVCT architecture.** A low-saturation schematic-led composite that separates offline MVC construction from online joint detection.
2. **Fig. 2 — From view-level consistency to token-level attention bias.** A compact mechanism schematic using the same channel colors as Fig. 1.
3. **Fig. 3 — Predictive performance and contribution of the consistency prior.** A quantitative grid combining the primary Macro-F1 comparison, paired test-set effects, and channel ablation.
4. **Fig. 4 — Subgroup behavior and error-balance boundary.** A two-panel conditional-accuracy figure with Wilson intervals for HR, MR, HF, and MF.
5. **Fig. 5 — Efficiency and resource cost.** A three-panel figure showing warm-cache latency, peak allocated GPU memory, and the performance–latency trade-off.
6. **Fig. 6 — Auditable illustrative case.** A three-panel descriptive analysis linking the active view-relation matrices, learned channel weights, and comparator predictions for the selected article. The article text is reported separately in the Results or supplementary material.

The three-seed validation comparison between the four-channel and three-channel variants should remain a compact table or supplementary figure. With only three validation runs, it is architecture-selection evidence rather than a main performance claim. A confusion matrix is also not recommended for the main text because the subgroup figure already exposes the practically relevant error asymmetry more directly.

## Unified visual system

- Canvas: white; no gradients, drop shadows, or decorative outer frames.
- Font: Arial, with Helvetica and DejaVu Sans fallbacks.
- Final width: 180 mm for all five main figures; use 86 mm only for a genuinely simple single-column figure.
- Text: 7.2 pt body/ticks, 7.6 pt panel titles, and 8.2 pt bold lowercase panel labels.
- Method colors: MVCT `#1F5A85`; w/o MVCM `#88B4D1`; RoBERTa `#4F5B66`; ELECTRA-base `#5F7F71`; remaining baselines in neutral greys.
- Task colors: Fake/Real `#1F5A85`; AI/Human `#D27A32`.
- Channel colors in schematics: semantic `#3B78A5`; logical `#7356A8`; local `#2A8C82`; fused bias `#C7772A`.
- Legends: one shared frameless legend when possible; direct labels for stable identities; canonical model names preserved.
- Axes: top and right spines removed; only light reference grids; percentages shown explicitly.
- Export: editable SVG and PDF plus 600 dpi PNG and TIFF. IJML requires at least 300 dpi and figures embedded close to first citation.

## Manuscript-ready English captions

**Fig. 1. Overall architecture of the final three-channel MVCT framework.** During offline consistency construction, title, body, and description are compared along semantic, logical, and local dimensions, producing three cached 3 × 3 view-pair matrices. During online joint detection, masked-softmax channel weights fuse these matrices into a view-level consistency bias. A pretrained BERT encoder represents the concatenated views, and view identities expand the fused bias to token pairs. Two consistency-aware Transformer layers inject the token-pair bias into self-attention. The shared [CLS] representation is then passed to separate Fake/Real and AI/Human classification heads.

**Fig. 2. Conversion of multi-view consistency into token-level attention guidance.** The semantic, logical, and local matrices are combined by learnable masked-softmax weights and centered to form a signed 3 × 3 view-level bias. View identifiers then expand each view-pair entry to the corresponding token pairs, while special and padding positions receive zero bias. The resulting matrix is added to the scaled dot-product attention logits with learnable strength λ before the softmax operation. Positive values strengthen and negative values suppress the corresponding cross-view attention links.

**Fig. 3. Predictive performance and contribution of the consistency prior.** (a, b) Locked test-set Macro-F1 for Fake/Real and AI/Human detection, respectively, including the post-hoc input-compatible ELECTRA-base comparator. Values are from one deterministic model run per method and are not multi-seed means. (c) Paired Macro-F1 effects of MVCT relative to RoBERTa and the architecture-matched model without MVCM. Points denote differences in percentage points, and horizontal lines denote sample-level paired 95% confidence intervals; ELECTRA-base is not included because it was not part of this pre-specified paired comparison family. (d) Decrease in Macro-F1 after removing each active consistency channel or the complete MVCM. The ablation values are descriptive results from the locked model and do not estimate variation across training seeds.

**Fig. 4. Conditional subgroup performance across article type and provenance.** Conditional correctness for (a) veracity and (b) provenance classification in the human-written real (HR), machine-generated real (MR), human-written fake (HF), and machine-generated fake (MF) subgroups. Points denote subgroup accuracies and error bars denote Wilson 95% confidence intervals for each proportion. The intervals are not paired confidence intervals for differences between models, and no subgroup hypothesis tests were performed.

**Fig. 5. Warm-cache efficiency and resource cost.** (a) Median and 95th-percentile latency per article for batch sizes 1 and 8; circles denote medians and terminal ticks denote 95th percentiles. (b) Peak allocated GPU memory. (c) Fake/Real Macro-F1 versus median latency, with marker shape indicating batch size. Measurements were obtained on one NVIDIA GeForce RTX 4080 SUPER after 20 warm-up batches, using five repetitions of 100 measured batches per condition. Timing includes batch retrieval, tokenization and cache reading, host-to-device transfer, and the frozen forward pass; model loading is excluded. Offline MVC construction required an observed 6.85 h for 19,378 newly built and 100 reused samples. Because 100 samples were reused, this value is not a strict cold-start estimate.

**Fig. 6. Internal consistency patterns and prediction shift for an illustrative HF case.** The human-written fake-news test article (sample 3894) was selected post hoc from 61 candidates satisfying the strict correctness criterion. Selection used a deterministic median-gain and article-length rule and did not use consistency matrices, attention patterns, or visual appearance. (a) Extracted semantic, logical, and local view-relation matrices and the resulting fused signed view bias for title (T), description (D), and body (B). (b) Masked-softmax weights of the three active channels; the factual channel is inactive in the final model and is therefore not shown. (c) Fake-class probabilities and predicted veracity labels from RoBERTa, the architecture-matched variant without MVCM, and MVCT; the dashed line marks the fixed Fake/Real decision threshold of 0.55. MVCT correctly predicts Fake with probability 0.621, whereas the two controls predict Real with Fake-class probabilities of 0.406 and 0.441, respectively. The matrices and weights are descriptive internal states and do not establish attention faithfulness or a uniquely identified causal mechanism. Values are from one frozen illustrative test case, not repeated experimental estimates. Source data are provided with the figure.

## Source-data mapping

- Fig. 3a–b: main locked test performance table plus the audited post-hoc ELECTRA locked-test result.
- Fig. 3c: paired test-prediction analysis with Holm-adjusted inference reported in the manuscript table.
- Fig. 3d: strict three-channel ablation table.
- Fig. 4: verified subgroup result file; all 1,948 test articles are represented.
- Fig. 5: clean fixed-thread efficiency rerun; the earlier preliminary benchmark is not used.
- Fig. 6: frozen representative-case selection and read-only diagnostic evidence JSON files; all plotted values are also exported to `Fig6_representative_case_source_data.csv`.
