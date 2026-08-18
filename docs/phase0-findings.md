# Phase 0 Findings — Multilingual Post-Training Safety Evaluation Framework

Status: **Phase 0 complete, pending user approval to proceed to Phase 1.**
Date: 2026-08-18
Scope: resource, checkpoint, benchmark, and methodology validation only. No evaluation code was written.

Verification method note: findings below are marked either **verified** (confirmed
directly against a primary source — HF Hub API, HF datasets-server API, a fetched
dataset/model card, or a fetched GitHub/paper page) or **unconfirmed** (surfaced by
search but not independently fetched). Nothing in the "verified" category rests on
a search-result snippet alone. Two research forks did the bulk of the checkpoint
and benchmark investigation; their most consequential claims (the final RLVR
checkpoint's config/license, and PolyGuardPrompts' Korean coverage) were
independently re-verified directly in this session, and one claim (an XSTest HF
mirror) was checked and found to be wrong, then corrected — see §2.

---

## 1. Final checkpoint lineage

**Two complete, independently verified 4-stage lineages exist** for OLMo 3 7B: `Instruct` and `Think`. Both share the same base weights (`allenai/Olmo-3-1025-7B`), the same architecture (`Olmo3ForCausalLM`, 7.298B params), and the same license (`apache-2.0`) at every stage. Neither lineage was assumed valid from naming — every stage transition was confirmed via HuggingFace's own `base_model` / `base_model:finetune` tags, which HF derives from each model card's declared `base_model:` field, not from repo-name pattern matching.

**Recommendation: use the `Instruct` lineage for the MVP.**

| Stage | HF Model ID | Revision | `base_model` (HF-derived tag) | Verified via |
|---|---|---|---|---|
| Base | `allenai/Olmo-3-1025-7B` | `main` | — | HF API + `config.json` |
| SFT | `allenai/Olmo-3-7B-Instruct-SFT` | `main` | `allenai/Olmo-3-1025-7B` | HF API tag |
| DPO | `allenai/Olmo-3-7B-Instruct-DPO` | `main` | `allenai/Olmo-3-7B-Instruct-SFT` | HF API tag |
| RLVR (final) | `allenai/Olmo-3-7B-Instruct` | `main` | `allenai/Olmo-3-7B-Instruct-DPO` | HF API tag + AI2's own README table, which literally labels this row **"Final Models (RLVR)"** |

Also verified directly in this session (not just by the fork):
- `allenai/Olmo-3-7B-Instruct`'s `config.json`: `architectures: ["Olmo3ForCausalLM"]`, `dtype: bfloat16`, sharded into 3 safetensors files, mixed sliding/full attention pattern.
- `chat_template.jinja`: ChatML-style (`<|im_start|>`/`<|im_end|>`). **Important prompting detail**: when no system message is supplied, the template silently injects a default system prompt framing the model as *"a helpful function-calling AI assistant"* — even when no tools are passed. This must be applied identically (or deliberately overridden and documented identically) across all four stages and both languages, or it becomes an uncontrolled confound.
- The base model (`allenai/Olmo-3-1025-7B`) has 1,487 branches (pretraining checkpoints at `stage1-step*`, `stage2-*`, `stage3-step*`); `main` is the correct "Base" representative and is exactly what the SFT model's `base_model` tag points to.
- A minor upstream data-quality note: `Olmo-3-7B-Instruct-DPO`'s model card carries the tag `dataset:allenai/Dolci-Think-DPO-7B` — almost certainly a copy-paste error in AI2's card (should likely read `Dolci-Instruct-DPO-7B`). Confirmed present via direct API check. This does **not** affect checkpoint identity — the `base_model` chain (the authoritative link) is internally consistent — but it's documented here so it isn't mistaken for our own error later, and as a reminder that upstream metadata isn't infallible even from a careful lab.

**Why `Instruct`, not `Think`:** both lineages are equally complete and verifiable. `Think`'s chat template forces a `<think>...</think>` reasoning block before every answer. That's a genuine confound for this project's purposes: a stage-over-stage behavioural shift could reflect reasoning-trace length/content changes rather than the post-training stage itself, and every refusal/compliance label would first require parsing out the CoT. `Instruct` gives directly parseable single-turn output. `Think` is recorded as a verified, viable could-have extension — arguably a *better* fit for any future instrumental-convergence-style question, since that construct is closer to what a reasoning/agentic model does — but not the MVP default.

Machine-readable record: `configs/models/olmo3_lineages.yaml` (includes both lineages, verification notes, and the MLX quantization findings from §8).

---

## 2. Benchmark bake-off

Evaluated InstrumentalEval plus five alternatives (XSTest, WildGuardMix, Aya Red-teaming, Do-Not-Answer, and PolyGuardPrompts, the last added after it turned up as the only candidate with confirmed native-field Korean coverage). Every row below reflects a fetched primary source (dataset card, GitHub repo, or arXiv HTML) unless flagged.

| Dimension | **InstrumentalEval** | **XSTest** | **WildGuardMix** | **Aya Red-teaming** | **Do-Not-Answer** | **PolyGuardPrompts** |
|---|---|---|---|---|---|---|
| Construct | Instrumental convergence (shutdown evasion, self-replication, deception, etc.) under adversarial prompt steering | Exaggerated refusal / over-refusal vs. genuine harm refusal | Prompt harm / response harm / refusal (general moderation) | Culturally-grounded harm elicitation, 10 categories | "Should-refuse" harmful-question taxonomy, 61 harm types | Prompt harm / response harm / refusal (general moderation) |
| Relevance to RLVR specifically | Direct — the paper's own finding is that RL-trained models show *higher* instrumental-convergence rates than RLHF models | Indirect | Indirect | Indirect | Indirect | Indirect |
| Relevance to general safety | Narrow (one agentic-misalignment subtype) | Yes — directly tests the helpfulness/harmlessness tradeoff | Yes — canonical general construct | Yes | Yes | Yes |
| Size | 76 items (6 categories, 10–20 each) | 450 (250 safe incl. scary-sounding-but-benign + 200 unsafe) | 86.7K train / 1.7K test | 7,419 (~900/language) | 939 | 29,300 |
| Scoring | **LLM-judge only**, no rule-based fallback in the paper | Not shipped — commonly rule-based refusal-pattern matching in downstream work | GPT-4 labels + human audit; 3-annotator human labels on test (Fleiss κ 0.50–0.72) | Human-annotated at creation, no built-in scorer | Fine-tuned 600M classifier (English-only) | **Ships its own multilingual PolyGuard classifier**, validated across all 17 languages |
| Code/data availability | GitHub, **CC BY 4.0** | GitHub (original) + several unverified HF mirrors, **CC BY 4.0** | HF, **odc-by**, gated behind AI2 Responsible Use acceptance | HF, **Apache 2.0** | Data CC BY-NC-SA 4.0, code Apache 2.0 | HF, **CC BY 4.0**, ungated |
| Applied to open-weight models before | **No** — only o1/o3-mini/Gemini-Thinking/DeepSeek-R1 vs. GPT-4o/Claude-3.5; never an open-weight model, never across one model family's stages | Yes, widely | Yes — AI2's own dataset | Not verified | Yes, widely | Not yet common, but purpose-built for exactly this use |
| English coverage | Native | Native | Native | Native | Native | Native |
| **Korean coverage** | **None** | **None** — requires translation | **None** | **None** (8 languages: EN/Hindi/French/Spanish/Russian/Arabic/Serbian/Filipino — Korean not included) | **None** (only a Chinese localization exists) | **Yes** — `ko` is one of 17 languages, **confirmed directly** via the dataset card and cross-checked live via the HF datasets-server API |
| Korean provenance | N/A | N/A (self-translate required) | N/A | N/A (disqualified on coverage) | N/A | Mixed sourcing per the card ("naturally occurring" + "human-verified MT," reported mean quality ~81/100 across 16 non-English languages) — **Korean-specific provenance not confirmed from the card alone; open item for Phase 1** |
| Translation risk if self-translated | High — multi-sentence agentic scenarios, framing easily distorted | Low — short single-sentence templates, easiest of the self-translate candidates | Low–medium | N/A | Medium | N/A — already exists |
| Compute (item count) | Low (76 × 4 × 2 = 608 generations) | Low (450 × 4 × 2 = 3,600, subsampled if needed) | High unless subsampled | N/A | Low–medium | Low if subsampled (recommended, given 29.3K total) |
| Statistical limitation | Very small per-category N (10–20/category) — category-level claims will be noisy | Small but balanced | Large enough if subsampled well | N/A | Moderate | Manageable if subsampled |

### Recommendation

- **Primary MVP benchmark: PolyGuardPrompts** (`ToxicityPrompts/PolyGuardPrompts`), filtered to English and Korean. It's the only candidate with confirmed, licensed, ready-to-use Korean items rather than a translation protocol invented from zero; it measures the general harm/refusal construct the research question is actually about; it ships its own multilingual classifier as scorer, which is a materially better mitigation against "does the judge even understand Korean" than a generic LLM-judge would be; it's CC BY 4.0 and ungated; and at 29.3K items it's comfortably subsample-able to whatever the compute budget in §8 allows.
- **Secondary benchmark: XSTest**, self-translated to Korean under the protocol in §4. It adds something PolyGuardPrompts doesn't test on its own: **over-refusal**. Without it, a rise in refusal rate post-RLVR reads as "safer" by default with no check on whether it's genuine harm-avoidance or blanket caution.
- **Rejected: InstrumentalEval** — construct mismatch with "safety-relevant behaviour" as framed (it's a narrow agentic-misalignment subtype, not general harm refusal); English-only with zero multilingual precedent; never validated on an open-weight model or across one model's post-training stages; LLM-judge-only scoring. Retained as a possible **English-only supplementary probe for the RLVR stage specifically** in a later phase — its own headline finding (RL training raises instrumental-convergence rates) is directly on-topic for this project's interest in RLVR — but out of MVP scope and out of any multilingual claim.
- **Rejected: WildGuardMix** — no Korean; otherwise a strong in-family (AI2) fit. Candidate for a later English-only cross-check against PolyGuardPrompts' English subset.
- **Rejected: Aya Red-teaming** — genuinely multilingual with *native* (not translated) items in 8 languages, methodologically the gold standard for provenance — but Korean isn't one of the 8. Disqualified on coverage, not quality.
- **Rejected: Do-Not-Answer** — no Korean version (only Chinese); English-only scorer.

**Correction made during my own verification pass:** the benchmark fork's report referenced `natolambert/xstest-v2-copy` as a candidate HF mirror for XSTest. I checked it directly via the datasets-server API: it is **not** the raw 450-prompt set — it's 2,700 rows of prompt+completion+human-annotation data (an annotated-responses artifact from a different downstream project). No HF mirror was confirmed in Phase 0 to be a faithful copy of just the 450 XSTest v2 prompts. **Phase 1 must pull the prompt set from the original GitHub source (`github.com/paul-rottger/xstest`) or verify a specific HF mirror against it row-for-row before use.** This is flagged in `configs/benchmarks/registry.yaml` and is a concrete example of why every "found by search" ID in this project needs its own direct check rather than trust-by-association.

Machine-readable record: `configs/benchmarks/registry.yaml`.

---

## 3. Final research question

**Recommended framing (broad, but precisely scoped):**

> How do SFT, DPO, and RLVR post-training stages change OLMo 3 7B's harmful-request refusal and over-refusal behaviour (as measured by PolyGuardPrompts and XSTest), and do these behavioural changes transfer consistently between English and Korean once general Korean task-competence and response-validity are controlled for?

**Broad vs. narrow, evaluated against the evidence:**

- *Broad ("safety-relevant behaviour across languages")* — defensible **only because** the benchmark choice (PolyGuardPrompts + XSTest) genuinely covers a general harm-refusal/over-refusal construct with real, licensed Korean grounding. Had InstrumentalEval been selected instead, this framing would **not** have been defensible.
- *Narrow ("instrumental-convergence-related behaviour")* — the necessary framing **if and only if** InstrumentalEval were the primary benchmark, and even then would have to stay English-only pending a validated Korean translation with no existing precedent to build on. Not needed given the benchmark choice above; noted as the correct framing for a possible future English-only RLVR-focused supplementary study.

The recommended wording avoids the unbounded word "safety" (which could otherwise imply bias, toxicity generation, hallucination, or other constructs this project isn't testing) in favor of naming the two specific behaviours actually measured: refusal and over-refusal. This is a construct-validity choice, not a hedge — it's the version of the claim the evidence gathered in Phase 0 actually supports.

---

## 4. Korean evaluation strategy

**PolyGuardPrompts**: native `language` field with `ko` as one of 17 values, CC BY 4.0. Confirmed directly via dataset card fetch and cross-checked live via the HF datasets-server API. Open items for Phase 1 (not yet resolved, do not assume clean): exact per-language item count for `ko` (the 29,300 total isn't yet broken down by language), harmful/unharmful label balance within the Korean subset, and — most importantly — reading the underlying PolyGuard paper directly for a Korean-specific translation-quality figure, since the dataset card's mixed-sourcing description ("naturally occurring" + "human-verified MT," ~81/100 mean quality across 16 non-English languages) was not confirmed as applying to Korean specifically.

**XSTest**: no existing Korean version. Minimal but defensible translation protocol for the Phase 1 self-translation:

1. **Source**: pull the canonical 450-item English set directly from `github.com/paul-rottger/xstest` (not an unverified HF mirror — see §2 correction).
2. **Translator**: professional or highest-quality-available translation, never raw/unreviewed MT, given these are short single-sentence templates where a native speaker can efficiently produce and check the whole translated set.
3. **Semantic equivalence & behavioural-framing preservation**: the translator must be briefed on XSTest's actual test logic (a "safe" item is deliberately worded to *sound* alarming while being benign — e.g. "how do I kill a process" — and the test measures whether a model over-refuses it). A literal translation is not sufficient if it loses that engineered ambiguity.
4. **Concrete translation-construct risk to flag explicitly**: some XSTest items work only because of an English-specific lexical double-meaning (kill/process, execute/program) that may have no Korean equivalent. Where the ambiguity doesn't survive translation, the item stops testing over-refusal and starts testing something else entirely. **Do not force a translation in these cases** — exclude the item and log the exclusion with a reason in the translation-provenance file, rather than silently keeping a broken item in the set.
5. **Answer-choice / scoring-logic preservation**: XSTest's scoring is refusal-vs-compliance, not multiple-choice, so this reduces mainly to: the translated prompt must still map cleanly to the same safe/unsafe ground-truth label used for scoring.
6. **Parallel IDs**: every translated item keeps a `parallel_item_id` linking it 1:1 to its English source, propagated through `BenchmarkItem` in the eventual implementation.
7. **Human spot-check**: for the MVP, translate a stratified subsample (~100–150 items across all 10 XSTest categories, not the full 450) rather than the whole set — full manual review of 450 items is not warranted before Phase 1/2 have even run once. A native Korean speaker reviews 100% of the translated subsample for naturalness and construct preservation before first use.
8. **Provenance documentation**: record translator identity/method, date, and protocol version in a small provenance file alongside the translated set — this is what makes "translation sensitivity" an auditable methods note instead of an invisible assumption.

---

## 5. Language-competence control

**Concrete MVP proposal: Belebele Korean (`facebook/belebele`, config `kor_Hang`)**, verified directly via the HF datasets-server API: **900 items**, 4-way multiple-choice reading comprehension over FLORES-200 passages, CC BY-SA-4.0. Every language config (122 total, including `kor_Hang`) is a professional translation of the same underlying English passages with shared `question_number`/`link` fields — meaning genuine parallel EN–KO items already exist with zero translation work required from this project.

- **Subsample**: ~40–50 items, stratified, paired to their English counterpart via `question_number`. Adds only ~50 × 4 checkpoints × 2 languages = 400 generations — small relative to the main benchmark budget.
- **Scoring**: exact-match on `correct_answer_num`. Fully deterministic, zero judge dependency, trivial to automate.
- **Gating rule** (this is the actual mechanism that answers "post-training behavioural divergence vs. the model just understanding Korean less well," not a separate feature bolted on): for any observed stage-transition effect on a Tier-2 safety metric in Korean, first check whether the Belebele-KO competence score *also* moved significantly at that same stage relative to its English counterpart and relative to other stages. If competence dropped at the same point, the safety-metric shift **must be reported as confounded**, not as a clean finding. If competence stayed flat across all four stages while a safety metric moved in Korean but not English, that's the genuinely interesting result the whole project is built to surface.

---

## 6. Scoring strategy

Hierarchy, cheapest/most-deterministic first:

1. **Rule-based / deterministic** (used wherever valid):
   - Parseability / valid-response check — non-empty, schema-conforming output
   - Language-match check — KO prompt → KO-script response (regex/langid)
   - Belebele-KO control — exact-match on `correct_answer_num`
   - XSTest refusal/compliance — rule-based refusal-phrase pattern matching (documented pattern list per language), the standard approach in XSTest follow-up literature
2. **Benchmark-native scoring**:
   - PolyGuardPrompts — its own shipped multilingual PolyGuard classifier, used as primary scorer for prompt/response harm and refusal labels. This is the single best available mitigation against "the judge doesn't understand Korean," since it's a purpose-built classifier validated across all 17 covered languages rather than a general-purpose LLM asked to judge in a language it may handle unevenly.
3. **LLM-as-judge — fallback only**, used to (a) validate the rule-based XSTest refusal classifier on a subsample, or (b) resolve cases the PolyGuard classifier or rule-based patterns flag as low-confidence. If invoked:
   - **Judge model**: a strong instruction-following model with independently confirmed Korean competence (e.g., a current Claude or GPT-4-class model). Note explicitly: this is a paid external API call, a different cost/dependency category from the rented-GPU eval compute in §8, and needs separate sign-off before it's wired in.
   - **Structured output**: forced JSON schema (Pydantic-validated at the code level once implemented) — label + one-sentence rationale, no free text.
   - **Korean-specific validation**: never assume English-validated judge quality transfers to Korean. Validate on a human-labeled subset **per language**, not just once overall.
   - **Validation subset size**: ~50 items per language (100 total), human-double-checked against judge output, reporting agreement (e.g. Cohen's κ) before any judge-derived number is used in a reported result.

No LLM judge is used unless rule-based/benchmark-native scoring proves insufficient on a validation pass — this is a default-off, not default-on, component.

---

## 7. Statistical scope

**Required for MVP** (all cheap — computed post-hoc from stored per-item scores, no extra generation cost):
- Raw proportions per (checkpoint × language) for every Tier-1 and Tier-2 metric
- Stage-over-stage deltas (Base→SFT, SFT→DPO, DPO→RLVR) per language
- Cross-lingual delta-of-deltas (how differently a stage transition moved the metric in EN vs. KO)
- Bootstrap confidence intervals on every proportion and delta — essential given small-to-moderate N (a 100–150-item XSTest-KO subsample, a PolyGuard subsample, 40–50 Belebele-KO items); without CIs, "regression" claims will overstate what the sample size actually supports
- Paired significance testing (e.g. McNemar's test) **within a single language**, across stage transitions on the same items — the design is naturally paired here (same item, same language, different checkpoint), so this is nearly free once bootstrap infrastructure exists

**Explicitly NOT required for MVP, and explicitly NOT valid, for cross-lingual comparisons**: a formal paired significance test comparing an English item to its Korean counterpart. Translated/parallel items are not the *same* measurement instrument — EN and KO versions of "the same" item are related but not identical text — so cross-lingual comparison stays at the level of descriptive deltas with overlapping/non-overlapping bootstrap CIs, not a claimed paired statistical test.

**Valuable later extension, not MVP:**
- Multiple seeds / repeated sampling at temperature > 0, to measure generation-level variance — multiplies compute cost by the sample count, deferred until budget is confirmed after one full run
- Any causal framing — this design is **observational**: it compares checkpoints AI2 already trained, not a controlled ablation this project runs itself. Findings must be stated as "associated with" / "co-occurring with" a post-training stage transition, never "caused by," since confounds in what changed between AI2's stages beyond the stage label itself (e.g. data composition differences unrelated to safety) can't be ruled out by this design.

---

## 8. Compute strategy

Development machine: MacBook, Apple M4, **16GB unified memory, 47GB free disk, no cached HF token.** A single bf16 7B checkpoint is ~14GB of weights alone — doesn't comfortably fit for inference in 16GB total memory alongside macOS and KV cache, and four such checkpoints (~56GB) exceed the 47GB free disk. This was flagged as a likely hard blocker in the initial design conversation; the findings below refine rather than reverse that.

| Option | Feasibility for OLMo 3 7B (4 stages) | Memory/VRAM | Setup complexity | Reproducibility | Rough cost | Verdict |
|---|---|---|---|---|---|---|
| **A. Cloud GPU rental** (RunPod/Lambda/etc.) | High — single L4 (24GB) or A100 (80GB) comfortably holds a 14GB bf16 checkpoint + KV cache for the item counts here | 24–80GB | Low–medium (rent, pull image, run) | High — same container image reproducible on demand | **Verified pricing (2026)**: RunPod L4 ≈ $0.39/hr, RunPod A100 80GB ≈ $1.19–1.39/hr on-demand. Full MVP matrix (4 checkpoints × 2 languages × ~1,000–1,500 short-prompt/short-response generations across the PolyGuard subsample + XSTest-KO subsample + Belebele-KO control) is plausibly a few GPU-hours — likely **single-digit to low-double-digit dollars total** | **Recommended for real research runs** |
| **B. Hosted inference endpoint** | Low/unconfirmed — no evidence any hosted provider serves the two *intermediate* stage checkpoints (Instruct-SFT, Instruct-DPO); at most the final Instruct model might be hosted somewhere, covering only 1 of 4 required stages | N/A | N/A | N/A | N/A | **Rejected** — doesn't cover the full stage matrix |
| **C. Local quantized inference (MLX)** | Partial — **confirmed** 8-bit MLX conversions exist for the Base (`mlx-community/Olmo-3-1025-7B-8bit`, ~7.75GB) and final RLVR (`mlx-community/Olmo-3-7B-Instruct-8bit`) checkpoints, which fit the M4's 16GB comfortably. MLX conversions for the two **intermediate** checkpoints (Instruct-SFT, Instruct-DPO) are **unconfirmed** — not checked for existence in Phase 0 | ~8GB (8-bit) | Low (mlx-lm is straightforward) | Medium — quantized output ≠ full-precision output; must not be mixed with cloud bf16 numbers in the same comparison | Free (local compute) | **Recommended for local dev/plumbing only**, and only for the two stages with confirmed MLX builds |

**Recommendation:**
- **Real research runs**: rented cloud GPU (a single L4 or A100 via RunPod), full bf16 precision, one documented instance type/image, all four stages run identically. This is the authoritative number source for anything reported as a finding.
- **Local development/testing on the M4**: code and pipeline correctness only, against a **tiny stub HF model** (not real OLMo weights) for automated unit/integration tests — this keeps CI fast, free, and independent of Hub/network availability. The confirmed MLX 8-bit builds of Base and RLVR-final may be used for **optional manual smoke-testing** with real weights on two of the four stages, but are not part of the automated test suite, and any number produced this way must be labeled as quantized and never merged into a table of full-precision cloud results.

**Explicitly rejected**: any assumption that this project can run its authoritative evaluation entirely on the local M4. It cannot, for the reasons above — this needs to be a conscious, approved decision (cloud spend, however small) before Phase 2 begins.

---

## 9. Known limitations

- Small-to-moderate sample sizes (XSTest-KO subsample, PolyGuard subsample) limit statistical power; confidence intervals will be wide, and category-level (rather than benchmark-level) claims will be especially noisy.
- PolyGuardPrompts' Korean provenance (native vs. machine-translated, per-language quality score) is not fully confirmed from the dataset card alone — flagged for a Phase 1 close read of the source paper before treating EN–KO deltas on this benchmark as fully clean.
- XSTest-KO is a self-produced translation (Phase 1 task under the §4 protocol), introducing translator-dependent variance despite the documented process, and necessarily excludes any item whose over-refusal test depended on an English-only lexical ambiguity.
- The overall design is **observational**, comparing AI2's already-released checkpoints — not a controlled ablation this project runs. No causal claims are supportable; see §7.
- Two benchmark families (general harm-refusal/over-refusal) — findings do not generalize to other safety-relevant constructs (bias, toxicity generation, instrumental convergence, hallucination) without additional benchmarks explicitly added later.
- Two languages only (EN, KO) — this design cannot distinguish "Korean-specific" effects from "any non-English language" effects; a third language would be required for that and is explicitly out of MVP scope.
- The local development machine cannot run the authoritative evaluation (see §8) — the compute recommendation depends on an external paid cloud GPU rental that needs explicit user approval before Phase 2, not just Phase 0 sign-off.
- No HF token is currently cached on the dev machine; several of the repos above are ungated public repos so this hasn't blocked verification, but should be set up before Phase 1 (some AI2 datasets in the broader ecosystem, e.g. WildGuardMix, are gated behind license acceptance even though the ones selected here are not).

---

## 10. Explicit assumptions we must NOT make

- Do **not** assume a checkpoint name implies a valid training trajectory without verifying the `base_model` chain — this was the actual failure mode this project needed to avoid, and the verification method above must remain policy for any future checkpoint additions.
- Do **not** assume PolyGuardPrompts' Korean items are native just because the dataset markets itself as multilingual — treat as translated-with-partial-provenance until the Phase 1 paper read confirms otherwise.
- Do **not** assume a rise in refusal rate equals "safer" without checking the XSTest over-refusal control in parallel.
- Do **not** assume a behavioural metric difference between EN and KO reflects a genuine post-training effect without first checking that the Belebele-KO competence control didn't also move at that same stage/language (§5's gating rule).
- Do **not** assume a single-seed, temperature-0 result is a stable estimate — always pair with the bootstrap CI (§7) before calling something a finding.
- Do **not** assume findings from the `Think` lineage transfer to `Instruct`-lineage claims or vice versa — they are separate, non-interchangeable trajectories, both verified but only one selected for MVP claims.
- Do **not** treat LLM-judge output as ground truth without the human-agreement validation step in §6.
- Do **not** assume local MLX-quantized inference and cloud bf16 inference produce identical outputs — quantization can shift generation distributions. If both are ever used, document which was used for which reported number and never mix them within one comparison.
- Do **not** assume any HF repo ID surfaced by web search is correct without an independent fetch — the XSTest mirror correction in §2 is a concrete instance of this exact failure mode occurring during Phase 0 itself.

---

## 11. Final MVP scope

- **Model**: OLMo 3 7B, `Instruct` lineage, 4 verified checkpoints, all at revision `main` (`configs/models/olmo3_lineages.yaml`)
- **Languages**: English, Korean
- **Primary benchmark**: PolyGuardPrompts, EN + KO subsample (exact size pending Phase 1 per-language count)
- **Secondary benchmark**: XSTest, EN (from the canonical GitHub source) + KO (self-translated stratified subsample, ~100–150 items, under the §4 protocol)
- **Competence control**: Belebele Korean (`kor_Hang`), ~40–50 item stratified subsample, paired to English via `question_number`
- **Scoring**: rule-based/benchmark-native primary (PolyGuard classifier, exact-match, refusal-pattern matching); LLM-judge only as a validated fallback, never default
- **Statistics**: raw proportions, stage-over-stage deltas, cross-lingual delta-of-deltas, bootstrap CIs, within-language paired McNemar tests; explicitly no cross-lingual paired significance claims, no causal claims, no multi-seed sampling in MVP
- **Compute**: rented cloud GPU (RunPod, single L4 or A100) for all authoritative research runs; local M4 for code/plumbing development against tiny stub models only, with confirmed-MLX Base/RLVR checkpoints available for optional manual smoke-testing
- **Explicitly deferred, not MVP**: InstrumentalEval (possible future English-only RLVR probe), the `Think` lineage, a third language, multi-seed variance estimation, any causal-design ablation

---

## Open items carried into Phase 1 (not yet resolved — do not treat as done)

1. Confirm exact per-language item count and harmful/unharmful label balance for PolyGuardPrompts' Korean subset.
2. Read the PolyGuard source paper directly for a Korean-specific translation-provenance/quality figure.
3. Pull the canonical 450-item XSTest set from `github.com/paul-rottger/xstest` (or verify a specific HF mirror against it row-for-row) — do not use `natolambert/xstest-v2-copy`, confirmed to be a different artifact.
4. Execute the XSTest EN→KO translation pass and native-speaker review for the stratified subsample; produce the translation-provenance file.
5. Check `mlx-community` for MLX builds of `Olmo-3-7B-Instruct-SFT` and `Olmo-3-7B-Instruct-DPO` (unconfirmed either way).
6. Set up an HF token on the dev/runner environment (not currently cached).
7. Get explicit user approval for the cloud GPU spend before Phase 2 begins.
