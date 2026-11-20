# Novelty check: multilingual perception-vs-compliance in visual prompt injection

**Research question.** When a VLM agent resists a non-English visual prompt injection, is it refusal or failure to read? Language-dependent OCR ability may make multilingual visual-injection robustness a measurement artifact.

**Scope.** 2024–2026 work on (1) visual/typographic prompt injection in VLMs and VLM agents, (2) multilingual/low-resource jailbreaks and prompt injection, (3) work that separates perception (OCR) from compliance in injection attacks, especially across languages. Both jailbreak (user-driven) and indirect-injection (third-party data hijacks the agent's goal — the target setting) are included.

_Date of review: 2026-11-20._

## Landscape

| Paper | Year | Venue | Languages | Modality | Agent/chat | Sep. perception vs compliance | Setting |
|---|---|---|---|---|---|---|---|
| FigStep (Gong et al.)[^fig] | 2023/25 | AAAI'25 | English | Image (typo) | Chat | No | Jailbreak |
| MM-SafetyBench (Liu et al.)[^mm] | 2024 | ECCV 2024 | English | Image (typo+SD) | Chat | No | Jailbreak |
| Yong et al., Low-Resource Jailbreak GPT-4[^yong] | 2023 | NeurIPS'23 SoLaR WS | 12 low-resource | Text | Chat | No | Jailbreak |
| Deng et al., Multilingual Jailbreak[^deng] | 2024 | ICLR 2024 | Multilingual low-res | Text | Chat | No | Jailbreak |
| MLingualFC[^ml] | 2026 | preprint | Hindi, Punjabi, Spanish, Romanian, German (+En) | Image (flowchart) | Chat | Hypothesized, not measured | Jailbreak |
| JOCR / weak-OOD (Zhou et al.)[^woo] | 2025 | preprint | English (implied) | Image (OCR text) | Chat | Yes (mechanistic) | Jailbreak |
| Jailbreaking VLMs via Visual Modality[^vm] | 2026 | preprint | English | Image | Chat | Yes (4-way judge) | Jailbreak |
| AgentTypo (Li et al.)[^at] | 2025 | ICLR'26 WS | English | Image (typo) | Agent | No | Injection |
| VPI-Bench[^vpi] | 2025 | ICLR 2026 | English | Image (web UI) | Agent (CUA/BUA) | No | Injection |
| Chang et al., Trust-Boundary Confusion[^chang] | 2026 | preprint | English | Image | Agent (VLAS) | Yes — RQ1 | Injection |
| Kimura et al., Goal Hijacking via VPI[^kim] | 2024 | preprint | English (unverified) | Image | Chat/early-agent | Partial (read vs follow), unverified | Injection |
| BabelDojo (AgentDojo ext.)[^babel] | 2025/26 | GitHub (WIP) | ~10 langs incl. low-res, romanized, code-switched | Text | Agent | No | Injection |

### What the two target papers measure

- **MLingualFC** measures per-language ASR only (`ASR_i = jailbreak responses / samples in language i`; GPT-oss-120b judge, ≥4/5 = harmful), five languages + English, harmful steps rendered inside flowchart PNGs, single-turn chat jailbreak. It does **not** measure OCR ability per language; the OCR-vs-safety explanation is a hypothesis with only indirect support and unreconciled counter-evidence (Gemma-4 hits 74% Hindi ASR; Pangea's reading is never tested).
- **JOCR / weak-OOD** decomposes jailbreak outcomes into *intent perception* (hidden-state cosine similarity, original vs shuffled input) and *refusal triggering* (cosine similarity to a 50-refusal-token vocabulary), finding perception robust and refusal fragile. It is monolingual — no language/script axis at all — and is a jailbreak on chat VLMs, not an agent.

### Injection-side findings

- **Chang et al. (2604.19844)** RQ1 explicitly measures OCR capability (>80% detection, 0.85 Jaccard) separately from action planning and reports "modality laziness": models read the injected text but use it in <10% of decisions. This is a perception-vs-compliance separation in the agent-injection setting — but monolingual, on embodied/driving/UAV systems.
- **VPI-Bench**: visual injection for computer-use/browser agents, 306 cases, English only, no OCR-ability or language axis.
- **AgentDojo / InjecAgent** are text-only. The only multilingual extension, **BabelDojo**, is text (not visual) and a work-in-progress. No visual extension of either benchmark exists.

### "None found": non-English text injected *inside images* for *tool-using agents*

No paper occupies all three of {non-English, in-image, tool-using agent}. The pieces exist separately — multilingual+text+agent (BabelDojo), visual+injection+agent (VPI-Bench, AgentTypo, Chang), multilingual+visual+jailbreak (MLingualFC) — but not combined.

Queries run: `multilingual visual prompt injection image text VLM OCR cross-lingual`; `non-English prompt injection image VLM agent low-resource language visual attack 2025`; `non-English text injection inside image tool-using agent indirect prompt injection low-resource language`; `AgentDojo InjecAgent indirect prompt injection benchmark multilingual visual extension`.

## The two closest papers

The cross-lingual perception-vs-compliance confound is stated but never measured in jailbreak work (MLingualFC hypothesizes it; JOCR measures the split but monolingually), and on the injection side Chang et al. prove the method is feasible — they measure OCR separately from action and find "modality laziness" — but only in English on embodied systems, while every multilingual agent-injection effort (BabelDojo) is text-only and every visual agent-injection benchmark (VPI-Bench, AgentTypo) is English-only. The two closest papers are **Chang et al. 2026 (2604.19844)** — the same perception-vs-compliance-via-OCR separation in the agent-injection setting, but monolingual — and **MLingualFC (2606.07706)** — multilingual visual payloads that name the OCR confound but never measure OCR per language. The contribution beats both by doing the one thing neither does: measure per-language OCR/readability as a controlled covariate and show that multilingual *in-image injection* robustness on tool-using agents is predicted by it — ideally reusing BabelDojo's language set but rendering the payloads as images.

## To verify before citing

- [ ] **Kimura et al. (2408.03554)** — read first-hand; this review relied on citations only. Confirm languages tested, whether it separates reading from following, and the chat-vs-agent setting.
- [ ] **Chang et al. RQ1 numbers** — confirm the OCR detection rate, Jaccard score, and "modality laziness" <10% figure directly from the PDF (2604.19844); this review took them from secondary summaries.
- [ ] **Acceptance status** — confirm peer-review venue/acceptance for MLingualFC (2606.07706), Chang et al. (2604.19844), VPI-Bench (2506.02456, listed as ICLR 2026), and Jailbreaking VLMs via Visual Modality (2605.00583) before citing as published.
- [ ] **Repeat the "none found" search before submission** — re-run the four queries above to confirm the {non-English, in-image, tool-using agent} gap is still open at submission time.

## Footnotes

[^fig]: FigStep — arXiv:2311.05608 (AAAI'25 oral). https://arxiv.org/abs/2311.05608
[^mm]: Liu et al., MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal LLMs — ECCV 2024, arXiv:2311.17600. https://arxiv.org/abs/2311.17600
[^yong]: Yong, Menghini & Bach, "Low-Resource Languages Jailbreak GPT-4" — NeurIPS 2023 SoLaR Workshop, arXiv:2310.02446. https://arxiv.org/abs/2310.02446
[^deng]: Deng et al., Multilingual Jailbreak Challenges in LLMs — ICLR 2024, arXiv:2310.06474. https://arxiv.org/abs/2310.06474
[^ml]: Makwana, Mamta, Varshney & Cocarascu, MLingualFC — arXiv:2606.07706; code https://github.com/Rishabhpm23/MLingualFC
[^woo]: Zhou et al., "Why does weak-OOD help?" (JOCR) — arXiv:2511.08367. https://arxiv.org/abs/2511.08367
[^vm]: "Jailbreaking Vision-Language Models Through the Visual Modality" — arXiv:2605.00583. https://arxiv.org/html/2605.00583v1
[^at]: AgentTypo (Li et al.) — ICLR'26 Workshop; arXiv:2510.04257. https://www.emergentmind.com/papers/2510.04257
[^vpi]: VPI-Bench: Visual Prompt Injection Attacks for Computer-Use Agents — ICLR 2026, arXiv:2506.02456. https://arxiv.org/abs/2506.02456
[^chang]: Chang, Xue, Sun, Pang, Kanhere & Pearce, "Mitigating Trust Boundary Confusion from Visual Injections on Vision-Language Agentic Systems" — arXiv:2604.19844 (RQ1: OCR capability vs action planning; "modality laziness"). https://arxiv.org/html/2604.19844
[^kim]: Kimura et al., "Empirical Analysis of LVLMs against Goal Hijacking via Visual Prompt Injection" — arXiv:2408.03554 (not verified first-hand). https://arxiv.org/abs/2408.03554
[^babel]: BabelDojo: multilingual extension of AgentDojo — https://github.com/himanshubeniwal/BabelDojo (text-only, WIP); InjecAgent, ACL 2024 Findings, https://aclanthology.org/2024.findings-acl.624/
