# Who's building on VQASynth & the Space\* models

[VQASynth](https://github.com/remyxai/VQASynth) and the open-weight **Space\*** VLM family
(SpaceLLaVA, SpaceQwen2.5-VL, SpaceThinker, SpaceOm) seeded an open sub-field of spatial reasoning for
vision-language models. This page tracks works that **build on the pipeline/data** or **evaluate the released
models** — every arXiv ID below resolves and was checked against the paper's full text (evaluation tables vs
citation-only). **Last updated: 2026-11-16.**

Three buckets: **(A)** built on the VQASynth pipeline or OpenSpaces data · **(B)** evaluate a released Space\*
model in experiments · **(C)** cite / benchmark the broader family.

---

## A. Built on the VQASynth pipeline / OpenSpaces data ★

| arXiv | Work | How it builds on VQASynth |
|---|---|---|
| [2506.01371](https://arxiv.org/abs/2506.01371) | **SVQA-R1** | Trains on `Vqasynth_Spacellava`, evaluates OOD on OpenSpaces; +31pp over SpaceThinker |
| [2601.03309](https://arxiv.org/abs/2601.03309) | **VLM4VLA** | Runs the VQASynth pipeline to annotate Bridge-v2 / Fractal / Calvin → VLA training data |
| [2605.25802](https://arxiv.org/abs/2605.25802) | **Rethinking VLM Representation for VLA** | Consumes VQASynth/OpenSpaces (+ SpaceThinker/SpaceLLaVA datasets) as Stage-1 VLA adaptation data |
| [2503.11094](https://arxiv.org/abs/2503.11094) | **Open3D-VQA** | Builds its own pipeline "following a similar approach to RemyXAI"; cites VQASynth (methodologically inspired) |
| [2510.12693](https://arxiv.org/abs/2510.12693) | **ERA** | Trains on the **SpaceThinker dataset** (OpenSpaces-lineage reasoning corpus) as an embodied prior |

## B. Evaluate a released Space\* model (benchmarks & experiments)

### SpaceThinker — 19 works
SVQA-R1 [2506.01371](https://arxiv.org/abs/2506.01371) · OmniSpatial [2506.03135](https://arxiv.org/abs/2506.03135) · SpaCE-10 [2506.07966](https://arxiv.org/abs/2506.07966) · Visual&Textual Thinking [2507.20529](https://arxiv.org/abs/2507.20529) · Object Properties [2508.10956](https://arxiv.org/abs/2508.10956) · Ego3D-Bench [2509.06266](https://arxiv.org/abs/2509.06266) · Perception-Time Scaling [2510.08964](https://arxiv.org/abs/2510.08964) · Spatial-DISE [2510.13394](https://arxiv.org/abs/2510.13394) · SpatialThinker [2511.07403](https://arxiv.org/abs/2511.07403) · CityCube [2601.14339](https://arxiv.org/abs/2601.14339) · SpatiaLab (ICLR'26) [2602.03916](https://arxiv.org/abs/2602.03916) · MultihopSpatial [2603.18892](https://arxiv.org/abs/2603.18892) · CVSBench [2606.22476](https://arxiv.org/abs/2606.22476) · SATURN [2606.22694](https://arxiv.org/abs/2606.22694) · UAV-DualCog [2607.16193](https://arxiv.org/abs/2607.16193) · SpatialQuery [2608.01709](https://arxiv.org/abs/2608.01709) · SpatialCORE [2609.38716](https://arxiv.org/abs/2609.38716) · Soft Spatial Reasoning [2609.38717](https://arxiv.org/abs/2609.38717) · Beyond Omni [2610.26751](https://arxiv.org/abs/2610.26751)

### SpaceOm — 16 works
Understanding Space [2509.02175](https://arxiv.org/abs/2509.02175) · Spatial-DISE [2510.13394](https://arxiv.org/abs/2510.13394) · SpatialThinker [2511.07403](https://arxiv.org/abs/2511.07403) · CityCube [2601.14339](https://arxiv.org/abs/2601.14339) · SpatiaLab [2602.03916](https://arxiv.org/abs/2602.03916) · 3ViewSense [2603.07751](https://arxiv.org/abs/2603.07751) · GoalVLM [2603.18210](https://arxiv.org/abs/2603.18210) · MultihopSpatial [2603.18892](https://arxiv.org/abs/2603.18892) · Reasmory [2606.00963](https://arxiv.org/abs/2606.00963) · SATURN [2606.22694](https://arxiv.org/abs/2606.22694) · UAV-DualCog [2607.16193](https://arxiv.org/abs/2607.16193) · SpatialQuery [2608.01709](https://arxiv.org/abs/2608.01709) · SCOUT [2608.12220](https://arxiv.org/abs/2608.12220) · SpatialCORE [2609.38716](https://arxiv.org/abs/2609.38716) · Soft Spatial Reasoning [2609.38717](https://arxiv.org/abs/2609.38717) · Beyond Omni [2610.26751](https://arxiv.org/abs/2610.26751)

### SpaceQwen2.5-VL-3B — 15 works
OmniSpatial [2506.03135](https://arxiv.org/abs/2506.03135) · SpaCE-10 [2506.07966](https://arxiv.org/abs/2506.07966) · SIRI-Bench [2506.14512](https://arxiv.org/abs/2506.14512) · MindCube [2506.21458](https://arxiv.org/abs/2506.21458) · Ego3D-Bench [2509.06266](https://arxiv.org/abs/2509.06266) · G²VLM [2511.21688](https://arxiv.org/abs/2511.21688) · MMSI-Video-Bench [2512.10863](https://arxiv.org/abs/2512.10863) · Lost in Space? [2601.22228](https://arxiv.org/abs/2601.22228) · SpatiaLab [2602.03916](https://arxiv.org/abs/2602.03916) · 3ViewSense [2603.07751](https://arxiv.org/abs/2603.07751) · SpatialForge [2605.11462](https://arxiv.org/abs/2605.11462) · CVSBench [2606.22476](https://arxiv.org/abs/2606.22476) · SATURN [2606.22694](https://arxiv.org/abs/2606.22694) · SpatialCORE [2609.38716](https://arxiv.org/abs/2609.38716) · Soft Spatial Reasoning [2609.38717](https://arxiv.org/abs/2609.38717)

### SpaceLLaVA — 10 works
SpatialReasoner [2504.20024](https://arxiv.org/abs/2504.20024) · SpaRE [2504.20648](https://arxiv.org/abs/2504.20648) · SpatialLLM [2505.00788](https://arxiv.org/abs/2505.00788) · SpatialMQA [2505.19015](https://arxiv.org/abs/2505.19015) · SpatialThinker [2511.07403](https://arxiv.org/abs/2511.07403) · SpaceTools [2512.04069](https://arxiv.org/abs/2512.04069) · Spatial Inconsistencies [2604.00799](https://arxiv.org/abs/2604.00799) · VLM3 [2605.30561](https://arxiv.org/abs/2605.30561) · DiSR [2608.05242](https://arxiv.org/abs/2608.05242) · SCOUT [2608.12220](https://arxiv.org/abs/2608.12220)

## C. Cite / benchmark the broader family
SPHERE [2412.12693](https://arxiv.org/abs/2412.12693) (cites SpaceLLaVA/SpaceMantis) · and spatial benchmarks adjacent to the family that did not evaluate a specific Space\* model in this pass: MARS2 Challenge [2509.14142](https://arxiv.org/abs/2509.14142), SpinBench [2509.25390](https://arxiv.org/abs/2509.25390), SpatiaLQA [2602.20901](https://arxiv.org/abs/2602.20901), HiSpatial [2603.25411](https://arxiv.org/abs/2603.25411), LAST [2604.09712](https://arxiv.org/abs/2604.09712), SpatialBlock [2609.07064](https://arxiv.org/abs/2609.07064).

---
*Design-informing frontier work (not adopters): SenseNova-SI [2511.13719](https://arxiv.org/abs/2511.13719), VST [2511.05491](https://arxiv.org/abs/2511.05491), Cambrian-S [2511.04670](https://arxiv.org/abs/2511.04670), VSI-Bench [2412.14171](https://arxiv.org/abs/2412.14171), SpatialScore [2505.17012](https://arxiv.org/abs/2505.17012).*
