# Implementation spec — drafted by Remyx Recommendation

**Recommended paper**: [Can These Views Be One Scene? Evaluating Multiview 3D Consistency when 3D Foundation Models Hallucinate](https://arxiv.org/abs/2605.18754v1)
**Confidence**: high (Remyx relevance 0.87)
**Research interest**: VQASynth

---

## Team's research focus

# remyxai/VQASynth

## Project summary
VQASynth is a Python-based pipeline for generating synthetic Visual Question Answering (VQA) datasets focused on spatial reasoning. It implements and extends the methodology from the SpatialVLM paper, enabling the creation of rich, instruction-tuning data for Vision Language Models (VLMs). The project processes standard image datasets by performing 3D scene reconstruction—including metric depth estimation, object segmentation, and grounded captioning—to automatically generate question-answer pairs about object distances, orientations, and relative positions, complete with chain-of-thought reasoning.

## Evolution stages
1. **Stage 1: Initial SpatialVLM Pipeline** (2024-02-21 - 2024-10-07, prototype)
   This initial phase established the core Docker-based pipeline inspired by the SpatialVLM paper, focusing on basic 3D scene understanding and prompt generation from local image files.

   | Component | Before | After |
   | :--- | :--- | :--- |
   | segmentation | n/a | ClipSeg + SAM |
   | data_pipeline | n/a | Local file processing |
   | infrastructure | n/a | Docker Compose |

   Key commits: `PR #1`, `PR #2`

2. **Stage 2: Modernizing the Pipeline** (2024-10-08 - 2025-02-23, iterating)
   This stage involved a significant overhaul of core components, replacing initial models with more powerful alternatives and integrating the pipeline with the Hugging Face Hub for more scalable data handling.

   | Component | Before | After |
   | :--- | :--- | :--- |
   | depth | Initial Estimator | DepthPro |
   | segmentation | ClipSeg + SAM | Florence + SAM2 |
   | filtering | n/a | CLIP-based tag filtering |
   | data_pipeline | Local file processing | Hugging Face Datasets |

   Key commits: `PR #14`, `PR #15`, `PR #18`, `PR #20`

3. **Stage 3: Advanced Models and CoT Reasoning** (2025-02-24 - 2025-05-31, converging)
   The project advanced its capabilities by integrating state-of-the-art models for captioning (Molmo) and depth estimation (VGGT), which simplified the pipeline, and introduced a new stage to generate explicit Chain-of-Thought (CoT) reasoning traces.

   | Component | Before | After |
   | :--- | :--- | :--- |
   | depth | DepthPro | VGGT |
   | captioning | Florence | Molmo (as high-end option) |
   | reasoning | VQA Templating | VQA Templating + CoT Generation |

   Key commits: `PR #35`, `PR #44`, `PR #45`

4. **Stage 4: Performance Tuning and Hardening** (2025-06-01 - present, hardening)
   The current stage focuses on improving the pipeline's robustness and efficiency, addressing performance bottlenecks like multi-GPU memory usage and fixing inconsistencies in data processing to support larger-scale dataset generation.

   | Component | Before | After |
   | :--- | :--- | :--- |
   | infrastructure | Basic multi-GPU support | Hardened multi-GPU inference |
   | fusion | Naive image resizing | Aspect-aware image handling |

   Key commits: `PR #56`, `PR #62`

## Current techniques
- **depth** → VGGT (for direct point cloud generation from images)
- **segmentation** → SAM2 (for object mask generation)
- **captioning** → Molmo / Florence (for object-grounded descriptions)
- **filtering** → CLIP (for content-based filtering of input images)
- **fusion** → Custom point cloud and mask fusion logic
- **reasoning** → Template-based VQA generation with an optional Chain-of-Thought formatting stage
- **data_pipeline** → Hugging Face Datasets integration for input and output
- **infrastructure** → Docker Compose for orchestrating the multi-stage pipeline

## Open problems & research directions
*   **Scalability and Efficiency:** The pipeline requires significant VRAM and faces OOM errors on multi-GPU setups. Research into model quantization, distillation, or more efficient 3D reconstruction techniques would be highly relevant.
*   **Reasoning Quality and Complexity:** The current CoT generation is template-based. There is a need to generate more diverse, complex, and logically sound reasoning chains that cover a wider range of spatial concepts.
*   **Generalization to Diverse Scenes:** The pipeline's performance on non-standard scenes (e.g., cluttered, low-light, aerial, non-rigid objects) is an open question. Work on robust segmentation and depth estimation in challenging domains would be actionable.
*   **Extension to Video:** The current framework is static. Extending the data generation pipeline to video to create datasets for spatial-temporal reasoning (e.g., object tracking, action understanding) is a natural next step.
*   **Evaluation of Synthetic Data:** Developing robust benchmarks and metrics to evaluate the true spatial reasoning capabilities of VLMs trained on VQASynth's output is crucial to measure its effectiveness and identify biases.
*   **Finer-Grained Scene Understanding:** Moving beyond object-level relationships to reason about object parts, affordances, and physical interactions (e.g., "can object A fit inside object B?") remains a significant challenge.

## Suggested paper topics
- Efficient monocular 3D scene reconstruction
- Generating synthetic chain-of-thought for vision-language models
- Robust object segmentation in cluttered environments
- Video-based spatial-temporal reasoning dataset generation
- Evaluating synthetic data for visual question answering
- Bias and fairness in synthetic multimodal datasets
- Fine-tuning VLMs for metric-aware spatial reasoning
- Compositional 3D scene understanding from single images
- Zero-shot depth estimation with large vision models
- Affordance reasoning in visual language models

## Why this paper for this team

VQASynth generates synthetic data from 3D scene reconstructions, making 'Evaluation of Synthetic Data' and the 'Add multi-benchmark evaluation stage' critical. This paper directly addresses the reliability of multiview 3D consistency, revealing that 3D foundation models can hallucinate. It introduces controlled benchmarks and COLMAP-based metrics to assess 3D consistency more reliably. This research significantly improves VQASynth's evaluation strategy by providing robust, human-correlated metrics to verify the geometric integrity and cross-view consistency of the underlying 3D scenes from which QA pairs are generated. This is crucial for ensuring the high quality of synthetic data and preventing training VLMs on inconsistent 3D information.

## Suggested experiment

For a small batch of VQASynth's input images, process them through the depth and fusion stages to obtain a 3D scene. Apply the COLMAP-based consistency metrics proposed in this paper to assess the geometric integrity and multi-view consistency of the reconstructed scene. Identify potential instances of 'hallucination' or inconsistency to inform improvements in VQASynth's 3D reconstruction pipeline.

## Paper abstract

Multiview 3D evaluation assumes that the images being scored are observations of one static 3D scene. This assumption can fail in NVS and sparse-view reconstruction: inputs or generated outputs may contain artifacts, outlier frames, repeated views, or noise, yet still receive high 3D consistency scores. Existing reference-based metrics require ground truth, while ground-truth-free metrics such as MEt3R depend on learned reconstruction backbones whose failure modes are poorly characterized. We study this reliability problem by comparing neural reconstruction priors with classical geometric verification. We introduce \benchmark, a controlled robustness benchmark for multiview 3D consistency, and a parametric family that decomposes neural metrics into backbone, residual, and aggregation components. This family recovers MEt3R and yields variants up to $3\times$ more robust. Our analysis shows that VGGT, MASt3R, DUSt3R, and Fast3R can hallucinate dense geometry and cross-view support for unrelated scenes, repeated images, and random noise. We introduce COLMAP-based metrics that use matches, registration, dense support, and reconstruction failure as failure-aware consistency signals. On real NVS outputs and a structured human study, these metrics achieve up to $4\times$ higher correlation with human judgments than MEt3R.
