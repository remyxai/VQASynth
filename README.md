# VQASynth 🎹 

![GIF Description](./assets/vqasynth-example.gif)

**Spatial Reasoning** is fundamental to interacting within and navigating physical environments for embodied AI applications like robotics. However, data samples suitable for learning these capabilities are rare in AI pretraining datasets.
Don't be limited by what your model can do out-of-the-box, curate any image dataset from the Huggingface Hub for Spatial VQA with tools for 3D scene understanding. 

VLMs trained using VQASynth 🎹 
* estimate 3D distances between objects in an image
* describe distances colloquially, convert between common units of measurement
* answer queries about the orientation and spatial relationships between objects
* base responses on consistent references like floors and surfaces
* apply CoT "thinking" for more robust reasoning and better estimates

## Run a Pipeline on Your Image Dataset

### Environment

Before running the demo scripts, ensure you have the following installed:
- Python 3.10 or later
- [Docker](https://docs.docker.com/engine/install/), [Docker Compose V2](https://docs.docker.com/compose/migrate/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- At least 24GB VRAM (A10 or larger) and 16GB RAM

Use Docker Compose to transform Image datasets from Huggingface Hub into VQA datasets describing spatial relations between objects.
You can process different datasets after updating the [config.yaml](config/config.yaml).

Then run the spatial VQA pipeline locally with Docker:

```bash
# Authenticate to push to hub
huggingface-cli login

# Run the pipeline
cd /path/to/VQASynth
bash run.sh
```

You can run the colab notebook (requires A100 runtime) or compose the pipeline modules in-process to turn one image into grounded spatial VQAs:

```python
from PIL import Image
from vqasynth.localize import Localizer
from vqasynth.scene_fusion import SpatialSceneConstructor
from vqasynth.prompts import PromptGenerator

image = Image.open("warehouse.jpg").convert("RGB")

# Detect + segment task-relevant objects
masks, _, captions = Localizer(captioner_type="florence").run(image)

# Lift to 3D — VGGT emits per-object point clouds, depth, and intrinsics in one pass
pcd_filepaths, canonicalized, _, _ = SpatialSceneConstructor().run(
    "warehouse_0", image, masks, output_dir="./scenes"
)

# Generate spatial VQAs from the reconstructed 3D scene
qa_pairs = PromptGenerator().run(captions, pcd_filepaths, canonicalized)

# Example output:
#   "How close is the man in red hat walking from the wooden pallet with boxes?"
#   → "Approximately 60.13 centimeters."
```

The resulting Huggingface dataset is in the cache directory and you can push to hub with:
```python
from vqasynth.datasets import Dataloader
Dataloader(cache_dir).push_to_hub(final_dataset, target_repo_name)
```

## Description

Fusing semantic and metric data into templated VQA chat, Vision Language Models can be instruction-tuned with low-rank adapters to enhance their baseline spatial reasoning capabilities. 
VQASynth 🎹 provides an open-source reproduction of [SpatialVLM](https://spatial-vlm.github.io/), which describes a 3D scene reconstruction pipeline and prompt templates for enhancing the spatial reasoning abilities of VLMs including:

![VQASynth pipeline](https://github.com/user-attachments/assets/6acd4695-8fde-40e0-b0b4-42361e92c303)

🪄 Object-grounded captions by [point prompting with Molmo](https://github.com/remyxai/VQASynth/issues/27)

📐 Improves metric depth estimation speed & accuracy by replacing DepthPro with [VGGT](https://vgg-t.github.io/).

🪢 [SAM2](https://github.com/facebookresearch/sam2) replaces SAM in the localization refinement stage.

🧠 Multimodal "Thinking" by CoT Reasoning

## Datasets made using VQASynth 🎹

* [SpaceOm](https://huggingface.co/datasets/remyxai/SpaceOm)
* [SpaceThinker](https://huggingface.co/datasets/remyxai/SpaceThinker)
* [OpenSpaces_MC_R1](https://huggingface.co/datasets/remyxai/OpenSpaces_MC_R1)
* [OpenSpaces](https://huggingface.co/datasets/remyxai/OpenSpaces)
* [vqasynth_spacellava](https://huggingface.co/datasets/remyxai/vqasynth_spacellava)

## Models trained using VQASynth 🎹

* [SpaceOm](https://huggingface.co/remyxai/SpaceOm) - the best overall
* [SpaceThinker-Qwen2.5VL-3B](https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B) - the most accurate distance estimates
* [SpaceQwen2.5-VL-3B-Instruct - 3B](https://huggingface.co/remyxai/SpaceQwen2.5-VL-3B-Instruct) - the most popular
* [SpaceLLaVA - 13B](https://huggingface.co/remyxai/SpaceLLaVA) - the original
* [SpaceLLaVA - 7B](https://huggingface.co/salma-remyx/spacellava-1.5-7b) - the most interpretable

Examples from [SpaceThinker](https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B):

| <img src="assets/thinking_example_1.jpg"  alt="sample_1" style="max-height: 350px;" > | <img src="assets/thinking_example_2.png"  alt="sample_2" style="max-height: 350px;"> | <img src="assets/thinking_example_3.png" alt="sample_3" style="max-height: 350px;"> |
|----------|-------------|--------|
| **Prompt:** How far is the man in the red hat from the pallet of boxes in feet? | **Prompt:** How far is the Goal from the penalty taker in this image? | **Prompt:** Approximately how many meters apart are the chair and bookshelf? |
| **Reasoning:** To determine how far the man in the red hat is from the pallet of boxes, I need to consider the spatial arrangement in the warehouse. The man is walking on the floor, and there's a visible gap between him and the pallet. First, I'll estimate the size of the pallet. A standard pallet is typically around 4 feet wide and 8 feet long. Given that the man is standing near the edge of the pallet, he might be about 2–3 feet away from the front edge of the pallet. Next, I'll consider the distance from the pallet to the man. The man appears to be walking along the aisle, which seems to be about 6–7 feet wide. If he is standing at the edge of the pallet, he would likely be about 2–3 feet from the front edge of the pallet. Combining these observations, the man in the red hat is approximately 2–3 feet from the pallet of boxes. | **Reasoning:** First, I observe the scene where a player in a red uniform stands near the goal. The goal itself is a standard soccer goalpost, which typically extends 2.5 meters beyond the line of the goalposts. The player appears to be standing close to the goal, but not directly at the goal line. To estimate the distance, I consider that the player's position relative to the goal suggests he is likely standing closer than 2.5 meters away. Given the typical setup of a penalty kick scenario, the player would be positioned further back from the goal than the goal itself. By visual estimation, the player seems to be standing approximately 10–15 meters from the goal. This range aligns with the common distance for a penalty kick, considering the player's stance and the goal's structure. Thus, the distance from the goal to the penalty taker is likely around 10–15 meters. | **Reasoning:** To determine the distance between the chair and the bookshelf, I'll consider the typical dimensions of these objects. A standard office chair is around 60–70 cm tall, and a bookshelf can be anywhere from 1.2 to 1.8 meters tall. Given that the chair is likely placed on a desk or floor level, the height difference between the chair and the top of the bookshelf would be approximately 1 meter. Assuming the bookshelf is at least 1 meter high, and the chair is at ground level, the vertical distance between them is indeed 1 meter. The horizontal distance might vary depending on the layout, but given the room's dimensions and the placement of the chair relative to the bookshelf, the total distance between the chair and the top of the bookshelf is likely around 1 meter. Therefore, the chair and bookshelf are approximately 1 meter apart. |
| **Answer:** 2–3 feet | **Answer:** 10–15 meters | **Answer:** 1 meter |

Synthetic spatial reasoning traces suffice to train VLMs capable of grounded, quantitative spatial reasoning—inferring accurate distances, interpreting 3D scene context, and formatting open-ended answers precisely by integrating visual cues, real-world object priors, and human-centric spatial logic.

## Model Evaluation
Report VLM spatial reasoning on [benchmarks](https://github.com/remyxai/VQASynth/blob/main/pipelines/spatialvqa_eval.yaml
): SpatialScore, OmniSpatial, SpaCE-10, MindCube

## Notebooks
We've hosted some notebooks visualizing and experimenting with the techniques included in this repo.

| Notebook | Description | Launch |
|----------|-------------|--------|
| Generate Spatial VQA Dataset | Augment an HF Image Dataset with Spatial VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sJUsJ5-UR-3Bydgg-thJ59KSNxRG8Q30?usp=sharing) |
| Spatial Reasoning with Point Clouds | Visualize point clouds and evaluate spatial relationships | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f3rr-y233GvxWVzPE7_mK-DY52pG0fsm?usp=sharing) |
| Evaluate SpaceThinker on QSpatial++ | Assess SpaceThinker's quantitative spatial reasoning on the QSpatial++ benchmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1buEe2QC4_pnrJwQ9XyRAH7RfaIa6pbex?usp=sharing) |
| SpaceLLaVA Attention with TransformerLens | Visualize SpaceLLaVA-7B attention patterns using TransformerLens | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19H_hOL8gc1nFQKpDoioJR8nDWX1lsNZM?usp=sharing) |
| Agent with VQASynth Tools | Dynamic tool composition for spatial questions beyond template + VLM ceilings | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nEWs0eVJPW-mmh5PMJFWx8kenPILRlvE?usp=sharing) |

## Agent with VQASynth Tools

The VQASynth tool inventory — Florence-2 detection/OCR, DepthPro/VGGT metric depth, 3D distance measurement — is also exposed as an [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)-based agent (`experiments/nooa_agent/`) that composes tool calls dynamically per prompt rather than following a pre-templated pipeline. Useful when the input question isn't known at pipeline-design time.

- **Dynamic tool composition** — the LLM decides which tools to call and in what order per question. `detect_objects → metric_depth → distance_3d` for a "how far apart" query; `caption_scene → dense_region_captions` for a scene inventory. No pre-coded question types.
- **Dense reward annotation** — plug into a robot-learning pipeline as a per-frame reward or CoT source. The agent satisfies [`lerobot`](https://github.com/huggingface/lerobot)'s `VlmClient` protocol via `SpatialAnnotatorVlmClient`, so it drops into any annotation module (VQA, plan, ECoT) as a tool-grounded alternative to a raw VLM call.
- **Traces grounded on tool calls** — every `annotate()` streams a JSONL row in OpenAI-messages format (ready for Qwen2.5-VL / Qwen3-VL fine-tuning) preserving the full tool-call chain, not just the final answer. Reasoning stays auditable.

Quick example:

```python
from PIL import Image
from experiments.nooa_agent.spatial_annotator import SpatialAnnotator, SceneContext
from experiments.nooa_agent.trace import TraceWriter
from nooa.unifiedllm.registry import get_llm_client

agent = SpatialAnnotator(llm=get_llm_client("gemini/gemini-2.5-pro"))
img = Image.open("warehouse.jpg").convert("RGB")

with TraceWriter("traces.jsonl") as writer:
    scene = SceneContext(agent, img, trace_writer=writer, image_ref="warehouse.jpg")
    result = await scene.annotate("How far apart are the two workers in the foreground?")

print(result.answer)            # "The two workers in the foreground are 2.04 meters apart."
print(result.confidence)        # "high"
print(result.tool_calls_used)   # 4
```
<img width="2752" height="1536" alt="complex_image_reasoning_20x" src="https://github.com/user-attachments/assets/4edc92cf-0bff-461d-8f4e-f94a174f4b14" />

See `experiments/nooa_agent/README.md` for install, resource tiers (CPU with DepthPro / GPU with VGGT), the full tool inventory, and the lerobot integration path.

## References
This project was inspired by or utilizes concepts discussed in the following research paper(s):
```
@article{chen2024spatialvlm,
  title = {SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities},
  author = {Chen, Boyuan and Xu, Zhuo and Kirmani, Sean and Ichter, Brian and Driess, Danny and Florence, Pete and Sadigh, Dorsa and Guibas, Leonidas and Xia, Fei},
  journal = {arXiv preprint arXiv:2401.12168},
  year = {2024},
  url = {https://arxiv.org/abs/2401.12168},
}
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```

## Region Captioning with Describe Anything

VQASynth's localization stage already produces one [SAM2](https://github.com/facebookresearch/sam2) mask per detected object, but the captions attached to those masks are short class labels. The `vqasynth.describe_anything` stage ([issue #51](https://github.com/remyxai/VQASynth/issues/51)) feeds each mask to NVIDIA's [Describe Anything Model (DAM)](https://describe-anything.github.io/) to produce a detailed region caption — useful for distinguishing subjects by small visual features — and turns each `(image, mask, caption)` triple into spatial-VQA training samples.

🖼️ Per-region detail via [Describe Anything (DAM)](https://arxiv.org/abs/2504.16072), prompted with the masks SAM2 already produces. No re-segmentation: DAM consumes `vqasynth.localize.Localizer` output directly.

```python
from PIL import Image
from vqasynth.localize import Localizer
from vqasynth.describe_anything import DescribeAnything

image = Image.open("warehouse.jpg").convert("RGB")
masks, _, _ = Localizer(captioner_type="molmo").run(image)

# A DAM caption for every SAM mask, then spatial-VQA pairs
dam = DescribeAnything()  # model_id defaults to nvidia/DAM-3B-Self-Contained
detailed_captions = dam.describe_regions(image, masks)
qa_prompts, messages = dam.generate_qa_pairs(masks, detailed_captions)
```

`docker/describe_anything_stage/` runs this over a Hugging Face dataset via `datasets.map`, adding `dam_captions` and `dam_messages` columns downstream of the location-refinement stage. DAM is loaded lazily through `transformers` (self-contained variant), and the module + tests import and run without a GPU using an injected stub callable — see `tests/test_describe_anything.py`. Override the DAM variant with `--model_id` (or the `DAM_MODEL_ID` env var) on the Docker stage.

If you build on it, cite the [DAM paper](https://arxiv.org/abs/2504.16072) (ICCV 2025).

