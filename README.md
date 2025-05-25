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

## Description

Fusing semantic and metric data into templated VQA chat, Vision Language Models can be instruction-tuned with low-rank adapters to enhance their baseline spatial reasoning capabilities. 
VQASynth 🎹 provides an open-source reproduction of [SpatialVLM](https://spatial-vlm.github.io/), which describes a 3D scene reconstruction pipeline and prompt templates for enhancing the spatial reasoning abilities of VLMs including:

![VQASynth-diagram.png](https://github.com/user-attachments/assets/913fbc87-f778-4675-83b8-357c28c56c4d)


🪄 Object-grounded captions by [point prompting with Molmo](https://github.com/remyxai/VQASynth/issues/27)

📐 Improves metric depth estimation speed & accuracy by replacing DepthPro with [VGGT](https://vgg-t.github.io/).

🪢 [SAM2](https://github.com/facebookresearch/sam2) replaces SAM in the localization refinement stage.

🧠 Multimodal "Thinking" by CoT Reasoning


## Datasets made using VQASynth 🎹

* [SpaceThinker](https://huggingface.co/datasets/remyxai/SpaceThinker)
* [OpenSpaces_MC_R1](https://huggingface.co/datasets/remyxai/OpenSpaces_MC_R1)
* [OpenSpaces](https://huggingface.co/datasets/remyxai/OpenSpaces)
* [vqasynth_spacellava](https://huggingface.co/datasets/remyxai/vqasynth_spacellava)
  
Here are some samples from the VQASynth pipeline:

| <img src="https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"  alt="sample_1" style="max-height: 350px;" > | <img src="https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_2.jpeg?raw=true"  alt="sample_2" style="max-height: 350px;"> | <img src="https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_3.jpeg?raw=true" alt="sample_3" style="max-height: 350px;"> |
|----------|-------------|--------|
| Does the red forklift in warehouse appear on the left side of the brown cardboard boxes stacked? | How close is the man in red hat walking from the wooden pallet with boxes? | Does the man in blue shirt working have a greater height compared to the wooden pallet with boxes on floor? |
| <think>Okay, let me analyze the layout of the scene to figure this out. The red forklift is clearly visible and it's facing the stack of brown cardboard boxes. The boxes are loaded onto a pallet, and the forklift's forks are inserted under this pallet, indicating that the forklift is preparing to lift or place the boxes...To verify scale and orientation: the yellow safety rails and barriers are on the left side of the boxes. The ladder and wall panels are further left. The forklift is operating from the right-hand side of the image, engaging the pallet head-on...</think> <answer>Correct, the red forklift in the warehouse is not on the left side of the brown cardboard boxes stacked; it is on the right side, actively engaging the boxes from that direction.</answer> | <think>Alright, let me break this down. The man in the red hat is walking in a warehouse aisle, and there's a wooden pallet loaded with boxes right behind him. The pallet is attached to a manual pallet jack, which appears to have been pulled forward...Taking the average stride length of an adult male into account (about 0.75 meters or 75 cm), it looks like the distance from the man's heel to the nearest edge of the pallet is slightly shorter than a full stride. That puts us at just under that 75 cm mark. I'd estimate somewhere around 60 cm is reasonable here, based on visual proximity...</think> <answer>The man in the red hat walking is approximately 60.13 centimeters from the wooden pallet with boxes.</answer> | <think>Let me take a close look at this warehouse scene. The man in the blue shirt is leaning over, reaching toward boxes on a shelf, and there's a pallet with cardboard boxes on the floor near him. To compare their heights, I need to reason about how tall the man would be if he were standing straight, and how tall the pallet is with boxes stacked on top. A standard wooden pallet is about 15 cm (6 inches) tall. The boxes on top add some height, maybe around 30–50 cm depending on their size. Altogether, the total height of the loaded pallet might be roughly 45–65 cm (about 1.5 to 2 feet)...</think> <answer>Indeed, the man in the blue shirt working is taller compared to the wooden pallet with boxes on the floor.</answer> |

  
## Models trained using VQASynth 🎹

[![Try the SpaceThinker Space](https://img.shields.io/badge/🤗%20Spaces-Demo-blue)](https://huggingface.co/spaces/remyxai/SpaceThinker-Qwen2.5VL-3B)

* [SpaceThinker-Qwen2.5VL-3B](https://huggingface.co/remyxai/SpaceThinker-Qwen2.5VL-3B) - the most accurate
* [SpaceQwen2.5-VL-3B-Instruct - 3B](https://huggingface.co/remyxai/SpaceQwen2.5-VL-3B-Instruct) - the most popular
* [SpaceLLaVA - 13B](https://huggingface.co/remyxai/SpaceLLaVA) - the original

| <img src="assets/thinking_example_1.jpg"  alt="sample_1" style="max-height: 350px;" > | <img src="assets/thinking_example_2.png"  alt="sample_2" style="max-height: 350px;"> | <img src="assets/thinking_example_3.png" alt="sample_3" style="max-height: 350px;"> |
|----------|-------------|--------|
| **Prompt:** How far is the man in the red hat from the pallet of boxes in feet? | **Prompt:** How far is the Goal from the penalty taker in this image? | **Prompt:** Approximately how many meters apart are the chair and bookshelf? |
| **Reasoning:** To determine how far the man in the red hat is from the pallet of boxes, I need to consider the spatial arrangement in the warehouse. The man is walking on the floor, and there's a visible gap between him and the pallet. First, I'll estimate the size of the pallet. A standard pallet is typically around 4 feet wide and 8 feet long. Given that the man is standing near the edge of the pallet, he might be about 2–3 feet away from the front edge of the pallet. Next, I'll consider the distance from the pallet to the man. The man appears to be walking along the aisle, which seems to be about 6–7 feet wide. If he is standing at the edge of the pallet, he would likely be about 2–3 feet from the front edge of the pallet. Combining these observations, the man in the red hat is approximately 2–3 feet from the pallet of boxes. | **Reasoning:** First, I observe the scene where a player in a red uniform stands near the goal. The goal itself is a standard soccer goalpost, which typically extends 2.5 meters beyond the line of the goalposts. The player appears to be standing close to the goal, but not directly at the goal line. To estimate the distance, I consider that the player's position relative to the goal suggests he is likely standing closer than 2.5 meters away. Given the typical setup of a penalty kick scenario, the player would be positioned further back from the goal than the goal itself. By visual estimation, the player seems to be standing approximately 10–15 meters from the goal. This range aligns with the common distance for a penalty kick, considering the player's stance and the goal's structure. Thus, the distance from the goal to the penalty taker is likely around 10–15 meters. | **Reasoning:** To determine the distance between the chair and the bookshelf, I'll consider the typical dimensions of these objects. A standard office chair is around 60–70 cm tall, and a bookshelf can be anywhere from 1.2 to 1.8 meters tall. Given that the chair is likely placed on a desk or floor level, the height difference between the chair and the top of the bookshelf would be approximately 1 meter. Assuming the bookshelf is at least 1 meter high, and the chair is at ground level, the vertical distance between them is indeed 1 meter. The horizontal distance might vary depending on the layout, but given the room's dimensions and the placement of the chair relative to the bookshelf, the total distance between the chair and the top of the bookshelf is likely around 1 meter. Therefore, the chair and bookshelf are approximately 1 meter apart. |
| **Answer:** 2–3 feet | **Answer:** 10–15 meters | **Answer:** 1 meter |

Synthetic spatial reasoning traces suffice to train VLMs capable of grounded, quantitative spatial reasoning—inferring accurate distances, interpreting 3D scene context, and formatting open-ended answers precisely by integrating visual cues, real-world object priors, and human-centric spatial logic.


## Run a Pipeline on Your Image Dataset

### Environment

Before running the demo scripts, ensure you have the following installed:
- Python 3.10 or later
- [Docker](https://docs.docker.com/engine/install/), [Docker Compose V2](https://docs.docker.com/compose/migrate/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


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
You can run the colab notebook using free-tier CPU or GPU acceleration or customize your own pipeline:
```python
from vqasynth.datasets import Dataloader
from vqasynth.embeddings import EmbeddingGenerator, TagFilter

dataloader = Dataloader(cache_dir)
dataset = dataloader.load_dataset(dataset_name)
embedding_generator = EmbeddingGenerator()
tag_filter = TagFilter()

include_tags = include_tags.strip().split(",")
exclude_tags = exclude_tags.strip().split(",")

# Extract embeddings
dataset = dataset.map(lambda example: embedding_generator.apply_transform(example, images))

# Extract tags
dataset = dataset.map(lambda example: tag_filter.apply_transform(example, include_tags + exclude_tags))

# Filter by tags
dataset_filtered = dataset.filter(
    lambda example: tag_filter.filter_by_tag(
        example['tag'], include_tags, exclude_tags
        )
    )
```

The resulting Huggingface dataset is in the cache directory and you can push to hub with:
```python
dataloader.push_to_hub(final_dataset, target_repo_name)
```


## Notebooks
We've hosted some notebooks visualizing and experimenting with the techniques included in this repo.

| Notebook | Description | Launch |
|----------|-------------|--------|
| Generate Spatial VQA Dataset | Augment an HF Image Dataset with Spatial VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sJUsJ5-UR-3Bydgg-thJ59KSNxRG8Q30?usp=sharing) |
| Spatial Reasoning with Point Clouds | Visualize point clouds and evaluate spatial relationships | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f3rr-y233GvxWVzPE7_mK-DY52pG0fsm?usp=sharing) |

Try SpaceLLaVA in [Discord](http://discord.gg/b2yGuCNpuC)

![image](<https://github.com/remyxai/VQASynth/assets/9044907/8d99db2a-6b93-4123-85bd-8c91e795a5ef> "SpaceThinker-Qwen2.5VL-3B")

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
