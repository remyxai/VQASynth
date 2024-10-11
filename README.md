# VQASynth 🎹

![GIF Description](./assets/vqasynth-example.gif)

**Spatial Reasoning** is fundamental to navigating and interacting in a physical environment for embodied AI applications like robotics. However, data samples suitable for learning these capabilities are rare in AI pretraining datasets.
Don't be limited by what your model can do out-of-the-box, curate any image dataset from the Huggingface Hub for Spatial VQA with tools for scene understanding. 

VLMs trained using VQASynth 🎹 
* estimate 3D distances between objects in an image
* describe distances colloquially, convert between common units
* answer queries about the orientation and spatial relationships between objects
* base responses on consistent references like floors and surfaces

## Description

Fusing semantic and metric data into templated VQA chat, Vision Language Models can be instruction-tuned with low-rank adapters to enhance their baseline spatial reasoning capabilities. 
VQASynth 🎹 provides an open-source reproduction of [SpatialVLM](https://arxiv.org/abs/2401.12168), which describes a 3D scene reconstruction pipeline and templates to enhance the spatial reasoning abilities of VLMs including:

* Semantic filtering with [CLIP](https://github.com/openai/CLIP) to normalize the image distribution and attributes
* Metric Depth Estimation with [ZoeDepth](https://github.com/isl-org/ZoeDepth) to lift the 2D image to 3D point cloud
* Object-level captioning with [FlexCap](https://flex-cap.github.io/) for precise 2D region proposal
* Plane-fitting with RANSAC for consistent 3D reference coordinates

Initial VQASynth 🎹 pipelines prompted [LLaVA](https://github.com/haotian-liu/LLaVA) for JSON-formatted object-level detailed captions or tags using [RAM](https://github.com/xinyu1205/recognize-anything). Accordingly, we evaluated caption/tag based region proposal with publicly available models like [CLIPSeg](https://github.com/timojl/clipseg) and [groundingDINO](https://github.com/IDEA-Research/GroundingDINO).

![VQASynth-diagram.png](https://github.com/remyxai/VQASynth/blob/main/assets/VQASynth-diagram.png?raw=true)

## What's New 👀 in VQASynth 🎹

🪶 Faster & lighter using [Florence-2](https://arxiv.org/abs/2311.06242) for detailed image captions **and** region proposal grounded on text captions.

📐 Improves metric depth estimation speed & accuracy by replacing ZoeDepth with [DepthPro](https://github.com/apple/ml-depth-pro).

🎓 [SAM2](https://github.com/facebookresearch/sam2) replaces SAM in the localization refinement stage.


### Environment

Before running the demo scripts, ensure you have the following installed:
- Python 3.10 or later
- [Docker](https://docs.docker.com/engine/install/), [Docker Compose V2](https://docs.docker.com/compose/migrate/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

CLIPSeg-based SpatialVLM data processing (recommended):
```bash
cd tests/data_processing/
docker build -f clipseg_data_processing.dockerfile -t vqasynth:clipseg-dataproc-test .
docker run --gpus all -v /path/to/output/:/path/to/output vqasynth:clipseg-dataproc-test --input_image="warehouse_rgb.jpg" --output_dir "/path/to/output" 
```

GroundingDINO-based SpatialVLM data processing:
```bash
cd tests/data_processing/
docker build -f groundingDino_data_processing.dockerfile -t vqasynth:dino-dataproc-test .
docker run --gpus all -v /path/to/output/:/path/to/output vqasynth:dino-dataproc-test --input_image="warehouse_rgb.jpg" --output_dir "/path/to/output" 
```

The scripts will produce 3D point clouds, segmented images, labels, and prompt examples for a test image.


## Run a Pipeline on Your Images

The main pipeline uses Docker Compose to process a Hugging Face dataset into a VQA dataset including spatial relations between objects. The dataset follows conventions for training models like [LLaVA](https://llava-vl.github.io/). We recommend using an A10 GPU or larger for processing.

Make sure to update the [config.yaml](config/config.yaml) file by adding the following details: an output directory path, the repository ID for the dataset to be processed, and a dataset name to store the results to the hub. You can also optionally add `include_tags` and/or `exclude_tags` as comma-separated lists in the config file for filtering the dataset based on tags. If no tags are provided, the filtering will not be applied.

Then launch the pipeline with:

```bash
# Authenticate to push to hub
huggingface-cli login

# Run the pipeline
cd /path/to/VQASynth
bash run.sh
```

In your designated output directory, you'll find a json file `processed_dataset.json` containing the formatted dataset.

Here are some examples:

| <img src="https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_1.jpeg?raw=true"  alt="sample_1" style="max-height: 350px;" > | <img src="https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_2.jpeg?raw=true"  alt="sample_2" style="max-height: 350px;"> | <img src="https://github.com/remyxai/VQASynth/blob/main/assets/warehouse_sample_3.jpeg?raw=true" alt="sample_3" style="max-height: 350px;"> |
|----------|-------------|--------|
| Does the red forklift in warehouse appear on the left side of the brown cardboard boxes stacked? | How close is the man in red hat walking from the wooden pallet with boxes? | Does the man in blue shirt working have a greater height compared to the wooden pallet with boxes on floor? |
| Incorrect, the red forklift in warehouse is not on the left side of the brown cardboard boxes stacked. | The man in red hat walking is 60.13 centimeters from the wooden pallet with boxes. | Indeed, the man in blue shirt working is taller compared to the wooden pallet with boxes on floor. |

Here's a sample of warehouse images captioned with spatial relationships similar to the table above. 

```bash
wget https://remyx.ai/assets/vqasynth/vqasynth_warehouse_spaces.zip

# Data is formatted for LLaVA fine-tuning
unzip vqasynth_warehouse_spaces.zip 
```

Once completed, you can follow this resource on [fine-tuning LLaVa](https://github.com/haotian-liu/LLaVA/blob/5d8f1760c08b7dfba3ae97b71cbd4c6f17d12dbd/docs/Finetune_Custom_Data.md#L4).

## Datasets from VQASynth 🎹

* [vqasynth_spacellava](https://huggingface.co/datasets/remyxai/vqasynth_spacellava)

## Models tuned on VQASynth 🎹

* [SpaceLLaVA - 13B](https://huggingface.co/remyxai/SpaceLLaVA)
* [SpaceMantis - 8B](https://huggingface.co/remyxai/SpaceMantis)
* [SpaceFlorence-2 - <1B](https://huggingface.co/remyxai/SpaceFlorence-2)
* [SpaceVLMs Collection](https://huggingface.co/collections/remyxai/spacevlms-66a3dbb924756d98e7aec678)

Try SpaceMantis in the [HF Space](https://huggingface.co/spaces/remyxai/SpaceMantis) or SpaceLLaVA in [Discord](http://discord.gg/b2yGuCNpuC)

![image](https://github.com/remyxai/VQASynth/assets/9044907/8d99db2a-6b93-4123-85bd-8c91e795a5ef)


## Notebooks
We've hosted some notebooks visualizing and experimenting with the techniques included in this repo.

| Notebook | Description | Launch |
|----------|-------------|--------|
| Spatial Reasoning with Point Clouds | Visualize point clouds and evaluate spatial relationships | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f3rr-y233GvxWVzPE7_mK-DY52pG0fsm?usp=sharing) |

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
```
