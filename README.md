# VQASynth

Enhance the reasoning of multimodal models with pipelines to synthesize VQA datasets.

## Background
Inspired by SpatialVLM, this repo uses ZoeDepth to adapt Vision Langauge Models for spatial reasoning.
The demos feature pipelines using LLaVA for object captioning and SAM for segmentation. 
One uses CLIPSeg for region proposal, while the other uses GroundingDINO. 

![VQASynth-diagram.png](https://github.com/remyxai/VQASynth/blob/main/assets/VQASynth-diagram.png?raw=true)

### Environment

Before running the demo scripts, ensure you have the following installed:
- Python 3.9 or later
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

The main pipeline uses Docker Compose to process a directory of images into a VQA dataset including spatial relations between objects. The dataset follows conventions for training models like [LLaVA](https://llava-vl.github.io/). We recommend using an A10 GPU or larger for processing.

Make sure to update [.env](pipelines/.env) with the full path to your image directory and output directory. Then launch the pipeline with:

```bash
cd /path/to/VQASynth
docker compose -f pipelines/spatialvqa.yaml up --build
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

## Models

Check out our LLaVA 1.5 LoRA [SpaceLLaVA](https://huggingface.co/remyxai/SpaceLLaVA)
and MobileVLM-based [SpaceLLaVA-lite](https://huggingface.co/remyxai/SpaceLLaVA-lite)

Try SpaceLLaVA in [Discord](http://discord.gg/b2yGuCNpuC)

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
