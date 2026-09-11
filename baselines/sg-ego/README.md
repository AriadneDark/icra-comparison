<div align="center">

# SG-Ego: A training-free Scene Graph Annotation pipeline
[Francesca Pistilli](https://scholar.google.com/citations?user=7MJdvzYAAAAJ) • [Simone Alberto Peirone](https://scholar.google.com/citations?user=K0efPssAAAAJ) •  [Giuseppe Averta](https://scholar.google.com/citations?user=i4rm0tYAAAAJ)

Politecnico di Torino
</div>

<div align="center">
<a href='https://arxiv.org/abs/2607.02425' style="margin: 10px"><img src='https://img.shields.io/badge/Paper-Arxiv:2607.02425-red'></a>
<a href='https://francescapistilli.github.io/GLEN/' style="margin: 10px"><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://huggingface.co/datasets/francescapistilli/sg-ego' style="margin: 10px"><img src='https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor='></a>
</div>

<br>

**TL;DR:** a three stages pipeline to extract spatial and functional relations in egocentric videos, convert them into spatially grounded scene graph and consolidate them over time. 

## Overview

SG-Ego is released as part of our paper **"Learning to Evolve Scenes: Reasoning about Human Activities with Scene Graphs"**, which presents a novel graph edit formulation to model dynamic scenes of human activities. SG-Ego is a training-free pipeline that uses a combination of an LLM (Qwen3.5 9B) and two foundation models (GroundingDINO and SAM2) to extract grounded and temporally consistent scene graphs from videos of human activities.
This repository provides the code implementation of the automatic annotation pipeline. You can find our pre-extracted scene graph annotations for Ego4D at [here](https://huggingface.co/datasets/francescapistilli/sg-ego).

The pipeline is structured in three stages:
1. **📝 Stage 1: Frame-level relations extraction**: this stage leverages an open-source LLM (Qwen3.5 9B) to extract relevant spatial and functional relations appearing in each frame of the video.
2. **📌 Stage 2: Frame-level relations grounding**: captions from stage 1 are grounded in the frame with GroundingDINO and a frame-level scene graph is constructed from the grounded relations.
3. **⏱️ Stage 3: Graph consolidation across frames**: graphs are consolidated by tracking object instances over time with SAM2.

The output is a spatially grounded and temporally consistent scene graph of the input video.

## Getting started

### ⚙️ Environment Setup

This repository was tested with Python 3.10, Torch 2.6 and CUDA 12.6.
You can install all the required dependencies with the following commands:

```
conda create --name sg-ego python=3.10
conda activate sg-ego
pip install -r requirements.txt
```

### 🗄️ Data structure

This repository expects a data directory with the following structure:

```bash
data/
├── dataset_name
│   ├── videos
│   │   ├── 001.mp4
│   │   └── ...
│   ├── captions
│   │   ├── qwen3.5_9b
│   │   │   ├── 001.json
│   │   │   └── ...
│   ├── frame_graphs
│   │   ├── v1
│   │   │   ├── 001.json
│   │   │   └── ...
│   └── video_graphs
│   │   ├── v1
│   │   │   ├── 001.json
│   │   │   └── ...
│   └── input_files
│   │   ├── videos.txt
│   │   ├── frames.txt
│   │   └── windows.txt
├── other_dataset_name
│   ├── ...
└── ...
```

| Directory | Description |
|-----------|-------------|
| `videos` | Raw mp4 video files, resampled at the desired framerate for annotations. |
| `captions` | `(subject, relation, object)` triplets extracted from an LLM encoding the spatial and functional relations at each frame in the video. See **Stage 1** for captions generation. |
| `frame_graphs` | Frame-level scene graphs obtained by grounding the captions in the frame. See **Stage 2** for frame-level grounding. |
| `video_graphs` | Window-level scene graphs obtained by consolidating the frame-level scene graphs over time. See **Stage 3** for frame-level grounding. |
| `input_files`  | Custom input files for batched captioning, grounding and captioning. Check the dedicated README files for more information. |


## Annotations pipeline

### 📝 Stage 1: Frame-level relations extraction

Given a video, we sample frames at 5 fps and prompt an MLLM (Qwen3.5 9B) to describe the spatial and functional relations in each frame while ignoring irrelevant appearance details. 
The model directly outputs *(subject_x, relation, object_y)* triplets, where *_x* and *_y* identify entity instances. 
We then apply rule-based filtering to remove malformed and duplicate triplets. 

To generate the frame-level captions for all the frames of a video, use the following command:
```bash
    python -m captioning.main \
        --input-path data/example/videos/727ffce8-20ec-4111-af26-698eb306e8c7.mp4 \
        --output-path data/example/captions/qwen3.5_9b \
        --planning-goal "move the cup from the table to the sink" \
        --batch-size=8 \
        --model-name Qwen/Qwen3.5-9B
```

The adapted caption prompt emits only the induced subgraph on `robot`,
`manipulated_object`, `initial_support`, and `target`. It uses
`role::visual_name_N` identifiers so GroundingDINO still receives a visually
groundable noun phrase while the role survives grounding and consolidation.

More details in [captioning/README.md](captioning/README.md).

### 📌 Stage 2: Frame-level relations grounding

We ground each candidate triplet using GroundingDINO. 
For each triplet, we concatenate the subject, predicate, and object into a sentence and use the detector to localize the corresponding entities. 
We merge detections using the instance identifiers from the captions and apply heuristic filtering to remove duplicates and invalid spatial relations. Detected objects and relations define the nodes and edges of the frame-level graph, with node attributes given by object bounding boxes and semantic labels. Objects and relations are mapped to fixed vocabularies of $N{obj}=1480$ and $N_{rel}=387$ classes, respectively. 

To generate the frame-level scene graphs for the example video:
```bash
python -m grounding.main \
    --root data/example/ \
    --input-file 727ffce8-20ec-4111-af26-698eb306e8c7.mp4 \
    --captions-version qwen3.5_9b \
    --frame-graphs-version v1
```

More details in [grounding/README.md](grounding/README.md).

### ⏱️ Stage 3: Clip-level graph consolidation over time

Each frame-level scene graph provides a partial view of the ongoing interactions in the video, due to occlusions or omitted triplets by the captioner.
Given a sequence of frame-level scene graphs, the consolidation process C merges them into a single spatiotemporal scene graph that includes all nodes and relations observed during the timespan.
Objects correspondences are established via a semantic tracking pipeline based on SAM2 and DINOv2.

To consolidate the frame-level scene graphs over time:
```bash
python -m consolidation.main \
    --root data/example/ \
    --input-file 727ffce8-20ec-4111-af26-698eb306e8c7.mp4 \
    --frame-graphs-version v1 \
    --video-graphs-version v1
```

More details in [consolidation/README.md](consolidation/README.md).

## 💥 Scene graph annotations at scale
We provide additional scripts to run the annotations pipeline on a HPC cluster. More details in the dedicated README.md of each pipeline stage.

## 🧱 Pre-extracted SG-Ego Annotations for Ego4D
We provide pre-extracted SG-Ego Annotations for Ego4D [here](https://huggingface.co/datasets/francescapistilli/sg-ego).

## 🐋 Docker Support
You can also run the pipeline inside a CUDA container.
The docker image provides additional packages, e.g., `causal-conv1d` and `xformers`, that can significantly speedup parts of the pipeline.

Build the image from the repository root with:
```bash
sudo docker build -t sgego:cu126 .
```

Then, run each stage of the annotation pipeline in the docker container. 
For example, the following command executes the captioning stage inside the container:
```bash
sudo docker run --rm -it \
    --gpus all \
    -v "$PWD:/app" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    --shm-size=8g \
    sgego:cu126 \
    python -m captioning.main \
    --input-path data/example/videos/727ffce8-20ec-4111-af26-698eb306e8c7.mp4 \
    --output-path data/example/captions/qwen3.5_9b \
    --batch-size 8 --model-name Qwen/Qwen3.5-9B
```
Mounting the Hugging Face cache is optional, but it avoids downloading the model again for each run.
We also provide a pre-built docker image which can be downloaded with `docker pull sapeirone/sgego:cu126`.

> [!WARNING]
> If you are running behind a proxy, you may need to add the following arguments `--build-arg HTTP_PROXY=http://proxy.example.com:8080 --build-arg HTTPS_PROXY=http://proxy.example.com:8080` to the build command and to specify the following environment variables `-e HTTP_PROXY=http://proxy.example.com:8080 -e HTTPS_PROXY=http://proxy.example.com:8080` when running the container.


## Acknowledgements

This study was carried out within the FAIR - Future Artificial Intelligence Research and received funding from the European Union Next-GenerationEU (PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) – MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.3 – D.D. 1555 11/10/2022, PE00000013). This manuscript reflects only the authors’ views and opinions, neither the European Union nor the European Commission can be considered responsible for them. We acknowledge the CINECA award under the ISCRA initiative, for the availability of high performance computing resources and support.

## Cite Us

```
@article{pistilli2026sgego,
  title={Learning to Evolve Scenes: Reasoning about Human Activities with Scene Graphs},
  author={Pistilli, Francesca and Peirone, Simone Alberto and Averta, Giuseppe},
  journal={arXiv preprint arXiv:2607.02425},
  year={2026}
}
```
