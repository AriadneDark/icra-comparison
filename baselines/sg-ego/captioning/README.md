# Stage 1: Frame-level captioning

The first stage in the SG-Ego pipeline leverages a Large Language Model to generate frame-level captions describing the spatial and functional interactions in the video.

## Scripts overview

| Script file | Description |
|-------------|-------------|
| `captioning/main.py` | Local captioning using the Huggingface transformers library. |
| `captioning/main_sglang.py` | Large-scale captioning with the SGLang inference engine. |

## 📝 Usage
We provide two approaches for Qwen3.5-based captioning: local captioning and large scale cluster-based captioning.

### 📍 Local captioning with transformers

The `captioning.main` uses the `transformers` library and iterates over the video to generate the captions for each frame:
```bash
python -m captioning.main --input-path path/to/video.mp4 --output-path path/to/output/dir
```

**Example**:

```bash
python -m captioning.main \
    --input-path data/example/videos/64049385-6855-4eed-8b3c-ce3e8d704008.mp4 \
    --output-path data/example/captions/qwen3.5_9b
```

You can optionally change the captioning model with the `--model-name` argument:
```bash
python -m captioning.main \
    --input-path data/example/videos/64049385-6855-4eed-8b3c-ce3e8d704008.mp4 \
    --output-path data/example/captions/qwen3.5_0.8b \
    --model-name Qwen/Qwen3.5-0.8B
```

### 🌍 Video captioning at scale

We also provide a script for running large-scale captioning via the SGLang inference engine, which provides a set of optimizations out of the box, like KV caching. SGLang exposes and API endpoint that is compatible with OpenAI's API.
This is especially useful when running the captioning pipeline for very large scale datasets, e.g., Ego4D.

The script is meant to run on a cluster to parallelize the captioning effort into many small jobs. Each job reads a txt file containing the list of video files it has to process. The complete list of videos to caption can be obtained as follows:

#### Preparing the input files
```bash
ls data/example/videos/*.mp4 > data/example/input_files/videos.txt
```

Then, you can split the videos into a set of input files, one for each job using the `split` command:

```bash
split -d -a 3 -n l/128 --additional-suffix=.txt data/example/input_files/videos.txt data/example/input_files/videos_
```

The previous command generates a set of files like `data/example/input_files/videos_000.txt`, `data/example/input_files/videos_001.txt`, etc...

#### Running SLURM-based captioning
The scripts leverages SGLang packaged in a .sif container through [Singularity](https://sylabs.io/docs/). You can build and export the container file on a workstation using `apptainer`:
```bash
apptainer build sglang.sif docker://lmsysorg/sglang:latest
```

Then, move the `sglang.sif` on the cluster.

Finally, you can submit the captioning jobs on a SLURM cluster with following command:
```bash
mkdir -p logs
sbatch --array=0-127 captionins/scripts/run_slurm.sh
```


## 📜 Example output
```text
[
    (human_1, holding, knife_1)
    (human_1, holding, squash_piece_1)
    (human_1, cutting, squash_piece_1)
    (knife_1, on, cutting_board_1)
    (wooden_plate_1, on, countertop_1)
    (blue_container_1, on, countertop_1)
    (black_container_1, on, countertop_1)
    (black_container_2, on, countertop_1)
    (brown_paper_bag_1, on, countertop_1)
    (black_mortar_1, on, countertop_1)
]
```
The output is in raw text format and we use a combination of regex expressions and heuristics to convert this output into a python list of `(subject, relation, object)` triplets.