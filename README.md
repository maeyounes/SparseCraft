# SparseCraft

## [ECCV'24] SparseCraft: Few-Shot Neural Reconstruction through Stereopsis Guided Geometric Linearization

This repository contains the official code of SparseCraft, that learns a linearized implicit neural shape function using multi-view stereo cues, enabling robust training and efficient reconstruction from sparse views.

![Teaser](static/teaser.gif)

# [Project](https://sparsecraft.github.io/) | [Paper](https://arxiv.org/abs/2407.14257) | [Data](https://drive.google.com/file/d/1nuR1C80N8SvewFpggm3rqGQ8KghsgHIi)
## Requirements
- Python 3.8
- CUDA >= 11.7
- To utilize multiresolution hash encoding provided by tiny-cuda-nn, you should have least an RTX 2080Ti, see [https://github.com/NVlabs/tiny-cuda-nn#requirements](https://github.com/NVlabs/tiny-cuda-nn#requirements) for more details.
- This repo is tested with PyTorch 2.1.2, CUDA 11.8 on Ubuntu 22.04 system with NVIDIA RTX 3090 GPU.
- Make sure that the cuda paths are set in your shell session.
```bash
export CUDA_HOME=/usr/local/cuda-11.7  
export PATH=${CUDA_HOME}/bin:${PATH}  
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```
### Environment Installation

Step 1: Clone this repository:
```bash
git clone https://github.com/maeyounes/SparseCraft.git
cd SparseCraft
```

Step 2: Install the required packages using conda:
```bash
conda create -n sparsecraft python=3.8
conda activate sparsecraft
python -m pip install --upgrade pip
```

Step 3: Install PyTorch with CUDA 11.8:
```bash
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

Step 4: Install additional dependencies:
```bash
conda install -c conda-forge gcc=11 gxx_linux-64=11
conda install -c conda-forge libxcrypt
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
export LIBRARY_PATH="/usr/local/cuda-11.7/lib64/stubs:$LIBRARY_PATH"
```

Step 5: Install tiny-cuda-nn, set the correct CUDA architectures according to your GPU. a list of supported architectures can be found [here](https://developer.nvidia.com/cuda-gpus).
```bash
TCNN_CUDA_ARCHITECTURES=86 pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Step 6: Install project requirements:
```bash
pip install -r requirements.txt
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
```

Step 7: Install additional dependencies for DTU evaluation:
```bash
conda install -c conda-forge embree=2.17.7
conda install -c conda-forge pyembree
pip install pyglet==1.5
```

Step 8: Set environment variables in the conda environment:
```bash
conda env config vars set CUDA_HOME=/usr/local/cuda-11.7
conda env config vars set LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
conda env config vars set LIBRARY_PATH="/usr/local/cuda-11.7/lib64/stubs:$LIBRARY_PATH"
conda env config vars set CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
conda deactivate && conda activate sparsecraft
```

Remember to replace the path and the CUDA version with your own.

**Note** that the first time you run the code, it will take some time to compile the CUDA kernels.



## Data Preparation
### DTU Dataset
We provide preprocessed DTU data and results for the tasks of novel view synthesis and reconstruction. The data is available at [Google Drive](https://drive.google.com/file/d/1nuR1C80N8SvewFpggm3rqGQ8KghsgHIi)

Download the data and extract it to the root directory of this repository.

It contains the following directories:

```
sparsecraft_data
├── nvs # Novel View Synthesis task data and results
│   └── mvs_data
│       ├── scan103
│       ├── ...
│   └── results # Results for training using 3, 6, and 9 views
│       ├── 3v
│       │   ├── scan103
│       │   ├── ...
│       ├── 6v
│       │   ├── scan103
│       │   ├── ...
│       └── 9v
│           ├── scan103
│           ├── ...
└── reconstruction # Surface Reconstruction task data and results
    └── mvs_data # Surface reconstruction data uses a different set of scans and views than the novel view synthesis task
        ├── set0
        │   ├── scan105
        │   ├── ...
        └── set1
            ├── scan105
            ├── ...
    └── results
        ├── set0
        │   ├── scan105
        │   ├── ...
        └── set1
            ├── scan105
```


**Note**

The DTU dataset was preprocessed as follows:
- The original data is from the [NeuS Project](https://github.com/Totoro97/NeuS). We use the same camera poses and intrinsics as the original data.
- To obtain MVS data, we used the [Colmap](https://colmap.github.io/) initialized with the original camera poses and intrinsics.
- We provide a script that achieves this in `scripts` that you can run using the following command. Note that you will need to have Colmap installed on your machine:
```bash
python scripts/dtu_colmap_mvs.py --scan_dir sparsecraft_data/nvs/mvs_data/scan8 --task novel_view_synthesis --n_views 3
```

where `--scan_dir` specifies the directory of the scan that contains the images and the `cameras.npz` file, `--task` specifies the task: `novel_view_synthesis` or `surface_reconstruction`, and `--n_views` specifies the number of views to be used for the novel view synthesis task.

The script will generate the MVS data in the specified directory `scan_dir`.

### Custom data
Coming soon...

## Surface Reconstruction task on DTU dataset
### Training
The config files of the reconstruction task for running SparseCraft on the provided scenes are located in `configs/surface-reconstruction`.

You can find configurations for training on DTU with 3 in the `dtu-sparse-3v.yaml` file. Note that the reconstruction task uses a different set of scans and views than the novel view synthesis task.

You can use the following command to train the model on scan55 from the DTU dataset with 3 views:

```bash
python launch.py --config configs/surface-reconstruction/dtu-sparse-3v.yaml --gpu 0 --train tag=reconstruction_dtu_scan55_3v
```

where `--gpu` specifies the GPU to be used (GPU 0 will be used by default), and `--train` specifies the tag of the experiment. Use `python launch.py --help` to see all available options.

The checkpoints, and experiment outputs are saved to `exp_dir/[name]/[tag]@[timestamp]`, and tensorboard logs can be found at `runs_dir/[name]/[tag]@[timestamp]`.

You can change any configuration in the YAML config file by specifying arguments without `--`, for example:

```bash
python launch.py --config configs/surface-reconstruction/dtu-sparse-3v.yaml --gpu 0 --train tag=reconstruction_dtu_scan55_3v seed=0 trainer.max_steps=50000
```

The training procedure is followed by testing on the input images that you can just ignore, and exports the geometry as triangular meshes.

### Evaluation
To evaluate on the DTU dataset, you need to download SampleSet and Points from DTU's [website](http://roboimagedata.compute.dtu.dk/?page_id=36). Extract the data and make sure the directory structure is as follows:

```
SampleSet
├── MVS Data
      └── Points
```

Similar to previous works, we first clean the raw mesh with object masks by running:

```bash
python evaluation/clean_dtu_mesh.py --dtu_input_dir ./sparsecraft_data/reconstruction/mvs_data/set0/scan55 --mesh_dir exp_dir/reconstruction-3v-scan55/reconstruction_dtu_scan55_3v@[timestamp]/save
```

Then, run the evaluation script:

```bash
python evaluation/eval_dtu_python.py --dataset_dir PATH_to_SampleSet --mesh_dir ./exp_dir/reconstruction-3v-scan55/reconstruction_dtu_scan55_3v@[timestamp]/save/ --scan 55
```

The script computes the F-score, completeness, and accuracy metrics based on Chamfer distances between the generated mesh and the ground truth. 

The results are saved in the `exp_dir/[name]/[tag]@[timestamp]/save` directory.

**Reproducibility Note**


Please note that results can be slightly different across different runs due to the randomness in the training process. 

This is mainly due to the fact that the hash encoding library `tiny-cuda-nn` used in the codebase is not deterministic.

More details about this issue can be found in the tiny-cuda-nn repository [here](https://github.com/NVlabs/tiny-cuda-nn/issues/343).


## Novel View Synthesis task on DTU dataset
The config files of the novel view synthesis task for running SparseCraft on the provided scenes are located in `configs/novel-view-synthesis`.

You can find configurations for training on DTU with 3, 6 and 9 views in the `dtu-sparse-3v.yaml`, `dtu-sparse-6v.yaml`, and `dtu-sparse-9v.yaml` files, respectively.

Use the following command to train the model on scan55 from the DTU dataset with 3 views:

```bash
python launch.py --config configs/novel-view-synthesis/dtu-sparse-3v.yaml --gpu 0 --train tag=nvs_dtu_scan55_3v
```

The training procedure is followed by testing, which computes metrics on the provided test data, and exports the geometry as triangular meshes.

### Testing
If you want to do only testing, just resume the pretrained model and replace `--train` with `--test`, for example:
```bash
python launch.py --config path/to/your/exp/config/parsed.yaml --train --resume path/to/your/exp/ckpt/epoch=0-step=20000.ckpt --gpu 0
```

## Docker Image
Coming soon...

## Aknowledgment
This work is adapted from the repository [Instant Neural Surface Reconstruction](https://github.com/bennyguo/instant-nsr-pl). Many thanks to the author.

## Citation
If you find this codebase useful, please consider citing:
```
@article{younes2024sparsecraft,
        title={SparseCraft: Few-Shot Neural Reconstruction through Stereopsis Guided Geometric Linearization}, 
        author={Mae Younes and Amine Ouasfi and Adnane Boukhayma},
        year={2024},
        url={https://arxiv.org/abs/2407.14257}, 
}
@misc{instant-nsr-pl,
    Author = {Yuan-Chen Guo},
    Year = {2022},
    Note = {https://github.com/bennyguo/instant-nsr-pl},
    Title = {Instant Neural Surface Reconstruction}
}
```

