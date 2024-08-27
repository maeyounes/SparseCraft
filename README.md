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

#### Note

The first time you run the code, it will take some time to compile the CUDA kernels.



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


#### Note

The DTU dataset was preprocessed as follows:
- The original data is from the [NeuS Project](https://github.com/Totoro97/NeuS). We use the same camera poses and intrinsics as the original data.
- To obtain MVS data, we used [COLMAP](https://colmap.github.io/) initialized with the original camera poses and intrinsics.
- We provide a script that achieves this in `scripts` that you can run using the following command. Note that you will need to have COLMAP installed on your machine:
```bash
python scripts/dtu_colmap_mvs.py --scan_dir sparsecraft_data/nvs/mvs_data/scan8 --task novel_view_synthesis --n_views 3
```

where `--scan_dir` specifies the directory of the scan that contains the images and the `cameras.npz` file, `--task` specifies the task: `novel_view_synthesis` or `surface_reconstruction`, and `--n_views` specifies the number of views to be used for the novel view synthesis task.

The script will generate the MVS data in the specified directory `scan_dir`.

### Custom data
For custom data preparation, you can have the images in a folder `scene_name/images` and run the following command to obtain the camera calibration using [COLMAP](https://colmap.github.io/):
```bash
python scripts/run_colmap.py absolute_path/to/your/scene_name
```
then run the following script to generate the MVS data:
```bash
python scripts/custom_colmap_mvs.py --scene_dir absolute_path/to/your/scene_name
```
where `--scene_dir` specifies the directory of the scene that contains the images and the files generated by COLMAP.

This process can take some time depending on the number of images and the resolution of the images. Also make sure to use the CUDA version of COLMAP for faster processing.

After running the script, you will find the MVS data in the `scene_name/all_views/dense/fused.ply`.

You can visualize the generated point cloud using [Meshlab](https://www.meshlab.net) or any other point cloud viewer.

#### Note

In case COLMAP fails to generate the MVS data, or the generated point cloud is not dense enough and gives bad results, you can use more advanced MVS methods (traditional or deep learning-based methods) to generate the MVS data.

You can find some of the available MVS methods in the [Awesome-MVS](https://github.com/walsvid/Awesome-MVS).


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

#### Reproducibility Note

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
If you want to do only testing, given that you have a trained model, you can use the following command:
```bash
python launch.py --config path/to/your/exp/config/parsed.yaml --train --resume path/to/your/exp/ckpt/epoch=0-step=20000.ckpt --gpu 0
```

## Surface Reconstruction task on Custom data
### Training
You can find configurations for training on custom data in the `configs/surface-reconstruction/custom-sparse.yaml` file.

#### Dataset Notes
- Adapt the `root_dir` and `img_wh` (or `img_downscale`) option in the config file to your data;
- The scene is normalized so that cameras have a minimum distance `1.0` to the center of the scene. Setting `model.radius=1.0` works in most cases. If not, try setting a smaller/larger radius that wraps tightly to your foreground object.
- There are three choices to determine the scene center: `dataset.center_est_method=camera` uses the center of all camera positions as the scene center; `dataset.center_est_method=lookat` assumes the cameras are looking at the same point and calculates an approximate look-at point as the scene center; `dataset.center_est_method=point` uses the center of all points (reconstructed by COLMAP) that are bounded by cameras as the scene center. Please choose an appropriate method according to your capture.

#### Performance Notes
- Depending on your hardware, you may need to adjust the `model.num_samples_per_ray` and `model.max_train_num_rays` options in the config file to fit your GPU memory and to for a trade-off between performance and training time: a larger number of samples per ray and rays will improve the reconstruction quality but will require more memory and time.
- The `trainer.max_steps` option in the config file specifies the number of training steps. You can adjust this value according to the number of input images and the complexity of the scene. Checkout the differences between config files for the novel view synthesis task for a better understanding of the parameters.  
- You will also have to tune other hyperparameters for the hash encoding depending on the complexity of the scene and the number of input images. To regularize the geometry by progressively increasing the capacity of the hash encoding, you have to tune the hash encoding parameters `start_level` and `update_steps`. We show in the paper that for the few shot setting, it is beneficial to have `update_steps` it set such that the model is at full capacity at 80% of the training time.
- Keep in mind that because the capacity of the hash encoding is limited at the beginning of the training, it is important to regularize the specular component of the color by setting `lambda_specular_color` to a non-zero value. This will help the model to focus on the geometry at the beginning of the training.
- Depending on the density of the obtained MVS point cloud, you might need to adjust the sampling parameters used for Taylor based regularizations: `sampling_taylor_step`, `sampling_taylor_sigma` and `sampling_taylor_n`.


You can use the following command to train the model on a custom scene:

```bash
python launch.py --config configs/surface-reconstruction/custom-sparse.yaml --gpu 0 --train tag=reconstruction_custom_scene
```



## Docker Image
Coming soon...

## Aknowledgment
This work is mainly adapted from the repository [Instant Neural Surface Reconstruction](https://github.com/bennyguo/instant-nsr-pl), with some code snippets taken from [NeuS](https://github.com/Totoro97/NeuS), [Instant-angelo](https://github.com/hugoycj/Instant-angelo), [FSGS](https://github.com/VITA-Group/FSGS), [LLFF](https://github.com/Fyusion/LLFF), and [RegNerf](https://github.com/m-niemeyer/regnerf).

Many thanks to all the authors for their great work and contributions.

## Citation
If you find this codebase useful, please consider citing:
```
@article{younes2024sparsecraft,
        title={SparseCraft: Few-Shot Neural Reconstruction through Stereopsis Guided Geometric Linearization}, 
        author={Mae Younes and Amine Ouasfi and Adnane Boukhayma},
        year={2024},
        url={https://arxiv.org/abs/2407.14257}
}
```

