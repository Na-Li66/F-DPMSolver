# F-DPMSolver
Are First-Order Diffusion Samplers Really Slower? A Fast Forward-Value Approach

### Abstract

Higher-order ODE solvers have become a standard tool for accelerating diffusion probabilistic model (DPM) sampling, leading to the widespread belief that first-order methods are inherently slower and that increasing discretization order is the primary path to faster generation.  

In this work, we challenge this assumption and revisit acceleration from a complementary perspective: the placement of DPM evaluations along reverse-time dynamics. We show that sampling accuracy in the low neural function evaluation (NFE) regime is highly sensitive to evaluation locations, independent of solver order.

We propose a novel training-free first-order sampler whose leading discretization error has the opposite sign to that of DDIM. The method approximates the forward-value evaluation using a lightweight one-step lookahead predictor. We further provide theoretical guarantees showing that the proposed sampler provably approximates the ideal forward-value trajectory while retaining first-order convergence.

Extensive experiments on standard image generation benchmarks (CIFAR-10, ImageNet, FFHQ, and LSUN) demonstrate that our sampler consistently improves sample quality under the same NFE budget, and can be competitive with — and sometimes outperform — state-of-the-art higher-order samplers. Overall, our results highlight evaluation placement as an additional and largely orthogonal design dimension for accelerating diffusion sampling.

## Requirements

Please refer to [`environment.yaml`](./environment.yaml) for the complete list of dependencies.

You can create and activate the environment using Miniconda:

```bash
conda env create -f environment.yaml -n F-DPMSolver
conda activate F-DPMSolver
```

## Getting started

To reproduce the main results from our paper, simply run:

```bash
sh run.sh
```

## Pre-trained Models
We evaluate our method on four benchmark datasets using publicly available pre-trained diffusion models:
1. CIFAR-10 with EDM (Repository: https://github.com/NVlabs/edm)
2. ImageNet 64×64 with EDM2 (size S and L) (Repository: https://github.com/NVlabs/edm2)
3. ImageNet 512×512 with EDM2 (size XS and XXL) (Repository: https://github.com/NVlabs/edm2)
4. LSUN and FFHQ with Latent Diffusion (Repository: https://github.com/CompVis/latent-diffusion)

## Sample
To generate samples using a specified model and sampler, run:

```.bash
# Generate 1024 images using 4 GPUs
torchrun --standalone --nproc_per_node=4 main.py \
  --subdirs {images_dir} \
  --seeds=0-49999 \
  --NFE {NFE} \
  --batch {batch_size} \
  --algorithm_name "F-DPMSolver" \
  --model_name {model_name} \
  --order {order}
```

### Supported Models (model_name)
1. CIFAR10-cond
2. CIFAR10-uncond
3. ImageNet64-L
4. ImageNet64-S
5. ImageNet512-XXL
6. ImageNet512-XS
7. LSUN
8. FFHQ

### Sampler Order (order)
1 for F-DDIM and order>=2 for F-DPMSolver with higher order.

### Baseline Samplers
You can also reproduce baseline results by replacing algorithm_name with:
1. DDIM
2. DPMSolver
3. UniPC

## Calculating FID

To compute Fr&eacute;chet inception distance (FID) for a given model and sampler, first generate 50,000 random images and then compare them against the dataset reference statistics using `./fid/fid.py` for edm and `./fid/calculate_metrics_func.py`:

```.bash
# Calculate FID with images_dir of CIFAR10
torchrun --standalone --nproc_per_node=1 ./fid/fid.py  --ref_path=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --subdirs {images_dir}

# Calculate FID with images_dir of ImageNet
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img64.pkl --subdirs {images_dir}
torchrun --standalone --nproc_per_node=1 ./fid/calculate_metrics_func.py  --ref_path=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl --subdirs {images_dir}
```

Since latent diffusion does not provide official FID reference statistics, we preprocess the datasets and compute reference files manually. To facilitate evaluation, we provide the exact reference statistics that correspond to our pre-trained models: https://drive.google.com/drive/folders/1x1pEbZbD73giIOXUqLA3-h8W-GDORUym?usp=sharing.

The generated references are stored in the refs/ directory.

```.bash
# Image process and get reference of LSUN
unzip bedroom_train_lmdb
python ./fid/data_process/extract_data.py --lmdb bedroom_train_lmdb --out data/lsun/bedroom/1024 --fmt png --workers 8 --skip-existing
python ./fid/data_process/lsun_preprocess.py
python ./fid/fid.py ref --data=images_raw/bedroom/256 --dest=refs/lsun-bedroom.npz

# Image process and get reference of FFHQ
python ./fid/data_process/FFHQ_preprocess.py
python ./fid/fid.py ref --data= data/ffhq/256 --dest=refs/ffhq-256.npz
```

After generating the reference statistics, FID is calculated by
```.bash
# Calculate FID with images_dir of LSUN
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs {images_dir} --ref_path refs/lsun-bedroom.npz

# Calculate FID with images_dir of FFHQ
torchrun --standalone --nproc_per_node=1 ./fid/fid.py --subdirs {images_dir} --ref_path refs/ffhq-256.npz
```

## Results of F-DDIM

| NFE | CIFAR10-Cond | CIFAR10-uncond | ImageNet64-S | ImageNet64-L | ImageNet512-XS | ImageNet512-XXL | LSUN | FFHQ |
| --- | -----------: | -------------: | -----------: | -----------: | -------------: | --------------: | ---: | ---: |
| 4   | 18.44 | 25.01 | 20.55 | 22.36 | 50.96 | 52.32 | 12.18 | 15.79 |
| 5   | 12.29 | 16.04 | 10.70 | 11.98 | 22.69 | 23.56 | 6.98  | 9.75  |
| 6   | 7.57  | 9.47  | 6.67  | 7.21  | 14.33 | 14.04 | 5.57  | 7.89  |
| 8   | 4.32  | 4.91  | 3.50  | 3.64  | 6.18  | 5.04  | 4.93  | 6.80  |
| 10  | 3.18  | 3.46  | 2.56  | 2.51  | 4.57  | 3.06  | 4.66  | 6.46  |
