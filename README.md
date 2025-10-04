# Q1:

### Implement a Vision Transformer (ViT) and train it on the CIFAR-10 dataset (10 classes)
Reference Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., ICLR 2021).

## How to Run (in Colab)
* Upload q1.ipynb to Google Colab.

* Set runtime to GPU.

* All necessary dependencies are installed in the notebook, except for the dataset, which will install on running cell 6 in [q1.ipynb](https://github.com/shreshth3000/CIFARTask/blob/main/q1.ipynb), so cells can be run sequentially.

* Training will start automatically and use the CIFAR-10 dataset downloaded as a torchvision dataset.

**Best Model Config**
| Parameter           | Value                                |
|---------------------|--------------------------------------|
| patch_size          | 4                                    |
| embed_dim           | 384                                  |
| depth (blocks)      | 6                                    |
| num_heads           | 12                                   |
| mlp_ratio           | 4.0                                  |
| drop_rate           | 0.1                                  |
| epochs              | 200                                  |
| batch_size          | 128                                  |
| learning_rate       | 9e-4                                 |
| weight_decay        | 0.1                                  |
| optimizer           | AdamW                                |
| scheduler           | Cosine with 15 percent warmup        |
| label smoothing     | 0.1                                  |
| Data Augmentations  | RandomCrop, RandomHorizontalFlip, ColorJitter |

## Results
| Metric        | Value   |
|---------------|---------|
| Test Accuracy | 79.47%  |

## Short Analysis
- Dataset (CIFAR-10) proved challenging to train on due to low volumes of data, just 60000 32X32 images.
  
- Patch size: Smaller patches (4x4) helped the model capture more local detail, improving accuracy on CIFAR-10 compared to larger patch sizes.

- Depth/width trade-off: A moderately deep (6-layer) transformer with 12 heads and a larger embedding (384) was optimal. Increasing depth further led to overfitting, while reducing width reduced capacity.
  Other configurations included 6-layers with 8 attention heads per layer (thus having an embedding dimension of 256 as for all configurations above 4 layers the embed_dim/num_heads ratio has been kept at 32), which was slow to converge,
  and 3 layers with 4 attention heads per block, which underfit the data.

- Data augmentation: Basic augmentations like crop, flip, and color jitter improved generalization. Mixup and CutMix (batchwise) were tested but only destablised training. I also tested RandAugment, but
  it did not give any added benefits.

- Optimizer and schedule: AdamW with cosine scheduling and warmup stabilized training and improved convergence over simpler optimizers.

- Overlapping vs non-overlapping patches: Standard non-overlapping patches were used; attempts at overlap did not yield significant improvement for this dataset and model size (Such as shifted patch tokenization).

# Q2

### Implement text-prompted segmentation of objects using SAM 2

## How to Run (in Colab)

* Upload q2.ipynb to Google Colab
* Set runtime to GPU
* All necessary dependencies are installed in the notebook cells at the top, but will be prompted to restart the session in order to use required dependencies.
* Run cells sequentially beginning from the installation section.
* Upload your image when prompted.
* Enter your text prompt based on the image.

## Pipeline Overview

### Step 1: Text-to-Regions (GroundingDINO)
* Load GroundingDINO model for open-vocabulary object detection
* Input: Image + text prompt
* Output: Bounding boxes of detected objects matching the text description

### Step 2: Region-to-Masks (SAM 2)
* Load SAM 2 model for high-quality segmentation
* Input: Image + bounding boxes from Step 1
* Output: Precise pixel-level segmentation masks

### Step 3: Visualization
* Overlay segmentation masks on original image
* Display bounding boxes with masks for verification

## Model Configuration

| Component | Model |
|-----------|-------|
| Object Detection | GroundingDINO (Swin-T backbone) |
| Segmentation | SAM 2 Hiera-Large |
| Detection Threshold | 0.35 (box), 0.25 (text) |
| Device | CUDA (GPU recommended) |

## Limitations

* Performance depends on GroundingDINO's ability to detect objects from text descriptions, so complex or ambiguous text prompts may result in false positives or missed detections
* Small objects or crowded scenes may be challenging for accurate segmentation
* Model and dependency loading time can be significant on first run
* Text prompts must be in English and simple descriptions work best



