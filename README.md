# Deepfake Video Detection with Keyframe Selection

This project detects deepfake videos by extracting a small number of keyframes
instead of using all frames. It supports two keyframe selection methods and
trains CNN + LSTM models on the extracted keyframes.

## Highlights
- Keyframe selection via **K-Means** on CNN features.
- Keyframe selection via **Optical Flow** focused on face regions.
- CNN + LSTM training for **fake vs real** classification.
- Backbones included: **VGG16**, **ResNet50**, **EfficientNet-B0**, **MobileNetV2**.

## Project Structure
```
.
├── src/
│   ├── data_preprocessing/
│   │   ├── kmeans_keyframes.py
│   │   ├── optical_flow_keyframes.py
│   │   └── keyframe_dataset.py
│   ├── models/
│   │   └── cnn_lstm.py
│   ├── training/
│   │   ├── train.py
│   │   └── metrics.py
│   └── utils/
│       └── io.py
├── scripts/
│   ├── extract_keyframes_kmeans.py
│   ├── extract_keyframes_optical_flow.py
│   ├── infer.py
│   ├── test.py
│   └── train.py
├── LICENSE
├── requirements.txt
└── README.md
```

## Utility Helpers
`src/utils/io.py` contains small filesystem helpers used across scripts:
- `list_video_files()` to collect video files from a folder
- `ensure_dir()` to create output directories safely

`src/data_preprocessing/` contains keyframe extraction logic and dataset loading
utilities for training.

## Installation
```bash
pip install -r requirements.txt
```

## Data Layout (Keyframes)
This project uses a unified folder layout for both keyframe methods.
```
data/
  kmeans/
    fake/
      video_001/
        cluster_0.png
        cluster_1.png
    real/
      video_002/
        cluster_0.png
  opticalflow/
    fake/
      video_003/
        flow_face_0.png
        flow_face_1.png
    real/
      video_004/
        flow_face_0.png
```
If your dataset already has `train/val/test`, place those as an extra level:
```
data/kmeans/fake/train/<video_id>/*.png
data/kmeans/real/train/<video_id>/*.png
data/opticalflow/fake/train/<video_id>/flow_face_*.png
```
If you keep a single dataset at `data/fake` and `data/real`, just omit
`--method` when training.

## Step 1: Keyframe Selection (K-Means)
Extract keyframes by clustering frame features (ResNet50) and saving frames
closest to cluster centers.

```bash
python scripts/extract_keyframes_kmeans.py \
  --input_root data/videos \
  --output_root data \
  --method kmeans \
  --min_k 3 --max_k 15
```

Expected input:
```
data/videos/
  fake/
    *.mp4
  real/
    *.mp4
```
Both extraction scripts process **fake** and **real** in a single run when
those folders exist under the same `input_root`.

Output:
```
data/kmeans/
  fake/<video_id>/*.png
  real/<video_id>/*.png
```

## Step 2: Keyframe Selection (Optical Flow)
Extract keyframes using optical flow energy in face regions (MTCNN).

```bash
python scripts/extract_keyframes_optical_flow.py \
  --input_root data/videos \
  --output_root data \
  --method opticalflow \
  --threshold_ratio 0.3
```

Expected input:
```
data/videos/
  fake/
    *.mp4
  real/
    *.mp4
```
Both extraction scripts process **fake** and **real** in a single run when
those folders exist under the same `input_root`.

Output:
```
data/opticalflow/
  fake/<video_id>/flow_face_*.png
  real/<video_id>/flow_face_*.png
```

## Step 3: Training
Train a CNN + LSTM model on keyframe sequences. Choose a model and a keyframe
method by name; metrics are printed for train/val/test.

```bash
python scripts/train.py \
  --data_root data \
  --method kmeans \
  --class_folders fake real \
  --model resnet50 \
  --epochs 30
```

Optical Flow example:
```bash
python scripts/train.py \
  --data_root data \
  --method opticalflow \
  --class_folders fake real \
  --model efficientnet_b0
```

If your keyframes are directly in `data/fake` and `data/real`:
```bash
python scripts/train.py \
  --data_root data \
  --class_folders fake real \
  --model vgg16
```

### Training Options
Common hyperparameters you can tune:
- `--epochs`
- `--lr`
- `--batch_size`
- `--max_seq_length`
- `--img_size`
- `--patience`

## Step 4: Cross-dataset Evaluation

Use `scripts/test.py` to evaluate a checkpoint trained on Dataset A against
all videos in Dataset B. Dataset B is used **100% as a test set**: it is not
split, and no training or fine-tuning is performed.

Workflow:

```text
Dataset A
    ↓
scripts/train.py
    ↓
checkpoint
    ↓
scripts/test.py
    ↓
100% of Dataset B
    ↓
cross-dataset metrics
```

Unlike a training dataset, the cross-test dataset must not contain
`train/val/test` directories. Its extracted keyframes must use this layout:

```text
data/CelebDF/kmeans/
├── fake/
│   ├── video001/
│   │   ├── cluster_0.png
│   │   └── cluster_1.png
│   └── ...
└── real/
    ├── video100/
    │   ├── cluster_0.png
    │   └── cluster_1.png
    └── ...
```

For example, to test a checkpoint trained on FF++ against all of CelebDF:

```bash
python scripts/test.py \
  --data_root data/CelebDF \
  --method kmeans \
  --class_folders fake real \
  --model resnet50 \
  --weights output/FFPP/best_resnet50.pth \
  --batch_size 8 \
  --output_dir output/crosstest/FFPP_to_CelebDF
```

For optical-flow keyframes, change `--method` to `opticalflow` and point
`--data_root` at the dataset directory containing `opticalflow/fake` and
`opticalflow/real`.

The script prints loss, accuracy, precision, recall, F1, AUC, and EER. It also
writes `metrics.json` and per-video probabilities to `predictions.csv` inside
`--output_dir`. The class mapping is identical to training because both scripts
reuse the same dataset loader (`fake=0`, `real=1` for the default classes).

## Step 5: Inference (Single Video)
Run inference directly from a video file. The script will extract keyframes
using the selected method, load weights, and output the predicted label.

```bash
python scripts/infer.py \
  --video_path path/to/video.mp4 \
  --method opticalflow \
  --model resnet50 \
  --weights checkpoints/best_resnet50.pth
```

### Backbones
Use any of these values for `--model`:
- `vgg16`
- `resnet50`
- `efficientnet_b0`
- `mobilenet_v2`


## Notes
- Choose the extraction method with `--method kmeans` or `--method opticalflow`.
- If no `train/val/test` folders exist, the training script will split the data
  randomly using `--val_ratio` and `--test_ratio`.
- For large datasets, consider sampling frames during extraction to speed up
  processing.

## License

This project is licensed under the [MIT License](LICENSE).
