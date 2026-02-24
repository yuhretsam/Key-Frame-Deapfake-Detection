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
│   └── train.py
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
python scripts/extract_keyframes_kmeans.py ^
  --input_root data/videos ^
  --output_root data ^
  --method kmeans ^
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
python scripts/extract_keyframes_optical_flow.py ^
  --input_root data/videos ^
  --output_root data ^
  --method opticalflow ^
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
python scripts/train.py ^
  --data_root data ^
  --method kmeans ^
  --class_folders fake real ^
  --model resnet50 ^
  --epochs 30
```

Optical Flow example:
```bash
python scripts/train.py ^
  --data_root data ^
  --method opticalflow ^
  --class_folders fake real ^
  --model efficientnet_b0
```

If your keyframes are directly in `data/fake` and `data/real`:
```bash
python scripts/train.py ^
  --data_root data ^
  --class_folders fake real ^
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

## Step 4: Inference (Single Video)
Run inference directly from a video file. The script will extract keyframes
using the selected method, load weights, and output the predicted label.

```bash
python scripts/infer.py ^
  --video_path path/to/video.mp4 ^
  --method opticalflow ^
  --model resnet50 ^
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
MIT License

Copyright (c) 2026 Nguyễn Hồ Nhật Huy

Permission is hereby granted, free of charge, to any person obtaining a copy of this repository and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.