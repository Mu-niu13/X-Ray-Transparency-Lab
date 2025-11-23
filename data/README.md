# Data Directory

## Download Instructions

1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download the dataset (5.3 GB)
3. Extract the ZIP file here

After extraction, you should have this structure:

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/      (1,341 images)
    │   └── PNEUMONIA/   (3,875 images)
    ├── test/
    │   ├── NORMAL/      (234 images)
    │   └── PNEUMONIA/   (390 images)
    └── val/
        ├── NORMAL/      (8 images)
        └── PNEUMONIA/   (8 images)
```

## Dataset Information

- **Total Images**: 5,856 chest X-ray images
- **Classes**: NORMAL, PNEUMONIA
- **Source**: Guangzhou Women and Children's Medical Center
- **Format**: JPEG images (variable sizes)

## Notes

- The validation set is very small (16 images). Consider creating a larger validation split from the training set.
- Images are grayscale chest X-rays
- Some images may need preprocessing (resizing, normalization)