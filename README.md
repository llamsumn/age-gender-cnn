# CM52053 Coursework 2: Age and Gender Prediction

Backup repository for **CM52053 - Artificial Intelligence and Machine Learning** Coursework 2.

## Task

Train two CNN models to estimate a person's age and predict their gender from 128x128 face images (UTKFace dataset, 5000 images).

## Model Summary

### Model A - Custom CNN (trained from scratch)

- **Architecture**: 5 convolutional blocks (32 → 64 → 128 → 256 → 512 filters), each with Conv2D, BatchNormalization, ReLU, and MaxPooling. Separate Dense heads for age (regression) and gender (classification).
- **Data Augmentation**: Horizontal flip
- **Loss Weights**: `age_output: 0.1, gender_output: 1.0`
- **Callbacks**: EarlyStopping (patience 15), ReduceLROnPlateau (factor 0.5, patience 5)
- **Results**: Val Age MAE ~7.07 | Val Gender Accuracy ~86%

### Model B - EfficientNetB0 (transfer learning, two-phase)

- **Architecture**: EfficientNetB0 (ImageNet pre-trained) backbone with GlobalAveragePooling2D, followed by separate Dense heads for age and gender.
- **Data Augmentation**: Horizontal flip
- **Phase 1**: Backbone frozen, trained 30 epochs (Adam, lr=1e-3) to warm up heads
- **Phase 2**: Full network unfrozen, fine-tuned up to 100 epochs (Adam, lr=1e-4)
- **Loss Weights**: `age_output: 0.1, gender_output: 1.0`
- **Results**: Val Age MAE ~6.18 | Val Gender Accuracy ~88%

## Repository Structure

```
age_gender_submit.ipynb   # Main notebook (run on Google Colab) - contains frozen training outputs
local_test.ipynb          # Local visualisation notebook for reproducibility
plot_learning_curves.py   # Utility script to plot training history from .npy files
models/
  age_gender_A.keras      # Trained Model A
  age_gender_B.keras      # Trained Model B
  history_A.npy           # Model A training history
  history_B.npy           # Model B training history
```

## Reproducibility

- **`age_gender_submit.ipynb`**: The submitted notebook with frozen screenshot outputs from Google Colab. Contains all training code, model definitions, and learning curve plots as executed on Colab.
- **`local_test.ipynb`**: Used to visualise training outcomes locally by loading the saved models and history files. This allows verification of results without re-running training on Google Colab.
