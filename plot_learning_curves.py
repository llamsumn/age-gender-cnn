import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_learning_curves(history_path, model_name="Model"):
    """
    Plot 4 learning curve figures from a saved history .npy file.
    
    Args:
        history_path: path to the .npy history file
        model_name: label for plot titles (e.g. "Model A", "Model B")
    """
    history_dict = np.load(history_path, allow_pickle=True).item()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{model_name} - Learning Curves', fontsize=14)

    # Figure 1: Gender Loss
    axes[0, 0].plot(history_dict['gender_output_loss'], label='Train')
    axes[0, 0].plot(history_dict['val_gender_output_loss'], label='Validation')
    axes[0, 0].set_title('Gender Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Binary Crossentropy')
    axes[0, 0].legend()

    # Figure 2: Gender Accuracy
    axes[0, 1].plot(history_dict['gender_output_accuracy'], label='Train')
    axes[0, 1].plot(history_dict['val_gender_output_accuracy'], label='Validation')
    axes[0, 1].set_title('Gender Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Figure 3: Age Loss
    axes[1, 0].plot(history_dict['age_output_loss'], label='Train')
    axes[1, 0].plot(history_dict['val_age_output_loss'], label='Validation')
    axes[1, 0].set_title('Age Loss (MSE)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()

    # Figure 4: Age MAE
    axes[1, 1].plot(history_dict['age_output_mae'], label='Train')
    axes[1, 1].plot(history_dict['val_age_output_mae'], label='Validation')
    axes[1, 1].set_title('Age MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE (years)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_learning_curves.py <history.npy> [model_name]")
        print("Example: python plot_learning_curves.py history_A_backup.npy 'Model A'")
        sys.exit(1)

    history_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Model"
    plot_learning_curves(history_path, model_name)