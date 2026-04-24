import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

# ─────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────

@dataclass
class TrainingHistory:
    loss: list = field(default_factory=list)
    iou: list = field(default_factory=list)
    precision: list = field(default_factory=list)
    recall: list = field(default_factory=list)

    def update(self, loss, iou):
        self.loss.append(float(loss))
        self.iou.append(float(iou))

    def last(self):
        return self.loss[-1], self.iou[-1]


def plot_history(history: TrainingHistory, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.loss)
    ax1.set_title("Loss")
    ax1.set_xlabel("Step")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.iou)
    ax2.set_title("Mean IoU")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Graphs saved : {save_path}")

    plt.show()


# ─────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────

def save_checkpoint(model, epoch, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.weights.h5")
    model.save_weights(path)
    print(f"Checkpoint : {path}")
    return path


def load_latest_checkpoint(model, checkpoint_dir="./checkpoints"):
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint found, start from zero.")
        return 0

    checkpoints = sorted([
        f for f in os.listdir(checkpoint_dir) if f.endswith(".weights.h5")
    ])

    if not checkpoints:
        print("No checkpoint found, start from zero.")
        return 0

    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    model.load_weights(latest)
    epoch = int(checkpoints[-1].split("_")[1].split(".")[0])
    print(f"Resuming from : {latest} (epoch {epoch})")
    return epoch


# ─────────────────────────────────────────────
# Progressive display progression
# ─────────────────────────────────────────────

def print_step(epoch, total_epochs, step, total_steps, loss, iou, step_time):
    epoch_str = f"{epoch + 1:02d}/{total_epochs}"
    step_str = f"{step:04d}/{total_steps}"
    print(
        f"\r  Epoch {epoch_str} | Step {step_str} "
        f"| loss {loss:.4f} | iou {iou:.4f} "
        f"| {step_time:.2f}s/step",
        end=""
    )


def print_epoch(epoch, total_epochs, loss, iou, epoch_time):
    epoch_str = f"{epoch + 1:02d}/{total_epochs}"
    print(
        f"\r  Epoch {epoch_str} "
        f"| loss {loss:.4f} | iou {iou:.4f} "
        f"| {epoch_time:.1f}s"
    )