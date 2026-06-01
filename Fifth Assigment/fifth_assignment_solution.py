from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix


AEM = 6931
EPOCHS = 30
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.10

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train_dataset" / "train_dataset"
TEST_DIR = BASE_DIR / "test_dataset" / "test_dataset"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_CACHE_PATH = OUTPUT_DIR / f"sampled_mnist_aem_{AEM}.npz"


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def read_labels(dataset_dir: Path) -> pd.DataFrame:
    labels_path = dataset_dir / "labels.csv"
    df = pd.read_csv(labels_path)
    df["label"] = df["label"].astype(int)
    return df


def sample_dataframes(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_full = read_labels(TRAIN_DIR)
    test_full = read_labels(TEST_DIR)
    train_df = train_full.sample(frac=0.90, random_state=seed).reset_index(drop=True)
    test_df = test_full.sample(frac=0.80, random_state=seed).reset_index(drop=True)
    return train_full, test_full, train_df, test_df


def image_index(rel_path: str) -> int:
    return int(Path(rel_path).stem.split("_")[-1])


def validate_fallback_labels(df: pd.DataFrame, fallback_labels: np.ndarray, subset_name: str) -> None:
    for rel_path, label in zip(df["file"], df["label"]):
        idx = image_index(rel_path)
        if int(fallback_labels[idx]) != int(label):
            raise ValueError(
                f"Fallback MNIST label mismatch in {subset_name}: {rel_path} has "
                f"CSV label {label}, but fallback label {fallback_labels[idx]}"
            )


def count_missing_files(df: pd.DataFrame, dataset_dir: Path) -> int:
    return sum(1 for rel_path in df["file"] if not (dataset_dir / rel_path).is_file())


def load_images(
    df: pd.DataFrame,
    dataset_dir: Path,
    fallback_images: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    images = np.empty((len(df), 28, 28), dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    missing_count = 0

    for i, rel_path in enumerate(df["file"]):
        image_path = dataset_dir / rel_path
        if image_path.is_file():
            with Image.open(image_path) as image:
                images[i] = np.asarray(image.convert("L"), dtype=np.float32)
        elif fallback_images is not None:
            images[i] = fallback_images[image_index(rel_path)].astype(np.float32)
            missing_count += 1
        else:
            raise FileNotFoundError(f"Missing image and no fallback was provided: {image_path}")

        if (i + 1) % 10000 == 0:
            print(f"Loaded {i + 1}/{len(df)} images from {dataset_dir.name}")

    if missing_count:
        print(f"Used Keras MNIST fallback for {missing_count} missing files from {dataset_dir.name}")

    images /= 255.0
    return images, labels


def load_or_prepare_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if DATA_CACHE_PATH.is_file():
        print(f"Loading cached sampled arrays from {DATA_CACHE_PATH.name}")
        cached = np.load(DATA_CACHE_PATH)
        return cached["x_train"], cached["y_train"], cached["x_test"], cached["y_test"]

    x_train, y_train, x_test, y_test = load_or_prepare_arrays(train_df, test_df)
    np.savez_compressed(DATA_CACHE_PATH, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Saved sampled arrays cache to {DATA_CACHE_PATH.name}")
    return x_train, y_train, x_test, y_test


def save_class_balance(
    train_full: pd.DataFrame,
    test_full: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    subsets = {
        "train_full": train_full,
        "test_full": test_full,
        "train_sample_90pct": train_df,
        "test_sample_80pct": test_df,
    }

    for subset, df in subsets.items():
        counts = df["label"].value_counts().sort_index()
        total = len(df)
        for digit in range(10):
            count = int(counts.get(digit, 0))
            rows.append(
                {
                    "subset": subset,
                    "digit": digit,
                    "count": count,
                    "percent": 100 * count / total,
                }
            )

    balance_df = pd.DataFrame(rows)
    balance_df.to_csv(OUTPUT_DIR / "class_balance.csv", index=False)

    plt.figure(figsize=(10, 5.5))
    plot_df = balance_df[balance_df["subset"].isin(["train_sample_90pct", "test_sample_80pct"])]
    sns.barplot(data=plot_df, x="digit", y="count", hue="subset", palette=["#2b6cb0", "#dd6b20"])
    plt.title("Class balance after AEM sampling")
    plt.xlabel("Digit")
    plt.ylabel("Number of samples")
    plt.legend(title="Subset")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png", dpi=180)
    plt.close()
    return balance_df


def save_sample_digits(x_train: np.ndarray, y_train: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.6))
    for digit, ax in enumerate(axes.ravel()):
        idx = int(np.flatnonzero(y_train == digit)[0])
        ax.imshow(x_train[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {digit}")
        ax.axis("off")
    fig.suptitle("One sampled training image per digit", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_digits.png", dpi=180, bbox_inches="tight")
    plt.close()


def build_simple_nn() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(784,), name="input_784"),
            tf.keras.layers.Dense(784, activation="relu", name="dense_784_relu"),
            tf.keras.layers.Dense(256, activation="relu", name="dense_256_relu"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_128_relu"),
            tf.keras.layers.Dense(10, activation="softmax", name="output_softmax"),
        ],
        name="simple_nn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), name="input_image"),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_32_relu"),
            tf.keras.layers.MaxPooling2D((2, 2), name="maxpool_2x2"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_64_relu"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(64, activation="relu", name="dense_64_relu"),
            tf.keras.layers.Dense(10, activation="softmax", name="output_softmax"),
        ],
        name="cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model_summary(model: tf.keras.Model, filename: str) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    (OUTPUT_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_history_csv(history: tf.keras.callbacks.History, model_key: str) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(OUTPUT_DIR / f"{model_key}_history.csv", index=False)


def completed_epochs(model_key: str) -> int:
    history_path = OUTPUT_DIR / f"{model_key}_history.csv"
    if not history_path.is_file():
        return 0
    history_df = pd.read_csv(history_path)
    if history_df.empty:
        return 0
    return int(history_df["epoch"].max())


class HistoryCsvLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_key: str) -> None:
        super().__init__()
        self.path = OUTPUT_DIR / f"{model_key}_history.csv"

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        row = {"epoch": epoch + 1}
        for key in ["accuracy", "loss", "val_accuracy", "val_loss"]:
            row[key] = float(logs[key]) if key in logs else np.nan
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=not self.path.is_file(), index=False)


def plot_history(history: tf.keras.callbacks.History, filename: str, title: str) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    plot_history_df(history_df, filename, title)


def plot_history_csv(model_key: str, filename: str, title: str) -> None:
    history_df = pd.read_csv(OUTPUT_DIR / f"{model_key}_history.csv")
    plot_history_df(history_df, filename, title)


def plot_history_df(history_df: pd.DataFrame, filename: str, title: str) -> None:
    epochs = history_df["epoch"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(epochs, history_df["accuracy"], label="Train accuracy", color="#2b6cb0")
    axes[0].plot(epochs, history_df["val_accuracy"], label="Validation accuracy", color="#dd6b20")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history_df["loss"], label="Train loss", color="#2b6cb0")
    axes[1].plot(epochs, history_df["val_loss"], label="Validation loss", color="#dd6b20")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_key: str,
    title: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = list(range(10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels])
    cm_df.to_csv(OUTPUT_DIR / f"{model_key}_confusion_matrix.csv")

    plt.figure(figsize=(7.2, 6.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted digit")
    plt.ylabel("True digit")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_key}_confusion_matrix.png", dpi=180)
    plt.close()

    pairs = []
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    for true_digit in labels:
        for pred_digit in labels:
            count = int(off_diag[true_digit, pred_digit])
            if count > 0:
                pairs.append({"true_digit": true_digit, "predicted_digit": pred_digit, "count": count})
    pairs_df = pd.DataFrame(pairs).sort_values("count", ascending=False).reset_index(drop=True)
    pairs_df.to_csv(OUTPUT_DIR / f"{model_key}_top_confusions.csv", index=False)
    return cm, pairs_df


def train_and_evaluate(
    model: tf.keras.Model,
    model_key: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | int | str]:
    print(f"\nTraining {model.name}...")
    checkpoint_path = OUTPUT_DIR / f"{model_key}_checkpoint.keras"
    start_epoch = completed_epochs(model_key)
    if checkpoint_path.is_file() and 0 < start_epoch < EPOCHS:
        print(f"Resuming {model_key} from epoch {start_epoch}/{EPOCHS}.")
        model = tf.keras.models.load_model(checkpoint_path)
    elif checkpoint_path.is_file() and start_epoch >= EPOCHS:
        print(f"{model_key} already has {start_epoch} logged epochs. Loading checkpoint for evaluation.")
        model = tf.keras.models.load_model(checkpoint_path)

    save_model_summary(model, f"{model_key}_summary.txt")
    if start_epoch < EPOCHS:
        model.fit(
            x_train,
            y_train,
            epochs=EPOCHS,
            initial_epoch=start_epoch,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            shuffle=True,
            verbose=2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), save_best_only=False),
                HistoryCsvLogger(model_key),
            ],
        )
    plot_history_csv(model_key, f"{model_key}_history.png", f"{model.name}: training history")

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    save_confusion_matrix(y_test, predictions, model_key, f"{model.name}: confusion matrix")

    history_df = pd.read_csv(OUTPUT_DIR / f"{model_key}_history.csv")
    last_epoch = history_df.iloc[-1]
    result = {
        "model": model.name,
        "parameters": int(model.count_params()),
        "train_accuracy_epoch_30": float(last_epoch["accuracy"]),
        "validation_accuracy_epoch_30": float(last_epoch["val_accuracy"]),
        "train_loss_epoch_30": float(last_epoch["loss"]),
        "validation_loss_epoch_30": float(last_epoch["val_loss"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    pd.DataFrame([result]).to_csv(OUTPUT_DIR / f"{model_key}_metrics.csv", index=False)
    return result


def parse_total_params(model_key: str) -> int | None:
    summary_path = OUTPUT_DIR / f"{model_key}_summary.txt"
    if not summary_path.is_file():
        return None

    match = re.search(r"Total params:\s*([\d,]+)", summary_path.read_text(encoding="utf-8"))
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def infer_existing_metrics(model_key: str, model_name: str) -> dict[str, float | int | str] | None:
    metrics_path = OUTPUT_DIR / f"{model_key}_metrics.csv"
    if metrics_path.is_file():
        return pd.read_csv(metrics_path).iloc[0].to_dict()

    cm_path = OUTPUT_DIR / f"{model_key}_confusion_matrix.csv"
    if not cm_path.is_file():
        return None

    cm = pd.read_csv(cm_path, index_col=0).to_numpy()
    test_accuracy = float(np.trace(cm) / cm.sum())
    result: dict[str, float | int | str] = {
        "model": model_name,
        "parameters": parse_total_params(model_key) or -1,
        "train_accuracy_epoch_30": np.nan,
        "validation_accuracy_epoch_30": np.nan,
        "train_loss_epoch_30": np.nan,
        "validation_loss_epoch_30": np.nan,
        "test_loss": np.nan,
        "test_accuracy": test_accuracy,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate the 5th Data Analysis assignment models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["simple_nn", "cnn"],
        default=["simple_nn", "cnn"],
        help="Models to train. Use --models cnn to continue from an already completed simple NN run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a model when its confusion matrix already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    set_reproducibility(AEM)

    train_full, test_full, train_df, test_df = sample_dataframes(AEM)
    print(f"AEM seed: {AEM}")
    print(f"Original train samples: {len(train_full)}")
    print(f"Sampled train samples: {len(train_df)}")
    print(f"Original test samples: {len(test_full)}")
    print(f"Sampled test samples: {len(test_df)}")

    balance_df = save_class_balance(train_full, test_full, train_df, test_df)
    print(balance_df[balance_df["subset"].isin(["train_sample_90pct", "test_sample_80pct"])])

    train_missing = count_missing_files(train_df, TRAIN_DIR)
    test_missing = count_missing_files(test_df, TEST_DIR)
    if train_missing or test_missing:
        print(
            "Some PNG files referenced by labels.csv are missing. "
            "Loading Keras MNIST as an index-based fallback."
        )
        (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = (
            tf.keras.datasets.mnist.load_data()
        )
        validate_fallback_labels(train_df, mnist_train_labels, "train")
        validate_fallback_labels(test_df, mnist_test_labels, "test")
    else:
        mnist_train_images = None
        mnist_test_images = None

    x_train, y_train = load_images(train_df, TRAIN_DIR, mnist_train_images)
    x_test, y_test = load_images(test_df, TEST_DIR, mnist_test_images)
    save_sample_digits(x_train, y_train)

    shape_rows = [
        {"name": "x_train_images", "shape": str(x_train.shape)},
        {"name": "y_train", "shape": str(y_train.shape)},
        {"name": "x_test_images", "shape": str(x_test.shape)},
        {"name": "y_test", "shape": str(y_test.shape)},
        {"name": "x_train_flat", "shape": str((x_train.shape[0], 784))},
        {"name": "x_test_flat", "shape": str((x_test.shape[0], 784))},
        {"name": "x_train_cnn", "shape": str((x_train.shape[0], 28, 28, 1))},
        {"name": "x_test_cnn", "shape": str((x_test.shape[0], 28, 28, 1))},
    ]
    pd.DataFrame(shape_rows).to_csv(OUTPUT_DIR / "data_shapes.csv", index=False)

    x_train_flat = x_train.reshape((x_train.shape[0], 784))
    x_test_flat = x_test.reshape((x_test.shape[0], 784))
    x_train_cnn = x_train[..., np.newaxis]
    x_test_cnn = x_test[..., np.newaxis]

    trained_results: dict[str, dict[str, float | int | str]] = {}
    if "simple_nn" in args.models:
        if args.skip_existing and (OUTPUT_DIR / "simple_nn_confusion_matrix.csv").is_file():
            print("\nSkipping simple_nn because existing outputs were found.")
        else:
            trained_results["simple_nn"] = train_and_evaluate(
                build_simple_nn(),
                "simple_nn",
                x_train_flat,
                y_train,
                x_test_flat,
                y_test,
            )

    if "cnn" in args.models:
        if args.skip_existing and (OUTPUT_DIR / "cnn_confusion_matrix.csv").is_file():
            print("\nSkipping cnn because existing outputs were found.")
        else:
            trained_results["cnn"] = train_and_evaluate(
                build_cnn(),
                "cnn",
                x_train_cnn,
                y_train,
                x_test_cnn,
                y_test,
            )

    ordered_results = []
    model_names = {"simple_nn": "simple_nn", "cnn": "cnn"}
    for model_key in ["simple_nn", "cnn"]:
        result = trained_results.get(model_key) or infer_existing_metrics(model_key, model_names[model_key])
        if result is not None:
            ordered_results.append(result)

    metrics_df = pd.DataFrame(ordered_results)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print("\nFinal metrics:")
    print(metrics_df)


if __name__ == "__main__":
    main()
