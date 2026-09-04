"""Kịch bản huấn luyện và đánh giá mô hình Machine Learning Baseline (KNN, SVM, Random Forest).

Thực hiện phân chia dataset theo subject_id (GroupKFold / Leave-One-Subject-Out) để triệt tiêu data leakage.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.preprocessing import LandmarkPreprocessor

logger = logging.getLogger("train_baseline")


def load_and_preprocess_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc tệp CSV và chuyển đổi 21 landmarks 3D qua LandmarkPreprocessor.

    Args:
        csv_path: Đường dẫn tệp CSV dataset.

    Returns:
        Tuple[X, y, groups]: Vector đặc trưng X, nhãn y, nhóm người tham gia groups.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["subject_id", "gesture", "handedness"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu các cột quy định: {missing}")

    preprocessor = LandmarkPreprocessor(mirror_left_hand=True)
    features_list = []
    labels_list = []
    groups_list = []

    for _, row in df.iterrows():
        coords = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            coords[i, 0] = row[f"x{i}"]
            coords[i, 1] = row[f"y{i}"]
            coords[i, 2] = row[f"z{i}"]

        feat = preprocessor.transform(coords, handedness=row["handedness"])
        features_list.append(feat)
        labels_list.append(row["gesture"])
        groups_list.append(row["subject_id"])

    return np.array(features_list), np.array(labels_list), np.array(groups_list)


def evaluate_baselines(csv_path: str) -> None:
    """Chạy GroupKFold Cross-Validation đánh giá các mô hình KNN, SVM, Random Forest."""
    logger.info("Đang nạp và tiền xử lý dataset từ %s...", csv_path)
    X, y, groups = load_and_preprocess_dataset(csv_path)

    unique_subjects = np.unique(groups)
    logger.info("Dataset có tổng cộng %d mẫu từ %d subjects: %s", len(X), len(unique_subjects), list(unique_subjects))

    n_splits = min(5, len(unique_subjects))
    if n_splits < 2:
        logger.error("Cần ít nhất 2 subjects khác nhau trong dataset để thực hiện GroupKFold!")
        return

    gkf = GroupKFold(n_splits=n_splits)

    classifiers: Dict[str, Any] = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0),
        "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []

    for name, clf in classifiers.items():
        macro_f1_scores = []
        acc_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Đảm bảo không có data leakage giữa train_subjects và val_subjects
            train_subs = set(groups[train_idx])
            val_subs = set(groups[val_idx])
            assert train_subs.isdisjoint(val_subs), "Cảnh báo Data Leakage: Subject xuất hiện ở cả Train và Val!"

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")

            acc_scores.append(acc)
            macro_f1_scores.append(f1)

        avg_acc = np.mean(acc_scores)
        avg_f1 = np.mean(macro_f1_scores)

        results.append({
            "Phương pháp": name,
            "Accuracy": f"{avg_acc:.4f}",
            "Macro-F1": f"{avg_f1:.4f}",
        })

    logger.info("=== BẢNG KẾT QUẢ ĐÁNH GIÁ BASELINE MACHINE LEARNING ===")
    res_df = pd.DataFrame(results)
    print("\n", res_df.to_string(index=False), "\n")


def main() -> None:
    """Khởi chạy CLI cho train_baseline."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    parser = argparse.ArgumentParser(description="Kịch bản huấn luyện & đánh giá Baseline ML")
    parser.add_argument("--dataset", type=str, default="data/raw/landmarks_dataset.csv", help="Đường dẫn tệp CSV dataset")

    args = parser.parse_args()
    evaluate_baselines(args.dataset)


if __name__ == "__main__":
    main()
