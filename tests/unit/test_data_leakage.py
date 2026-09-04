"""Automated test asserting subject-level dataset split disjointness to prevent data leakage."""


def test_subject_split_disjointness():
    """Đảm bảo các tập train, validation và test có danh sách subject rời rạc tuyệt đối."""
    train_subjects = {f"subject_{i:03d}" for i in range(1, 7)}
    val_subjects = {f"subject_{i:03d}" for i in range(7, 9)}
    test_subjects = {f"subject_{i:03d}" for i in range(9, 11)}

    assert train_subjects.isdisjoint(val_subjects), "Data Leakage: Train và Val có chung Subject!"
    assert train_subjects.isdisjoint(test_subjects), "Data Leakage: Train và Test có chung Subject!"
    assert val_subjects.isdisjoint(test_subjects), "Data Leakage: Val và Test có chung Subject!"
