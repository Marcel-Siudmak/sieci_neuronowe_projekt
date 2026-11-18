from pathlib import Path

from scripts.prepare_data import get_base_scene_name, split_by_scenes


def test_get_base_scene_name():
    assert get_base_scene_name("kitchen_0019a_out") == "kitchen_0019"
    assert get_base_scene_name("kitchen_0019b_out") == "kitchen_0019"
    assert get_base_scene_name("kitchen_0019_out") == "kitchen_0019"
    assert get_base_scene_name("nyu_office_0_out") == "nyu_office_0"
    assert get_base_scene_name("official_test") == "official_test"


def test_split_by_scenes_no_cross_take_leakage():
    rooms = [
        ("room_0001a_out", 10),
        ("room_0001b_out", 8),
        ("room_0002a_out", 12),
        ("room_0002b_out", 6),
        ("room_0003_out", 15),
        ("room_0004a_out", 9),
        ("room_0004b_out", 7),
        ("room_0005_out", 11),
    ]
    pairs = []
    for scene, count in rooms:
        for idx in range(count):
            pairs.append((Path(f"{scene}/{idx}.jpg"), Path(f"{scene}/{idx}.png"), scene))

    train_set, val_set, test_set, _ = split_by_scenes(pairs, (0.6, 0.2, 0.2), seed=42)

    train_rooms = {get_base_scene_name(scene) for _, _, scene in train_set}
    val_rooms = {get_base_scene_name(scene) for _, _, scene in val_set}
    test_rooms = {get_base_scene_name(scene) for _, _, scene in test_set}

    assert len(train_set) + len(val_set) + len(test_set) == len(pairs)
    assert train_rooms.isdisjoint(val_rooms)
    assert train_rooms.isdisjoint(test_rooms)
    assert val_rooms.isdisjoint(test_rooms)
