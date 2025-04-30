import pytest
from datetime import datetime
from src.modules.Synchronizer import Synchronizer

@pytest.fixture
def dummy_filenames():
    return [
        "D1_S20240408093000_E20240408100000.mp4",
        "D1_S20240408100000_E20240408100423.mp4",
        "D2_S20240408093005_E20240408100005.mp4",
        "D2_S20240408100005_E20240408100445.mp4",
        "D3_S20240408092958_E20240408095958.mp4",
        "D4_S20240408093002_E20240408100002.mp4",
    ]

def test_parse_filename():
    filename = "D1_S20240408093000_E20240408100000.mp4"
    cam_id, start, end = Synchronizer.parse_filename(filename)
    assert cam_id == 1
    assert start == datetime(2024, 4, 8, 9, 30, 0)
    assert end == datetime(2024, 4, 8, 10, 0, 0)

def test_separate_by_cameras(dummy_filenames):
    sync = Synchronizer(rfid_data_path=None)
    sorted_videos = sync.separate_by_cameras(dummy_filenames)
    assert set(sorted_videos.keys()) == {1, 2, 3, 4}
    assert isinstance(sorted_videos[1], list)
    assert sorted_videos[1][0][0][0] == datetime(2024, 4, 8, 9, 30, 0)

def test_get_offsets(dummy_filenames):
    sync = Synchronizer(rfid_data_path=None)
    sync.separate_by_cameras(dummy_filenames)
    offsets = sync.get_offsets()
    assert offsets[3] == 7.0  
    assert offsets[1] == 5.0
    assert offsets[4] == 3.0
    assert offsets[2] == 0.0
