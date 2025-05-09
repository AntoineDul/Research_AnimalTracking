from src.modules.MultiTracker import MultiTracker
from src.modules.Mapper import Mapper
from src import config
import pytest

@pytest.fixture
def sample_paths_cam5():
    # One path from (0.2, 0) to ~(0.2, 1) in global coords
    return [[((1290, 206), 0),((1260, 350), 1),((1230, 500), 2),((1200, 650), 3),((1170, 800), 4),((1140, 950), 5),((1110, 1100), 6),((1080, 1250), 7),((1050, 1400), 8),((1030, 1550), 9), ((1000, 1667), 10)]]

@pytest.fixture
def sample_paths_cam7():
    # Two paths: One from (0.2, 0) to ~(0.2, 1) but only 8 frames, second is static pig at (2, 0)
    return [[((619, 880), 0),((1260, 800), 1),((1230, 720), 2),((1200, 640), 3),((1170, 560), 4),((1140, 480), 5),((1110, 400), 6),((1080, 320), 7)], [((2076, 948), 0), ((2076 ,948), 1), ((2076 ,948), 2), ((2076 ,948), 3), ((2076 ,948), 4), ((2076 ,948), 5), ((2076 ,948), 6), ((2076 ,948), 7), ((2076 ,948), 8), ((2076 ,948), 9), ((2076 ,948), 10)]] 

@pytest.fixture
def sample_paths_cam17():
    # Three paths:One from (0.2, 0) to ~(0.2, 1) but only 8 frames, second is static pig at (2, 0), last is a static pig in a corner (only visible by cam 17)
    return [[((558, 480), 0), ((600, 470), 1), ((650, 460), 2), ((700, 450), 3), ((750, 440), 4), ((800, 430), 5), ((850, 420), 6), ((900, 410), 7), ((950, 400), 8), ((1000, 390), 9), ((1011, 378), 10)], [((600, 1400), 0),((600, 1400), 1),((600, 1400), 2),((600, 1400), 4),((600, 1400), 5),((600, 1400), 6),((600, 1400), 8),((600, 1400), 9)], [((2400, 700), 0),((2400, 700), 1),((2400, 700), 3),((2400, 700), 4),((2400, 700), 5),((2400, 700), 7),((2400, 700), 8),((2400, 700), 10)]]

def test_batch_to_world_coords(sample_paths_cam17):
    mapper = Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION)
    irl_paths = mapper.batch_to_world_coords(sample_paths_cam17, 9)

    print(f"Translated paths from cam 17 : {irl_paths}")

    assert len(irl_paths) == 3
    assert len(irl_paths[0]) == 11


def test_batch_match(sample_paths_cam5, sample_paths_cam7, sample_paths_cam17):
    mapper = Mapper(config.MAPPINGS, config.RESOLUTION, config.DISTORTION)
    multi_tracker = MultiTracker(config.NUM_CAMERAS, config.FIRST_CAMERA, mapper, config.CLUSTER_EPSILON, config.CLUSTER_MIN_SAMPLES, config.MAX_GLOBAL_AGE, config.MAX_CLUSTER_DISTANCE, config.FRECHET_THRESHOLD, config.BATCH_SIZE, config.CAM_FULLY_OVERLAPPED, False)

    # Translate paths to global coords
    global_paths_cam5 = mapper.batch_to_world_coords(sample_paths_cam5, 5)
    global_paths_cam7 = mapper.batch_to_world_coords(sample_paths_cam7, 7)
    global_paths_cam17 = mapper.batch_to_world_coords(sample_paths_cam17, 9)

    # batch match paths from cam 5 and 7, static pig from 7 should be discarded and moving pig paths should be merged
    paths_7_17 = multi_tracker.batch_match(global_paths_cam7, global_paths_cam17, 7, 9)
    print(f"COMBINED PATHS FROM CAM 7 AND 17 :\n{paths_7_17}\n")
    assert len(paths_7_17) == 3

    print("\n\n\n")

    paths_5_7_17 = multi_tracker.batch_match(global_paths_cam5, paths_7_17, 5, None)
    print(f"COMBINED PATHS FROM CAMS 5, 7, 17 :{paths_5_7_17}")

    assert len(paths_5_7_17) == 3

def test_merge_paths():
    pass

def test_no_batch_size():
    pass

def test_pig_in_corner_detected():
    pass
