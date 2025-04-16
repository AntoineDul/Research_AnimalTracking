# test_mapper.py

import numpy as np
import pytest
from src.modules.Mapper import Mapper  # replace with the actual module name where Mapper is defined

@pytest.fixture
def sample_mappings():
    return {
        '5' : {
    'image_points': np.array(
        [[1558, 292], [1302, 254], [954, 222], [591, 214], [1108, 1654], [880, 1582], [587, 1457], [97, 1184], [212, 1921], [106, 271]], dtype=np.float32),
    'world_points': np.array(
        [[0, 0], [2, 0], [5, 0], [9, 0], [1, 1], [3, 1], [6, 1], [14, 1], [13, 2], [19, 0]], dtype=np.float32)
        }
    }

def test_mapper_initialization(sample_mappings):
    mapper = Mapper(sample_mappings)
    assert isinstance(mapper.homography_matrices, list)
    assert len(mapper.homography_matrices) == 1
    assert mapper.homography_matrices[0].shape == (3, 3)

def test_valid_transformation(sample_mappings):
    mapper = Mapper(sample_mappings)

    # Define image points to transform
    test_point_1 = np.array([[[854, 213]]], dtype=np.float32)  # shape (1, 1, 2), should map to (6, 0) and (9, 1)
    test_point_2 = np.array([[[340, 1355]]], dtype=np.float32)  # shape (1, 1, 2), should map to (3, 1)
    result1 = mapper.image_to_world_coordinates(5, test_point_1)
    result2 = mapper.image_to_world_coordinates(5, test_point_2)
    print(f"Result1: {result1}, Result2: {result2}")
    assert result1.shape == (2,)
    assert result2.shape == (2,)
    # Since homography can introduce some numerical noise, we allow a small tolerance
    expected1 = (6, 0)
    expected2 = (9, 1)
    assert np.allclose(result1[0], expected1[0], atol=2)
    assert np.allclose(result1[1], expected1[1], atol=3e-1)

    assert np.allclose(result1[0], expected1[0], atol=2)
    assert np.allclose(result2[1], expected2[1], atol=3e-1)

def test_invalid_camera_id(sample_mappings):
    mapper = Mapper(sample_mappings)
    test_points = np.array([[0.5, 0.5]], dtype=np.float32)
    
    with pytest.raises(ValueError, match="Camera ID 1 not found in mappings."):
        mapper.image_to_world_coordinates(1, test_points)
