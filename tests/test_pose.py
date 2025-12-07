import numpy as np
import torch

from har.pose.decode import decode, draw_poses, score_connections
from har.pose.model import PoseEstimation


def test_pose_network_output_channels():
    model = PoseEstimation().eval()
    with torch.inference_mode():
        pafs, heatmaps = model(torch.zeros(1, 3, 32, 40))
    assert pafs.shape == (1, 38, 4, 5)
    assert heatmaps.shape == (1, 19, 4, 5)


def test_no_people_keeps_image_unchanged():
    people = decode(np.zeros((32, 32, 38)), np.zeros((32, 32, 19)))
    assert people.shape == (0, 18, 3)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    np.testing.assert_array_equal(draw_poses(image, people), image)


def test_pafs_connect_four_joint_person():
    heatmaps = np.zeros((96, 96, 19), dtype=np.float32)
    yy, xx = np.mgrid[:96, :96]
    for joint, x in zip((1, 2, 3, 4), (16, 32, 48, 64)):
        heatmaps[:, :, joint] = np.exp(-((xx - x) ** 2 + (yy - 32) ** 2) / 32)
    pafs = np.zeros((96, 96, 38), dtype=np.float32)
    pafs[:, :, [12, 14, 16]] = 1.0
    people = decode(pafs, heatmaps)
    assert people.shape == (1, 18, 3)
    np.testing.assert_allclose(people[0, [1, 2, 3, 4], 0], [16, 32, 48, 64])
    assert np.count_nonzero(people[0, :, 2]) == 4


def test_coincident_joints_are_not_connected():
    peaks = [[] for _ in range(18)]
    peaks[1], peaks[2] = [0], [1]
    candidates = np.array([[10, 10, 1], [10, 10, 1]], dtype=np.float32)
    assert not any(score_connections(np.ones((32, 32, 38)), peaks, candidates))
