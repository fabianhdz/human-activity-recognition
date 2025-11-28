"""Associate COCO-18 joints using part affinity fields."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

# Joint and PAF channel indices follow the COCO-18 OpenPose layout.
LIMBS = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 16),
    (5, 17),
]
PAF_CHANNELS = [
    (12, 13),
    (20, 21),
    (14, 15),
    (16, 17),
    (22, 23),
    (24, 25),
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (28, 29),
    (30, 31),
    (34, 35),
    (32, 33),
    (36, 37),
    (18, 19),
    (26, 27),
]
COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]


def find_peaks(heatmaps: np.ndarray, threshold: float = 0.1):
    peaks = []
    candidates = []
    for joint in range(18):
        smoothed = gaussian_filter(heatmaps[:, :, joint], sigma=3)
        maxima = (smoothed == maximum_filter(smoothed, size=3)) & (smoothed > threshold)
        ys, xs = np.nonzero(maxima)
        ids = []
        for x, y in zip(xs, ys):
            ids.append(len(candidates))
            candidates.append((float(x), float(y), float(heatmaps[y, x, joint])))
        peaks.append(ids)
    return peaks, np.asarray(candidates, dtype=np.float32).reshape(-1, 3)


def score_connections(pafs, peaks, candidates, threshold=0.05):
    connections = []
    for (joint_a, joint_b), channels in zip(LIMBS, PAF_CHANNELS):
        possible = []
        for a in peaks[joint_a]:
            for b in peaks[joint_b]:
                start, end = candidates[a, :2], candidates[b, :2]
                vector = end - start
                length = float(np.linalg.norm(vector))
                if length < 1e-6:
                    continue
                points = np.rint(np.linspace(start, end, 10)).astype(int)
                field = pafs[points[:, 1], points[:, 0]][:, channels]
                scores = field @ (vector / length)
                score = float(scores.mean()) + min(0.5 * pafs.shape[0] / length - 1, 0)
                if np.count_nonzero(scores > threshold) > 0.8 * len(scores) and score > 0:
                    possible.append((a, b, score))
        selected, used_a, used_b = [], set(), set()
        for a, b, score in sorted(possible, key=lambda item: item[2], reverse=True):
            if a not in used_a and b not in used_b:
                selected.append((a, b, score))
                used_a.add(a)
                used_b.add(b)
        connections.append(selected)
    return connections


def assemble_people(connections, candidates, min_parts=4, min_score=0.4):
    bodies = []
    for limb, edges in enumerate(connections):
        joint_a, joint_b = LIMBS[limb]
        for a, b, edge_score in edges:
            matches = [
                body for body in bodies if body[0].get(joint_a) == a or body[0].get(joint_b) == b
            ]
            if not matches:
                if limb < 17:
                    score = float(candidates[a, 2] + candidates[b, 2]) + edge_score
                    bodies.append(({joint_a: a, joint_b: b}, score))
                continue
            merged = {}
            conflict = False
            for joints, _ in matches:
                for joint, peak in joints.items():
                    if joint in merged and merged[joint] != peak:
                        conflict = True
                    merged[joint] = peak
            if merged.get(joint_a, a) != a or merged.get(joint_b, b) != b or conflict:
                continue
            added = {joint_a: a, joint_b: b}
            score = sum(body[1] for body in matches) + edge_score
            score += sum(
                float(candidates[peak, 2]) for joint, peak in added.items() if joint not in merged
            )
            merged.update(added)
            for body in matches:
                bodies.remove(body)
            bodies.append((merged, score))
    people = []
    for joints, score in bodies:
        if len(joints) < min_parts or score / len(joints) < min_score:
            continue
        person = np.zeros((18, 3), dtype=np.float32)
        for joint, peak in joints.items():
            person[joint] = candidates[peak]
        people.append(person)
    return np.asarray(people, dtype=np.float32).reshape(-1, 18, 3)


def decode(pafs, heatmaps, peak_threshold=0.1, paf_threshold=0.05):
    peaks, candidates = find_peaks(heatmaps, peak_threshold)
    return assemble_people(score_connections(pafs, peaks, candidates, paf_threshold), candidates)


def draw_poses(image, people):
    canvas = image.copy()
    for person in people:
        for index, (a, b) in enumerate(LIMBS[:17]):
            if person[a, 2] > 0 and person[b, 2] > 0:
                cv2.line(
                    canvas,
                    tuple(person[a, :2].astype(int)),
                    tuple(person[b, :2].astype(int)),
                    COLORS[index],
                    3,
                    cv2.LINE_AA,
                )
        for joint, (x, y, confidence) in enumerate(person):
            if confidence > 0:
                cv2.circle(canvas, (int(x), int(y)), 4, COLORS[joint], -1, cv2.LINE_AA)
    return canvas
