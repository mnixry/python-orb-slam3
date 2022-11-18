from pathlib import Path

import cv2
import numpy

from python_orb_slam3 import ORBExtractor
from pytest_benchmark.fixture import BenchmarkFixture

source_image_file = Path(__file__).parent / "test_source.png"
target_image_file = Path(__file__).parent / "test_target.png"
unrelated_image_file = Path(__file__).parent / "test_unrelated.jpg"

assert (
    source_image_file.exists()
    and target_image_file.exists()
    and unrelated_image_file.exists()
)


def test_zero():
    extractor = ORBExtractor()

    image = numpy.zeros((100, 100), dtype=numpy.uint8)
    keypoints, descriptors = extractor.detectAndCompute(image)

    assert len(keypoints) == 0 and descriptors is None


def test_randoms():
    extractor = ORBExtractor()

    image = numpy.random.randint(0, 255, (100, 100), dtype=numpy.uint8)
    keypoints, descriptors = extractor.detectAndCompute(image)

    assert keypoints
    assert descriptors is not None and descriptors.shape[0] == len(keypoints)


def test_randoms_with_mask():
    extractor = ORBExtractor()

    image = numpy.random.randint(0, 255, (100, 100), dtype=numpy.uint8)
    mask = numpy.random.randint(0, 2, (100, 100), dtype=numpy.uint8)
    keypoints, descriptors = extractor.detectAndCompute(image, mask)

    assert keypoints
    assert descriptors is not None and descriptors.shape[0] == len(keypoints)
    assert descriptors.shape[1] == 32


def test_randoms_with_lapping_area():
    extractor = ORBExtractor()

    image = numpy.random.randint(0, 255, (100, 100), dtype=numpy.uint8)
    lapping_area = (10, 10)
    keypoints, descriptors = extractor.detectAndCompute(image, lappingArea=lapping_area)

    assert keypoints
    assert descriptors is not None and descriptors.shape[0] == len(keypoints)
    assert all(keypoint.pt[0] > lapping_area[0] for keypoint in keypoints) and all(
        keypoint.pt[1] > lapping_area[1] for keypoint in keypoints
    )


def test_real_image():
    extractor = ORBExtractor()

    image = cv2.imread(str(source_image_file), cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = extractor.detectAndCompute(image)

    assert keypoints
    assert descriptors is not None and descriptors.shape[0] == len(keypoints)
    assert descriptors.shape[1] == 32


def test_image_comparison():
    extractor = ORBExtractor()

    source_image = cv2.imread(str(source_image_file), cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(str(target_image_file), cv2.IMREAD_GRAYSCALE)
    unrel_image = cv2.imread(str(unrelated_image_file), cv2.IMREAD_GRAYSCALE)

    source_keypoints, source_descriptors = extractor.detectAndCompute(source_image)
    target_keypoints, target_descriptors = extractor.detectAndCompute(target_image)
    unrel_keypoints, unrel_descriptors = extractor.detectAndCompute(unrel_image)

    assert source_keypoints and target_keypoints and unrel_keypoints
    assert (
        source_descriptors is not None
        and target_descriptors is not None
        and unrel_descriptors is not None
    )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    related_matches = bf.match(source_descriptors, target_descriptors)
    related_matches = sorted(related_matches, key=lambda x: x.distance)

    unrelated_matches = bf.match(source_descriptors, unrel_descriptors)
    unrelated_matches = sorted(unrelated_matches, key=lambda x: x.distance)

    good_matches = related_matches[: int(len(related_matches) * 0.2)]
    bad_matches = unrelated_matches[: int(len(unrelated_matches) * 0.2)]
    assert len(good_matches) > len(bad_matches)

    good_matched = cv2.drawMatches(
        source_image,
        source_keypoints,
        target_image,
        target_keypoints,
        good_matches,
        None,
        flags=2,
    )
    bad_matched = cv2.drawMatches(
        source_image,
        source_keypoints,
        unrel_image,
        unrel_keypoints,
        bad_matches,
        None,
        flags=2,
    )

    cv2.imwrite("good_matched.png", good_matched)
    cv2.imwrite("bad_matched.png", bad_matched)


def test_performance(benchmark: BenchmarkFixture):
    extractor = ORBExtractor()

    image = cv2.imread(str(source_image_file), cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = benchmark(extractor.detectAndCompute, image)

    assert keypoints
    assert descriptors is not None and descriptors.shape[0] == len(keypoints)
