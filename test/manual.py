from tkinter.filedialog import askopenfilename

import cv2
from matplotlib import pyplot as plt

from python_orb_slam3 import ORBExtractor

source_file = askopenfilename()
target_file = askopenfilename()

extractor = ORBExtractor(1000, 1.2, 8, 20, 7, 1, True)

source_image = cv2.imread(source_file, cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)

source_keypoints, source_descriptors = extractor.detectAndCompute(source_image)
target_keypoints, target_descriptors = extractor.detectAndCompute(target_image)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(source_descriptors, target_descriptors)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[: int(len(matches) * 0.2)]
print("Good matches: ", len(good_matches))

matched = cv2.drawMatches(
    source_image,
    source_keypoints,
    target_image,
    target_keypoints,
    good_matches,
    None,
    flags=2,
)

plt.imshow(matched)
plt.show()
