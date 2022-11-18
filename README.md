# python-orb-slam3

A Python wrapper for the [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) feature extraction algorithm.

## Installation

### From PyPI

> **Note**
> This package's pre-built binaries are only available for AMD64 architectures.

```bash
pip install python-orb-slam3
```

### From source

There are a few steps to follow to install this package from the source code, please refer to the CI configuration file [here](.github/workflows/ci.yml) for more details.

## Usage

```python
import cv2
from matplotlib import pyplot as plt

from python_orb_slam3 import ORBExtractor

source = cv2.imread("path/to/image.jpg")
target = cv2.imread("path/to/image.jpg")

orb_extractor = ORBExtractor()

# Extract features from source image
source_keypoints, source_descriptors = orb_extractor.detectAndCompute(source)
target_keypoints, target_descriptors = orb_extractor.detectAndCompute(target)

# Match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(source_descriptors, target_descriptors)

# Draw matches
source_image = cv2.drawKeypoints(source, source_keypoints, None)
target_image = cv2.drawKeypoints(target, target_keypoints, None)
matches_image = cv2.drawMatches(source_image, source_keypoints, target_image, target_keypoints, matches, None)

# Show matches
plt.imshow(matches_image)
plt.show()
```

## License

This repository is licensed under the [GPLv3](LICENSE) license.

<!--markdownlint-disable-file MD046-->

    A Python wrapper for the ORB-SLAM3 feature extraction algorithm.
    Copyright (C) 2022  Johnny Hsieh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
