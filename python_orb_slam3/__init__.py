from typing import List, Tuple, Optional

import orb_slam3
import cv2

__all__ = ["ORBExtractor"]


class ORBExtractor(orb_slam3.ORBExtractor):
    def __init__(
        self,
        n_features: int = 1000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        ini_th_fast: int = 20,
        min_th_fast: int = 7,
        interpolation: int = 1,
        angle: bool = True,
    ) -> None:
        super().__init__(
            n_features,
            scale_factor,
            n_levels,
            ini_th_fast,
            min_th_fast,
            interpolation,
            angle,
        )

    def detectAndCompute(
        self,
        image: cv2.Mat,
        mask: Optional[cv2.Mat] = None,
        lappingArea: Tuple[int, int] = (0, 0),
    ) -> Tuple[List[cv2.KeyPoint], Optional[cv2.Mat]]:
        return super().detectAndCompute(image, mask, lappingArea)
