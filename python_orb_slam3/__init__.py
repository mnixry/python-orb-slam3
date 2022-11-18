import typing

import numpy
import orb_slam3

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
        image: numpy.ndarray,
        mask: typing.Optional[numpy.ndarray] = None,
        lappingArea: typing.Tuple[int, int] = (0, 0),
    ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        return super().detectAndCompute(image, mask, lappingArea)  # type: ignore
