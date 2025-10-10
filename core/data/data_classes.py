from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pydantic import ConfigDict


@dataclass
class Prediction:  # noqa: WPS210
    absolute_box: list[Tuple[int, int]]  # Формат (tl_x, tl_y, br_x, br_y)
    score: Optional[float]  # Конфиденс
    relative_box: list[float] = None
    crop: Optional[np.array] = None  # Нарезанные кропы
    text: Optional[str] = None
    text_score: Optional[float] = None

    @property
    def center(self, format="absolute") -> Tuple[float, float]:
        if format == "absolute":
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = self.absolute_box
        else:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = self.relative_box

        center_x = (top_left_x + bottom_right_x) / 2
        center_y = (top_left_y + bottom_right_y) / 2
        return center_x, center_y

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
