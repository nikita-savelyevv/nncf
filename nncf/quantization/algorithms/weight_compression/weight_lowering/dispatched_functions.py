# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

from nncf.tensor import Tensor

from ..config import WeightCompressionConfig
from .weight_lowering_dispatcher import ov_available_backend_selector
from .weight_lowering_dispatcher import weight_lowering_dispatcher


@weight_lowering_dispatcher(ov_available_backend_selector)
def do_int_quantization(
    weight: Tensor,
    reduction_axes: Tuple[int, ...],
    config: WeightCompressionConfig,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
    **kwargs,
):
    pass


@weight_lowering_dispatcher(ov_available_backend_selector)
def calculate_quantized_dequantized_weight(
    weight: Tensor, config: WeightCompressionConfig, scale: Tensor, zero_point: Optional[Tensor] = None, **kwargs
) -> Tensor:
    pass
