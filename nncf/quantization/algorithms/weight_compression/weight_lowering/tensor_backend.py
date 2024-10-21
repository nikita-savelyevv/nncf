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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import nncf
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType

from .common import calculate_integer_quantization_params
from .common import calculate_quantized_weight
from .common import do_int_dequantization
from .common import reshape_weight_for_grouped_quantization
from .dispatched_functions import calculate_quantized_dequantized_weight
from .dispatched_functions import do_int_quantization
from .dispatcher import WeightLoweringBackend

ReductionAxes = Tuple[int, ...]


@do_int_quantization.register(WeightLoweringBackend.TENSOR)
def _(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
    invert_division=False,
    **kwargs,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    The method quantizes the given weights to integer data type uniformly in accordance with the compression config.
    The config defines a quantization mode:
        INT8_SYM mode refers to signed int8 symmetric weight compression without zero point -
            quantization to [-128, 127] range.
        INT8_ASYM mode refers to unsigned int8 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 255] range.
        INT4_ASYM mode refers to unsigned int4 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 15] range.
        INT4_SYM mode refers to signed int4 symmetric weight compression without zero point -
            quantization to [-8, 7] range.
        NF4 or E2M1 mode requires a dedicated procedure and it is not supported in this method.
    One of the parameter of compression config is a group size. Quantization is per-channel, if group size equals to -1,
    otherwise it's per-group, i.e. group size number of weights in the channel dimension share quantization parameters
    (scales).

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :param precomputed_scale: Precomputed scale.
    :param precomputed_zero_point: Precomputed zero point.
    :param invert_scale: applies inversion for scale and then multiply by weights instead of division.
        Need as reference implementation for OV.
    :return: The compressed weights tensor of uint8 (asymmetric mode) or int8 (symmetric mode) type,
        scale tensor of float32 type and zero point tensor of int32 type that was used for its quantization.
    """
    assert config.is_integer(), "The function supports integer quantization only"
    group_size = config.group_size

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if group_size != -1:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, group_size)

    scale, zero_point = None, None
    if precomputed_zero_point is None or precomputed_zero_point is None:
        scale, zero_point = calculate_integer_quantization_params(weight, reduction_axes, config)
    if precomputed_scale is not None:
        scale = precomputed_scale
    if precomputed_zero_point is not None:
        zero_point = precomputed_zero_point

    compressed_weights = calculate_quantized_weight(weight, config, scale, zero_point, invert_division)
    return compressed_weights, scale, zero_point


@calculate_quantized_dequantized_weight.register(WeightLoweringBackend.TENSOR)
def _(
    weight: Tensor,
    config: WeightCompressionConfig,
    scale: Tensor,
    zero_point: Optional[Tensor] = None,
    invert_division=False,
    **kwargs,
) -> Tensor:
    compressed_weight = calculate_quantized_weight(weight, config, scale, zero_point, invert_division)
    decompressed_weight = do_int_dequantization(compressed_weight, scale, zero_point)
    return decompressed_weight
