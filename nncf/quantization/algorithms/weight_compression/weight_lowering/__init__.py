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


from .common import reshape_weight_for_grouped_quantization, calculate_nf4_scale, do_nf4_quantization, \
    do_nf4_dequantization, calculate_normalized_weight_and_fp4_scale, calculate_integer_quantization_params, \
    calculate_quantized_weight, compress_weight, do_int_dequantization

from .dispatched_functions import do_int_quantization, calculate_quantized_dequantized_weight
