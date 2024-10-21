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


from .common import WeightCompressionConfig
from .common import calculate_integer_quantization_params
from .common import calculate_nf4_scale
from .common import calculate_normalized_weight_and_fp4_scale
from .common import calculate_quantized_weight
from .common import compress_weight
from .common import do_int_dequantization
from .common import do_nf4_dequantization
from .common import do_nf4_quantization
from .common import get_integer_quantization_error
from .common import reshape_weight_for_grouped_quantization
from .dispatched_functions import calculate_quantized_dequantized_weight
from .dispatched_functions import do_int_quantization
