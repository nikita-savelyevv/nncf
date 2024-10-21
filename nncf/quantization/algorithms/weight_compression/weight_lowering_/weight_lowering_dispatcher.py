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
from enum import Enum
from functools import wraps
from typing import Dict, Any, Callable, Optional

from nncf.utils import is_openvino_available
from .ov_backend import do_int_quantization as do_int_quantization_ov
from .tensor_backend import do_int_quantization as do_int_quantization_tensor
from functools import singledispatch


class WeightLoweringBackend(Enum):
    TENSOR = "TENSOR"
    OV = "OV"


def ov_available_backend_selector(*args, **kwargs) -> WeightLoweringBackend:
    forced_backend = kwargs.get("forced_backend", None)
    if forced_backend is not None:
        return forced_backend
    return WeightLoweringBackend.OV if is_openvino_available() else WeightLoweringBackend.TENSOR


# Dispatcher decorator factory that accepts a backend selector function
def weight_lowering_dispatcher(backend_selector_fn: Callable):
    registry = {}  # Holds backend-specific implementations

    def decorator(fn: Callable):
        # Register method to add implementations for different backends
        def register(backend: WeightLoweringBackend):
            def wrapper(backend_fn: Callable):
                registry[backend] = backend_fn
                return backend_fn

            return wrapper

        # This is the wrapper that will dynamically select the backend and call the corresponding function
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Use the backend selector function to determine which backend to call
            backend = backend_selector_fn(*args, **kwargs)
            if backend in registry:
                # Call the appropriate registered backend-specific function
                return registry[backend](*args, **kwargs)
            else:
                raise ValueError(f"No implementation registered for backend {backend}")

        # Attach the register method to the wrapper
        wrapper.register = register
        return wrapper

    return decorator
