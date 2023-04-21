"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os
import pytest

import torchvision
import torch

from tests.shared.isolation_runner import ISOLATION_RUN_ENV_VAR


@torch.jit.script_if_tracing
def test_fn(x: torch.Tensor):
    return torch.empty(x.shape)


class TestModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return test_fn(x)


import nncf.torch


@pytest.mark.skipif(ISOLATION_RUN_ENV_VAR not in os.environ, reason="Should be run via isolation proxy")
def test_temp():
    torch.onnx.export(TestModel(), (torch.zeros((1,)),), "/tmp/jit_if_tracing_test_model.onnx")
