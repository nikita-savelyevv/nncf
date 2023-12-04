# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch import nn

from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.model_creation import wrap_model
from nncf.torch.nested_objects_traversal import objwalk


class ArgumentModel(nn.Module):
    def __init__(self, example_input) -> None:
        super().__init__()
        self._example_input = example_input

    def forward(self, x):
        with no_nncf_trace():
            assert x == self._example_input
        return x


class NonKeyWordArgumentsModel(nn.Module):
    def __init__(self, example_input) -> None:
        super().__init__()
        self._example_input = example_input

    def forward(self, x, y):
        with no_nncf_trace():
            assert x == self._example_input[0]
            assert y == self._example_input[1]
        return x, y


class KeyWordArgumentsModel(nn.Module):
    def __init__(self, example_input) -> None:
        super().__init__()
        self._example_input = example_input

    def forward(self, *, x, y):
        with no_nncf_trace():
            assert x == self._example_input["x"]
            assert y == self._example_input["y"]
        return x, y


@pytest.mark.parametrize(
    ("example_input", "model_cls"),
    [
        (torch.empty(1), ArgumentModel),
        ((torch.empty(1), torch.empty(1)), NonKeyWordArgumentsModel),
        ({"x": torch.empty(1), "y": torch.empty(1)}, KeyWordArgumentsModel),
    ],
    ids=("tensor", "tuple", "dict"),
)
def test_wrap_model_with_example_input(example_input, model_cls):
    model = model_cls(example_input)
    nncf_network = wrap_model(model, example_input)

    def check_type(x):
        assert type(x) == torch.Tensor
        return x

    objwalk(example_input, lambda x: True, check_type)

    nncf_graph = nncf_network.nncf.get_original_graph()
    all_nodes = nncf_graph.get_all_nodes()
    num_nodes = 2
    if isinstance(example_input, (tuple, dict)):
        num_nodes *= len(example_input)

    assert len(all_nodes) == num_nodes

    num_io_nodes = num_nodes // 2
    nodes = ["nncf_model_input"] * num_io_nodes + ["nncf_model_output"] * num_io_nodes
    assert sorted([node.node_type for node in all_nodes]) == nodes
