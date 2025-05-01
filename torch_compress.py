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
import gc
import json
import os
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Union, Type, Optional, Any
from weakref import WeakKeyDictionary

import numpy as np
import torch
import transformers
from datasets import load_dataset
from lm_eval.models.utils import get_dtype
from optimum.exporters.openvino.convert import export_from_model
from optimum.intel.openvino import OVModelForCausalLM
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
from transformers import AutoTokenizer
from optimum.gptq.data import get_dataset, prepare_dataset
from transformers.modeling_utils import SpecificPreTrainedModelType, load_state_dict, _load_state_dict_into_model
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils.hub import get_checkpoint_shard_files

from lm_eval import evaluator
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import nncf
from nncf.quantization.algorithms.smooth_quant.torch_backend import SQMultiply
from nncf.torch.function_hook import get_hook_storage
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import build_nncf_graph
from nncf.torch.model_graph_manager import get_const_data, split_const_name, get_module_by_name
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from nncf.torch.utils import is_multidevice

# MODEL_ID = "microsoft/Phi-4-mini-instruct"
# MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_ID = "facebook/opt-125m"
# MODEL_ID = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NNCF_CONFIG_FILENAME = "nncf_config.json"
print(f"Using device: {DEVICE}")


class ModelBacked(Enum):
    PT = "pt"
    OV = "ov"


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


class NNCFModelForCausalLM(PreTrainedModel):
    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        with open(save_directory / Path(NNCF_CONFIG_FILENAME), "w") as f:
            json.dump(self.get_nncf_config(self), f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path,
        *args,
        **kwargs
    ) -> SpecificPreTrainedModelType:
        model, loading_info = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
            output_loading_info=True
        )

        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        nncf_config_filepath = pretrained_model_name_or_path / NNCF_CONFIG_FILENAME
        if nncf_config_filepath.exists():
            with open(nncf_config_filepath, "r") as f:
                nncf_config = json.load(f)

            weights_filepath = pretrained_model_name_or_path / SAFE_WEIGHTS_NAME
            if weights_filepath.exists():
                # Single model shard case
                state_dict = load_state_dict(str(weights_filepath))
                state_dict = {k: v for k, v in state_dict.items() if k in loading_info["unexpected_keys"]}
            else:
                # Multiple model shards case
                cached_filenames, sharded_metadata = get_checkpoint_shard_files(
                    pretrained_model_name_or_path, f"{weights_filepath}.index.json"
                )
                state_dict = {
                    k: v
                    for shard_file in cached_filenames
                    for k, v in load_state_dict(shard_file).items()
                    if k in loading_info["unexpected_keys"]
                }

            model = nncf.torch.load_from_config(model, nncf_config, model.dummy_inputs)
            _load_state_dict_into_model(model, state_dict, "")

            if is_multidevice(model):
                def get_forward_wrapper(sq_multiply: SQMultiply):
                    orig_forward = sq_multiply.forward

                    @wraps(sq_multiply.forward)
                    def forward(x):
                        if sq_multiply.scale.device != x.device:
                            sq_multiply.scale.data = sq_multiply.scale.to(x.device)
                            if sq_multiply.scale.grad is not None:
                                sq_multiply.scale.grad.data = sq_multiply.scale.grad.to(x.device)
                        return orig_forward(x)

                    return forward

                for name, module in get_hook_storage(model).named_hooks():
                    if isinstance(module, SQMultiply):
                        module.forward = get_forward_wrapper(module)

        new_class = type("NNCFWrappedModelForCausalLM", (NNCFModelForCausalLM, model.__class__), {})
        model.__class__ = new_class
        return model

    @staticmethod
    def get_nncf_config(model: nn.Module) -> dict[str, Any]:
        """
        Returns serializable config which contains all information required to recover all additional modules placement.

        :param model: The model to serialize.
        :return: Serializable config.
        """

        from nncf.torch.function_hook.serialization import S_COMMAND
        from nncf.torch.layer_utils import COMPRESSION_MODULES
        from nncf.torch.layer_utils import StatefulModuleInterface
        from nncf.torch.function_hook.serialization import COMPRESSION_STATE_ATTR

        hook_storage = get_hook_storage(model)

        # Find shared modules
        modules_map: WeakKeyDictionary[nn.Module, list[str]] = WeakKeyDictionary()
        for name, module in hook_storage.named_hooks(remove_duplicate=False):
            if module not in modules_map:
                modules_map[module] = []
            modules_map[module].append(name)

        # Generate serialized transformation commands
        serialized_transformations: list[S_COMMAND] = []
        for module, names in modules_map.items():
            compression_module_name = module.__class__.__name__
            if compression_module_name not in COMPRESSION_MODULES.registry_dict:
                msg = (
                    f"Could not serialize compression module with name {compression_module_name}. "
                    "Please register your module in the COMPRESSION_MODULES registry."
                )
                # nncf_logger.info(msg)
                continue
                # raise nncf.InternalError(msg)
            if not isinstance(module, StatefulModuleInterface):
                msg = "Support only StatefulModuleInterface modules"
                raise nncf.InternalError(msg)

            serialized_transformations.append(
                {
                    "hook_names_in_model": names,
                    "module_cls_name": compression_module_name,
                    "module_config": module.get_config(),
                }
            )

        return {COMPRESSION_STATE_ATTR: serialized_transformations}



@register_model("nncf")
class NNCFHFLM(HFLM):
    def __init__(self, *args, **kwargs) -> None:
        if "backend" in kwargs:
            assert kwargs["backend"] == "causal"
        super().__init__(*args, **kwargs)

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs
    ) -> None:
        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        self._model = NNCFModelForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=get_dtype(dtype),
            **model_kwargs
        )


def export_to_ov(model, output_dir):
    if isinstance(model, OVModelForCausalLM):
        model.save_pretrained(output_dir)
    else:
        try:

            if hasattr(model, "__nncf_hooks"):
                hook_storage = get_hook_storage(model)
                for name, decompressor in hook_storage.named_hooks():
                    if isinstance(decompressor, BaseWeightsDecompressor):
                        decompressor.result_dtype = torch.float32
        except ImportError:
            if hasattr(model, "nncf"):
                for module in model.nncf.modules():
                    if isinstance(module, BaseWeightsDecompressor):
                        module.result_dtype = torch.float32

        export_from_model(model.to("cpu"), output_dir, compression_option="fp32", device="cpu")


def export_to_pt2(model, output_dir, weight_dtype=torch.float32):
    example_input = model.dummy_inputs
    for k in example_input:
        example_input[k] = example_input[k].to(DEVICE)
    graph = build_nncf_graph(model, example_input)

    hook_storage = get_hook_storage(model)
    named_hooks = {k: v for k,v in hook_storage.named_hooks()}
    for name, decompressor in named_hooks.items():
        if not isinstance(decompressor, BaseWeightsDecompressor):
            continue
        _, op_name, _ = decode_hook_name(name)
        weight_node = graph.get_node_by_name(op_name)
        weight = get_const_data(weight_node, model)

        qdq_weight = decompressor(weight)
        qdq_weight = qdq_weight.type(weight_dtype)

        module_name, weight_attr_name = split_const_name(weight_node.layer_attributes.name)
        module = get_module_by_name(module_name, model)
        weight_param = getattr(module, weight_attr_name)
        weight_param.requires_grad = False
        weight_param.data = qdq_weight

        hook_storage.set_submodule(name, torch.nn.Identity())

    model.save_pretrained(output_dir)


def compress_model(model, dataset, compression_kwargs):
    if isinstance(model, OVModelForCausalLM):
        nncf.compress_weights(
            model.model,
            dataset=nncf.Dataset(dataset, lambda x: model.prepare_inputs(**x)),
            **compression_kwargs
        )
        compressed_model = model
        compressed_model.request = None
    else:
        compressed_model = nncf.compress_weights(
            model,
            dataset=nncf.Dataset(dataset),
            **compression_kwargs
        )
    return compressed_model


def do_sample_generation(model: Union[str, PreTrainedModel], backend: ModelBacked):
    if isinstance(model, str):
        if backend == ModelBacked.OV:
            model = OVModelForCausalLM.from_pretrained(model)
        else:
            model = NNCFModelForCausalLM.from_pretrained(model).to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer("What is PyTorch?", return_tensors="pt")
    if not isinstance(model, OVModelForCausalLM):
        inputs = inputs.to(device=model.device)

    transformers.set_seed(42)
    start_t = time.time()
    output = model.generate(**inputs, max_new_tokens=100)
    print("Elapsed time: ", time.time() - start_t)

    output_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:])
    print("\n", "-"*50, "\n", output_text, "\n", "-"*50, "\n")
    return output_text


def run_lm_eval(model: Union[str, PreTrainedModel], backend: ModelBacked, task: str, device: str, limit=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    if backend == ModelBacked.OV:
        model = OVModelForCausalLM.from_pretrained(model)
    lm_eval_model = NNCFHFLM(
        model,
        tokenizer=tokenizer,
        batch_size=1,
        device=device,
        parallelize=True and isinstance(model, str),
        max_length=4096,
    )
    start_time = time.perf_counter()
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=[task],
        num_fewshot=0,
        batch_size=1,
        limit=limit,
        device=device,
    )
    print(f"Evaluation time: {time.perf_counter() - start_time:.2f} seconds")
    results["config"]["model_dtype"] = str(results["config"]["model_dtype"])
    results.pop("samples", None)
    return results


def main(input_backend, output_backend, compression_kwargs, save_dir, pt_dtype=torch.float32, export_compressed=False):
    # Create model
    if input_backend == ModelBacked.PT:
        model = NNCFModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=pt_dtype).to(DEVICE).eval()
    else:
        model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=True, load_in_8bit=False)

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = prepare_dataset(get_dataset("wikitext2", tokenizer, seqlen=32, nsamples=128))
    if input_backend == ModelBacked.PT:
        dataset = [{k: v.to(model.device) for k, v in x.items()} for x in dataset]

    # Compress model
    compressed_model = compress_model(model, dataset, compression_kwargs)
    # compressed_model = model

    # Export model
    if output_backend == ModelBacked.PT:
        if export_compressed:
            # Export PT model with compressed constants
            compressed_model.save_pretrained(save_dir / "compressed")
            tokenizer.save_pretrained(save_dir / "compressed")
            do_sample_generation(compressed_model, tokenizer)
            eval_results = run_lm_eval(compressed_model, ModelBacked.PT, "wikitext", DEVICE)
            with open(save_dir / "compressed" / "eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=4)

            save_dir = save_dir / "decompressed"

        export_to_pt2(compressed_model, save_dir, pt_dtype)
    else:
        if input_backend == ModelBacked.PT:
            compressed_model = nncf.strip(
                compressed_model,
                do_copy=False,
                strip_format=nncf.StripFormat.DQ,
                example_input=dataset[0]
            )

        export_to_ov(compressed_model.to("cpu"), save_dir)

    tokenizer.save_pretrained(save_dir)

    del dataset
    del model
    del compressed_model
    gc.collect()
    torch.cuda.empty_cache()

    # Make a demo generation
    # do_sample_generation(str(save_dir), output_backend)

    # Run evaluation
    with torch.no_grad():
        eval_results = run_lm_eval(str(save_dir), output_backend, "wikitext", DEVICE)
    with open(save_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    # parent_save_dir = "torch_compress" / Path(MODEL_ID.split("/")[1])
    parent_save_dir = "torch_compress" / Path("tmp")
    save_subdir = "int4_asym_awq"
    compression_kwargs = dict(
        mode=nncf.CompressWeightsMode.INT4_ASYM,
        # group_size=4,
        # ratio=0.8,
        awq=True,
        subset_size=1,
        # sensitivity_metric=nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION,
    )

    # try:
    #     main(ModelBacked.OV, ModelBacked.OV, compression_kwargs, parent_save_dir / save_subdir / "ov")
    # except Exception as e:
    #     print(f"OV-OV case failed: {e}")

    # try:
    #     main(ModelBacked.PT, ModelBacked.OV, compression_kwargs, parent_save_dir / save_subdir / "pt_ov")
    # except Exception as e:
    #     print(f"PT-OV case failed: {e}")

    try:
        main(ModelBacked.PT, ModelBacked.PT, compression_kwargs, parent_save_dir / save_subdir / "pt_pt_fp32", torch.float32, export_compressed=False)
    except Exception as e:
        raise e
        print(f"PT-PT FP32 case failed: {e}")

    # try:
    #     main(ModelBacked.PT, ModelBacked.PT, compression_kwargs, parent_save_dir / save_subdir / "pt_pt_bf16", torch.bfloat16, export_compressed=True)
    # except Exception as e:
    #     print(f"PT-PT BF16 case failed: {e}")
