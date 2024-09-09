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

import argparse
import gc
import os
import shutil
import time
from functools import partial
from pathlib import Path

import openvino as ov

import nncf
from nncf.openvino.quantization.compression_primitives import OV_COMPRESSION_PRIMITIVE_CACHE
from tools.memory_monitor import MemoryMonitor
from tools.memory_monitor import MemoryType


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True, help="Path where the model is stored")

    parser.add_argument("--log-dir", default="./compression_logs", type=str, help="Directory where logs will be saved")

    parser.add_argument("--numpy", action="store_true", help="Enable numpy compression")

    parser.add_argument("--dynamic", action="store_true", help="Enable compression with dynamic-shaped OV models")

    parser.add_argument("--end-to-end", action="store_true", help="Enable end-to-end OV compression")

    parser.add_argument("--input-dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="OV model input dtype")

    parser.add_argument("--int8-output", action="store_true", help="Output in (u)int8")

    parser.add_argument("--recompile", action="store_true", help="Recompile model every time")

    parser.add_argument("--not-shared-outputs", action="store_true", help="Do not share outputs")

    parser.add_argument("--save-model", action="store_true", help="Save compressed model")

    parser.add_argument("--compare-with-numpy", action="store_true", help="Compare compressed weight with the one computed with NumPy")

    parser.add_argument("--invert-numpy-division", action="store_true", help="Invert division when compressing with NumPy")

    parser.add_argument("--release-memory", action="store_true", help="Release memory")

    return parser.parse_args()


def log(mm, fz, log_dir):
    mm.save_memory_logs(
        *mm.get_data(memory_from_zero=fz), save_dir=Path(log_dir), filename_suffix="_from-zero" if fz else ""
    )


def main(args):
    model_path = Path(args.model_path)
    log_dir = Path(args.log_dir)

    numpy_compression = args.numpy
    dynamic_compression = args.dynamic
    end_to_end_compression = args.end_to_end
    input_dtype = args.input_dtype
    int8_output = args.int8_output
    recompile = args.recompile
    not_shared_outputs = args.not_shared_outputs
    save_model = args.save_model
    compare_with_numpy = args.compare_with_numpy
    invert_numpy_division = args.invert_numpy_division
    release_memory = args.release_memory
    if numpy_compression:
        log_dir_suffix = "numpy"
        if invert_numpy_division:
            log_dir_suffix += "_inverted"
    else:
        log_dir_suffix = "end-to-end_" if end_to_end_compression else ""
        log_dir_suffix = f"{log_dir_suffix}{'ov-dynamic' if dynamic_compression else 'ov-static'}"
        log_dir_suffix = f"{log_dir_suffix}_{'output-int8' if int8_output else 'output-fp32'}"
        log_dir_suffix = f"{log_dir_suffix}_{f'input-{input_dtype}'}"
        if recompile:
            log_dir_suffix = f"{log_dir_suffix}_recompile"
        if not_shared_outputs:
            log_dir_suffix = f"{log_dir_suffix}_not-shared-outputs"

    memory_monitors = []
    for memory_type, mem_from_zero in [(MemoryType.RSS, False), (MemoryType.SYSTEM, False), (MemoryType.SYSTEM, True)]:
        memory_monitor = MemoryMonitor(interval=1e-2, memory_type=memory_type, include_child_processes=bool(0))
        memory_monitor.start(at_exit_fn=partial(log, memory_monitor, mem_from_zero, log_dir / log_dir_suffix))
        memory_monitors.append(memory_monitor)

    core = ov.Core()
    # core.set_property({"ENABLE_MMAP": "NO"})
    model = core.read_model(model_path)

    os.environ["NUMPY_COMPRESSION"] = f"{int(numpy_compression)}"
    os.environ["DYNAMIC_COMPRESSION"] = f"{int(dynamic_compression)}"
    os.environ["END_TO_END_COMPRESSION"] = f"{int(end_to_end_compression)}"
    os.environ["INPUT_DTYPE"] = input_dtype
    os.environ["INT8_OUTPUT"] = f"{int(int8_output)}"
    os.environ["RECOMPILE"] = f"{int(recompile)}"
    os.environ["NOT_SHARED_OUTPUTS"] = f"{int(not_shared_outputs)}"
    os.environ["COMPARE_WITH_NUMPY"] = f"{int(compare_with_numpy)}"
    os.environ["INVERT_NUMPY_DIVISION"] = f"{int(invert_numpy_division)}"
    os.environ["RELEASE_MEMORY"] = f"{int(release_memory)}"

    start_time = time.perf_counter()
    compressed_model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_ASYM)
    compression_time = time.perf_counter() - start_time
    print(f"Compression Time: {compression_time:.2f} sec.")

    if save_model:
        ov.save_model(compressed_model, log_dir / log_dir_suffix / "openvino_model.xml")
        for filepath in model_path.parent.glob("*.json"):
            shutil.copy(str(filepath), str(log_dir / log_dir_suffix / filepath.name))

    del core
    del model
    del compressed_model
    gc.collect()
    time.sleep(0.5)

    before_cache_deletion = memory_monitors[2].get_data(True)[1][-1]
    if OV_COMPRESSION_PRIMITIVE_CACHE._compress_weight_model_cache or \
       OV_COMPRESSION_PRIMITIVE_CACHE._compress_weight_end_to_end_model_cache:
        OV_COMPRESSION_PRIMITIVE_CACHE._compress_weight_model_cache.clear()
        OV_COMPRESSION_PRIMITIVE_CACHE._compress_weight_end_to_end_model_cache.clear()
        gc.collect()
        time.sleep(memory_monitors[0].interval * 10)
        after_cache_deletion = memory_monitors[2].get_data(True)[1][-1]
    else:
        after_cache_deletion = before_cache_deletion
    cache_size = before_cache_deletion - after_cache_deletion
    print(f"Cache size: {cache_size:.2f} MiB")

    time.sleep(memory_monitors[0].interval * 10)

    leftover_memory = memory_monitors[2].get_data(True)[1][-1]
    peak_memory = max(memory_monitors[2].get_data(True)[1])
    print(f"Peak memory: {peak_memory:.2f} MiB")
    print(f"Leftover memory: {leftover_memory:.2f} MiB")
    print("Done")

    csv_path = log_dir / "results.csv"
    csv_exists = csv_path.exists()
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    with open(csv_path, "a") as f:
        if not csv_exists:
            f.write(
                "Model Path,"
                "Backend,"
                "End-to-end,"
                "Input Shapes,"
                "Input,"
                "Output,"
                "Compression Time,"
                "Peak Memory,"
                "Cache Size,"
                "Leftover Memory"
                "\n"
            )
        f.write(
            f"{model_path},"
            f"{'NumPy' if numpy_compression else 'OV'},"
            f"{end_to_end_compression},"
            f"{'-' if numpy_compression else 'Dynamic' if dynamic_compression else 'Static'},"
            f"{'-' if numpy_compression else input_dtype.upper()},"
            f"{'-' if numpy_compression else 'INT8' if int8_output else 'FP32'},"
            f"{compression_time:.2f},"
            f"{peak_memory:.2f},"
            f"{cache_size:.2f},"
            f"{leftover_memory:.2f}"
            f"\n"
        )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
