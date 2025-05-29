import copy
import dataclasses
import gc
import json
import shutil
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import nncf
from nncf.torch.model_creation import wrap_model
from torch_compress import main, extract_acc, ModelBacked

REPORT_FILENAME = "report.json"


@dataclasses.dataclass
class SearchDefinition:
    log_dir: Union[str, Path]
    model_id: str
    metric_name: str
    compression_mode: str
    start_group_size: int
    iter_group_size: int
    search_ratio: float


def extract_matmul_node_names(model: torch.nn.Module, compress_embeddings: bool = False) -> List[str]:
    from nncf.common.factory import NNCFGraphFactory
    from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend

    nncf_graph = NNCFGraphFactory.create(model)
    nodes_to_compress = []
    for node in nncf_graph.topological_sort():
        if PTWeightCompressionAlgoBackend.is_node_with_weights(node, nncf_graph):
            if node.metatype in PTWeightCompressionAlgoBackend.MATMUL_METATYPES:
                nodes_to_compress.append(node)
            if compress_embeddings and node.metatype in PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES:
                nodes_to_compress.append(node)

    node_names = [n.node_name for n in nodes_to_compress]

    return node_names


def run_main(model_id, compression_kwargs: dict, log_dir: Path, metric_name: str):
    # return np.random.normal()
    main(
        model_id,
        ModelBacked.PT,
        ModelBacked.PT,
        compression_kwargs,
        log_dir,
        metric_name,
        pt_dtype=torch.bfloat16,
        backup_device="cpu",
        do_sample_generation=False,
        cleanup_model_files=True,
    )
    metric = extract_acc(log_dir, metric_name)
    return metric


def load_report(log_dir: Path) -> dict:
    report_path = log_dir / REPORT_FILENAME
    if not report_path.exists():
        return None

    with open(report_path, "r") as f:
        report_data = json.load(f)

    return report_data


def save_report(log_dir: Path, report_data: dict):
    report_data_copy = copy.deepcopy(report_data)
    report_data_copy["log_dir"] = str(report_data_copy["log_dir"])
    with open(log_dir / REPORT_FILENAME, "w") as f:
        json.dump(report_data_copy, f, indent=4)
    shutil.copyfile(
        str(log_dir / REPORT_FILENAME), str(log_dir / f"{REPORT_FILENAME.replace('.json', '_backup.json')}")
    )

def run_search(search_def: SearchDefinition):
    assert search_def.metric_name in ["mmlu", "wikitext", "wikitext_validation"]
    higher_metric_is_better = search_def.metric_name in ["mmlu"]
    inf_metric = -1e9 if higher_metric_is_better else 1e9

    optimum_model = AutoModelForCausalLM.from_pretrained(search_def.model_id).to("cpu")
    optimum_model = wrap_model(optimum_model, optimum_model.dummy_inputs, True)
    matmul_node_names = extract_matmul_node_names(optimum_model)
    del optimum_model
    gc.collect()

    log_dir = Path(search_def.log_dir) / search_def.model_id.split("/")[-1]
    log_dir.mkdir(parents=True, exist_ok=True)

    report_data = dataclasses.asdict(search_def)

    loaded_report_data = load_report(log_dir)
    if loaded_report_data is not None:
        # Check if other fields match
        report_data_copy = copy.deepcopy(report_data)
        report_data_copy["log_dir"] = str(report_data_copy["log_dir"])
        report_data_copy["compression_mode"] = str(report_data_copy["compression_mode"])
        loaded_report_data_copy = loaded_report_data.copy()
        del loaded_report_data_copy["iterations"]
        del loaded_report_data_copy["iteration_results"]
        del loaded_report_data_copy["start_metric"]
        if report_data_copy != loaded_report_data_copy:
            raise ValueError(
                "Loaded search definition metadata does not match the current one. "
                f"Loaded: {loaded_report_data_copy}\nCurrent: {report_data}"
            )
        report_data = loaded_report_data

        start_iter = len(loaded_report_data["iterations"]) - 1
        if start_iter != -1:
            start_sub_iter = len(loaded_report_data["iterations"][start_iter])
            if start_sub_iter == len(matmul_node_names):
                start_iter += 1
                start_sub_iter = 0
        else:
            start_sub_iter = 0
    else:
        report_data["start_metric"] = None
        report_data["iterations"] = []
        report_data["iteration_results"] = []

        start_iter = 0
        start_sub_iter = 0

    if report_data["start_metric"] is None:
        print("Running initial model evaluation...")
        compression_kwargs = dict(mode=search_def.compression_mode, group_size=search_def.start_group_size)
        report_data["start_metric"] = run_main(
            search_def.model_id,
            compression_kwargs,
            log_dir / f"start",
            search_def.metric_name
        )
        save_report(log_dir, report_data)

    group_size_mapping = {}
    if start_iter > 0:
        for i in range(start_iter):
            group_size_mapping[report_data["iteration_results"][i]["best_node_name"]] = search_def.iter_group_size

    n_iters = int(len(matmul_node_names) * search_def.search_ratio)
    for i in range(start_iter, n_iters):
        best_metric = None
        best_node_name = None
        if start_sub_iter == 0:
            report_data["iterations"].append([])
        for j in range(len(matmul_node_names)):
            node_name = matmul_node_names[j]
            if j < start_sub_iter:
                metric, node_name = report_data["iterations"][-1][j]["metric"], report_data["iterations"][-1][j]["node_name"]
            else:
                print("Iteration:", i, "Sub-iteration:", j, "Node:", matmul_node_names[j])
                if node_name in group_size_mapping:
                    # The node has already been processed in a previous iteration
                    metric = inf_metric
                else:
                    iter_group_size_mapping = copy.deepcopy(group_size_mapping)
                    iter_group_size_mapping[node_name] = search_def.iter_group_size
                    compression_kwargs = dict(
                        mode=search_def.compression_mode,
                        group_size=search_def.start_group_size,
                        advanced_parameters=nncf.AdvancedCompressionParameters(group_size_mapping=iter_group_size_mapping)
                    )
                    metric = run_main(
                        search_def.model_id,
                        compression_kwargs,
                        log_dir / f"iter_{i:04}/{j:04}",
                        search_def.metric_name
                    )
                report_data["iterations"][-1].append({"node_name": node_name, "metric": metric})
                save_report(log_dir, report_data)

            if (best_metric is None
                or (higher_metric_is_better and metric > best_metric)
                or (not higher_metric_is_better and metric < best_metric)
            ):
                best_metric = metric
                best_node_name = node_name

        group_size_mapping[best_node_name] = search_def.iter_group_size
        report_data["iteration_results"].append(
            {
                "best_metric": best_metric,
                "best_node_name": best_node_name,
            }
        )
        save_report(log_dir, report_data)
        start_sub_iter = 0


def main_search(
    log_dir: Union[str, Path],
    model_id: str,
    metric_name: str,
    compression_mode: str,
    start_group_size: int,
    iter_group_size: int,
    search_ratio: float
):
    search_def = SearchDefinition(
        log_dir=log_dir,
        model_id=model_id,
        metric_name=metric_name,
        compression_mode=compression_mode,
        start_group_size=start_group_size,
        iter_group_size=iter_group_size,
        search_ratio=search_ratio
    )
    run_search(search_def)


if __name__ == "__main__":
    metric_name = "wikitext_validation"
    log_dir = Path("group_size_search")
    model_ids = [
        "meta-llama/Llama-3.2-1B-Instruct",
        # "microsoft/Phi-4-mini-instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
    ]
    for model_id in model_ids:
        main_search(
            log_dir / "256_64_0.25",
            model_id,
            metric_name,
            nncf.CompressWeightsMode.INT4_ASYM,
            start_group_size=256,
            iter_group_size=64,
            search_ratio=0.25,
        )

    for model_id in model_ids:
        main_search(
            log_dir / "64_256_0.25",
            model_id,
            metric_name,
            "int4_asym",
            start_group_size=64,
            iter_group_size=256,
            search_ratio=0.25,
        )
