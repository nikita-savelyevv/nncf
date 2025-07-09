import copy
import dataclasses
import gc
import json
import shutil
from pathlib import Path
from typing import Union, List

import openvino as ov
import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from optimum.intel import OVModelForCausalLM
from tqdm import tqdm
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


def extract_matmul_node_names(model: Union[torch.nn.Module, ov.Model], compress_embeddings: bool = False) -> List[str]:
    from nncf.common.factory import NNCFGraphFactory
    from nncf.quantization.algorithms.weight_compression.torch_backend import PTWeightCompressionAlgoBackend
    from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

    is_pt = isinstance(model, torch.nn.Module)
    ov_backend = None
    if is_pt:
        model = wrap_model(model, model.dummy_inputs, True)
    else:
        ov_backend = OVWeightCompressionAlgoBackend(model)

    nncf_graph = NNCFGraphFactory.create(model)
    nodes_to_compress = []
    for node in nncf_graph.topological_sort():
        if is_pt:
            if PTWeightCompressionAlgoBackend.is_node_with_weights(node, nncf_graph):
                if node.metatype in PTWeightCompressionAlgoBackend.MATMUL_METATYPES:
                    nodes_to_compress.append(node)
                if compress_embeddings and node.metatype in PTWeightCompressionAlgoBackend.EMBEDDING_METATYPES:
                    nodes_to_compress.append(node)
        else:
            if ov_backend.is_node_with_weights(node, nncf_graph):
                if node.metatype in ov_backend.matmul_metatypes:
                    nodes_to_compress.append(node)
                if compress_embeddings and node.metatype in ov_backend.embedding_metatypes:
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
        del loaded_report_data_copy["start_test_metric"]
        if report_data_copy != loaded_report_data_copy:
            raise ValueError(
                "Loaded search definition metadata does not match the current one. "
                f"Loaded: {loaded_report_data_copy}\nCurrent: {report_data}"
            )
        report_data = loaded_report_data

        start_iter = len(loaded_report_data["iterations"]) - 1
        if start_iter != -1:
            start_sub_iter = len(loaded_report_data["iterations"][start_iter])
            if (
                start_sub_iter == len(matmul_node_names) and
                len(loaded_report_data["iterations"]) == len(loaded_report_data["iteration_results"])
            ):
                start_iter += 1
                start_sub_iter = 0
        else:
            start_sub_iter = 0
    else:
        report_data["start_metric"] = None
        report_data["start_test_metric"] = None
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
        report_data["start_test_metric"] = run_main(
            search_def.model_id,
            compression_kwargs,
            log_dir / f"start",
            "wikitext"
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

        # Update current best size mapping
        group_size_mapping[best_node_name] = search_def.iter_group_size

        # Test validation
        compression_kwargs = dict(
            mode=search_def.compression_mode,
            group_size=search_def.start_group_size,
            advanced_parameters=nncf.AdvancedCompressionParameters(group_size_mapping=group_size_mapping)
        )
        test_metric = run_main(
            search_def.model_id,
            compression_kwargs,
            log_dir / f"iter_{i:04}/{j:04}",
            "wikitext"
        )

        report_data["iteration_results"].append(
            {
                "best_metric": best_metric,
                "best_node_name": best_node_name,
                "test_metric": test_metric,
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


def compress_model_from_report(report_path: Path):
    def torch_node_name_to_ov_node_name(torch_node_name: str) -> str:
        if "lm_head" in torch_node_name:
            # Special case for lm_head
            return "__module.lm_head/ov_ext::linear/MatMul"
        parts = torch_node_name.split('/')
        index = parts[-1]
        return f"__module.model.layers.{index}.{'.'.join(parts[1:-2])}/ov_ext::linear/MatMul"

    report_data = load_report(report_path)
    if report_data is None:
        raise ValueError(f"No report found at {report_path}")

    #
    # Run compression for PyTorch model
    #

    model_id = report_data["model_id"]
    torch_group_size_mapping = {}
    iteration_results = report_data["iteration_results"]
    for it_res in iteration_results:
        torch_group_size_mapping[it_res["best_node_name"]] = report_data["iter_group_size"]
    main(
        model_id,
        ModelBacked.PT,
        ModelBacked.PT,
        compression_kwargs=dict(
            mode=nncf.CompressWeightsMode(report_data["compression_mode"]),
            group_size=report_data["start_group_size"],
            advanced_parameters=nncf.AdvancedCompressionParameters(group_size_mapping=torch_group_size_mapping)
        ),
        save_dir=report_path / "final_compressed_model" / "torch",
        eval_task="wikitext",
        pt_dtype=torch.bfloat16,
        backup_device="cpu",
        do_sample_generation=False,
    )

    #
    # Run compression for OpenVINO model
    #

    ov_model = OVModelForCausalLM.from_pretrained(model_id, load_in_8bit=False)
    ov_matmul_node_names = extract_matmul_node_names(ov_model.model)

    ov_group_size_mapping = {}
    matched_nodes = set()
    for torch_node_name, group_size in torch_group_size_mapping.items():
        ov_node_name = torch_node_name_to_ov_node_name(torch_node_name)
        if ov_node_name not in ov_matmul_node_names:
            from Levenshtein import distance
            closest_node = min(ov_matmul_node_names, key=lambda x: distance(x, ov_node_name))
            raise ValueError(
                f"OpenVINO model does not contain node {ov_node_name} corresponding to torch node {torch_node_name}. "
                f"Closest node found: {closest_node}."
            )
        if ov_node_name in matched_nodes:
            raise ValueError(
                f"Multiple torch nodes map to the same OpenVINO node: {ov_node_name}. "
            )
        matched_nodes.add(ov_node_name)
        ov_group_size_mapping[ov_node_name] = group_size

    del ov_model
    gc.collect()

    main(
        model_id,
        ModelBacked.OV,
        ModelBacked.OV,
        compression_kwargs=dict(
            mode=nncf.CompressWeightsMode(report_data["compression_mode"]),
            group_size=report_data["start_group_size"],
            advanced_parameters=nncf.AdvancedCompressionParameters(group_size_mapping=ov_group_size_mapping)
        ),
        save_dir=report_path / "final_compressed_model" / "openvino",
        eval_task="wikitext",
        do_sample_generation=False,
    )


def plot_sorting_similarity(report_path: Path):
    inf_metrics = [-1e9, 1e9]
    report_data = load_report(report_path)

    iterations_data = report_data["iterations"]
    # iterations_data = iterations_data[:-1] # Remove the last iteration, which is not complete
    # node_names = sorted([it["node_name"] for it in iterations_data[0]])
    node_names = sorted([it["node_name"] for it in iterations_data[0]], key=lambda x: (int(x.split("/")[-1]), "/".join(x.split("/")[:-1])))
    node_ids = {n: i for i, n in enumerate(node_names)}
    metric_data = []
    order_data = []
    for it_res in iterations_data:
        it_data = [(it["metric"], node_ids[it["node_name"]]) for it in it_res]
        it_data = sorted(it_data)
        metric_data.append(it_data)
        it_order = [np.nan] * len(node_names)
        for i, (metric, node_id) in enumerate(it_data):
            if metric not in inf_metrics:
                it_order[node_id] = i
        order_data.append(it_order)

    # Compute pairwise Kendall's tau correlation
    from scipy.stats import kendalltau
    plot_data = [[None] * len(order_data) for _ in range(len(order_data))]
    for i in tqdm(range(len(order_data))):
        # for j in range(len(order_data)):
        for j in range(i + 1, len(order_data)):
            correlation = kendalltau(order_data[i], order_data[j], nan_policy="omit").statistic
            plot_data[i][j] = plot_data[j][i] = correlation

            # k = 2 - 1
            # top_k_best_metric_i = metric_data[i][k][0]
            # best_metric_j = metric_data[j][0][0]
            # diff = top_k_best_metric_i - best_metric_j
            # plot_data[i][j] = diff

    plot_data[0][0] = 1
    plot_data[1][1] = 0
    plot_data = np.array(plot_data, dtype=np.float32)
    ax = sns.heatmap(plot_data, linewidth=0.5, cmap="viridis")
    plt.savefig(report_path / "sorting_similarity.png")
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    metric_name = "wikitext_validation"
    log_dir = Path("group_size_search")

    # plot_sorting_similarity(log_dir / "256_64_0.25" / "Llama-3.2-1B-Instruct")
    # plot_sorting_similarity(log_dir / "64_256_0.25" / "Llama-3.2-1B-Instruct")
    # plot_sorting_similarity(log_dir / "256_64_0.25" / "Phi-4-mini-instruct")
    # plot_sorting_similarity(log_dir / "64_256_0.25" / "Phi-4-mini-instruct")
    # plot_sorting_similarity(log_dir / "256_64_0.1" / "Llama-3.1-8B-Instruct")
    # exit(0)

    # compress_model_from_report(log_dir / "256_64_0.25" / "Llama-3.2-1B-Instruct")
    # compress_model_from_report(log_dir / "64_256_0.25" / "Llama-3.2-1B-Instruct")
    # compress_model_from_report(log_dir / "256_64_0.25" / "Phi-4-mini-instruct")
    # compress_model_from_report(log_dir / "64_256_0.25" / "Phi-4-mini-instruct")

    # model_ids = [
    #     "meta-llama/Llama-3.2-1B-Instruct",
    #     "microsoft/Phi-4-mini-instruct",
    # ]
    # for model_id in model_ids:
    #     main_search(
    #         log_dir / "256_64_0.25",
    #         model_id,
    #         metric_name,
    #         nncf.CompressWeightsMode.INT4_ASYM,
    #         start_group_size=256,
    #         iter_group_size=64,
    #         search_ratio=0.25,
    #     )
    #
    # for model_id in model_ids:
    #     main_search(
    #         log_dir / "64_256_0.25",
    #         model_id,
    #         metric_name,
    #         nncf.CompressWeightsMode.INT4_ASYM,
    #         start_group_size=64,
    #         iter_group_size=256,
    #         search_ratio=0.25,
    #     )

    model_ids = [
        "meta-llama/Llama-3.1-8B-Instruct",
    ]
    for model_id in model_ids:
        main_search(
            log_dir / "256_64_0.1",
            model_id,
            metric_name,
            nncf.CompressWeightsMode.INT4_ASYM,
            start_group_size=256,
            iter_group_size=64,
            search_ratio=0.1,
        )

    for model_id in model_ids:
        main_search(
            log_dir / "64_256_0.1",
            model_id,
            metric_name,
            nncf.CompressWeightsMode.INT4_ASYM,
            start_group_size=64,
            iter_group_size=256,
            search_ratio=0.1,
        )
