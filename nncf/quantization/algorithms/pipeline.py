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

from typing import Dict, List, Optional, TypeVar, Union

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.algorithm import Algorithm

TModel = TypeVar("TModel")
PipelineStep = List[Algorithm]

import pickle
from pathlib import Path
overwrite_statistics = bool(0)
save_statistics_as_dict = bool(0)
save_statistics_as_obj = bool(0)
read_stats_file = Path("ptq_stats/obj_reversed.pkl")
write_stats_filepath = Path(f"ptq_stats/{'obj_' if save_statistics_as_obj else ''}stats.pkl")
write_statistics = {}


def process_algo_statistic_points(statistic_points, step_index: int):
    from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic, OVMeanTensorStatistic
    from nncf.common.graph.transformations.commands import TargetType

    if save_statistics_as_obj:
        with open(str(write_stats_filepath).replace(".pkl", f"_step{step_index}.pkl"), 'wb') as f:
            pickle.dump(statistic_points, f)

    if overwrite_statistics:
        with open(str(read_stats_file).replace(".pkl", f"_step{step_index}.pkl"), 'rb') as f:
            read_statistic_points = pickle.load(f)
        return read_statistic_points

    if not save_statistics_as_dict:
        return

    for node_name, s_ps in statistic_points.items():
        for s_p in s_ps:
            for algo, collectors in s_p.algorithm_to_tensor_collectors.items():
                algo_name = algo.split('_')[0]
                for collector in collectors:
                    statistics = collector.get_statistics()
                    if node_name not in write_statistics:
                        write_statistics[node_name] = {}
                    if algo_name not in write_statistics[node_name]:
                        write_statistics[node_name][algo_name] = {}
                    if isinstance(statistics, dict):
                        for stat_name in statistics.keys():
                            if save_statistics_as_dict:
                                assert stat_name not in write_statistics[node_name][algo_name]
                                write_statistics[node_name][algo_name][stat_name] = statistics[stat_name]
                    elif isinstance(statistics, OVMinMaxTensorStatistic):
                        if save_statistics_as_dict:
                            assert "min_values" not in write_statistics[node_name][algo_name]
                            assert "max_values" not in write_statistics[node_name][algo_name]
                            write_statistics[node_name][algo_name]["min_values"] = statistics.min_values
                            write_statistics[node_name][algo_name]["max_values"] = statistics.max_values
                            if hasattr(statistics, "no_op"):
                                assert "no_op" not in write_statistics[node_name][algo_name]
                                write_statistics[node_name][algo_name]["no_op"] = statistics.no_op
                    elif isinstance(statistics, OVMeanTensorStatistic):
                        if s_p.target_point.type == TargetType.PRE_LAYER_OPERATION:
                            suffix = 'pre'
                        elif s_p.target_point.type == TargetType.POST_LAYER_OPERATION:
                            suffix = 'post'
                        else:
                            raise Exception(f"Can't handle such target point type: {s_p.target_point.type}")
                        mean_name = f"mean_values_{suffix}"
                        shape_name = f"shape_values_{suffix}"
                        if save_statistics_as_dict:
                            assert mean_name not in write_statistics[node_name][algo_name]
                            assert shape_name not in write_statistics[node_name][algo_name]
                            write_statistics[node_name][algo_name][mean_name] = statistics.mean_values
                            write_statistics[node_name][algo_name][shape_name] = statistics.shape
                    else:
                        raise Exception(f"Can't handle such statistics type: {type(statistics)}")


def collect_statistics(
    containers: Union[StatisticPointsContainer, List[StatisticPointsContainer]],
    model: TModel,
    graph: NNCFGraph,
    dataset: Dataset,
) -> StatisticPointsContainer:
    """
    Utility method for collecting statistics by model.

    :param statistic_points: Statistic points that need to be collected.
    :param model: A model.
    :param graph: A graph assosiated with a model.
    :param dataset: A dataset.
    :return: Collected statistics.
    """
    if not isinstance(containers, list):
        containers = [containers]

    statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
    for container in containers:
        statistics_aggregator.register_statistic_points(container)
    statistics_aggregator.collect_statistics(model, graph)

    return statistics_aggregator.statistic_points


class Pipeline:
    """
    A class for creating pipelines that apply algorithms to a model.

    This class is used for creating custom model processing pipelines
    that encapsulate a series of algorithms to be applied to a model
    using a provided dataset.

    A pipeline consists of pipeline steps. Each pipeline step is a
    sequence of Algorithm class instances whose statistic points are
    combined and collected using the model obtained after the previous
    pipeline step. The collected statistic points are used for all
    algorithms in this step.
    """

    def __init__(self, pipeline_steps: List[PipelineStep]):
        """
        :param pipeline_steps: A sequence of pipeline steps to be executed in order.
        """
        self._pipeline_steps = pipeline_steps

    @property
    def pipeline_steps(self) -> List[PipelineStep]:
        """
        Property that defines the sequence of distinct pipeline steps to
        be executed in order.

        :return: A sequence of pipeline steps to be executed in order.
        """
        return self._pipeline_steps

    def run(self, model: TModel, dataset: Dataset) -> TModel:
        """
        Executes the pipeline on the provided model.

        :param model: A model to which pipeline will be applied.
        :param dataset: A dataset that holds the data items for algorithms.
        :return: The updated model after executing the entire pipeline.
        """
        return self.run_from_step(model, dataset)

    def run_step(
        self,
        step_index: int,
        step_statistics: StatisticPointsContainer,
        model: TModel,
        graph: NNCFGraph,
    ) -> TModel:
        """
        Executes a provided pipeline step on the provided model.

        :param step_index: Zero-based index of the pipeline step that should be executed
        :param step_statistics: Statistics required to execute a pipeline step.
        :param model: A model to which a pipeline step will be applied.
        :param graph: A graph assosiated with a model.
        :return: The updated model after executing the pipeline step.
        """
        current_model = model
        current_graph = graph

        pipeline_steps = self._remove_unsupported_algorithms(get_backend(model))
        pipeline_step = pipeline_steps[step_index]
        for algorithm in pipeline_step[:-1]:
            current_model = algorithm.apply(current_model, current_graph, step_statistics)
            current_graph = NNCFGraphFactory.create(current_model)
        current_model = pipeline_step[-1].apply(current_model, current_graph, step_statistics)

        return current_model

    def run_from_step(
        self,
        model: TModel,
        dataset: Dataset,
        graph: Optional[NNCFGraph] = None,
        start_step_index: int = 0,
        step_index_to_statistics: Optional[Dict[int, StatisticPointsContainer]] = None,
    ) -> TModel:
        """
        Executes the pipeline from the specified pipeline step to the end.

        :param model: This is the model after the (start_step_index - 1)-th pipeline
            step, or the initial model if start_step_index is 0.
        :param dataset: A dataset that holds the data items for pipeline steps.
        :param graph: A graph assosiated with a model.
        :param start_step_index: Zero-based pipeline step index from which the pipeline
            should be executed.
        :param step_index_to_statistics: A mapping from pipeline step index to statistics
            required to execute pipeline step.
        :return: The updated model after executing the pipeline from the specified pipeline
            step to the end.
        """
        pipeline_steps = self._remove_unsupported_algorithms(get_backend(model))
        if step_index_to_statistics is None:
            step_index_to_statistics = {}

        # The `step_model` and `step_graph` entities are required to execute `step_index`-th pipeline step
        step_model = model
        step_graph = graph
        for step_index in range(start_step_index, len(pipeline_steps)):
            # Create graph required to run current pipeline step
            if step_graph is None:
                step_graph = NNCFGraphFactory.create(step_model)

            # Collect statistics required to run current pipeline step
            step_statistics = step_index_to_statistics.get(step_index)
            if step_statistics is None:
                statistic_points = self.get_statistic_points_for_step(step_index, step_model, step_graph)
                step_statistics = collect_statistics(statistic_points, step_model, step_graph, dataset)

            new_statistic_points = process_algo_statistic_points(step_statistics, step_index)
            if new_statistic_points is not None:
                step_statistics = new_statistic_points

            # Run current pipeline step
            step_model = self.run_step(step_index, step_statistics, step_model, step_graph)

            step_graph = None  # We should rebuild the graph for the next pipeline step

        if save_statistics_as_dict:
            global write_statistics
            print(f"Saving statistics to {write_stats_filepath}")
            write_stats_filepath.parent.mkdir(exist_ok=True, parents=True)
            with open(write_stats_filepath, 'wb') as f:
                pickle.dump(write_statistics, f)
            write_statistics = {}

        return step_model

    def get_statistic_points_for_step(
        self, step_index: int, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        """
        Returns statistics that should be collected to execute `step_index`-th pipeline step.

        :param step_index: Zero-based index of the pipeline step.
        :param model: A model.
        :param graph: A graph assosiated with a model.
        :return: Statistics that should be collected to execute `step_index`-th pipeline step.
        """
        container = StatisticPointsContainer()
        pipeline_steps = self._remove_unsupported_algorithms(get_backend(model))
        pipeline_step = pipeline_steps[step_index]
        for algorithm in pipeline_step:
            for statistic_points in algorithm.get_statistic_points(model, graph).values():
                for statistic_point in statistic_points:
                    container.add_statistic_point(statistic_point)

        return container

    def _remove_unsupported_algorithms(self, backend: BackendType) -> List[PipelineStep]:
        pipeline_steps = []
        for pipeline_step in self._pipeline_steps:
            step = []
            for algorithm in pipeline_step:
                if backend not in algorithm.available_backends:
                    nncf_logger.debug(f"{backend.name} does not support {algorithm.__class__.__name__} algorithm yet.")
                    continue
                step.append(algorithm)

            if step:
                pipeline_steps.append(step)

        return pipeline_steps
