#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 23:47
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pandas as pd
import numpy as np
from axisfuzzy.analysis.pipeline import FuzzyPipeline
from axisfuzzy.analysis.components.basic import (
    NormalizationTool,
    StatisticsTool,
    SimpleAggregationTool
)


def create_preprocessing_sub_pipeline() -> FuzzyPipeline:
    """
    Creates a sub-pipeline for data preprocessing and basic statistics.
    This sub-pipeline will be nested within a larger pipeline.
    """
    sub_pipeline = FuzzyPipeline(name="PreprocessingAndStats")

    # Define the input for the sub-pipeline
    # 子管道的输入是一个名为 'input_data' 的 CrispTable
    input_data = sub_pipeline.input("input_data", contract='CrispTable')

    # Step 1: Normalize the data (Min-Max, column-wise)
    # 使用 NormalizationTool 对输入数据进行列向的 Min-Max 归一化
    normalizer = NormalizationTool(method='min_max', axis=0)
    normalized_data = sub_pipeline.add(normalizer.run, data=input_data)

    # Step 2: Calculate statistics on the normalized data
    # 使用 StatisticsTool 计算归一化后数据的整体统计信息
    stats_calculator = StatisticsTool(axis=0)  # axis=0 for column-wise stats
    statistics_output = sub_pipeline.add(stats_calculator.run, data=normalized_data)

    # Step 3: Aggregate the normalized data (e.g., calculate mean for each row)
    # 使用 SimpleAggregationTool 计算每行的平均值
    aggregator = SimpleAggregationTool(operation='mean', axis=1)
    aggregated_values = sub_pipeline.add(aggregator.run, data=normalized_data)

    # 注意：这个子管道有两个“末端”输出：statistics_output 和 aggregated_values
    # 当它被嵌套时，其输出将是一个字典，包含这两个结果。
    return sub_pipeline


def run_nested_pipeline_example():
    """
    Demonstrates how to use a FuzzyPipeline as a nested component within another FuzzyPipeline.
    """
    print("--- Starting Nested Pipeline Example ---")

    # 1. Prepare some sample crisp data
    data = pd.DataFrame({
        'Feature_X': [10, 20, 15, 25, 30],
        'Feature_Y': [100, 80, 120, 90, 110],
        'Feature_Z': [5, 8, 6, 7, 9]
    }, index=['Sample_1', 'Sample_2', 'Sample_3', 'Sample_4', 'Sample_5'])

    print("\nOriginal Data:")
    print(data)

    # 2. Create the sub-pipeline
    preprocessing_sub_pipeline = create_preprocessing_sub_pipeline()
    print(f"\nCreated Sub-Pipeline: {preprocessing_sub_pipeline}")

    # 3. Build the main pipeline
    main_pipeline = FuzzyPipeline(name="MainAnalysisFlow")

    # Define the input for the main pipeline
    # 主管道的输入是一个名为 'main_input_data' 的 CrispTable
    main_input_data = main_pipeline.input("main_input_data", contract='CrispTable')

    # Add the sub-pipeline as a step in the main pipeline!
    # 这是实现嵌套的关键一步。我们将子管道作为一个可调用的工具传递给 add 方法。
    # 注意：kwargs 的键 ('input_data') 必须匹配子管道的输入名称。
    sub_pipeline_results = main_pipeline.add(
        preprocessing_sub_pipeline,
        input_data=main_input_data  # 将主管道的输入连接到子管道的输入
    )

    # Now, `sub_pipeline_results` holds the outputs from the sub-pipeline.
    # 根据我们对 pipeline.py 的修改，如果子管道有多个末端输出，
    # `sub_pipeline_results` 将是一个字典，其键是子管道中末端步骤的 display_name。
    # 我们可以通过这些键来访问子管道的各个输出。
    # 让我们打印 sub_pipeline_results 的类型和内容来确认
    print(f"\nType of sub_pipeline_results: {type(sub_pipeline_results)}")
    # print(f"Keys available from sub_pipeline_results: {list(sub_pipeline_results.keys())}")

    # 假设我们想对子管道的聚合结果进行进一步处理
    # 我们可以从 sub_pipeline_results 中提取 'SimpleAggregationTool_run_...' 的输出
    # 实际的键会是 'SimpleAggregationTool_run_xxxxxxxx'，其中 xxxxxxxx 是随机ID
    # 为了示例的通用性，我们假设它返回一个名为 'aggregated_values' 的键，或者我们知道其完整的显示名称
    # 在实际使用中，你可能需要检查 sub_pipeline_results.keys() 来获取确切的键名
    # 或者，如果子管道只有一个输出，它会直接返回该输出。
    # 由于我们有两个末端节点，它会返回一个字典。
    # 让我们假设我们知道键的格式，或者通过打印 sub_pipeline_results.keys() 来获取

    # 提取子管道的聚合结果
    # 假设 'SimpleAggregationTool_run_...' 是一个键
    # 实际键名会是 'SimpleAggregationTool_run_xxxxxxxx'
    # 为了演示，我们先假设其 display_name 是 'SimpleAggregationTool.run'
    # 并且其输出是 'aggregated_values'
    # 那么组合后的键名可能是 'SimpleAggregationTool.run_aggregated_values'
    # 或者更简单，如果 get_output_contracts 逻辑将它们扁平化，则直接是 'aggregated_values'

    # 根据之前对 pipeline.py 的修改，如果子管道有多个末端输出，
    # 并且这些输出被合并到一个字典中，那么键名会是 `display_name_output_name` 的形式。
    # 比如：'NormalizationTool_run_normalized_data', 'StatisticsTool_run_statistics', 'SimpleAggregationTool_run_aggregated_values'
    # 并且，如果最终输出多于一个，整个子管道的输出会被包装成一个 'result': 'PipelineResult' 的字典。
    # 这意味着 sub_pipeline_results 将是一个 StepOutput，代表一个 PipelineResult。
    # 它的实际值会在运行时被解析。

    # 为了简化，我们直接运行主管道，并查看最终结果
    # 最终结果将包含子管道的所有末端输出
    final_results, intermediate_states = main_pipeline.run(
        initial_data={"main_input_data": data},
        return_intermediate=True
    )

    print("\n--- Main Pipeline Execution Results ---")

    print("\nFinal Outputs from Main Pipeline (includes sub-pipeline's terminal outputs):")
    # final_results 将是一个字典，其中包含子管道的末端输出
    # 键名会是子管道中步骤的 display_name，例如 'PreprocessingAndStats_result'
    # 然后 'PreprocessingAndStats_result' 的值又是一个字典，包含子管道的实际输出
    # 比如：{'StatisticsTool.run_statistics': {...}, 'SimpleAggregationTool.run_aggregated_values': {...}}

    # 让我们直接打印 final_results 来观察其结构
    print(final_results)

    print("\n--- All Intermediate States (including nested pipeline's internal results) ---")
    # 遍历所有中间状态，可以观察到子管道内部的执行步骤
    for step_id, result_data in intermediate_states.items():
        step_info = main_pipeline.get_step_info(step_id)
        print(f"\nStep '{step_info['display_name']}' (ID: {step_id[:8]}...):")
        # 对于嵌套管道步骤，其结果是子管道的最终输出
        if step_info['display_name'] == preprocessing_sub_pipeline.name:
            print(f"  (This is the nested pipeline step. Its result is the sub-pipeline's final output.)")
        print(result_data)

    print("\n--- Nested Pipeline Example Finished ---")


if __name__ == "__main__":
    run_nested_pipeline_example()
