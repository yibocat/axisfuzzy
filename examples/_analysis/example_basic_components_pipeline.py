#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/23 23:35
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import pandas as pd
import numpy as np
from axisfuzzy.analysis._pipeline import FuzzyPipeline
from axisfuzzy.analysis._components.basic import (
    NormalizationTool,
    WeightNormalizationTool,
    StatisticsTool,
    SimpleAggregationTool
)


def run_basic_components_example():
    """
    Demonstrates the usage of basic analysis _components from axisfuzzy.analysis._components.basic
    within a FuzzyPipeline.
    """
    print("--- Starting Basic Components Pipeline Example ---")

    # 1. Prepare some sample crisp data (CrispTable)
    # 假设我们有三个备选方案（行）和三个评估标准（列）
    data = pd.DataFrame({
        'Criteria_A': [10, 25, 15, 30],
        'Criteria_B': [100, 80, 120, 90],
        'Criteria_C': [5, 8, 6, 7]
    }, index=['Alt_1', 'Alt_2', 'Alt_3', 'Alt_4'])

    print("\nOriginal Data (CrispTable):")
    print(data)

    # 2. Build the FuzzyPipeline
    pipeline = FuzzyPipeline(name="BasicDataProcessingPipeline")

    # Define the input for the pipeline
    # 管道的输入是一个名为 'raw_data' 的 CrispTable
    raw_data_input = pipeline.input("raw_data", contract='CrispTable')

    # Step 1: Normalize the data using Min-Max scaling (column-wise)
    # 使用 NormalizationTool 对原始数据进行列向的 Min-Max 归一化
    normalizer = NormalizationTool(method='min_max', axis=0)
    normalized_data_output = pipeline.add(normalizer.run, data=raw_data_input)

    # Step 2: Calculate statistics on the normalized data
    # 使用 StatisticsTool 计算归一化后数据的整体统计信息
    stats_calculator = StatisticsTool(axis=0)  # axis=0 for column-wise stats
    statistics_output = pipeline.add(stats_calculator.run, data=normalized_data_output)

    # Step 3: Aggregate the normalized data (e.g., calculate mean for each row)
    # 使用 SimpleAggregationTool 计算每行的平均值
    aggregator = SimpleAggregationTool(operation='mean', axis=1)
    aggregated_values_output = pipeline.add(aggregator.run, data=normalized_data_output)

    # Step 4: Demonstrate WeightNormalizationTool separately (as it takes a WeightVector, not CrispTable)
    # 创建一个独立的权重归一化步骤，它不直接依赖于前面的 CrispTable 流程
    # 假设我们有一个初始权重向量
    initial_weights = np.array([0.2, 0.5, 0.3, 0.1])  # 这是一个 WeightVector
    weights_input = pipeline.input("initial_weights", contract='WeightVector')

    weight_normalizer = WeightNormalizationTool()
    normalized_weights_output = pipeline.add(weight_normalizer.run, weights=weights_input)

    # 3. Run the pipeline
    # 运行管道，传入原始数据和初始权重
    # 注意：如果管道有多个输入，initial_data 必须是一个字典
    results = pipeline.run(
        initial_data={
            "raw_data": data,
            "initial_weights": initial_weights
        },
        return_intermediate=True  # 返回所有中间结果，便于查看
    )

    final_outputs, intermediate_results = results

    print("\n--- Pipeline Execution Results ---")

    # Print final outputs (from terminal nodes)
    print("\nFinal Outputs (from terminal nodes):")
    # 根据 _pipeline.py 中 _format_final_output 的逻辑，如果只有一个输出，直接返回；
    # 如果有多个，返回一个字典，键是步骤的 display_name。
    # 这里我们有三个末端节点：statistics_output, aggregated_values_output, normalized_weights_output
    # 所以 final_outputs 会是一个字典
    for key, value in final_outputs.items():
        print(f"\nOutput '{key}':")
        print(value)

    # Print intermediate results (for all steps)
    print("\nIntermediate Results (all steps):")
    for step_id, result_data in intermediate_results.items():
        step_info = pipeline.get_step_info(step_id)
        print(f"\nStep '{step_info}:")
        print(result_data)

    print("\n--- Basic Components Pipeline Example Finished ---")


if __name__ == "__main__":
    run_basic_components_example()
