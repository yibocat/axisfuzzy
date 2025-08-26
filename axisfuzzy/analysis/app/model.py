#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/26 14:12
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy
import inspect
from abc import ABC, abstractmethod

from collections import OrderedDict
from typing import Any, get_type_hints, Dict, List

# 导入组件模块, 契约系统, 管道 DAG 引擎
from ..component import AnalysisComponent
from ..contracts import Contract
from ..pipeline import FuzzyPipeline, StepOutput, StepMetadata


class Model(AnalysisComponent, ABC):
    """
    所有模糊分析模型的基类，其设计灵感来源于 PyTorch 的 nn.Module。

    用户应继承此类，在 `__init__` 中将分析组件（或其他 Model）定义为属性，
    并在 `forward` 方法中实现数据流图。本类会自动将 `forward` 方法中
    的调用“追踪”并编译成一个底层的 `FuzzyPipeline` 实例。
    """
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self._pipeline: FuzzyPipeline | None = None
        self._modules = OrderedDict()
        self.built = False  # 增加一个状态标志

    @property
    def pipeline(self) -> FuzzyPipeline:
        """公开访问内部构建的 FuzzyPipeline。"""
        if not self.built or self._pipeline is None:
            raise RuntimeError("Model has not been built yet. Call .build() first.")
        return self._pipeline

    def add_module(self, name: str, module: AnalysisComponent | None):
        """
        将一个分析组件或子模型注册为当前模型的子模块。

        Args:
            name (str): 子模块的名称。
            module (AnalysisComponent | None): 要注册的分析组件或子模型。
        """
        if not isinstance(module, AnalysisComponent) and module is not None:
            raise TypeError(f"{module} is not a valid AnalysisComponent")
        if '.' in name:
            raise KeyError("Module name cannot contain '.'")
        self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        """
        重载属性设置，自动将 AnalysisComponent 注册为子模块。
        """
        if isinstance(value, AnalysisComponent):
            self.add_module(name, value)
        super().__setattr__(name, value)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        定义模型前向传播逻辑（即数据流）。子类必须实现此方法。

        定义复杂,非线性, 嵌套模型的核心. 在编写模型子类的时候必须实现该方法
        """
        raise NotImplementedError

    def build(self):
        """
        【第二步】: 构建模型的计算图 (DAG)。
        此方法会追踪 forward 逻辑并生成底层的 FuzzyPipeline。
        """
        if self.built:
            print(f"Warning: Model '{self.name}' has already been built. Re-building...")

        print(f"--- Building FuzzyPipeline for '{self.name}'... ---")
        p = FuzzyPipeline(name=self.name)

        sig = inspect.signature(self.forward)
        type_hints = get_type_hints(self.forward)
        symbolic_inputs = {}

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            contract = type_hints.get(param_name, Contract.get('Any'))
            if not isinstance(contract, Contract):
                raise TypeError(f"Type hint for '{param_name}' must be a Contract object.")
            symbolic_inputs[param_name] = p.input(param_name, contract=contract)

        original_calls = {}
        try:
            for name, module in self._modules.items():
                if module is None:
                    continue

                # 对子模型递归调用 build
                if isinstance(module, Model) and not module.built:
                    module.build()

                original_calls[name] = module.__class__.__call__

                # 猴子补丁
                def _tracing_call(mod_self, *call_args, **call_kwargs):
                    print(f"  > Tracing call to '{mod_self.__class__.__name__}'...")

                    # 准备要传递给 p.add 的参数
                    add_kwargs = {}

                    if isinstance(mod_self, Model):
                        # --- 处理嵌套模型 ---
                        callable_tool = mod_self.pipeline
                        # FuzzyPipeline 的输入节点名称是其参数名
                        # 我们需要将调用时的参数映射到管道的输入节点上
                        pipeline_input_names = list(callable_tool.input_nodes.keys())

                        if len(pipeline_input_names) == 1 and len(call_args) == 1 and not call_kwargs:
                            add_kwargs = {pipeline_input_names[0]: call_args[0]}
                        else:
                            add_kwargs = {**dict(zip(pipeline_input_names, call_args)), **call_kwargs}

                    else:
                        # --- 处理普通组件 ---
                        callable_tool = mod_self.run
                        # 使用 inspect.signature 来获取普通 run 方法的参数
                        run_sig = inspect.signature(callable_tool)
                        run_params = list(run_sig.parameters.keys())
                        if 'self' in run_params: run_params.remove('self')

                        if len(run_params) == 1 and len(call_args) == 1 and not call_kwargs:
                            add_kwargs = {run_params[0]: call_args[0]}
                        else:
                            add_kwargs = {**dict(zip(run_params, call_args)), **call_kwargs}

                    # 现在 p.add() 会收到正确的 callable_tool 和参数
                    return p.add(callable_tool, **add_kwargs)

                module.__class__.__call__ = _tracing_call

            print("--- Starting symbolic trace of 'forward' method... ---")
            self.forward(**symbolic_inputs)
            print("--- Symbolic trace complete. ---")

        finally:
            for name, original_call in original_calls.items():
                if name in self._modules and self._modules[name] is not None:
                    self._modules[name].__class__.__call__ = original_call

        self._pipeline = p
        self.built = True
        print(f"--- FuzzyPipeline for '{self.name}' built successfully. ---\n")

    def run(self, *args, **kwargs) -> Any:
        """
        【第三步】: 执行实际计算。
        """
        if not self.built:
            raise RuntimeError("Model has not been built yet. Call .build() before running.")

        # 确定初始数据格式
        sig = inspect.signature(self.forward)
        input_names = list(sig.parameters.keys())
        if 'self' in input_names:
            input_names.remove('self')

        if len(input_names) == 1 and len(args) == 1 and not kwargs:
            initial_data = args[0]
        else:
            initial_data = {**dict(zip(input_names, args)), **kwargs}

        return self.pipeline.run(initial_data)

    def __call__(self, *args, **kwargs) -> Any:
        """调用模型实例等同于执行 run 方法。"""
        return self.run(*args, **kwargs)

    def visualize(self, **kwargs):
        """可视化计算图。必须在 build() 之后调用。"""
        if not self.built:
            raise RuntimeError("Model has not been built yet. Call .build() before visualizing.")
        return self.pipeline.visualize(**kwargs)

    def summary(self, line_length: int = 90, positions: list = None):
        """
        打印模型的摘要，基于构建后的 FuzzyPipeline 结构。

        Parameters
        ----------
        line_length : int, default 65
            表格的总宽度。
        positions : list, optional
            各列的相对位置，格式为 [name_width, input_width, output_width]。
            如果为 None，将自动计算合适的列宽。
        """
        if not self.built:
            raise RuntimeError("Model has not been built yet. Call .build() first.")

        # 计算列宽
        if positions is None:
            # 自动计算：根据 line_length 自动分配列宽
            # 预留分隔符空间（通常每列后有1-2个空格）
            available_width = line_length - 4  # 预留一些边距
            name_width = int(available_width * 0.5)  # 名称列占50%
            input_width = int(available_width * 0.25)  # 输入契约列占25%
            output_width = available_width - name_width - input_width  # 输出契约列占剩余
        else:
            if len(positions) != 3:
                raise ValueError("positions must contain exactly 3 elements: [name_width, input_width, output_width]")
            name_width, input_width, output_width = positions
            # 验证宽度是否合理
            total_width = name_width + input_width + output_width + 4  # 加上分隔符空间
            if total_width > line_length:
                print(f"Warning: Column widths ({total_width}) exceed line_length ({line_length})")

        # 生成表格头部
        title = f'Model: "{self.name}"'
        print(title)
        print("=" * line_length)

        # 表头
        header = f"{'Layer (type)':<{name_width}} {'Input Contracts':<{input_width}} {'Output Contracts':<{output_width}}"
        print(header)
        print("=" * line_length)

        # 获取执行顺序（包括输入节点）
        full_order = self._pipeline._build_execution_order()

        total_layers = 0
        nested_models_count = 0
        sub_layers_count = 0

        for step_id in full_order:
            step_meta = self._pipeline.get_step_info(step_id)

            if step_meta.is_input_node:
                # 显示输入节点
                input_name = None
                for name, sid in self._pipeline.input_nodes.items():
                    if sid == step_id:
                        input_name = name
                        break

                output_contract = list(step_meta.output_contracts.values())[0]
                layer_display = f"Input: {input_name}"

                # 截断过长的文本
                layer_display = self._truncate_text(layer_display, name_width)
                input_display = self._truncate_text("-", input_width)
                output_display = self._truncate_text(output_contract.name, output_width)

                print(f"{layer_display:<{name_width}} {input_display:<{input_width}} {output_display:<{output_width}}")

            else:
                # 显示计算节点
                display_name = step_meta.display_name

                # 获取输入契约（从依赖关系推断）
                if step_meta.input_contracts:
                    input_contracts_display = ", ".join([c.name for c in step_meta.input_contracts.values()])
                else:
                    input_contracts_display = "None"

                # 获取输出契约
                if step_meta.output_contracts:
                    output_contracts_display = ", ".join([c.name for c in step_meta.output_contracts.values()])
                else:
                    output_contracts_display = "None"

                # 检查是否是嵌套管道
                if self._is_nested_pipeline_step(step_meta):
                    # 这是一个嵌套管道步骤
                    nested_models_count += 1

                    # 截断文本并显示主层
                    layer_display = self._truncate_text(display_name, name_width)
                    input_display = self._truncate_text(input_contracts_display, input_width)
                    output_display = self._truncate_text(output_contracts_display, output_width)

                    print(f"{layer_display:<{name_width}} {input_display:<{input_width}} {output_display:<{output_width}}")

                    # 显示嵌套管道的内部结构
                    nested_model = self._find_nested_model_by_name(display_name)
                    if nested_model:
                        nested_sub_layers = self._print_nested_components(
                            nested_model, name_width, input_width, output_width
                        )
                        sub_layers_count += nested_sub_layers
                else:
                    # 普通组件
                    component_name = self._extract_component_name(display_name)

                    # 截断文本并显示
                    layer_display = self._truncate_text(component_name, name_width)
                    input_display = self._truncate_text(input_contracts_display, input_width)
                    output_display = self._truncate_text(output_contracts_display, output_width)

                    print(f"{layer_display:<{name_width}} {input_display:<{input_width}} {output_display:<{output_width}}")

                total_layers += 1

        # 表格尾部
        print("=" * line_length)
        if nested_models_count > 0:
            summary_text = f"Total layers: {total_layers} (including {nested_models_count} nested model(s) with {sub_layers_count} sub-layers)"
        else:
            summary_text = f"Total layers: {total_layers}"

        print(summary_text)
        print("=" * line_length)

    def _truncate_text(self, text: str, max_width: int) -> str:
        """
        截断过长的文本，确保不超过指定宽度。

        Parameters
        ----------
        text : str
            要截断的文本
        max_width : int
            最大宽度

        Returns
        -------
        str
            截断后的文本
        """
        if len(text) <= max_width:
            return text
        elif max_width <= 3:
            return "..." if max_width >= 3 else text[:max_width]
        else:
            return text[:max_width-3] + "..."

    def _is_nested_pipeline_step(self, step_meta) -> bool:
        """
        判断步骤是否为嵌套管道步骤。
        """
        return (isinstance(step_meta.callable_tool, type(lambda: None)) and
                hasattr(step_meta.callable_tool, '__name__') and
                'pipeline_runner' in step_meta.callable_tool.__name__)

    def _find_nested_model_by_name(self, display_name: str):
        """
        根据显示名称查找对应的嵌套模型实例。
        """
        for name, module in self._modules.items():
            if isinstance(module, Model) and module.name == display_name:
                return module
        return None

    def _extract_component_name(self, display_name: str) -> str:
        """
        从完整显示名称中提取组件类名。
        """
        if '.' in display_name:
            component_name = display_name.split('.')[-1]
            if component_name == 'run':
                parts = display_name.split('.')
                if len(parts) >= 2:
                    component_name = parts[-2]
        else:
            component_name = display_name
        return component_name

    def _print_nested_components(self, nested_model, name_width: int, input_width: int, output_width: int) -> int:
        """
        打印嵌套模型的组件信息。

        Returns
        -------
        int
            嵌套组件的数量
        """
        nested_order = nested_model._pipeline._build_execution_order()
        nested_sub_layers = 0

        for nested_step_id in nested_order:
            nested_step_meta = nested_model._pipeline.get_step_info(nested_step_id)
            if not nested_step_meta.is_input_node:
                nested_display_name = nested_step_meta.display_name
                nested_input_contracts = ", ".join([c.name for c in nested_step_meta.input_contracts.values()]) if nested_step_meta.input_contracts else "None"
                nested_output_contracts = ", ".join([c.name for c in nested_step_meta.output_contracts.values()]) if nested_step_meta.output_contracts else "None"

                # 提取组件类名
                component_name = self._extract_component_name(nested_display_name)

                # 添加树形前缀
                nested_layer_display = f"  └─ {component_name}"

                # 截断文本
                layer_display = self._truncate_text(nested_layer_display, name_width)
                input_display = self._truncate_text(nested_input_contracts, input_width)
                output_display = self._truncate_text(nested_output_contracts, output_width)

                print(f"{layer_display:<{name_width}} {input_display:<{input_width}} {output_display:<{output_width}}")
                nested_sub_layers += 1

        return nested_sub_layers
