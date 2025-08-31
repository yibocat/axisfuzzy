#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.12.7
#  Date: 2025/8/26 14:12
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: AxisFuzzy

import importlib
import inspect
import json
from abc import ABC, abstractmethod

from collections import OrderedDict
from pathlib import Path
from typing import Any, get_type_hints, Dict, Union

# 导入组件模块, 契约系统, 管道 DAG 引擎
from ..component import AnalysisComponent
from ..contracts import Contract
from ..pipeline import FuzzyPipeline, FuzzyPipelineIterator


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

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Returns the serializable configuration of the model.

        When implementing a custom `Model`, you must override this method.
        The returned dictionary should contain the arguments needed to
        re-create the model instance via its `__init__` method.

        For example, if your model's `__init__` is `def __init__(self, num_layers, activation):`,
        then `get_config` should return `{'num_layers': self.num_layers, 'activation': self.activation}`.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the model's configuration.
        """
        raise NotImplementedError("Custom Models must implement the 'get_config' method.")

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

    def run(self, *args, return_intermediate: bool = False, **kwargs) -> Any:
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

        # 处理输入数据
        if len(input_names) == 1 and len(args) == 1 and not kwargs:
            initial_data = args[0]
        else:
            initial_data = {**dict(zip(input_names, args)), **kwargs}

        return self.pipeline.run(initial_data, return_intermediate=return_intermediate)
    
    def step_by_step(self, *args, **kwargs) -> 'FuzzyPipelineIterator':
        """
        【第四步（可选）】: 创建一个迭代器用于单步执行。

        此方法返回一个迭代器，允许您逐步执行模型的计算图，
        从而可以观察每个中间步骤的结果。这对于调试或创建交互式分析非常有用。

        在使用此方法前，必须先调用 `.build()`。

        Parameters
        ----------
        *args
            传递给模型 `forward` 方法的位置参数。
        **kwargs
            传递给模型 `forward` 方法的关键字参数。

        Returns
        -------
        FuzzyPipelineIterator
            一个迭代器，每次迭代都会执行图中的一个步骤。

        Raises
        ------
        RuntimeError
            如果模型尚未构建。
        """
        if not self.built:
            raise RuntimeError("Model has not been built yet. Call .build() before using step_by_step.")

        # 此输入处理逻辑与 `run` 方法一致
        sig = inspect.signature(self.forward)
        input_names = list(sig.parameters.keys())
        if 'self' in input_names:
            input_names.remove('self')

        if len(input_names) == 1 and len(args) == 1 and not kwargs:
            initial_data = args[0]
        else:
            initial_data = {**dict(zip(input_names, args)), **kwargs}

        return self.pipeline.step_by_step(initial_data)

    def __call__(self, *args, **kwargs) -> Any:
        """调用模型实例等同于执行 run 方法。"""
        return self.run(*args, **kwargs)

    def visualize(self, **kwargs):
        """可视化计算图。必须在 build() 之后调用。"""
        if not self.built:
            raise RuntimeError("Model has not been built yet. Call .build() before visualizing.")
        return self.pipeline.visualize(**kwargs)

    def summary(self):
        """
        打印模型的摘要，基于构建后的 FuzzyPipeline 结构。
        """
        if not self.built:
            raise RuntimeError("Model has not been built yet. Call .build() first.")

        table_content = []

        # 获取执行顺序（包括输入节点）
        full_order = self._pipeline._build_execution_order()

        total_layers = 0
        nested_models_count = 0
        sub_layers_count = 0

        for step_id in full_order:
            step_meta = self._pipeline.get_step_info(step_id)

            if step_meta.is_input_node:
                # 显示输入节点
                input_name = next((name for name, sid in self._pipeline.input_nodes.items() if sid == step_id), None)

                output_contract = list(step_meta.output_contracts.values())[0]
                layer_display = f"Input: {input_name}"
                input_display = "-"
                output_display = output_contract.name
                table_content.append((layer_display, input_display, output_display))

            else:
                # 显示计算节点
                display_name = step_meta.display_name
                input_contracts_display = ", ".join(
                    [c.name for c in step_meta.input_contracts.values()]) if step_meta.input_contracts else "None"
                output_contracts_display = ", ".join(
                    [c.name for c in step_meta.output_contracts.values()]) if step_meta.output_contracts else "None"

                # 检查是否是嵌套管道
                if self._is_nested_pipeline_step(step_meta):
                    # 这是一个嵌套管道步骤
                    nested_models_count += 1
                    table_content.append((display_name, input_contracts_display, output_contracts_display))

                    # 显示嵌套管道的内部结构
                    nested_model = self._find_nested_model_by_name(display_name)
                    if nested_model:
                        nested_rows, nested_sub_layers = self._get_nested_components_data(nested_model)
                        table_content.extend(nested_rows)
                        sub_layers_count += nested_sub_layers
                else:
                    # 普通组件
                    component_name = self._extract_component_name(display_name)
                    table_content.append((component_name, input_contracts_display, output_contracts_display))

                total_layers += 1

        # 计算列宽
        if not table_content:
            name_width = len('Layer (type)')
            input_width = len('Input Contracts')
            output_width = len('Output Contracts')
        else:
            name_width = max(len(row[0]) for row in table_content)
            input_width = max(len(row[1]) for row in table_content)
            output_width = max(len(row[2]) for row in table_content)

        # 确保表头也能被容纳
        name_width = max(name_width, len('Layer (type)'))
        input_width = max(input_width, len('Input Contracts'))
        output_width = max(output_width, len('Output Contracts'))

        line_length = name_width + input_width + output_width + 20  # 2 spaces between columns

        # 生成表格头部
        title = f'Model: "{self.name}"'
        print(title)
        print("=" * line_length)

        # 表头
        header = f"{'Layer (type)':<{name_width + 10}}{'Input Contracts':<{input_width + 10}}{'Output Contracts':<{output_width}}"
        print(header)
        print("-" * line_length)

        # 打印内容
        for row in table_content:
            layer_display, input_display, output_display = row
            print(f"{layer_display:<{name_width + 10}}{input_display:<{input_width + 10}}{output_display:<{output_width}}")

        # 表格尾部
        print("-" * line_length)
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

    def _get_nested_components_data(self, nested_model) -> tuple[list[tuple[str, str, str]], int]:
        """
        收集嵌套模型的组件信息。

        Returns
        -------
        tuple[list[tuple[str, str, str]], int]
            一个元组，包含一个行数据列表和嵌套组件的数量。
            每个行数据是一个 (layer_display, input_contracts, output_contracts) 的元组。
        """
        nested_rows = []
        nested_order = nested_model._pipeline._build_execution_order()
        nested_sub_layers = 0

        for nested_step_id in nested_order:
            nested_step_meta = nested_model._pipeline.get_step_info(nested_step_id)
            if not nested_step_meta.is_input_node:
                nested_display_name = nested_step_meta.display_name
                nested_input_contracts = ", ".join(
                    [c.name for c in nested_step_meta.input_contracts.values()]) if nested_step_meta.input_contracts else "None"
                nested_output_contracts = ", ".join(
                    [c.name for c in nested_step_meta.output_contracts.values()]) if nested_step_meta.output_contracts else "None"

                # 提取组件类名
                component_name = self._extract_component_name(nested_display_name)

                # 添加树形前缀
                nested_layer_display = f"  └─ {component_name}"

                nested_rows.append((nested_layer_display, nested_input_contracts, nested_output_contracts))
                nested_sub_layers += 1

        return nested_rows, nested_sub_layers

    def save(self, filepath: Union[str, Path]):
        """
        将模型架构和配置保存到 JSON 文件。

        此方法仅序列化重建模型所需的最小信息：
        - 模型的类路径，用于动态加载。
        - 模型的初始化参数 (`get_config()`)。
        - 所有子模块的配置（递归）。

        Parameters
        ----------
        filepath : Union[str, Path]
            保存模型配置的 JSON 文件路径。
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = self._serialize()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)

        print(f"Model '{self.name}' configuration saved to: {filepath}")

    def _serialize(self) -> Dict[str, Any]:
        """
        递归地将模型及其子模块序列化为字典。

        Returns
        -------
        Dict[str, Any]
            包含模型重建所需信息的字典。
        """
        # 序列化当前模型
        data = {
            'class_path': f"{self.__class__.__module__}.{self.__class__.__name__}",
            'config': self.get_config(),
            'modules': {}
        }

        # 递归序列化所有子模块
        for name, module in self._modules.items():
            if module is None:
                continue  # None 值不需要保存，加载时会自动处理

            if isinstance(module, Model):
                # 嵌套模型，递归调用
                data['modules'][name] = module._serialize()
            elif isinstance(module, AnalysisComponent):
                # 普通组件
                data['modules'][name] = {
                    'class_path': f"{module.__class__.__module__}.{module.__class__.__name__}",
                    'config': module.get_config()
                }
            else:
                # 如果有非 AnalysisComponent 的模块，可以选择忽略或报错
                print(f"Warning: Skipping non-serializable module '{name}' of type {type(module).__name__}.")

        return data

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Model':
        """
        从 JSON 配置文件加载模型。

        加载后，模型处于未构建状态，需要手动调用 `.build()` 方法。

        Parameters
        ----------
        filepath : Union[str, Path]
            模型配置文件的路径。

        Returns
        -------
        Model
            加载并实例化的模型。
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)

        model_instance = cls._deserialize(model_data)
        print(f"Model loaded from: {filepath}. Please call .build() before use.")
        return model_instance

    @staticmethod
    def _deserialize(data: Dict[str, Any]) -> 'Model':
        """
        根据序列化字典递归地重建模型实例。

        Parameters
        ----------
        data : Dict[str, Any]
            包含模型信息的字典。

        Returns
        -------
        Model
            重建的模型实例。
        """
        # 动态导入主模型类
        class_path = data['class_path']
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import model class '{class_path}': {e}")

        # 实例化主模型
        instance = model_class(**data['config'])

        # 递归反序列化并设置子模块
        for name, module_data in data.get('modules', {}).items():
            if 'modules' in module_data:  # 这是一个嵌套模型
                sub_module = Model._deserialize(module_data)
            else:  # 这是一个普通组件
                sub_class_path = module_data['class_path']
                try:
                    sub_module_path, sub_class_name = sub_class_path.rsplit('.', 1)
                    sub_module_obj = importlib.import_module(sub_module_path)
                    sub_class = getattr(sub_module_obj, sub_class_name)
                    sub_module = sub_class(**module_data['config'])
                except (ImportError, AttributeError) as e:
                    raise ImportError(f"Failed to import submodule class '{sub_class_path}': {e}")

            # 使用 setattr 会触发 __setattr__，自动将其添加到 _modules 中
            setattr(instance, name, sub_module)

        return instance
