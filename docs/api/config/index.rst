config(axisfuzzy.config)
========================

Introduction
------------
The configuration subsystem of AxisFuzzy is responsible for maintaining global and local configuration items (such as default `mtype`, numerical precision, backend selection, etc.), and provides a unified access interface for various subsystems of the library. The configuration module includes user-facing APIs (convenience functions) and a lower-level configuration manager (used for programmatic and persistent operations).

.. toctree::

    :maxdepth: 2
    :caption: Configuration Modules

    manager
    api
    defaults

手动补充说明与使用示例
----------------------

下面示例以假设性的高层 API 展示典型用法（请根据实际函数/类名调整）：

示例：快速使用高层 API
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 高层便捷 API（示例）
   import axisfuzzy.config.api as cfg

   # 读取配置项（若无则返回默认）
   value = cfg.get("default_mtype", default="qrofn")

   # 设置全局配置项
   cfg.set("precision", 1e-6)

示例：使用配置管理器（面向程序化操作）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from axisfuzzy.config.manager import ConfigManager

   # 创建或加载配置管理器
   manager = ConfigManager()
   manager.set("default_mtype", "qrofn")
   manager.load_from_file("path/to/config.toml")  # 若支持文件加载

   # 读取与遍历配置
   for k in manager.keys():
       print(k, manager.get(k))

注意与建议
--------------

- 推荐在程序入口（如应用的初始化处）统一加载与注入配置，避免运行时零散地修改全局状态。
- 若模块提供 `load_from_file` / `save_to_file` 等接口，请使用受控格式（YAML/TOML/JSON）以便版本管理与可读性。
- 对于需要实验可复现性的配置（如随机种子），建议在 `axisfuzzy.init()` 或主程序初始化时设置并记录到日志/配置文件中。

扩展点（供开发者参考）
----------------------

- 配置项校验：如果需要严格类型/范围校验，建议在 manager 层添加验证钩子（validate callback）。
- 动态回调：支持对关键配置项注册回调（当配置改变时触发），方便在运行时同步调整子系统状态。
- 与环境/CLI 集成：支持从环境变量、命令行参数覆盖配置项，便于 CI/部署时动态控制行为。

如何在文档中更细粒度控制展示
------------------------------

- 若你希望对某个类或函数进行更细粒度展示（例如只显示部分方法、手写属性说明），请用 ``.. autoclass::`` / ``.. autofunction::`` 指令替代上面的 ``automodule``，并在该页面中手写示例与参数说明。
- 可在 conf.py 中启用 ``sphinx.ext.autosummary``（并设置 ``autosummary_generate = True``）来自动生成可编辑的 stub 页面，然后在生成后的 stub 上补充内容。

下一步
------

1. 将本文件保存为 docs/api/config.rst。
2. 在 docs/ 目录运行：

.. code-block:: bash

   make clean
   make html

或使用热更新：

.. code-block:: bash

   sphinx-autobuild . _build/html

3. 打开生成的 API 页面，检查自动抓取的类/函数列表，针对性用 ``autoclass`` / ``autofunction`` 替换并在页面中补充示例与参数说明直到满意。