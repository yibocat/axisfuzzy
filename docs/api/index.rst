=========
API 参考
=========

完整的 AxisFuzzy API 文档和使用指南。

.. toctree::
   :maxdepth: 2
   
   core
   fuzzify
   extension
   mixin
   random

核心组件概览
============

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - 模块
     - 描述
   * - :doc:`core`
     - 核心数据结构 (Fuzznum, Fuzzarray, FuzznumStrategy)
   * - :doc:`fuzzify`
     - 模糊化系统，将精确值转换为模糊数
   * - :doc:`extension`
     - 扩展系统，为不同类型添加特化功能
   * - :doc:`mixin`
     - Mixin 系统，提供类似 NumPy 的数组操作
   * - :doc:`random`
     - 随机模糊数生成系统