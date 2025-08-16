====================================
AxisFuzzy - 模糊计算库
====================================

**AxisFuzzy** 是一个专业的 Python 模糊逻辑计算框架，为科研人员和工程师提供高性能、模块化、可扩展的模糊数学运算能力。

.. note::
   AxisFuzzy 的设计理念源于 NumPy，致力于让模糊数运算像操作普通数组一样简单直观。

核心特性
========

🎯 **统一接口**
   通过 ``Fuzznum`` 和 ``Fuzzarray`` 提供类似 NumPy 的模糊数和模糊数组操作体验

⚡ **高性能**
   后端采用 Struct of Arrays (SoA) 架构，实现批量高效计算

🔌 **完全可扩展**
   插件化架构支持自定义模糊数类型、运算规则和扩展功能

🔢 **丰富类型**
   内置支持 q-rung 直觉模糊数、区间二型模糊数等多种模糊数类型

快速开始
========

.. code-block:: bash

   pip install axisfuzzy

.. code-block:: python

   import axisfuzzy as af

   # 创建模糊数
   fuzz1 = af.Fuzznum([0.8, 0.2], mtype='qrofn')
   fuzz2 = af.Fuzznum([0.7, 0.3], mtype='qrofn')

   # 模糊运算
   result = fuzz1 + fuzz2
   print(f"运算结果: {result}")

文档导览
========

.. toctree::
   :maxdepth: 1
   :caption: 新手指南

   getting_started/index

.. toctree::
   :maxdepth: 2
   :caption: 用户手册

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: 开发指南

   development/index

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api/index

索引与搜索
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`