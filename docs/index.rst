====================================
axisfuzzy
====================================

AxisFuzzy is a professional Python fuzzy logic computing framework that provides researchers and engineers with high-performance, modular, and scalable fuzzy mathematical operation capabilities.

.. note::
   The design concept of AxisFuzzy originates from NumPy, aiming to make fuzzy number operations as simple and intuitive as manipulating regular arrays.

Core Features
========

🎯 **Unified interface**
   Provide NumPy-like fuzzy number and fuzzy array operations through ``Fuzznum`` and ``Fuzzarray``.

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
   :maxdepth: 1
   :caption: 用户手册

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   development/index

.. toctree::
   :maxdepth: 1
   :caption: API 参考

   api/index

索引与搜索
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`