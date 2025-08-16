================
AxisFuzzy 架构
================

本页详细介绍 AxisFuzzy 的核心设计理念和架构分层。

核心理念
========

AxisFuzzy 的核心设计理念是 **模块化、可扩展性与高性能**。
整个系统围绕着 :class:`~axisfuzzy.core.Fuzznum` 和 :class:`~axisfuzzy.core.Fuzzarray` 这两个核心数据结构构建。

.. note::
   ``Fuzznum`` 是一个门面类，实际的计算由其内部的 ``FuzznumStrategy`` 策略类完成。
   这种设计模式分离了接口和实现，是系统可扩展性的关键。

核心组件
--------

下面是几个关键组件的说明：

1. **FuzznumStrategy**:
   这是实现新模糊数类型的关键抽象。详情请参阅 :doc:`api/core` 页面。

2. **扩展系统**:
   允许为不同 ``mtype`` 的模糊数动态注入功能，如距离计算。

   .. code-block:: python

      from axisfuzzy.extension import extension

      @extension('distance', mtype='qrofn')
      def qrofn_distance(a, b):
          # ... 实现 q-rung 直觉模糊数的距离计算
          pass

3. **分发器**:
   负责处理不同类型（``Fuzznum``, ``Fuzzarray``, ``scalar``）之间的二元运算。

.. seealso::
   完整的 API 文档可以在 :doc:`api/index` 找到。