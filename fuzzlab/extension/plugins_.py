#  Copyright (c) yibocat 2025 All Rights Reserved
#  Python: 3.10.9
#  Date: 2025/8/7 20:52
#  Author: yibow
#  Email: yibocat@yeah.net
#  Software: FuzzLab
"""
高级特性:插件管理器

插件系统 (PluginManager)

核心思想: 允许 FuzzLab 的核心库保持精简，而将特定领域或不常用的功能作为独立的“插件”提供。
用户可以根据需要安装和加载这些插件，从而扩展 FuzzLab 的能力。

实现方式:
--------
PluginManager 类:

- 负责发现、加载和管理插件。
- discover_plugins: 扫描特定的包前缀（例如 fuzzlab_plugins_），查找符合命名约定的插件模块。
- load_plugin: 动态导入插件模块。当插件模块被导入时，其中定义的 @extension 装饰器会自动执行，将插件的功能注册到主 ExtensionRegistry 中。
- load_all_plugins: 批量加载所有发现的插件。

优点:
------
    - 模块化和可插拔: 将功能分解为独立的、可选择的组件。
    - 减小核心库体积: 核心库只包含最基本的功能，高级或特定功能通过插件提供。
    - 社区贡献: 鼓励第三方开发者创建和分享自己的 FuzzLab 扩展，形成生态系统。
    - 按需加载: 用户只加载他们需要的功能，减少内存占用和启动时间。

示例:
------
用户可以创建一个名为 fuzzlab_plugins_mycustom 的 Python 包，
并在其中定义自己的扩展。当 PluginManager 扫描到这个包并加载它时，
其中的 @extension 装饰器就会生效。
"""
import importlib
import pkgutil
from typing import Dict, Any, List

from .registry import get_extension_registry


class ExtensionPluginManager:

    def __init__(self):
        self.loaded_plugins: Dict[str, Any] = {}
        self.registry = get_extension_registry()

    @staticmethod
    def discover_plugins(package_name: str = 'fuzzlab_plugins') -> List[str]:
        """发现插件"""
        plugins = []
        try:
            # 查找所有以fuzzlab_plugins开头的包
            for finder, name, ispkg in pkgutil.iter_modules():
                if name.startswith(package_name):
                    plugins.append(name)
        except ImportError:
            pass
        return plugins

    def load_plugin(self, plugin_name: str) -> bool:
        """加载插件"""
        try:
            plugin_module = importlib.import_module(plugin_name)

            # 检查插件是否有注册函数
            if hasattr(plugin_module, 'register_extensions'):
                plugin_module.register_extensions()
                self.loaded_plugins[plugin_name] = plugin_module
                return True
            else:
                # 自动加载插件中的所有扩展
                # 插件模块的导入会触发装饰器执行
                self.loaded_plugins[plugin_name] = plugin_module
                return True

        except ImportError as e:
            print(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def load_all_plugins(self):
        """加载所有发现的插件"""
        plugins = self.discover_plugins()
        for plugin in plugins:
            self.load_plugin(plugin)

    def unload_plugin(self, plugin_name: str):
        """卸载插件（注意：Python中模块卸载有限制）"""
        if plugin_name in self.loaded_plugins:
            # 这里可以添加清理逻辑
            del self.loaded_plugins[plugin_name]


# 全局插件管理器
_plugin_manager = ExtensionPluginManager()


def get_extension_plugin_manager() -> ExtensionPluginManager:
    """获取插件管理器"""
    return _plugin_manager
