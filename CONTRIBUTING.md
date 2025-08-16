# Contributing to AxisFuzzy

首先，感谢你对 AxisFuzzy 的贡献兴趣！  
为了保持代码库整洁和开发流程有序，请在提交代码前遵循以下 **Commit Message 规范**。

---

## 📝 Commit Message 规范

我们采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范。每一条提交消息需符合以下格式：

`<type>(<scope>): <subject>`


### 1. 提交类型 (type)

以下是常见的 `type` 列表：

- **feat**: 新功能 (feature)
- **fix**: 修复 bug
- **docs**: 文档修改 (仅限文档，不修改代码)
- **style**: 代码风格修改 (格式化、空格、注释调整，不影响逻辑)
- **refactor**: 重构 (不新增功能、不修 bug，而是优化/改善代码/架构)
- **perf**: 提升性能的修改
- **test**: 测试相关修改 (新增测试、修改已有测试)
- **chore**: 杂务 (构建系统、CI 配置、更改依赖、不影响最终产物的脚本)

---

### 2. 作用范围 (scope)

`scope` 用于指定影响到的模块或子系统。在 AxisFuzzy 项目中推荐使用以下 scope：

#### 核心 (core 层)
- `core` → 核心框架通用改动
- `fuzznum` → `Fuzznum` 抽象与策略
- `fuzzarray` → `Fuzzarray` 容器与后端
- `strategy` → `FuzznumStrategy` 抽象
- `registry` → 核心注册表
- `operation` → 运算系统核心 (`OperationMixin`, `OperationScheduler`)
- `dispatcher` → 运算分发器
- `tnorm` → t-范数与协范数函数

#### 功能扩展层
- `membership` → 隶属函数系统
- `fuzzify` → 模糊化系统
- `extension` → 扩展系统 (注册、分发、注入)
- `mixin` → Mixin 系统 (通用数组方法)
- `random` → 随机系统

#### 类型实现层
- 各种具体 `mtype` → 直接用缩写作为 scope，例如：
  - `qrofs` → q-rung 直觉模糊数
  - `ivfn` → 区间值模糊数
  - `type2` → 二型模糊数

#### 辅助与基础设施
- `tests` → 测试
- `docs` → 文档
- `build` → 构建系统、打包配置
- `infra` → 基础设施 (脚本、CI/CD、工具链)

---

### 3. 标题 (subject)

- 使用 **命令式语气** (建议英文)：  
  - ✅ `fix(dispatcher): correct type promotion`  
  - ❌ `fixed dispatcher bug`  
- 不要超过 72 个字符  
- 简洁清晰地说明 **做了什么**（而不是怎么做）

---

### 4. 可选部分

完整的 commit message 可以包含三部分：

```
<type>(<scope>): <subject>
[Body - 可选: 补充说明、动机、影响范围]
[Footer - 可选: issue 关联、重大声明]
```

示例：

```
feat(membership): add Gaussian membership function

Implement GaussianMF class
Register into factory for string-based creation
Update fuzzifier to support Gaussian as an option
Closes #42
```


---

## 🚦 规范化工具

为了帮助开发者更轻松遵循规范，推荐使用以下工具：

- [Commitizen](https://github.com/commitizen/cz-cli)：交互式生成符合规范的提交消息
- [commitlint](https://github.com/conventional-changelog/commitlint)：在 CI/CD 中校验提交消息
- [semantic-release](https://semantic-release.gitbook.io/semantic-release/)：基于规范提交自动生成版本 & Changelog

---

## ✅ 提交示例

```
feat(core): add FuzznumStrategy base class fix(dispatcher): resolve broadcasting issue in binary ops docs: update README with installation instructions refactor(fuzzarray): improve backend SoA performance test(extension): add unit tests for similarity function
```


---

> **注意**: 如果提交包含破坏性变更 (Breaking Change)，请在 Body 或 Footer 中注明：
>
> ```
> BREAKING CHANGE: Fuzznum interface changed, requires mtype args explicitly
> ```


这样在自动生成 Changelog 和版本号时，可以正确地 bump major version。
