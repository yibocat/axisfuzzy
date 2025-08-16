# Contributing to AxisFuzzy

First of all, thank you for your interest in contributing to AxisFuzzy!  
To keep the codebase clean and the development process organized, please follow the guidelines below before submitting your code.

---

## 📝 Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Each commit message must adhere to the following format:

`<type>(<scope>): <subject>`

### 1. Commit Type (`type`)

The following is a list of common `types`:

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes (documentation only, no code changes)
- **style**: Code style changes (formatting, spacing, comment adjustments, does not affect logic)
- **refactor**: Refactoring (not adding new features or fixing bugs, but optimizing/improving code/architecture)
- **perf**: Performance improvements
- **test**: Test-related changes (adding new tests, modifying existing tests)
- **chore**: Miscellaneous tasks (build system, CI configuration, dependency changes, scripts that do not affect the final product)

---

### 2. Scope (`scope`)

`scope` specifies the module or subsystem affected by the change. In the AxisFuzzy project, we recommend the following scopes:

#### Core Layer
- `core` → General changes to the core framework
- `fuzznum` → `Fuzznum` abstraction and strategy
- `fuzzarray` → `Fuzzarray` container and backend
- `strategy` → `FuzznumStrategy` abstraction
- `registry` → Core registry
- `operation` → Operation system core (`OperationMixin`, `OperationScheduler`)
- `dispatcher` → Operation dispatcher
- `tnorm` → t-norm and co-norm functions

#### Functional Extension Layer
- `membership` → Membership function system
- `fuzzify` → Fuzzification system
- `extension` → Extension system (registration, dispatch, injection)
- `mixin` → Mixin system (general array methods)
- `random` → Random system

#### Type Implementation Layer
- Various specific `mtype` → Use the abbreviation as the scope, for example:
  - `qrofs` → q-rung orthopair fuzzy set
  - `ivfn` → interval-valued fuzzy number
  - `type2` → type-2 fuzzy number

#### Auxiliary & Infrastructure
- `tests` → Tests
- `docs` → Documentation
- `build` → Build system, packaging configuration
- `infra` → Infrastructure (scripts, CI/CD, toolchain)

---

### 3. Subject

- Use **imperative mood** (recommended in English):  
  - ✅ `fix(dispatcher): correct type promotion`  
  - ❌ `fixed dispatcher bug`  
- Do not exceed 72 characters  
- Clearly and concisely state **what was done** (not how)

---

### 4. Optional Sections

A complete commit message can include three parts:

```
<type>(<scope>): <subject>
[Body - optional: additional explanation, motivation, impact]
[Footer - optional: issue references, breaking changes]
```

Example:

```
feat(membership): add Gaussian membership function

- Implement GaussianMF class
- Register into factory for string-based creation
- Update fuzzifier to support Gaussian as an option

Closes #42
```

---

## 🚦 Tooling

To help developers follow the convention more easily, we recommend the following tools:

- [Commitizen](https://github.com/commitizen/cz-cli): interactively generate standardized commit messages
- [commitlint](https://github.com/conventional-changelog/commitlint): validate commit messages in CI/CD
- [semantic-release](https://semantic-release.gitbook.io/semantic-release/): automatically generate version & changelog based on conventional commits

---

## ✅ Commit Examples

```
feat(core): add FuzznumStrategy base class
fix(dispatcher): resolve broadcasting issue in binary ops
docs: update README with installation instructions
refactor(fuzzarray): improve backend SoA performance
test(extension): add unit tests for similarity function
```

---

> **Note**: If your commit contains a breaking change, please indicate it in the Body or Footer:
>
> ```
> BREAKING CHANGE: Fuzznum interface changed, requires mtype args explicitly
> ```

This ensures that changelogs and version numbers are correctly bumped during automated release.