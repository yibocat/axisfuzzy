---
applyTo: '**'
---
## Project Framework & Version
- **Framework**: Python 3.12+ Scientific Computing Framework
- **Project Version**: from pyproject.toml
- **Core Dependencies**: 
  - numpy>=2.2.6 (Numerical Computing)
  - numba>=0.61.2 (JIT Compilation Optimization)
- **Optional Dependencies**:
  - analysis: pandas, matplotlib, networkx (Data Analysis and Visualization)
  - dev: pytest, notebook (Development and Testing)
  - docs: sphinx (Documentation Building)

## Git Flow
The project adopts the Git Flow branching strategy:

- **Main branch**: master (stable version for production environment)
- **Development branch**: develop (integration of feature development)
- **Feature branches**: feature (new feature development)
- **Release branch**: release (preparation for version release)
- **Hotfix branch**: hotfix (urgent bug fixing)

### Key Command Reference
- `git flow feature start <name>` - Start a new feature
- `git flow release start <version>` - Start a release
- `git flow hotfix start <version>` - Start a hotfix
- Detailed command reference: `./doc/private/COMMANDS.md`
- Complete workflow guide: `./doc/private/workflow_guide.md`

## Testing Framework
- **Testing Framework**: pytest
- **Testing Structure**: Modular test suite
- **Testing Documentation**: `./tests/README.md`
- **Testing Types**:
  - Dependency tests (test_dependencies/)
  - Core functionality tests (test_core/)
  - Documentation tests (test_docs/)
  - Fuzzification tests (test_fuzzifier/)
  - Membership function tests (test_membership/)