import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保 `axisfuzzy` 可被导入
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


