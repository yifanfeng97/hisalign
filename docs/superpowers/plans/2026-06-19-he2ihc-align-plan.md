# he2ihc_align 实施计划

> **给执行代理：** 必需子技能：superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans。每一步用复选框（`- [ ]`）跟踪进度。

**目标：** 构建一个轻量的 HE 到多 IHC 病理切片配准工具，输出每个 HE patch 对应到每张 IHC 切片位置的 CSV 映射表。HE 通过 kfbslide/openslide 读取，配准核心从 VALIS 裁剪而来。

**架构：** 一个小型 Python 包，包含统一的 `slide_io` 读取层、从 VALIS 裁剪精简的 `registration` 模块、`patching`/`mapping` 坐标映射层、可选的 `viz` 可视化，以及一个 CLI/脚本入口。所有坐标均使用 level-0 像素坐标。

**技术栈：** Python 3.11、uv、kfbslide、openslide-python、numpy、pandas、pillow、matplotlib、pyvips、opencv-python-headless、scikit-image、scipy、scikit-learn、torch/torchvision、pyyaml。

---

## 文件结构

```text
he2ihc_align/
├── pyproject.toml
├── .gitignore
├── README.md
├── configs/
│   └── default.yaml
├── src/he2ihc_align/
│   ├── __init__.py
│   ├── slide_io/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── kfb_backend.py
│   │   └── openslide_backend.py
│   ├── registration/
│   │   ├── __init__.py
│   │   ├── feature_detectors.py
│   │   ├── feature_matcher.py
│   │   ├── rigid.py
│   │   ├── non_rigid.py
│   │   ├── warp.py
│   │   └── registrar.py
│   ├── patching.py
│   ├── mapping.py
│   ├── viz.py
│   └── cli.py
├── scripts/
│   └── run_pipeline.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_slide_io.py
│   ├── test_patching.py
│   └── test_mapping.py
└── outputs/                   # gitignored
```

---

## 测试数据

使用 `/home/fengyifan/disk/code/valis/test_SCCE`（软链接到 `escc-h2ihc/data/test_SCCE`）。

每个病例的结构：

```text
test_SCCE/
├── 174162-1/
│   ├── 174162-1-第一批/
│   │   ├── 174162-1.kfb          # HE
│   │   ├── 174162-1 CD3.svs      # IHC
│   │   ├── 174162-1 CD68.svs
│   │   └── ...
│   └── 174162-1-第二批/
│       └── 174162-1.kfb          # HE 重复扫描
├── 98140-6/
│   └── ...
```

MVP 阶段每个病例只使用 `第一批` 目录，忽略 `第二批`。

---

## Task 1：使用 uv 初始化项目

**涉及文件：**
- 创建：`pyproject.toml`
- 创建：`.gitignore`
- 创建：`src/he2ihc_align/__init__.py`

- [ ] **步骤 1：创建项目骨架**

执行：

```bash
cd /home/fengyifan/disk/code/he2ihc_align
uv init --python 3.11 --name he2ihc_align .
```

这会生成 `pyproject.toml`、`README.md`、`.python-version` 和 `hello.py`。删除 `hello.py`。

- [ ] **步骤 2：编辑 pyproject.toml**

替换为以下内容：

```toml
[project]
name = "he2ihc-align"
version = "0.1.0"
description = "Lightweight HE to multi-IHC registration and patch mapping"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "pandas",
    "pillow",
    "matplotlib",
    "pyvips",
    "opencv-python-headless",
    "scikit-image",
    "scipy",
    "scikit-learn",
    "torch",
    "torchvision",
    "tqdm",
    "pyyaml",
    "openslide-python",
    "kfbslide",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff"]

[project.scripts]
he2ihc-align = "he2ihc_align.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/he2ihc_align"]

[tool.ruff]
line-length = 100
```

- [ ] **步骤 3：创建包目录**

执行：

```bash
mkdir -p src/he2ihc_align/slide_io
mkdir -p src/he2ihc_align/registration
mkdir -p scripts
mkdir -p tests
mkdir -p configs
mkdir -p outputs
```

- [ ] **步骤 4：创建根 `__init__.py`**

创建 `src/he2ihc_align/__init__.py`：

```python
"""he2ihc_align: lightweight HE to multi-IHC patch mapping."""

__version__ = "0.1.0"
```

- [ ] **步骤 5：创建 .gitignore**

创建 `.gitignore`：

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
dist/
build/

# Outputs
outputs/
*.csv
*.tiff
*.tif

# IDE
.vscode/
.idea/
*.swp

# Misc
.DS_Store
```

- [ ] **步骤 6：使用 uv 安装依赖**

执行：

```bash
uv sync
uv pip install -e .
```

预期结果：创建 `.venv`，并且 `python -c "import he2ihc_align; print(he2ihc_align.__version__)"` 输出 `0.1.0`。

- [ ] **步骤 7：提交**

```bash
git add .
git commit -m "chore: initialize project with uv"
```

---

## Task 2：定义 Slide 抽象接口

**涉及文件：**
- 创建：`src/he2ihc_align/slide_io/base.py`
- 创建：`src/he2ihc_align/slide_io/__init__.py`
- 测试：`tests/test_slide_io.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_slide_io.py`：

```python
from he2ihc_align.slide_io.base import Slide


def test_slide_protocol_has_required_methods():
    # Protocol 不可实例化，但可以检查类上是否存在属性
    assert hasattr(Slide, "level_count")
    assert hasattr(Slide, "level_dimensions")
    assert hasattr(Slide, "level_downsamples")
    assert hasattr(Slide, "read_region")
```

执行：

```bash
uv run pytest tests/test_slide_io.py -v
```

预期结果：`ModuleNotFoundError` 或 `ImportError`，因为 `he2ihc_align.slide_io.base` 还不存在。

- [ ] **步骤 2：实现 Slide 协议**

创建 `src/he2ihc_align/slide_io/base.py`：

```python
"""Abstract slide interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Slide(Protocol):
    """Minimal slide reader interface."""

    @property
    def level_count(self) -> int: ...

    @property
    def level_dimensions(self) -> list[tuple[int, int]]: ...

    @property
    def level_downsamples(self) -> list[float]: ...

    @property
    def properties(self) -> dict[str, str]: ...

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Read RGB region as HWC uint8 numpy array."""
        ...

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Return pyramid level closest to requested downsample."""
        ...
```

创建 `src/he2ihc_align/slide_io/__init__.py`：

```python
from he2ihc_align.slide_io.base import Slide

__all__ = ["Slide"]
```

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_slide_io.py -v
```

预期结果：通过。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/slide_io tests/test_slide_io.py
git commit -m "feat(slide_io): add Slide protocol"
```

---

## Task 3：实现 OpenSlide 后端

**涉及文件：**
- 创建：`src/he2ihc_align/slide_io/openslide_backend.py`
- 修改：`src/he2ihc_align/slide_io/__init__.py`
- 测试：`tests/test_slide_io.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_slide_io.py` 末尾追加：

```python
from pathlib import Path

import pytest

from he2ihc_align.slide_io.openslide_backend import OpenSlide

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


def test_openslide_reads_ihc():
    ihc_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.svs"))
    slide = OpenSlide(ihc_path)
    assert slide.level_count > 0
    assert len(slide.level_dimensions) == slide.level_count
    assert len(slide.level_downsamples) == slide.level_count

    img = slide.read_region((0, 0), level=0, size=(512, 512))
    assert img.shape == (512, 512, 3)
    assert img.dtype == "uint8"
```

执行：

```bash
uv run pytest tests/test_slide_io.py::test_openslide_reads_ihc -v
```

预期结果：`ImportError` 或 `NameError`。

- [ ] **步骤 2：实现 OpenSlide 后端**

创建 `src/he2ihc_align/slide_io/openslide_backend.py`：

```python
"""OpenSlide-backed slide reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openslide


class OpenSlide:
    """Thin wrapper around openslide.OpenSlide returning RGB numpy arrays."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._osr = openslide.OpenSlide(str(self.path))

    @property
    def level_count(self) -> int:
        return self._osr.level_count

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._osr.level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        return list(self._osr.level_downsamples)

    @property
    def properties(self) -> dict[str, str]:
        return dict(self._osr.properties)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        pil_img = self._osr.read_region(location, level, size)
        img = np.array(pil_img.convert("RGB"))
        return img

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self._osr.get_best_level_for_downsample(downsample)

    def __repr__(self) -> str:
        return f"OpenSlide({self.path.name})"
```

更新 `src/he2ihc_align/slide_io/__init__.py`：

```python
from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.openslide_backend import OpenSlide

__all__ = ["Slide", "OpenSlide"]
```

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_slide_io.py::test_openslide_reads_ihc -v
```

预期结果：通过（系统需已安装 libopenslide）。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/slide_io tests/test_slide_io.py
git commit -m "feat(slide_io): add OpenSlide backend"
```

---

## Task 4：实现 kfbslide 后端

**涉及文件：**
- 创建：`src/he2ihc_align/slide_io/kfb_backend.py`
- 修改：`src/he2ihc_align/slide_io/__init__.py`
- 测试：`tests/test_slide_io.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_slide_io.py` 末尾追加：

```python
from he2ihc_align.slide_io.kfb_backend import KfbSlide


def test_kfbslide_reads_he():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    slide = KfbSlide(he_path)
    assert slide.level_count > 0
    assert len(slide.level_dimensions) == slide.level_count
    assert len(slide.level_downsamples) == slide.level_count

    img = slide.read_region((0, 0), level=0, size=(512, 512))
    assert img.shape == (512, 512, 3)
    assert img.dtype == "uint8"
```

执行：

```bash
uv run pytest tests/test_slide_io.py::test_kfbslide_reads_he -v
```

预期结果：`ImportError`。

- [ ] **步骤 2：实现 KfbSlide 后端**

创建 `src/he2ihc_align/slide_io/kfb_backend.py`：

```python
"""kfbslide-backed slide reader for KFBio .kfb files."""

from __future__ import annotations

from pathlib import Path

import kfbslide
import numpy as np


class KfbSlide:
    """Thin wrapper around kfbslide.KfbSlide returning RGB numpy arrays."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._kfb = kfbslide.KfbSlide(str(self.path))

    @property
    def level_count(self) -> int:
        return self._kfb.level_count

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._kfb.level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        return list(self._kfb.level_downsamples)

    @property
    def properties(self) -> dict[str, str]:
        return dict(self._kfb.properties)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        pil_img = self._kfb.read_region(location, level, size)
        img = np.array(pil_img.convert("RGB"))
        return img

    def get_best_level_for_downsample(self, downsample: float) -> int:
        downsamples = self.level_downsamples
        if not downsamples:
            return 0
        best = min(range(len(downsamples)), key=lambda i: abs(downsamples[i] - downsample))
        return best

    def __repr__(self) -> str:
        return f"KfbSlide({self.path.name})"
```

更新 `src/he2ihc_align/slide_io/__init__.py`：

```python
from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.kfb_backend import KfbSlide
from he2ihc_align.slide_io.openslide_backend import OpenSlide

__all__ = ["Slide", "OpenSlide", "KfbSlide"]
```

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_slide_io.py::test_kfbslide_reads_he -v
```

预期结果：通过。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/slide_io tests/test_slide_io.py
git commit -m "feat(slide_io): add kfbslide backend"
```

---

## Task 5：实现 Slide 工厂与病例发现

**涉及文件：**
- 创建：`src/he2ihc_align/slide_io/factory.py`
- 创建：`src/he2ihc_align/case_io.py`
- 修改：`src/he2ihc_align/slide_io/__init__.py`
- 测试：`tests/test_slide_io.py`

- [ ] **步骤 1：编写失败测试**

在 `tests/test_slide_io.py` 末尾追加：

```python
from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.case_io import discover_case


def test_open_slide_selects_backend_by_extension():
    svs_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.svs"))
    kfb_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))

    svs_slide = open_slide(svs_path)
    kfb_slide = open_slide(kfb_path)

    assert type(svs_slide).__name__ == "OpenSlide"
    assert type(kfb_slide).__name__ == "KfbSlide"


def test_discover_case_finds_he_and_markers():
    case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
    he_path, markers = discover_case(case_dir)

    assert he_path.suffix == ".kfb"
    assert len(markers) >= 3
    assert all(p.suffix == ".svs" for p in markers.values())
```

执行：

```bash
uv run pytest tests/test_slide_io.py::test_open_slide_selects_backend_by_extension tests/test_slide_io.py::test_discover_case_finds_he_and_markers -v
```

预期结果：`ImportError`。

- [ ] **步骤 2：实现工厂与病例发现**

创建 `src/he2ihc_align/slide_io/factory.py`：

```python
"""Factory for opening slides by format."""

from __future__ import annotations

from pathlib import Path

from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.kfb_backend import KfbSlide
from he2ihc_align.slide_io.openslide_backend import OpenSlide


SUPPORTED_EXTS = {
    ".kfb": KfbSlide,
    ".svs": OpenSlide,
    ".tif": OpenSlide,
    ".tiff": OpenSlide,
    ".mrxs": OpenSlide,
    ".ndpi": OpenSlide,
    ".scn": OpenSlide,
    ".vms": OpenSlide,
    ".vmu": OpenSlide,
}


def open_slide(path: str | Path) -> Slide:
    path = Path(path)
    ext = path.suffix.lower()
    backend = SUPPORTED_EXTS.get(ext)
    if backend is None:
        raise ValueError(f"Unsupported slide format: {ext} for {path}")
    return backend(path)
```

创建 `src/he2ihc_align/case_io.py`：

```python
"""Discover HE and IHC slides within a case directory."""

from __future__ import annotations

from pathlib import Path


def discover_case(
    case_dir: str | Path,
    he_ext: str = ".kfb",
    ihc_exts: tuple[str, ...] = (".svs", ".tif", ".tiff", ".mrxs", ".ndpi", ".scn", ".vms", ".vmu"),
) -> tuple[Path, dict[str, Path]]:
    """在病例批次目录中查找 HE 切片和所有 IHC marker 切片。

    Returns
    -------
    he_path : Path
        HE 切片路径（与病例 ID 匹配的 .kfb）。
    markers : dict[str, Path]
        marker 名称到 IHC 切片路径的映射。
    """
    case_dir = Path(case_dir)

    case_id = case_dir.parent.name.strip()
    kfb_candidates = sorted(case_dir.glob(f"*{he_ext}"))
    he_path = next(
        (p for p in kfb_candidates if p.stem.strip() == case_id),
        kfb_candidates[0] if kfb_candidates else None,
    )
    if he_path is None:
        raise FileNotFoundError(f"No HE slide (*{he_ext}) found in {case_dir}")

    he_stem_lower = he_path.stem.lower()
    marker_paths: dict[str, Path] = {}
    for ext in ihc_exts:
        for p in sorted(case_dir.glob(f"*{ext}")):
            if p.stem.lower() == he_stem_lower:
                continue
            # marker 名称：文件名 stem 的最后一个空格分隔 token，例如 "CD3"
            marker = p.stem.strip().split()[-1]
            marker_paths[marker] = p

    if not marker_paths:
        raise FileNotFoundError(f"No IHC slides found in {case_dir}")

    return he_path, marker_paths
```

更新 `src/he2ihc_align/slide_io/__init__.py`：

```python
from he2ihc_align.slide_io.base import Slide
from he2ihc_align.slide_io.factory import open_slide
from he2ihc_align.slide_io.kfb_backend import KfbSlide
from he2ihc_align.slide_io.openslide_backend import OpenSlide

__all__ = ["Slide", "open_slide", "OpenSlide", "KfbSlide"]
```

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_slide_io.py -v
```

预期结果：全部通过。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/slide_io src/he2ihc_align/case_io.py tests/test_slide_io.py
git commit -m "feat(slide_io): add slide factory and case discovery"
```

---

## Task 6：迁移 VALIS 配准核心（特征检测）

**涉及文件：**
- 创建：`src/he2ihc_align/registration/feature_detectors.py`
- 创建：`src/he2ihc_align/registration/__init__.py`

- [ ] **步骤 1：从 valis 复制 feature_detectors.py**

源文件：`/home/fengyifan/disk/code/valis/valis/feature_detectors.py`
目标：`src/he2ihc_align/registration/feature_detectors.py`

执行：

```bash
cp /home/fengyifan/disk/code/valis/valis/feature_detectors.py src/he2ihc_align/registration/feature_detectors.py
```

- [ ] **步骤 2：裁剪导入并移除不用的检测器**

编辑 `src/he2ihc_align/registration/feature_detectors.py`：
- 只保留需要支持的检测器（例如 `DISK` 和/或 `LightGlue`）。
- 移除不用的检测器（例如 `SIFT`、`ORB` 等）。
- 确保导入在包内可解析。

MVP 阶段保留 `DISK` 和 `LightGlue`，因为当前 valis 脚本使用它们。

- [ ] **步骤 3：创建 `__init__.py`**

创建 `src/he2ihc_align/registration/__init__.py`：

```python
"""Trimmed registration core adapted from VALIS."""
```

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/registration
git commit -m "feat(registration): import feature detectors from VALIS"
```

---

## Task 7：迁移 VALIS 特征匹配器

**涉及文件：**
- 创建：`src/he2ihc_align/registration/feature_matcher.py`

- [ ] **步骤 1：复制文件**

```bash
cp /home/fengyifan/disk/code/valis/valis/feature_matcher.py src/he2ihc_align/registration/feature_matcher.py
```

- [ ] **步骤 2：裁剪导入**

编辑文件，移除对已删除模块的引用（例如 `serial_rigid`、`viz` 等）。
只保留与所选特征检测器配合使用的匹配器类。

- [ ] **步骤 3：提交**

```bash
git add src/he2ihc_align/registration/feature_matcher.py
git commit -m "feat(registration): import feature matcher from VALIS"
```

---

## Task 8：迁移 VALIS 刚性配准

**涉及文件：**
- 创建：`src/he2ihc_align/registration/rigid.py`

- [ ] **步骤 1：从 serial_rigid.py 复制**

```bash
cp /home/fengyifan/disk/code/valis/valis/serial_rigid.py src/he2ihc_align/registration/rigid.py
```

- [ ] **步骤 2：简化为单参考图 + 多 moving 图**

编辑 `rigid.py`：
- 移除多切片序列配准逻辑。
- 保留核心 `RigidRegistrar` 或等效类。
- 只保留 `reference <-> moving` 对齐所需的内容。
- 如 MVP 不需要，移除 `viz`、`micro_rigid_registrar` 等导入。

- [ ] **步骤 3：提交**

```bash
git add src/he2ihc_align/registration/rigid.py
git commit -m "feat(registration): import rigid registration from VALIS"
```

---

## Task 9：迁移 VALIS 非刚性配准

**涉及文件：**
- 创建：`src/he2ihc_align/registration/non_rigid.py`

- [ ] **步骤 1：从 serial_non_rigid.py 复制**

```bash
cp /home/fengyifan/disk/code/valis/valis/serial_non_rigid.py src/he2ihc_align/registration/non_rigid.py
```

- [ ] **步骤 2：简化为单参考图 + 多 moving 图**

编辑 `non_rigid.py`：
- 移除多切片序列配准逻辑。
- 保留 `NonRigidRegistrar` 和形变场生成。
- 移除无用导入。

- [ ] **步骤 3：提交**

```bash
git add src/he2ihc_align/registration/non_rigid.py
git commit -m "feat(registration): import non-rigid registration from VALIS"
```

---

## Task 10：迁移 VALIS 变形工具

**涉及文件：**
- 创建：`src/he2ihc_align/registration/warp.py`

- [ ] **步骤 1：从 warp_tools.py 复制**

```bash
cp /home/fengyifan/disk/code/valis/valis/warp_tools.py src/he2ihc_align/registration/warp.py
```

- [ ] **步骤 2：裁剪为坐标变换工具**

保留：
- `warp_xy`
- `getInverseTform`
- `registrar.py` 会用到的形变场/网格辅助函数。

移除：
- 坐标映射不需要的整图 warping 工具。

- [ ] **步骤 3：提交**

```bash
git add src/he2ihc_align/registration/warp.py
git commit -m "feat(registration): import warp utilities from VALIS"
```

---

## Task 11：实现 HEIHCRegistrar

**涉及文件：**
- 创建：`src/he2ihc_align/registration/registrar.py`
- 测试：`tests/test_registration.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_registration.py`：

```python
from pathlib import Path

from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


def test_registrar_fits_and_warps_corners():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    ihc_paths = sorted((TEST_DATA / "174162-1/174162-1-第一批").glob("*.svs"))[:2]

    he_slide = open_slide(he_path)
    ihc_slides = {p.stem.split()[-1]: open_slide(p) for p in ihc_paths}

    registrar = HEIHCRegistrar(
        he_slide=he_slide,
        ihc_slides=ihc_slides,
        registration_level=3,
    )
    registrar.fit()

    he_corners = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    for marker in ihc_slides:
        ihc_corners = registrar.warp_xy_from_he_to_ihc(he_corners, marker=marker)
        assert ihc_corners.shape == (4, 2)
```

执行：

```bash
uv run pytest tests/test_registration.py -v
```

预期结果：`ImportError` 或 `NameError`。

- [ ] **步骤 2：实现 registrar**

创建 `src/he2ihc_align/registration/registrar.py`：

```python
"""High-level registrar for HE reference + multiple IHC moving slides."""

from __future__ import annotations

from typing import Any

import numpy as np

from he2ihc_align.registration.rigid import RigidRegistrar
from he2ihc_align.registration.non_rigid import NonRigidRegistrar
from he2ihc_align.slide_io.base import Slide


class HEIHCRegistrar:
    """Register N IHC slides to a single HE reference."""

    def __init__(
        self,
        he_slide: Slide,
        ihc_slides: dict[str, Slide],
        registration_level: int = 3,
        max_image_dim_px: int = 1024,
        max_non_rigid_dim_px: int = 2048,
    ):
        self.he_slide = he_slide
        self.ihc_slides = ihc_slides
        self.registration_level = registration_level
        self.max_image_dim_px = max_image_dim_px
        self.max_non_rigid_dim_px = max_non_rigid_dim_px

        self.rigid_registrar: RigidRegistrar | None = None
        self.non_rigid_registrar: dict[str, NonRigidRegistrar] | None = None

    def _read_level_image(self, slide: Slide, level: int) -> np.ndarray:
        dims = slide.level_dimensions[level]
        return slide.read_region((0, 0), level, dims)

    def fit(self) -> "HEIHCRegistrar":
        """Run rigid + non-rigid registration for all IHC slides."""
        he_img = self._read_level_image(self.he_slide, self.registration_level)

        ihc_imgs = {
            marker: self._read_level_image(slide, self.registration_level)
            for marker, slide in self.ihc_slides.items()
        }

        self.rigid_registrar = RigidRegistrar(
            reference_img=he_img,
            moving_imgs=ihc_imgs,
            max_image_dim_px=self.max_image_dim_px,
        )
        self.rigid_registrar.fit()

        self.non_rigid_registrar = {}
        for marker, moving_img in ihc_imgs.items():
            rigid_aligned = self.rigid_registrar.warp_image(marker, moving_img)
            nr = NonRigidRegistrar(
                reference_img=he_img,
                moving_img=rigid_aligned,
                max_dim_px=self.max_non_rigid_dim_px,
            )
            nr.fit()
            self.non_rigid_registrar[marker] = nr

        return self

    def warp_xy_from_he_to_ihc(
        self,
        xy: np.ndarray,
        marker: str,
        src_pt_level: int = 0,
        dst_slide_level: int = 0,
    ) -> np.ndarray:
        """Map (x, y) points from HE to the specified IHC slide.

        Parameters
        ----------
        xy : np.ndarray
            Nx2 array of points in HE level-0 coordinates.
        marker : str
            Target IHC marker name.
        src_pt_level, dst_slide_level : int
            Pyramid levels for source points and destination slide.
        """
        if self.rigid_registrar is None or self.non_rigid_registrar is None:
            raise RuntimeError("Must call fit() before warp_xy_from_he_to_ihc()")

        he_downsample = self.he_slide.level_downsamples[src_pt_level]
        ihc_downsample = self.ihc_slides[marker].level_downsamples[dst_slide_level]

        reg_downsample = self.he_slide.level_downsamples[self.registration_level]
        xy_reg = xy * (he_downsample / reg_downsample)

        xy_nr = self.non_rigid_registrar[marker].inverse_warp_xy(xy_reg)
        xy_rigid = self.rigid_registrar.inverse_warp_xy(marker, xy_nr)

        xy_ihc = xy_rigid * (reg_downsample / ihc_downsample)
        return xy_ihc
```

注意：`RigidRegistrar` 和 `NonRigidRegistrar` 的实际 API（`fit`、`warp_image`、`inverse_warp_xy`）需与从 VALIS 复制过来的类保持一致，必要时调整 wrapper 调用。

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_registration.py -v
```

预期结果：通过（调整方法名后）。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/registration/registrar.py tests/test_registration.py
git commit -m "feat(registration): add HEIHCRegistrar"
```

---

## Task 12：实现 patching

**涉及文件：**
- 创建：`src/he2ihc_align/patching.py`
- 测试：`tests/test_patching.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_patching.py`：

```python
from pathlib import Path

from he2ihc_align.patching import sample_grid_patches
from he2ihc_align.slide_io.factory import open_slide

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


def test_grid_sampling_returns_bboxes():
    he_path = next(TEST_DATA.glob("174162-1/174162-1-第一批/*.kfb"))
    slide = open_slide(he_path)

    patches = sample_grid_patches(
        slide,
        patch_size=512,
        stride=512,
        level=3,
        max_white_ratio=1.0,
    )
    assert len(patches) > 0
    x, y, w, h = patches[0]
    assert w == 512
    assert h == 512
```

执行：

```bash
uv run pytest tests/test_patching.py -v
```

预期结果：`ImportError`。

- [ ] **步骤 2：实现 patching**

创建 `src/he2ihc_align/patching.py`：

```python
"""HE patch sampling utilities."""

from __future__ import annotations

import numpy as np

from he2ihc_align.slide_io.base import Slide


def _white_ratio(img: np.ndarray, threshold: int = 230) -> float:
    """Return fraction of pixels that are nearly white/blank."""
    if img.size == 0:
        return 1.0
    gray = img.mean(axis=2) if img.ndim == 3 else img
    return float((gray > threshold).sum() / gray.size)


def sample_grid_patches(
    slide: Slide,
    patch_size: int,
    stride: int | None = None,
    level: int = 0,
    margin: int = 0,
    white_threshold: int = 230,
    max_white_ratio: float = 0.95,
) -> list[tuple[int, int, int, int]]:
    """在 slide 上按网格采样 patch。

    返回 level-0 像素坐标的 bbox (x, y, w, h)。
    """
    stride = stride or patch_size
    downsample = slide.level_downsamples[level]
    width, height = slide.level_dimensions[level]

    margin_level = int(margin / downsample)

    patches = []
    for y in range(margin_level, height - patch_size - margin_level + 1, stride):
        for x in range(margin_level, width - patch_size - margin_level + 1, stride):
            img = slide.read_region(
                (int(x * downsample), int(y * downsample)),
                level,
                (patch_size, patch_size),
            )
            if _white_ratio(img, white_threshold) <= max_white_ratio:
                x0 = int(x * downsample)
                y0 = int(y * downsample)
                w0 = int(patch_size * downsample)
                h0 = int(patch_size * downsample)
                patches.append((x0, y0, w0, h0))

    return patches
```

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_patching.py -v
```

预期结果：通过。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/patching.py tests/test_patching.py
git commit -m "feat(patching): add grid patch sampling with white filtering"
```

---

## Task 13：实现 mapping 表生成

**涉及文件：**
- 创建：`src/he2ihc_align/mapping.py`
- 测试：`tests/test_mapping.py`

- [ ] **步骤 1：编写失败测试**

创建 `tests/test_mapping.py`：

```python
from pathlib import Path

from he2ihc_align.case_io import discover_case
from he2ihc_align.mapping import build_mapping_table
from he2ihc_align.patching import sample_grid_patches
from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


def test_build_mapping_table_outputs_expected_columns():
    case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
    he_path, marker_paths = discover_case(case_dir)

    he_slide = open_slide(he_path)
    ihc_slides = {m: open_slide(p) for m, p in list(marker_paths.items())[:2]}

    registrar = HEIHCRegistrar(he_slide=he_slide, ihc_slides=ihc_slides, registration_level=3)
    registrar.fit()

    he_patches = sample_grid_patches(
        he_slide, patch_size=512, stride=2048, level=3, max_white_ratio=1.0
    )[:3]

    df = build_mapping_table(
        registrar=registrar,
        he_slide=he_slide,
        ihc_slides=ihc_slides,
        he_patch_bboxes=he_patches,
        slide_id="174162-1",
    )

    expected_cols = {
        "patch_id",
        "slide_id",
        "marker",
        "he_x",
        "he_y",
        "he_w",
        "he_h",
        "ihc_x",
        "ihc_y",
        "ihc_w",
        "ihc_h",
        "clipped",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == len(he_patches) * len(ihc_slides)
```

执行：

```bash
uv run pytest tests/test_mapping.py -v
```

预期结果：`ImportError`。

- [ ] **步骤 2：实现 mapping**

创建 `src/he2ihc_align/mapping.py`：

```python
"""Build HE patch to IHC coordinate mapping table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.base import Slide


def build_mapping_table(
    registrar: HEIHCRegistrar,
    he_slide: Slide,
    ihc_slides: dict[str, Slide],
    he_patch_bboxes: list[tuple[int, int, int, int]],
    slide_id: str,
    he_level: int = 0,
    ihc_level: int = 0,
) -> pd.DataFrame:
    """将每个 HE patch 映射到每张 IHC 切片的包络 bbox。

    输出中所有坐标均为 level-0 像素坐标。
    """
    rows = []
    for patch_id, (he_x, he_y, he_w, he_h) in enumerate(he_patch_bboxes, start=1):
        he_corners = np.array(
            [[he_x, he_y], [he_x + he_w, he_y], [he_x + he_w, he_y + he_h], [he_x, he_y + he_h]],
            dtype=float,
        )

        for marker, ihc_slide in ihc_slides.items():
            ihc_corners = registrar.warp_xy_from_he_to_ihc(
                he_corners,
                marker=marker,
                src_pt_level=he_level,
                dst_slide_level=ihc_level,
            )

            ixmin = float(ihc_corners[:, 0].min())
            iymin = float(ihc_corners[:, 1].min())
            ixmax = float(ihc_corners[:, 0].max())
            iymax = float(ihc_corners[:, 1].max())

            ihc_w = int(np.ceil(ixmax - ixmin))
            ihc_h = int(np.ceil(iymax - iymin))
            ihc_x = int(np.floor(ixmin))
            ihc_y = int(np.floor(iymin))

            ihc_w_full, ihc_h_full = ihc_slide.level_dimensions[ihc_level]
            clipped = bool(
                ixmin < 0 or iymin < 0 or ixmax > ihc_w_full or iymax > ihc_h_full
            )

            rows.append(
                {
                    "patch_id": patch_id,
                    "slide_id": slide_id,
                    "marker": marker,
                    "he_x": he_x,
                    "he_y": he_y,
                    "he_w": he_w,
                    "he_h": he_h,
                    "ihc_x": ihc_x,
                    "ihc_y": ihc_y,
                    "ihc_w": ihc_w,
                    "ihc_h": ihc_h,
                    "clipped": clipped,
                }
            )

    return pd.DataFrame(rows)
```

- [ ] **步骤 3：运行测试**

```bash
uv run pytest tests/test_mapping.py -v
```

预期结果：通过。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/mapping.py tests/test_mapping.py
git commit -m "feat(mapping): add HE to IHC coordinate mapping table"
```

---

## Task 14：添加配置文件

**涉及文件：**
- 创建：`configs/default.yaml`

- [ ] **步骤 1：编写配置**

创建 `configs/default.yaml`：

```yaml
# 数据路径
data_root: /home/fengyifan/disk/code/valis/test_SCCE
output_root: ./outputs

# 病例发现：每个病例下哪个子目录包含第一批切片
batch_dir_name: "第一批"

# Patch 采样
patch_size: 512
stride: 512
he_level: 0
registration_level: 3
max_white_ratio: 0.95

# 配准
max_image_dim_px: 1024
max_non_rigid_dim_px: 2048

# 输出
mapping_csv_name: mapping.csv
viz_sample_n: 5
```

- [ ] **步骤 2：提交**

```bash
git add configs/default.yaml
git commit -m "chore(config): add default yaml configuration"
```

---

## Task 15：实现 CLI

**涉及文件：**
- 创建：`src/he2ihc_align/cli.py`
- 修改：`pyproject.toml`

- [ ] **步骤 1：实现 CLI 入口**

创建 `src/he2ihc_align/cli.py`：

```python
"""Command-line interface for he2ihc_align."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from he2ihc_align.case_io import discover_case
from he2ihc_align.mapping import build_mapping_table
from he2ihc_align.patching import sample_grid_patches
from he2ihc_align.registration.registrar import HEIHCRegistrar
from he2ihc_align.slide_io.factory import open_slide


def run_case(case_dir: Path, config: dict, output_dir: Path) -> Path:
    """对一个病例批次目录运行完整流程。"""
    slide_id = case_dir.parent.name.strip()
    he_path, marker_paths = discover_case(case_dir)

    he_slide = open_slide(he_path)
    ihc_slides = {m: open_slide(p) for m, p in marker_paths.items()}

    registrar = HEIHCRegistrar(
        he_slide=he_slide,
        ihc_slides=ihc_slides,
        registration_level=config["registration_level"],
        max_image_dim_px=config["max_image_dim_px"],
        max_non_rigid_dim_px=config["max_non_rigid_dim_px"],
    )
    print(f"[{slide_id}] Running registration...")
    registrar.fit()

    print(f"[{slide_id}] Sampling HE patches...")
    he_patches = sample_grid_patches(
        he_slide,
        patch_size=config["patch_size"],
        stride=config["stride"],
        level=config["registration_level"],
        max_white_ratio=config["max_white_ratio"],
    )
    print(f"[{slide_id}] {len(he_patches)} HE patches kept")

    print(f"[{slide_id}] Building mapping table...")
    df = build_mapping_table(
        registrar=registrar,
        he_slide=he_slide,
        ihc_slides=ihc_slides,
        he_patch_bboxes=he_patches,
        slide_id=slide_id,
        he_level=config["he_level"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / config["mapping_csv_name"]
    df.to_csv(csv_path, index=False)
    print(f"[{slide_id}] Saved mapping to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="HE to multi-IHC patch mapping")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--case-dir", help="Run a single case batch directory")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_root = Path(args.output_dir or config["output_root"])

    if args.case_dir:
        case_dir = Path(args.case_dir)
        run_case(case_dir, config, output_root / case_dir.parent.name)
    else:
        data_root = Path(config["data_root"])
        batch_name = config.get("batch_dir_name", "第一批")
        for case_dir in sorted(data_root.iterdir()):
            if not case_dir.is_dir():
                continue
            batch_dir = case_dir / batch_name
            if not batch_dir.exists():
                print(f"Skipping {case_dir.name}: no {batch_name} directory")
                continue
            run_case(batch_dir, config, output_root / case_dir.name)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：添加 CLI 入口点**

在 `pyproject.toml` 中添加：

```toml
[project.scripts]
he2ihc-align = "he2ihc_align.cli:main"
```

- [ ] **步骤 3：重新安装包**

```bash
uv pip install -e .
```

- [ ] **步骤 4：在单个病例上运行 CLI**

```bash
he2ihc-align --case-dir /home/fengyifan/disk/code/valis/test_SCCE/174162-1/174162-1-第一批 --output-dir ./outputs
```

预期结果：生成 `outputs/mapping.csv`，列符合预期。

- [ ] **步骤 5：提交**

```bash
git add src/he2ihc_align/cli.py pyproject.toml
git commit -m "feat(cli): add end-to-end command-line interface"
```

---

## Task 16：实现可视化

**涉及文件：**
- 创建：`src/he2ihc_align/viz.py`
- 修改：`src/he2ihc_align/cli.py`

- [ ] **步骤 1：实现可视化辅助函数**

创建 `src/he2ihc_align/viz.py`：

```python
"""Visualization helpers for registration quality checks."""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from he2ihc_align.slide_io.base import Slide


def read_patch_rgb(slide: Slide, x: int, y: int, w: int, h: int, level: int = 0) -> np.ndarray:
    """Read a region clamped to slide bounds."""
    img_w, img_h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    x_lvl = int(x / downsample)
    y_lvl = int(y / downsample)
    w_lvl = int(w / downsample)
    h_lvl = int(h / downsample)

    x_lvl = max(0, min(x_lvl, img_w - 1))
    y_lvl = max(0, min(y_lvl, img_h - 1))
    w_lvl = min(w_lvl, img_w - x_lvl)
    h_lvl = min(h_lvl, img_h - y_lvl)

    if w_lvl <= 0 or h_lvl <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    patch = slide.read_region((x_lvl, y_lvl), level, (w_lvl, h_lvl))
    if patch.shape[0] != h or patch.shape[1] != w:
        patch = np.array(Image.fromarray(patch).resize((w, h), Image.Resampling.LANCZOS))
    return patch


def make_patch_figure(
    he_patch: np.ndarray,
    ihc_patches: dict[str, np.ndarray],
    title: str = "HE → IHC",
) -> plt.Figure:
    n = len(ihc_patches) + 1
    n_cols = min(n, 4)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    axes[0, 0].imshow(he_patch)
    axes[0, 0].set_title("HE")
    axes[0, 0].axis("off")

    for idx, marker in enumerate(sorted(ihc_patches), start=1):
        row, col = divmod(idx, n_cols)
        axes[row, col].imshow(ihc_patches[marker])
        axes[row, col].set_title(marker)
        axes[row, col].axis("off")

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_to_data_uri(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def create_html_gallery(
    output_path: Path,
    slide_id: str,
    entries: list[dict],
) -> Path:
    """Create a minimal HTML gallery of patch comparisons."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!DOCTYPE html>",
        "<html lang='zh-CN'><head><meta charset='UTF-8'>",
        f"<title>HE → IHC – {html.escape(slide_id)}</title>",
        "<style>body{font-family:sans-serif;margin:2rem;background:#f7f8fa}.card{background:#fff;border-radius:8px;padding:1rem;margin-bottom:1.5rem;box-shadow:0 2px 6px rgba(0,0,0,0.06)}img{width:100%;border-radius:4px}</style>",
        "</head><body>",
        f"<h1>HE → IHC: {html.escape(slide_id)}</h1>",
    ]

    for entry in entries:
        parts.append("<div class='card'>")
        parts.append(f"<h2>Patch #{entry['patch_id']}</h2>")
        parts.append(f"<p>HE bbox: ({entry['he_x']}, {entry['he_y']}) {entry['he_w']}x{entry['he_h']}</p>")
        parts.append(f"<img src='{entry['uri']}'>")
        parts.append("</div>")

    parts.append("</body></html>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
```

- [ ] **步骤 2：在 CLI 中集成可视化**

在 `src/he2ihc_align/cli.py` 中添加导入：

```python
from he2ihc_align import viz
```

在 `run_case` 保存 CSV 之后添加：

```python
    sample_n = min(config.get("viz_sample_n", 5), len(he_patches))
    if sample_n > 0:
        entries = []
        for patch_id, (he_x, he_y, he_w, he_h) in enumerate(he_patches[:sample_n], start=1):
            he_patch = viz.read_patch_rgb(he_slide, he_x, he_y, he_w, he_h, level=config["he_level"])
            ihc_patches = {}
            for marker, ihc_slide in ihc_slides.items():
                row = df[(df["patch_id"] == patch_id) & (df["marker"] == marker)].iloc[0]
                ihc_patch = viz.read_patch_rgb(
                    ihc_slide,
                    int(row["ihc_x"]),
                    int(row["ihc_y"]),
                    int(row["ihc_w"]),
                    int(row["ihc_h"]),
                )
                ihc_patches[marker] = ihc_patch
            fig = viz.make_patch_figure(he_patch, ihc_patches, title=f"{slide_id} – Patch {patch_id}")
            entries.append({
                "patch_id": patch_id,
                "he_x": he_x,
                "he_y": he_y,
                "he_w": he_w,
                "he_h": he_h,
                "uri": viz.fig_to_data_uri(fig),
            })
        gallery_path = output_dir / "gallery.html"
        viz.create_html_gallery(gallery_path, slide_id, entries)
        print(f"[{slide_id}] Gallery saved to {gallery_path}")
```

- [ ] **步骤 3：运行 CLI 并检查 gallery**

```bash
he2ihc-align --case-dir /home/fengyifan/disk/code/valis/test_SCCE/174162-1/174162-1-第一批 --output-dir ./outputs
```

预期结果：生成 `outputs/gallery.html`。

- [ ] **步骤 4：提交**

```bash
git add src/he2ihc_align/viz.py src/he2ihc_align/cli.py
git commit -m "feat(viz): add HTML patch comparison gallery"
```

---

## Task 17：添加端到端脚本

**涉及文件：**
- 创建：`scripts/run_pipeline.py`

- [ ] **步骤 1：创建脚本**

创建 `scripts/run_pipeline.py`：

```python
"""Convenience wrapper around he2ihc_align.cli."""

from he2ihc_align.cli import main

if __name__ == "__main__":
    main()
```

- [ ] **步骤 2：运行**

```bash
uv run python scripts/run_pipeline.py --case-dir /home/fengyifan/disk/code/valis/test_SCCE/174162-1/174162-1-第一批 --output-dir ./outputs
```

预期结果：与 CLI 输出一致。

- [ ] **步骤 3：提交**

```bash
git add scripts/run_pipeline.py
git commit -m "chore(scripts): add run_pipeline.py convenience script"
```

---

## Task 18：更新 README

**涉及文件：**
- 修改：`README.md`

- [ ] **步骤 1：重写 README**

替换 `README.md` 为：

```markdown
# he2ihc_align

轻量化的 HE 到多 IHC 全切片配准工具。输出一个 CSV，记录每个 HE patch 在每张 IHC 切片上的对应位置。

## 快速开始

```bash
uv sync
uv pip install -e .

# 单个病例
he2ihc-align --case-dir /path/to/case/第一批 --output-dir ./outputs

# 批量运行所有病例（编辑 configs/default.yaml）
he2ihc-align --config configs/default.yaml
```

## 输出 CSV

`mapping.csv` 列说明：

| 列 | 含义 |
|----|------|
| `patch_id` | HE patch 编号 |
| `slide_id` | 病例 ID |
| `marker` | IHC marker 名称 |
| `he_x`, `he_y`, `he_w`, `he_h` | HE patch 在 level-0 的 bbox |
| `ihc_x`, `ihc_y`, `ihc_w`, `ihc_h` | IHC 上映射后包络 bbox（level-0） |
| `clipped` | 映射区域是否部分超出 IHC 边界 |

## 配置

编辑 `configs/default.yaml`：

- `patch_size`, `stride`：HE patch 采样参数。
- `registration_level`：VALIS 配准使用的金字塔层级。
- `max_image_dim_px`, `max_non_rigid_dim_px`：配准分辨率。

## 项目结构

详见 `docs/superpowers/specs/2026-06-19-he2ihc-align-design.md`。
```

- [ ] **步骤 2：提交**

```bash
git add README.md
git commit -m "docs(readme): rewrite quick-start and output format"
```

---

## Task 19：最终集成测试

**涉及文件：**
- 测试：`tests/test_pipeline.py`

- [ ] **步骤 1：编写集成测试**

创建 `tests/test_pipeline.py`：

```python
from pathlib import Path

from he2ihc_align.cli import run_case

TEST_DATA = Path("/home/fengyifan/disk/code/valis/test_SCCE")


def test_run_case_outputs_csv():
    case_dir = TEST_DATA / "174162-1" / "174162-1-第一批"
    config = {
        "patch_size": 512,
        "stride": 2048,
        "he_level": 0,
        "registration_level": 3,
        "max_white_ratio": 1.0,
        "max_image_dim_px": 1024,
        "max_non_rigid_dim_px": 2048,
        "mapping_csv_name": "mapping.csv",
    }
    output_dir = Path("./outputs/test_pipeline")
    csv_path = run_case(case_dir, config, output_dir)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
```

- [ ] **步骤 2：运行集成测试**

```bash
uv run pytest tests/test_pipeline.py -v
```

预期结果：通过（配准可能耗时数分钟）。

- [ ] **步骤 3：提交**

```bash
git add tests/test_pipeline.py
git commit -m "test(integration): add end-to-end pipeline test"
```

---

## 自查清单

- [ ] **Spec 覆盖**：设计文档 `2026-06-19-he2ihc-align-design.md` 的每个章节都有对应的实现任务。
- [ ] **Placeholder 检查**：计划中没有 `TBD`、`TODO`、`implement later`、模糊步骤。
- [ ] **类型一致性**：`HEIHCRegistrar.warp_xy_from_he_to_ihc` 的签名与 `mapping.py`、测试中一致。
- [ ] **文件路径**：所有创建/修改的路径都是相对于仓库根的精确路径。
- [ ] **测试命令**：每个任务都包含具体的 `uv run pytest ...` 命令和预期结果。
