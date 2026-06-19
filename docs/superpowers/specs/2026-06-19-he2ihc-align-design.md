# he2ihc_align 设计文档

> 目标：将 HE 病理切片与多 marker IHC 切片配准，输出每个 HE patch 对应每张 IHC patch 位置的 CSV 映射表。

---

## 1. 项目定位

`he2ihc_align` 是一个**针对 HE ↔ 多 IHC 场景裁剪的轻量配准工具**。

- 不依赖完整 VALIS 仓库，而是将 VALIS 的配准核心复制进来并精简。
- HE 读取原生支持 `kfbslide`（`.kfb`）和 `openslide`（`.svs/.mrxs/.tiff` 等）。
- 不处理褐色点检测标签，只输出坐标映射表供下游使用。
- 提供极简可视化，用于人工抽检配准质量。

---

## 2. 背景与约束

- **HE 切片**：原始格式为 KFBio `.kfb`，可通过 `kfbslide` 直接读取。
- **IHC 切片**：通常为 `.svs`、`.mrxs`、`.tiff` 等，可通过 `openslide` 读取。
- **一个病例**：1 张 HE + 20+ 张不同 marker 的 IHC。
- **坐标要求**：统一使用原始 HE level 0 和 IHC level 0 的像素坐标。
- **非刚性变换**：矩形 patch 映射后会变形，输出包络 bbox。

---

## 3. 整体架构

```text
he2ihc_align/
├── src/he2ihc_align/
│   ├── slide_io/               # 统一 slide 读取层
│   │   ├── base.py             # Slide 抽象接口
│   │   ├── kfb_backend.py      # kfbslide 后端
│   │   ├── openslide_backend.py # OpenSlide 后端
│   │   └── factory.py          # 按扩展名自动选择后端
│   ├── registration/           # 从 VALIS 裁剪的配准核心
│   │   ├── feature_detectors.py
│   │   ├── feature_matcher.py
│   │   ├── rigid.py
│   │   ├── non_rigid.py
│   │   ├── registrar.py        # 主入口：HE reference + N 张 IHC
│   │   └── warp.py             # 坐标变换工具
│   ├── patching.py             # HE 上 grid/random 采样
│   ├── mapping.py              # HE patch → IHC bbox 映射
│   ├── viz.py                  # 可视化（patch 对比图 / HTML gallery）
│   ├── cli.py                  # 命令行入口
│   └── __init__.py
├── scripts/
│   └── run_pipeline.py         # 端到端 pipeline 脚本
├── configs/
│   └── default.yaml            # 数据路径、patch size、marker 列表等
├── pyproject.toml
└── README.md
```

---

## 4. 核心组件

### 4.1 `slide_io` — 统一读取层

目标：让上层代码不关心 HE/IHC 的具体文件格式。

```python
class Slide(Protocol):
    @property
    def level_count(self) -> int: ...
    @property
    def level_dimensions(self) -> list[tuple[int, int]]: ...
    @property
    def level_downsamples(self) -> list[float]: ...
    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> np.ndarray: ...
```

- `KfbSlide`：封装 `kfbslide.KfbSlide`，输出 RGB numpy 数组。
- `OpenSlide`：封装 `openslide.OpenSlide`，输出 RGB numpy 数组。
- `open_slide(path)`：根据扩展名自动选择后端。

### 4.2 `registration` — 精简配准核心

从 VALIS 复制并保留以下模块：

- 特征检测：`feature_detectors.py`
- 特征匹配：`feature_matcher.py`
- 刚性配准：`rigid.py`
- 非刚性配准：`non_rigid.py`
- 坐标 warping：`warp.py`
- 主入口：`registrar.py` 提供 `HEIHCRegistrar`

删除以下内容：

- 完整的 HTML/JSON 报告
- serial_rigid / serial_non_rigid 多序列配准
- ome-tiff 写入
- annotation warp
- 大量内置可视化

### 4.3 `patching.py` — HE patch 采样

```python
def sample_patches(
    slide: Slide,
    patch_size: int,
    stride: int | None = None,
    level: int = 0,
    mode: str = "grid",           # "grid" | "random"
    n_patches: int | None = None, # random 模式用
    margin: int = 0,
    white_threshold: int = 230,
    max_white_ratio: float = 0.9,
) -> list[tuple[int, int, int, int]]:
    ...
```

- `mode="grid"`：按 stride 滑动采样。
- `mode="random"`：随机采样 `n_patches` 个。
- 过滤空白比例过高的 patch。

### 4.4 `mapping.py` — 坐标映射

```python
def build_mapping_table(
    registrar,
    he_slide: Slide,
    ihc_slides: dict[str, Slide],
    he_patch_bboxes: list[tuple[int, int, int, int]],
    he_level: int = 0,
    ihc_level: int = 0,
    non_rigid: bool = True,
) -> pd.DataFrame:
    ...
```

输出 DataFrame 列：

| 字段 | 含义 |
|------|------|
| `patch_id` | HE patch 全局编号 |
| `slide_id` | 病例 ID |
| `marker` | IHC marker 名称 |
| `he_x`, `he_y` | HE patch 左上角（level 0） |
| `he_w`, `he_h` | HE patch 尺寸 |
| `ihc_x`, `ihc_y` | IHC 上映射后包络 bbox 左上角（level 0） |
| `ihc_w`, `ihc_h` | IHC 包络 bbox 尺寸 |
| `clipped` | 是否部分超出 IHC 边界 |

### 4.5 `viz.py` — 可视化

- `make_patch_figure(he_patch, ihc_patches, ...) -> plt.Figure`
- `create_html_gallery(entries, output_path)`：生成简易 HTML 抽检页面。
- 默认只对少量 sample patch 生成可视化，避免 IO 爆炸。

### 4.6 `cli.py` / `scripts/run_pipeline.py`

端到端流程：

```text
读取配置
  │
  ▼
打开 HE + 所有 IHC slide
  │
  ▼
运行配准（HE 为 reference）
  │
  ▼
在 HE 上采样 patch
  │
  ▼
映射每个 patch 到每张 IHC
  │
  ▼
输出 mapping.csv
  │
  ▼
可选：输出可视化 gallery
```

---

## 5. 数据流与坐标系统

1. **配准阶段**
   - 从 HE 的某个低分辨率 level（如 level 3）读取图像作为 reference。
   - 从每张 IHC 的对应低分辨率 level 读取图像作为 moving。
   - 运行刚性 + 非刚性配准，得到 transformation。

2. **映射阶段**
   - 在 HE level 0 上按 grid/random 采样 patch。
   - 将每个 patch 的四个角点从 HE level 0 映射到 IHC level 0。
   - 取包络 bbox 作为 IHC 读取区域。

3. **输出阶段**
   - CSV 中所有坐标均为 level 0 像素坐标。
   - 下游读取时直接使用 `kfbslide`/`openslide` 的 `read_region`。

---

## 6. 依赖

```toml
[project]
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
    "openslide-python",   # IHC 读取
    "kfbslide",           # HE 读取，uv 安装
]
```

- 尽量去掉 VALIS 原本需要的 `scyjava`、`jpype1`、`ome-types` 等重型依赖（如果能完全用 numpy/torch 实现配准核心）。
- 如果裁剪后发现某些子模块仍需要 Java/Bio-Formats，则保留最小集合。

---

## 7. 实施计划（概要）

### Phase 1：基础设施
- 搭建 `pyproject.toml` 和项目骨架。
- 实现 `slide_io` 统一读取层。
- 写一个最小端到端脚本验证 HE/IHC 都能打开。

### Phase 2：迁移配准核心
- 从 VALIS 复制刚性/非刚性配准相关代码。
- 删除无用模块和依赖。
- 替换 Slide 读取入口，使其走 `slide_io`。
- 单病例跑通配准。

### Phase 3：Mapping + CSV
- 实现 `patching.py` 和 `mapping.py`。
- 输出 CSV 并验证坐标合理性。

### Phase 4：可视化与配置
- 实现 `viz.py` 和 HTML gallery。
- 添加 `configs/default.yaml`。
- 完善 `cli.py` 和 `scripts/run_pipeline.py`。

### Phase 5：验证与文档
- 多病例批量测试。
- 更新 README，记录使用方式和参数含义。

---

## 8. 风险与关键决策

| 风险 | 应对 |
|------|------|
| VALIS 裁剪时误删关键依赖 | 每次裁剪后都跑通单病例配准 |
| 去掉 Java 依赖后某些格式无法读取 | IHC 限定为 openslide 可读格式；HE 用 kfbslide |
| 非刚性变换导致 IHC bbox 过大 | 输出包络 bbox，并在 CSV 中标记 `clipped` |
| 多病例批量运行时内存/IO 爆炸 | 配准结果序列化；patch 映射分批生成 |

---

## 9. 成功标准

- 给定一个病例目录（1 张 HE + N 张 IHC），运行一条命令即可输出 `mapping.csv`。
- CSV 中每个 HE patch 对每个 marker 都有对应的 IHC level 0 坐标。
- 可视化 gallery 中随机抽样的 patch 与 IHC 区域在肉眼上对应对齐。
