# he2ihc_align

从 HE 原始病理切片配准到多标记 IHC/荧光染色切片，并生成训练数据映射表。

---

## 一、项目目标

训练一个深度学习模型，能够**直接从原始 HE 切片预测对应的 20+ 个 IHC/荧光 marker 的染色图**。

为了获取训练标签，本项目负责完成前半部分：

1. **配准与映射**：将 HE 切片与每个 marker 的 IHC 切片进行 WSI 级配准。
2. **坐标映射表生成**：对 HE 切片上的每一个 patch，计算其在每张 IHC 切片上的对应位置。
3. **标签对接**：将上述映射表输入下游的褐色点检测算法，得到每个 patch 的 `num_pos / num_neg / percent_pos` 等标签。

最终产出：一个结构化的 `mapping.csv`，记录 HE 上每个 patch 对应每张 IHC 的精确位置，供训练使用。

---

## 二、背景数据

- **HE 切片**：原始格式为 KFBio `.kfb`，需先转换为 VALIS 可读的金字塔 TIFF。
- **IHC 切片**：通常为 `.svs`、`.mrxs` 等格式，可由 OpenSlide/pyvips 读取。
- **一个病例包含**：1 张 HE + 20+ 张不同 marker 的 IHC。
- **多个病例**：如 `174162-1`、`396752-6`、`397354-4`、`98140-6` 等。

---

## 三、整体流程

```
原始 HE (KFB)
    │
    ▼
convert_he_kfb_to_tiff()        # KFB level N → pyramidal TIFF
    │
    ▼
register_case()                 # VALIS: HE 作为 reference，所有 IHC 作为 moving
    │
    ▼
build_he_to_ihc_mapping()       # HE patch 网格 → IHC 坐标映射表
    │
    ▼
patching / filtering            # 过滤空白、越界 patch
    │
    ▼
labeling()                      # 褐色点检测 → num_pos / num_neg / percent_pos
    │
    ▼
dataset.py                      # 生成训练用 Dataset / 多通道标签
```

---

## 四、核心模块设计

### 1. `conversion.py`

将 KFB 格式的 HE 切片转换为 pyramidal TIFF，供 VALIS 读取。

```python
def convert_he_kfb_to_tiff(
    kfb_path: Path,
    output_dir: Path,
    level: int = 3,
) -> Path:
    ...
```

- `level` 控制转换分辨率。`level=3` 约为 8× 下采样，适合快速配准；后续如需更精细的 HE patch，可改用 `level=2` 或 `level=1`。

### 2. `registration.py`

调用 VALIS，以 HE 为 reference，所有 marker 的 IHC 为 moving，完成一次性配准。

```python
def register_case(
    case_dir: Path,
    output_dir: Path,
    markers: list[str],
    he_level: int = 3,
) -> Valis:
    ...
```

- 输出包含 pickled registrar、summary CSV、overlap 图等。
- 所有 IHC 共享同一个 HE 参考坐标系。

### 3. `mapping.py`（核心）

生成 HE patch 到 IHC patch 的坐标映射表。

```python
def build_he_to_ihc_mapping(
    registrar,
    he_slide_name: str,
    ihc_slide_names: list[str],
    patch_size: int = 512,
    stride: int | None = None,
    non_rigid: bool = True,
) -> pd.DataFrame:
    ...
```

输入：
- HE 上按 grid 滑动的 patch 左上角坐标。
- 每个 patch 的四个角点。

输出 DataFrame 字段：

| 字段 | 含义 |
|------|------|
| `slide_id` | 病例 ID |
| `marker` | IHC marker 名称 |
| `he_x`, `he_y` | HE patch 左上角坐标（原始 KFB level 0） |
| `he_w`, `he_h` | HE patch 尺寸 |
| `ihc_x`, `ihc_y` | IHC 上对应包络 bbox 左上角（IHC level 0） |
| `ihc_w`, `ihc_h` | IHC 包络 bbox 尺寸 |
| `patch_id` | patch 全局编号 |

注意：非刚性变换不保持矩形，因此将 HE patch 四角映射到 IHC 后，取包络 bbox 作为 IHC 读取区域。

### 4. `patching.py`

- 在 HE level 0 上按 stride 滑动切 patch。
- 过滤空白比例过高的 patch（`white_ratio`）。
- 过滤 IHC 上映射后超出边界的 patch。

### 5. `labeling.py`

对接下游褐色点检测算法：

```python
def detect_brown_points(ihc_patch: np.ndarray) -> tuple[int, int]:
    """返回 num_pos, num_neg"""
    ...
```

最终生成：

| 字段 | 含义 |
|------|------|
| `num_pos` | 阳性（褐色）点数 |
| `num_neg` | 阴性（非褐色）点数 |
| `num_total` | 总细胞数 |
| `percent_pos` | 阳性比例 |
| `white_ratio` | 空白/无效区域比例 |

### 6. `dataset.py`

将 mapping 表封装为 PyTorch Dataset，供模型训练：

```python
class HE2IHCDataset(Dataset):
    def __init__(self, mapping_csv, he_slide_path, ihc_slide_dir):
        ...

    def __getitem__(self, idx):
        he_patch = ...          # 从 HE 读取
        labels = ...            # 20+ 维向量，每维对应一个 marker 的 percent_pos
        return he_patch, labels
```

---

## 五、关键设计决策

### 1. 坐标系统一

所有 mapping 表的坐标统一使用 **原始 KFB level 0** 和 **IHC level 0** 的像素坐标。

原因：
- 配准是在转换后的 HE TIFF（如 KFB level 3）上进行的。
- 配准完成后，将 HE TIFF 上的坐标乘以 downsample factor（如 ×8）即可还原到 KFB level 0。
- 这样后续读取原始数据时不需要关心配准用的 level。

### 2. 一个病例只跑一次配准

以 HE 为 reference，所有 marker 的 IHC 同时作为 moving 输入 VALIS。

优点：
- 所有 marker 共享同一个 reference 坐标系。
- 避免每个 marker 单独配准带来的误差累积。

### 3. 非刚性变换的处理

VALIS 默认使用刚性 + 非刚性配准。非刚性变换会导致矩形 patch 映射后变形。

处理方式：
- 将 HE patch 的四个角映射到 IHC。
- 取包络 bbox 作为 IHC 读取区域。
- 如果后续做 **像素级 image-to-image 翻译**，需要在 IHC 包络区域上做反向 warp，重采样成与 HE patch 严格对齐的矩形。
- 如果只做 **patch-level 分类/回归**（预测 `percent_pos`），包络 bbox 直接读取即可。

### 4. IHC 读取层级

褐色点检测需要高精度时，建议读取 IHC **level 0**。

如果 patch 数量巨大导致 IO 瓶颈，可配置 `ihc_read_level`，但需在标签生成阶段验证精度损失。

### 5. 输出格式

模型输出可以是：
- **多通道回归**：每个通道对应一个 marker 的染色强度图。
- **多标签向量**：每个 HE patch 输出 20+ 维向量，每维为该 marker 的 `percent_pos`。

本项目第一阶段只负责产出 mapping 表，不绑定具体模型输出格式。

---

## 六、实施计划（MVP）

### 阶段 1：单病例端到端打通

- 选择 1 个病例（如 `174162-1`）和 3-5 个 marker。
- 完成：convert → register → mapping → labeling → CSV。
- 目标：验证整个数据流正确。

### 阶段 2：Mapping 质量验证

- 可视化大量 HE patch 与对应 IHC patch 的对比图。
- 检查非刚性映射后是否对齐。
- 确认褐色点检测算法与 mapping 结果一致。

### 阶段 3：批量多病例

- 将脚本改为可配置化。
- 跑通所有病例和所有 20+ marker。
- 生成完整的训练 mapping 表。

### 阶段 4：训练数据生成

- 将 mapping 表转换为 PyTorch Dataset。
- 开始训练 HE → multi-marker IHC 的预测模型。

---

## 七、与 valis 的关系

`valis` 是配准依赖库，包含 VALIS 源码及我们之前调试用的临时脚本。

`he2ihc_align` 是新的独立仓库，职责是：

- 封装 VALIS 的调用逻辑。
- 管理病例、marker、patch 配置。
- 生成结构化的训练数据映射表。
- 不包含 VALIS 源码本身，VALIS 通过 `pyproject.toml` 作为依赖引入。

---

## 八、目录规划

```text
he2ihc_align/
├── README.md                  # 本文件
├── pyproject.toml             # 项目依赖
├── configs/
│   └── default.yaml           # 数据路径、patch size、marker 列表等
├── src/
│   ├── __init__.py
│   ├── conversion.py          # KFB → TIFF
│   ├── registration.py        # VALIS 配准封装
│   ├── mapping.py             # HE → IHC 坐标映射
│   ├── patching.py            # patch 采样与过滤
│   ├── labeling.py            # 褐色点检测对接
│   └── dataset.py             # PyTorch Dataset
├── scripts/
│   ├── 01_convert_he.py
│   ├── 02_register.py
│   ├── 03_build_mapping.py
│   ├── 04_extract_labels.py
│   └── 05_create_dataset.py
├── tests/
└── outputs/                   # 配准结果、mapping 表、数据集（gitignored）
```

---

## 九、参考

- VALIS: Virtual Alignment of pathology Image Series
- DISK: Learning local features with policy gradient (Tyszkiewicz et al., NeurIPS 2020)
- LightGlue: Local Feature Matching at Light Speed (Lindenberger et al., CVPR 2023)
