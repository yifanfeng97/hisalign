# hisalign

Whole-slide image alignment for H&E and IHC markers.

将 H&E（HE）病理切片与多标记 IHC/荧光染色切片进行 WSI 级配准，并支持离线坐标映射。

---

## 项目目标 / Project Purpose

hisalign 完成数据预处理的前半部分：

1. **WSI 级配准**：将 HE 切片作为参考，把每个 marker 的 IHC 切片配准到 HE 空间。
2. **可序列化模型**：把配准结果保存为一个独立的 `.pkl` 模型文件，不依赖原始切片句柄。
3. **离线坐标映射**：基于保存的模型，在 HE 与任意 IHC marker 之间双向映射 level-0 像素坐标。
4. **可视化**：可选生成 patch 级 `gallery.html` 与 slide 级 `report.html`。

最终产出：一个 `*.pkl` 模型文件，记录从 HE 到各 IHC marker 的完整空间变换，供下游任务使用。

---

## 安装 / Installation

```bash
uv sync && uv pip install -e .
```

安装完成后，命令行入口为：

```bash
hisalign --help
```

---

## 快速开始 / Quickstart

### Python API

```python
from hisalign import HisAlign, HisAlignModel

# 1. 配准并保存模型
aligner = HisAlign(
    he_path="HE.kfb",
    ihc_paths={"CD3": "CD3.svs", "Ki67": "Ki67.svs"},
    registration_level=3,
    max_image_dim_px=1024,
    preprocessing="od",
    feature_detector="kaze",
    feature_n_levels=3,
    match_max_ratio=1.0,
    mpp=0.25,  # 如果切片元数据中没有 MPP，需要显式指定
)
model = aligner.fit()
model.save("model.pkl")

# 2. 离线坐标映射（不需要打开切片）
loaded = HisAlignModel.load("model.pkl")
mapped = loaded.warp_xy(
    coords=[[1000, 2000]],  # HE level-0 坐标
    marker="CD3",
    direction="he_to_ihc",
)
print(mapped)  # IHC level-0 坐标
```

### 命令行 / CLI

#### 1. 配准并保存模型

```bash
hisalign register \
  --he HE.kfb \
  --ihc CD3=CD3.svs \
  --ihc Ki67=Ki67.svs \
  --output model.pkl \
  --config configs/default.yaml \
  --mpp 0.25
```

如果省略 `marker=`，marker 名称会从文件名中自动提取（取文件 stem 的最后一段）。

#### 2. 基于模型做坐标映射

```bash
hisalign warp \
  --model model.pkl \
  --marker CD3 \
  --direction he_to_ihc \
  --coords coords.csv \
  --output mapped.csv
```

输入 `coords.csv` 需要包含 `x`、`y` 两列；输出会额外包含 `marker`、`direction` 列。

#### 3. 生成可视化报告

```bash
hisalign visualize \
  --model model.pkl \
  --output-dir ./out \
  --config configs/default.yaml
```

`register` 命令也会根据配置自动生成 `gallery.html` 和 `report.html`。

---

## 坐标约定 / Coordinate Convention

- **所有坐标默认都是 level-0（最高分辨率）像素坐标。**
- 坐标顺序为 `(x, y)`，即 `(列, 行)`。
- 坐标原点为切片左上角。
- 如果需要传入其他层级的坐标，请先手动缩放到 level-0 再调用 `warp_xy`。

---

## 支持格式 / Supported Formats

### KFBio 原生

- `.kfb`

### OpenSlide 支持格式

- `.svs` (Aperio)
- `.tif` / `.tiff` (Leica/Aperio TIFF, generic tiled TIFF)
- `.ndpi` (Hamamatsu)
- `.vms` / `.vmu` (Hamamatsu)
- `.mrxs` (Mirax)
- `.scn` (Sakura)

只要 OpenSlide 能打开，hisalign 通常也能使用。

---

## 输出 / Outputs

运行 `hisalign register` 后，默认输出：

- `model.pkl`：可序列化的配准模型，包含所有坐标变换信息。

如果配置中启用了可视化，还会在同目录生成：

- `gallery.html`：**patch 级别**可视化，随机展示若干 HE patch 及其对应的各 marker IHC patch。
- `report.html`：**slide 级别**配准质量报告，包含全图叠加对比、配准误差统计、每个 marker 的缩略图和形变场。

---

## 配置文件 / Configuration

`configs/default.yaml` 示例：

```yaml
# Registration parameters
registration_level: 3
max_image_dim_px: 1024
preprocessing: "od"          # "od" or "gray"

# Feature detection / matching
feature_detector: "kaze"      # brisk, akaze, kaze, orb, sift, vgg
feature_n_levels: 3
match_max_ratio: 1.0         # Lowe's ratio test; 1.0 disables

# Physical units (set explicitly when slide metadata lacks MPP, e.g. KFBio)
mpp: null                    # microns per pixel at level 0

# Patch-level gallery
patch_size: 512
stride: 512
he_level: 0
max_white_ratio: 0.95
viz_sample_n: 5
viz_random_seed: 42
viz_sample_clipped: true

# Slide-level report
generate_report: true
report_rtre_threshold: 5.0
```

关键配置项说明：

| 配置项 | 说明 |
|--------|------|
| `registration_level` | 配准使用的金字塔层级 |
| `max_image_dim_px` | 配准图像最大边长 |
| `preprocessing` | 预处理：光学密度 `"od"` 或普通灰度 `"gray"` |
| `feature_detector` | 特征检测器 |
| `feature_n_levels` | 多尺度特征检测层数 |
| `match_max_ratio` | Lowe ratio test 阈值，1.0 表示关闭 |
| `mpp` | level-0 像素尺寸（µm/px），切片元数据缺失时需显式设置 |
| `patch_size` / `stride` | patch 尺寸与步长 |
| `viz_sample_n` | gallery 中展示的 patch 数量，0 表示不生成 |
| `viz_sample_clipped` | 是否允许采样超出 IHC 边界的 patch |
| `generate_report` | 是否生成 slide 级别 `report.html` |
| `report_rtre_threshold` | rTRE 阈值，低于该值标记为良好 |

---

## 项目结构 / Project Structure

```text
hisalign/
├── README.md                  # 本文件
├── pyproject.toml             # 项目依赖与配置
├── configs/
│   └── default.yaml           # 默认配置文件
├── src/
│   └── hisalign/
│       ├── __init__.py        # 公开 API 导出
│       ├── api.py             # HisAlign / HisAlignModel
│       ├── cli.py             # 命令行入口
│       ├── case_io.py         # 病例发现工具
│       ├── mapping.py         # HE → IHC 坐标映射表
│       ├── patching.py        # patch 采样与过滤
│       ├── preprocessing.py   # 预处理
│       ├── viz.py             # 可视化
│       ├── registration/      # 配准模块
│       │   ├── registrar.py
│       │   ├── rigid.py
│       │   ├── non_rigid.py
│       │   ├── feature_detectors.py
│       │   ├── feature_matcher.py
│       │   └── warp_tools.py
│       └── slide_io/          # 切片读取模块
│           ├── base.py
│           ├── factory.py
│           ├── kfb_backend.py
│           └── openslide_backend.py
├── tests/                     # 测试套件
└── outputs/                   # 输出目录（gitignored）
```

---

## 测试 / Tests

运行快速单元测试（跳过慢速测试）：

```bash
uv run pytest tests/ -v
```

运行包含慢速集成测试的全部测试：

```bash
uv run pytest tests/ -v -m slow
```

---

## 参考 / References

- VALIS: Virtual Alignment of pathology Image Series
- DISK: Learning local features with policy gradient (Tyszkiewicz et al., NeurIPS 2020)
- LightGlue: Local Feature Matching at Light Speed (Lindenberger et al., CVPR 2023)
