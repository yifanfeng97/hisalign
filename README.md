# he2ihc_align

HE-to-IHC whole-slide image alignment pipeline.

从 HE 原始病理切片配准到多标记 IHC/荧光染色切片，并生成训练数据映射表。

---

## 项目目标 / Project Purpose

训练一个深度学习模型，能够**直接从原始 HE 切片预测对应的 20+ 个 IHC/荧光 marker 的染色图**。

本项目负责完成数据预处理的前半部分：

1. **配准与映射**：将 HE 切片与每个 marker 的 IHC 切片进行 WSI 级配准。
2. **坐标映射表生成**：对 HE 切片上的每一个 patch，计算其在每张 IHC 切片上的对应位置。
3. **标签对接**：将上述映射表输入下游的褐色点检测算法，得到每个 patch 的 `num_pos / num_neg / percent_pos` 等标签。

最终产出：一个结构化的 `mapping.csv`，记录 HE 上每个 patch 对应每张 IHC 的精确位置，供训练使用。

---

## 安装 / Installation

```bash
uv sync && uv pip install -e .
```

---

## 使用 / Usage

### 1. 单病例运行 / Single Case

```bash
he2ihc-align --case-dir /path/to/case/第一批 --output-dir ./outputs
```

### 2. 批量运行 / Batch Processing

```bash
he2ihc-align --config configs/default.yaml
```

### 3. 脚本方式 / Script Wrapper

```bash
uv run python scripts/run_pipeline.py --case-dir /path/to/case/第一批 --output-dir ./outputs
```

---

## 输出 CSV 格式 / Output CSV Format

`mapping.csv` 包含以下列：

| 列名 | 含义 |
|------|------|
| `patch_id` | patch 全局编号，格式为 `{slide_id}_{idx:04d}` |
| `slide_id` | 病例 ID |
| `marker` | IHC marker 名称 |
| `he_x`, `he_y` | HE patch 左上角坐标（level 0 像素坐标） |
| `he_w`, `he_h` | HE patch 尺寸 |
| `ihc_x`, `ihc_y` | IHC 上对应包络 bbox 左上角（level 0 像素坐标） |
| `ihc_w`, `ihc_h` | IHC 包络 bbox 尺寸 |
| `clipped` | 是否超出 IHC 切片边界 |

注意：非刚性变换不保持矩形，因此将 HE patch 四角映射到 IHC 后，取包络 bbox 作为 IHC 读取区域。

---

## 配置文件 / Configuration

`configs/default.yaml` 示例：

```yaml
data_root: /path/to/test_SCCE
output_root: ./outputs
batch_dir_name: "第一批"

patch_size: 512
stride: 512
he_level: 0
registration_level: 3
max_white_ratio: 0.95

max_image_dim_px: 1024
max_non_rigid_dim_px: 2048

mapping_csv_name: mapping.csv
viz_sample_n: 5
```

关键配置项说明：

| 配置项 | 说明 |
|--------|------|
| `data_root` | 病例根目录，每个子目录为一个病例 |
| `batch_dir_name` | 病例下的批次目录名（如 `"第一批"`） |
| `patch_size` | HE patch 尺寸（像素） |
| `stride` | patch 滑动步长 |
| `he_level` | HE 采样金字塔层级 |
| `registration_level` | 配准使用的金字塔层级 |
| `max_white_ratio` | 最大允许空白比例，超过则丢弃该 patch |
| `max_image_dim_px` | 刚性配准最大图像尺寸 |
| `max_non_rigid_dim_px` | 非刚性配准最大图像尺寸 |
| `viz_sample_n` | 可视化 gallery 中展示的 patch 数量，设为 0 则不生成 |

---

## 项目结构 / Project Structure

```text
he2ihc_align/
├── README.md                  # 本文件
├── pyproject.toml             # 项目依赖与配置
├── configs/
│   └── default.yaml           # 默认配置文件
├── src/
│   └── he2ihc_align/
│       ├── __init__.py
│       ├── cli.py             # 命令行入口
│       ├── case_io.py         # 病例发现工具
│       ├── mapping.py         # HE → IHC 坐标映射
│       ├── patching.py        # patch 采样与过滤
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
├── scripts/
│   └── run_pipeline.py        # 便捷脚本入口
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
