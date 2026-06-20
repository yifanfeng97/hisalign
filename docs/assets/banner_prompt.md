# Banner Prompt for HISAlign

> 可直接把下面的英文/中英混合提示词丢给 Gemini（或任何支持图像生成的模型）来生成仓库 banner。

---

## 推荐提示词（英文，模型理解更稳）

```text
A high-quality, modern, minimalist hero banner for a GitHub open-source project named "HISAlign".

Theme: computational alignment of histopathology whole-slide images (H&E stain and IHC immunohistochemistry stain).

Visual style:
- Flat vector illustration with subtle gradients, clean tech aesthetic, no photorealism, no gore, no realistic cells.
- Dark navy-to-charcoal background (#0f172a to #1e293b) with very subtle glowing particles.
- Two overlapping semi-transparent microscope slide rectangles, one tinted in H&E pink/magenta (#e94560 / #a855f7), the other in IHC blue/counterstain (#3b82f6 / #f59e0b for positive markers).
- The slides should look slightly misaligned at first, with thin alignment grid lines and small glowing anchor points at the corners showing they are being registered.
- A few abstract deformation-field flow lines or a soft mesh grid between the slides to suggest non-rigid warping.
- Small bidirectional arrows implying coordinate mapping between the two slides.
- Accent colors: teal (#14b8a6) and soft white for highlights.

Composition:
- Wide aspect ratio, 1440 x 320 pixels, landscape orientation.
- Main visual centered or slightly right-of-center; left side kept relatively clean so the project title "HISAlign" can be overlaid later.
- No text in the image.
- Safe for professional academic / biomedical software presentation.

Style keywords: vector art, minimalist, scientific illustration, dark mode, neon accents, registration, alignment, pathology, whole-slide imaging.
```

---

## 中文提示词（备用）

```text
为 GitHub 开源项目 "HISAlign" 生成一张高质量的现代简约风格横幅图。

主题：病理全切片图像（H&E 染色与 IHC 免疫组化染色）的计算机配准。

视觉风格：
- 扁平矢量插画，带微妙渐变，科技感、干净，不要写实、不要血腥、不要真实细胞。
- 深蓝到炭黑渐变背景（#0f172a 到 #1e293b），带 faint 发光粒子。
- 两张半透明的载玻片矩形相互交叠：一张偏 H&E 粉紫色（#e94560 / #a855f7），另一张偏 IHC 蓝/棕阳性标记色（#3b82f6 / #f59e0b）。
- 载玻片微微错位，角点有发光对齐锚点，中间有薄网格或形变场流线，暗示非刚性配准。
- 加入双向小箭头，表示坐标映射。
- 点缀色：青绿色（#14b8a6）和白色高光。

构图：
- 宽屏比例，1440 x 320 像素，横向。
- 主视觉居中或略偏右，左侧留出较干净区域以便后续叠加项目名称 "HISAlign"。
- 图片内不要出现文字。
- 适合学术/生物医学软件的专业展示。

风格关键词：矢量插画、极简、科学插画、深色模式、霓虹点缀、配准、对齐、病理、全切片成像。
```

---

## 输出建议

- 首选 **1440 × 320 px PNG**，同时可要求提供 **2880 × 640 px** 的 Retina 版本。
- 生成后替换 `docs/assets/banner.svg` 为新生成的 `banner.png`（或 `banner.svg`），并同步修改 `README.md` 与 `README.zh.md` 中的图片链接。
- 如果生成模型不擅长文字，请让 banner **不含文字**，由 README 中的 HTML `<h1>` 来显示标题。
