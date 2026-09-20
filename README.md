# 语言引导目标定位与空间关系错误分析

目标：输入图片与描述（例如“找到笔记本左边的杯子。”），用已有文本引导检测模型输出目标框，在固定标注数据上评估定位，并分析漏检、参照物错误、空间关系错误。若错误分析支持，再尝试简单左右关系重排序，比较改动前后结果；不预设一定提高准确率，不从头训练大模型。

## 当前完成范围

今晚已实现主项目的 **IoU 评估与可视化模块**。图片是 Pillow 绘制的示意图，真实框和预测框都是手动坐标，JSON 使用 `prediction_source: "simulated"`。**当前程序没有理解文字或检测物体，结果不代表任何模型性能。** 描述目前只被读取、保留和打印。

尚未实现模型接入、真实数据基线、失败类型标注和关系重排序。没有安装 PyTorch 或 Grounding DINO，没有下载模型权重或大型数据集。

## 文件结构

```text
src/
  __init__.py
  evaluation.py          # 验证坐标、计算面积与 IoU，包含少量断言
  visualization.py       # 画框、图例和指标
data/examples/
  scene.png              # 小型示意图，需要保留在 Git 中
  samples.json           # 描述、真实框、模拟预测框、来源
  create_scene.py         # 可选：重建相同示意图
run_example.py           # 串联读取、评估、可视化与结果保存
outputs/                 # 生成结果，不加入版本控制
requirements.txt
.gitignore
README.md
```

## 最短运行步骤

先打开 `data/examples/samples.json` 看坐标，再读 `run_example.py`。本机已有 `.venv` 时，在项目根目录运行：

```bash
.venv/bin/python -m src.evaluation
.venv/bin/python run_example.py
```

从零设置 Mac（在项目目录运行）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.evaluation
.venv/bin/python run_example.py
```

Windows PowerShell（先安装 Python，进入项目目录）：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.evaluation
.\.venv\Scripts\python.exe run_example.py
```

不用激活环境，直接调用 `.venv` 内的 Python，避免误用全局解释器。**不要跨电脑复制 `.venv`；Windows 上重新创建并安装依赖。** 今晚模块只需 CPU，不依赖 GPU。Mac 当前使用 Python 3.13.9；明天需另行确认模型支持的 Python 版本、NVIDIA 型号和显存。

从其他工作目录运行时，给出解释器和脚本的完整路径即可，例如 Mac：

```bash
/Users/dandanliu/Desktop/project1/.venv/bin/python /Users/dandanliu/Desktop/project1/run_example.py
```

`ROOT = Path(__file__).resolve().parent` 表示入口脚本所在文件夹。JSON 的 `image_path` **相对于项目根目录**，不相对于终端当前目录。结果也始终写到项目的 `outputs/`。

## 评估约定与关键代码

框统一为 `[x_min, y_min, x_max, y_max]`，原点在原图左上角，x 向右、y 向下。使用连续坐标：面积 `(x_max-x_min)*(y_max-y_min)`，宽高不加 1。Pillow 绘图的像素边缘不参与面积计算。

`validate_box` 检查四个有限实数及坐标顺序；颠倒、NaN、无穷大等会抛出 `ValueError`。零宽或零高允许，但任一框面积为零时 IoU 返回 0。评估函数不按图片裁剪框；未来接入模型时应明确统一越界框的处理规则。

`evaluate_boxes(gt, pred)` 先分别计算两个框的面积。交集左边取两框左边的较大值，右边取右边的较小值；高度同理。`max(0, ...)` 将不相交的宽高变为 0。并集是两个面积之和减去交集，IoU 是交集除以并集。`calculate_iou(a, b)` 只返回 IoU，方便模型接入。

初版严格使用 **IoU > 0.5** 判定正确，等于 0.5 不通过。判断用完整数值，屏幕小数仅用于显示。单例的通过/不通过不是数据集准确率。

默认样本两个框面积均为 12000，交集 `75*105=7875`，并集 `24000-7875=16125`，IoU 约 `0.488372093`，定位不正确。

完整过程：`run_example.py` 用 `json.load` 读取样本 → 取两个框交给 `evaluate_boxes` → 将指标交给 `draw_result` → 原图上画绿色真实框和红色模拟预测框 → 保存 `outputs/simulated_scene_001_result.png` 及 `outputs/evaluation_results.json`。结果 JSON 保留描述、原始框和模拟来源。图例单独放在原图上方，绘图时只给 y 加上图例高度，评估坐标不变。

## 修改与复现

修改 `data/examples/samples.json` 中的 `prediction_box`，保存后重新运行入口。结果文件会被重新生成。不要为了让预测通过而修改真实框。

如需重建示意图（可选）：

```bash
.venv/bin/python data/examples/create_scene.py
```

断言覆盖重合、分离、边缘接触、部分重叠、零面积、非法坐标、恰好 0.5。部分重叠的手算案例：两个 10×10 框，第二个向右移 5，交集为 50，并集 150，因此 IoU 为 1/3。

## 明天如何接入真实模型

保留 `evaluation.py` 和 `visualization.py`。模型输出转换为四个普通 Python 数值后，将目标预测框传给 `evaluate_boxes(ground_truth_box, prediction_box)`，再交给绘图函数。真实预测使用 `prediction_source: "model"`，同时保存模型名称、版本、权重标识、提示词、选框规则和运行配置；切勿只把模拟样本的来源标签改掉。

模型可能输出 `[x,y,width,height]`、中心点格式 `[cx,cy,w,h]`、0–1 归一化坐标，或缩放/填充后图像坐标。必须转换回**原图像素 xyxy** 再评估。例如 xywh 转换为 `[x,y,x+w,y+h]`；归一化 xyxy 的 x 乘原图宽，y 乘原图高；有缩放或填充时还要先逆变换。

模型可能返回多个候选框或没有框。应提前固定选框规则、置信阈值和漏检处理，不能依据真实框挑最大 IoU 的候选。当前接口评估单个合法框；真实基线需补充“无预测”记录与统计，不能伪造预测框。

正式实验先固定样本清单，再按**图片 ID** 隔离开发集与留出评估集；同一图片的不同描述也必须在同一侧。只在开发集分析和调整规则，固定后再评估留出集，并将模拟演示与真实模型结果分开保存。后续错误类型分析需要查看案例，不能仅根据 IoU 自动认定是空间关系错误。

## Git

项目仓库：https://github.com/Dandan3455/language-guided-localization 。今晚的模块和后续模型、评估及错误分析代码在同一仓库持续开发。忽略虚拟环境、缓存、密钥文件、模型权重和大型下载目录，但保留小型示例数据。`outputs/` 只保留 `.gitkeep`。

检查后自行提交：

```bash
git status --short
git diff
git add README.md requirements.txt .gitignore src data/examples run_example.py outputs/.gitkeep
git diff --cached
git commit -m "Add simulated localization evaluation and visualization"
```

新文件未暂存时 `git diff` 不显示其内容，因此也要检查暂存后的差异。若 Git 提示缺少身份，可用 `git config user.name "你的名字"` 和 `git config user.email "你的邮箱"` 仅配置当前仓库，不加 `--global`。

## 两个小练习

保持真实框不变，先预测再修改并运行：

1. 将预测框设为 `[100,170,200,290]`：IoU 会怎样，判断会怎样？
2. 将预测框设为 `[150,170,250,290]`：手算交集宽高与并集，预测 IoU 和是否通过。

## 三个理解检查问题

1. 为什么并集要减去一次交集面积？
2. 若 IoU 恰好为 0.5，当前代码如何判定？哪行代码决定它？
3. 为什么同一张图片的不同描述不能分别放进开发集和留出评估集？
