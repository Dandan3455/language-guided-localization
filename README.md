# 语言引导目标定位与空间关系错误分析

目标：输入图片与描述（例如“找到笔记本左边的杯子。”），用已有文本引导检测模型输出目标框，在固定标注数据上评估定位，并分析漏检、参照物错误、空间关系错误。若错误分析支持，再尝试简单左右关系重排序，比较改动前后结果；不预设一定提高准确率，不从头训练大模型。

## 当前完成范围

**已跑通一个真实样本的完整流程：图片和文字 → 模型候选框 → 固定规则选框 → 与人工标注比较 → IoU 评分与可视化。** 当前使用别人训练好的 Grounding DINO Tiny 权重做推理，没有训练或更新模型权重。写代码、下载模型和运行预测都不等于训练模型。

`run_detection.py` 输出全部候选框和分数；`annotate.html` 用于人工标注正确框；`evaluate_detection.py` 按固定规则选一个候选，再调用现有评分和画图模块。尚未完成多样本真实数据基线、失败类型标注和关系重排序。

原有 `run_example.py` 仍保留为独立的模拟示例：使用 Pillow 绘制的示意图，正确框和预测框都是手写坐标，来源为 `simulated`。描述只被读取、保留和打印，结果不代表模型性能。

## 项目现在怎样工作

```text
原图 + 英文 query                         同一张原图
        ↓                                     ↓
run_detection.py                        annotate.html
        ↓                                     ↓
模型候选框 JSON                         人工正确框 JSON
        └─────────────────┬───────────────────┘
                          ↓
                evaluate_detection.py
              选择置信分数最高的目标候选
                          ↓
              src/evaluation.py 计算 IoU
                          ↓
          src/visualization.py 画对比图和评分
```

参与比较的只有两个框：**人工标注的参考答案**和**按固定规则选出的模型预测框**。没有额外一份“最准确的数据”，也不将预测和标注合并生成正确答案。人工标注可能有误差，应独立、仔细地完成，不为提高分数而修改答案。

Query 是模型收到的文字，可以用 `--text` 修改；标注文件中的中文描述不会自动传给模型或翻译。当前选框规则本身不判断左右位置，只在标签包含目标单词的候选里选择最高分。

| 数值 | 含义 |
| --- | --- |
| 置信分数 | 模型给候选的分数，用于选框 |
| IoU | 选中的预测框与人工正确框的重合程度 |
| 数据集准确率 | 固定样本集合中定位正确的比例，目前尚未统计 |

**IoU > 0.5 → CORRECT；IoU ≤ 0.5 → INCORRECT。** 当前桌面样本的 IoU 约 0.907574，定位正确；这不等于整体准确率为 90.8%，也不能证明模型理解了“左边”。

## Windows 真实模型入门

### 1. 准备图片并用鼠标标注正确框

将图片放入 `data/downloads/`。当前例子使用 `desk.png`，图中笔记本左右各有一个杯子。下载图片不提交 Git；克隆仓库后需要自行准备原图。已有标注只适用于原来的图片，更换图片必须重新标注，正式实验还需记录图片来源及使用许可。

在 Windows 文件资源管理器中双击项目根目录的 `annotate.html`，用浏览器打开（无需安装依赖或启动服务器）。默认显示 `data/downloads/desk.png`；如果未显示，点击“选择其他图片”选择原图。

1. 确认图片路径和描述。在原图上按住鼠标左键拖出矩形，框住笔记本左边的完整杯子，包含杯柄。
2. 松开后显示原图像素坐标 `[x_min, y_min, x_max, y_max]`。可反向拖动或重新拖框；页面缩放不改变坐标含义。
3. 点击“下载标注 JSON”，把下载的文件放到项目 `data/annotations/`。默认文件名为 `desk_left_cup_001.json`。

页面只显示原图，不显示模型预测。导出的 JSON 是样本列表，包含图片路径、描述、人工正确框、原图尺寸和 `annotation_source: "manual"`，不包含预测框。更换图片后，请确认路径确实指向项目里的对应图片。

当前工具的说明和导出文件名后缀针对左侧杯子示例。标注右侧或其他目标时，还需同步修改描述、导出 JSON 的 `sample_id` 和文件名，防止不同样本覆盖或混用。下载后需手动将 JSON 放到项目目录。

### 2. 根据 query 运行真实模型

已安装依赖后，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 run_detection.py --image data/downloads/desk.png --text "the cup to the left of the laptop."
```

该命令要求找笔记本左边的杯子。将 `left` 改成 `right` 可改变要求。**不传 `--text` 时，程序默认找 `a cat. a remote control.`，不是左边的杯子。** 首次安装方法见后文。

打开 `outputs/detection/result.png` 查看所有候选框；`outputs/detection/results.json` 保存完整候选、分数和实际 prompt。再次运行会覆盖这些文件，做 query 对照实验前应将每次结果另存到独立目录。

### 3. 对一个真实样本评分

将标注放到 `data/annotations/desk_left_cup_001.json`，并确认 `outputs/detection/results.json` 是同一张 `desk.png` 的预测后运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 evaluate_detection.py
```

该入口按固定规则选择标签包含完整单词 `cup` 的最高分候选（同分取第一个），不使用正确框选候选。然后复用 `evaluation.py` 和 `visualization.py` 计算 IoU、生成对比图。默认文件保存到 `outputs/evaluation/desk_left_cup_001_result.png` 和 `desk_left_cup_001_evaluation.json`，JSON 保留完整标注、预测来源和选框规则。没有目标候选时记录空预测、IoU 0 和失败，不伪造框，也不生成新的对比图。可通过 `--annotation`、`--prediction` 和 `--target` 指定输入。

这是开发样本的端到端检查，不是数据集准确率，也不能证明模型理解了左右关系。当前用户标注与最高分杯子框的 IoU 约为 0.907574。

当前人工正确框为 `[152, 297, 241, 377]`，选中预测框约为 `[150.616, 293.811, 238.738, 375.578]`，该候选的置信分数约为 0.568。绿色显示人工框，红色显示预测框。

评分程序只读取已保存的预测，不重新运行模型。它检查图片路径和尺寸，但不会自动确认中文描述与英文 prompt 的语义一致，操作者需确认两者指定同一目标。当前每次只支持一个标注样本。没有候选时不生成新图片，旧同名图片可能仍存在，应以本次 JSON 的 `visualization_path` 为准。

### 安装和运行模型

使用 Python 3.12，模型为 `IDEA-Research/grounding-dino-tiny`，通过 Transformers 接入。本机安装组合为 PyTorch 2.6.0 / torchvision 0.21.0（CUDA 12.4）和 Transformers 4.51.3；其他电脑应根据驱动选择 PyTorch 安装包。

在项目根目录的 PowerShell 中依次执行（已经安装后不必重复）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-model.txt
```

用自己的照片运行：

```powershell
.\.venv\Scripts\python.exe run_detection.py --image "D:\photos\desk.jpg" --text "a cup. a laptop."
```

`--image` 是照片路径，`--text` 是英文提示词。首次运行自动下载权重到项目的 `models/huggingface/`，以后复用缓存。结果为 `outputs/detection/result.png` 和 `outputs/detection/results.json`，再次运行会覆盖它们。JSON 保存候选框、分数、提示词、模型版本和阈值；没有候选时如实记录。框是未裁剪的原图像素 xyxy 坐标。

测试照片来自 [Transformers 官方教程](https://huggingface.co/docs/transformers/v4.51.3/model_doc/grounding-dino) 使用的 COCO 图片 `000000039769.jpg`，本地路径为 `data/downloads/cats.jpg`（不提交 Git）。下载后可运行：

```powershell
.\.venv\Scripts\python.exe run_detection.py --image data/downloads/cats.jpg
```

此演示只验证模型能运行，不评估空间关系或定位准确率；正式实验仍需固定标注数据和选框规则。安装来源：[PyTorch 官方版本说明](https://docs.pytorch.org/get-started/previous-versions/)。

本机已完成 CUDA 推理验证：上述照片在默认阈值下输出两只猫和一个遥控器，共 3 个候选框。依赖检查和原有评估断言通过。Windows 终端若输出中文报 `UnicodeEncodeError`，在命令的 `python.exe` 后添加 `-X utf8`。

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
data/annotations/        # 人工标注的正确框 JSON
data/downloads/          # 本地真实图片 desk.png、cats.jpg，不提交 Git
annotate.html            # 鼠标拖框页面，下载人工标注 JSON
run_detection.py         # 模型推理，保存全部候选并自行画框
evaluate_detection.py    # 读取真实预测和标注，选框、评分、画对比图
run_example.py           # 串联读取、评估、可视化与结果保存
outputs/                 # 生成结果，不加入版本控制
models/                  # 模型缓存，不加入版本控制
.venv/                   # Python 环境与已安装工具，不加入版本控制
requirements.txt
requirements-model.txt   # 真实模型依赖清单
.gitignore
README.md
```

## 模拟示例的运行步骤

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

不用激活环境，直接调用 `.venv` 内的 Python，避免误用全局解释器。**不要跨电脑复制 `.venv`；在新电脑上重新创建并安装依赖。** 模拟示例只需 CPU。真实模型在本机 Python 3.12、8 GB 显存的 RTX 3070 系列 GPU 上完成过 CUDA 推理验证。

从其他工作目录运行时，给出解释器和脚本的完整路径即可，例如 Mac：

```bash
/Users/dandanliu/Desktop/project1/.venv/bin/python /Users/dandanliu/Desktop/project1/run_example.py
```

`ROOT = Path(__file__).resolve().parent` 表示入口脚本所在文件夹。模拟与人工标注 JSON 的 `image_path` **相对于项目根目录**；真实检测 JSON 保存原图绝对路径。命令行传入的相对路径以终端当前目录为起点，因此建议在项目根目录运行。结果始终写到项目的 `outputs/`。

## 评估约定与关键代码

框统一为 `[x_min, y_min, x_max, y_max]`，原点在原图左上角，x 向右、y 向下。使用连续坐标：面积 `(x_max-x_min)*(y_max-y_min)`，宽高不加 1。Pillow 绘图的像素边缘不参与面积计算。

`validate_box` 检查四个有限实数及坐标顺序；颠倒、NaN、无穷大等会抛出 `ValueError`。零宽或零高允许，但任一框面积为零时 IoU 返回 0。真实评分入口另要求人工框面积大于零且位于图内。目前不按图片裁剪模型框，正式实验应保持统一规则。

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

## 模型接口与实验约定

保留 `evaluation.py` 和 `visualization.py`。模型输出转换为四个普通 Python 数值后，将目标预测框传给 `evaluate_boxes(ground_truth_box, prediction_box)`，再交给绘图函数。真实预测使用 `prediction_source: "model"`，同时保存模型名称、版本、权重标识、提示词、选框规则和运行配置；切勿只把模拟样本的来源标签改掉。

模型可能输出 `[x,y,width,height]`、中心点格式 `[cx,cy,w,h]`、0–1 归一化坐标，或缩放/填充后图像坐标。必须转换回**原图像素 xyxy** 再评估。例如 xywh 转换为 `[x,y,x+w,y+h]`；归一化 xyxy 的 x 乘原图宽，y 乘原图高；有缩放或填充时还要先逆变换。

模型可能返回多个候选框或没有框。应提前固定选框规则、置信阈值和漏检处理，不能依据真实框挑最大 IoU 的候选。当前检测后处理阈值为 `box_threshold=0.4` 和 `text_threshold=0.3`，它们不是 IoU 判定阈值。评分入口已记录无目标预测的情况，多样本汇总统计仍未实现。

正式实验先固定样本清单，再按**图片 ID** 隔离开发集与留出评估集；同一图片的不同描述也必须在同一侧。只在开发集分析和调整规则，固定后再评估留出集，并将模拟演示与真实模型结果分开保存。后续错误类型分析需要查看案例，不能仅根据 IoU 自动认定是空间关系错误。

## 下一步：空间关系对照实验

1. 使用同一张图片，分别输入“左边的杯子”“右边的杯子”和“杯子”，保存各次候选框、分数和最终选择。
2. 左右任务分别人工标注正确目标；无关系词且图中有多个杯子的描述可用于观察检测结果，不能随意指定唯一正确答案。
3. 增加图片并固定样本、提示词、选框规则和标注标准，统计表现并查看失败原因。不能仅凭 IoU 低判断为空间关系错误。
4. 若证据支持，在开发集尝试基于目标与参照物坐标的左右关系重排序，再在留出集比较。这属于“模型 + 空间规则”，要与模型直接预测明确区分。

目前完成的是实验工具和单样本验证，多样本评估、错误分析和改进对比仍未完成。

## Git

项目仓库：https://github.com/Dandan3455/language-guided-localization 。代码在同一仓库持续开发。忽略虚拟环境、缓存、密钥文件、模型权重、下载图片和生成结果，保留小型模拟数据及人工标注。`outputs/` 只保留 `.gitkeep`。当前真实样本依赖本地原图和预测文件，仓库尚未包含可独立复现的完整真实评估数据集。

检查后按实际改动范围提交，例如仅更新 README：

```bash
git status --short
git diff
git add README.md
git diff --cached
git commit -m "Update project workflow and evaluation documentation"
git push origin main
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
