# 基于 RAG 架构的卤化物固态电解质知识问答系统

北京化工大学大学生创新创业训练计划项目。面向卤化物固态电解质(Halide Solid-State Electrolytes, HSSEs)领域的检索增强生成(RAG)问答系统: 用中文自然语言提问, 系统从本地文献库检索相关片段, 由大模型生成**带文献引用编号**的答案。

## 功能特性

- 本地文献知识库: PyMuPDF 抽取 + 清洗(含各出版商下载水印清除) + 语义分块(600字符/重叠120)
- 跨语言检索: paraphrase-multilingual-MiniLM-L12-v2 嵌入模型, 支持**中文提问检索英文文献**
- 混合检索: 向量语义排序 + 化学式精确匹配(`$contains` 过滤) 经 RRF 加权融合——解决向量模型对 Li3YCl6/Li3ScCl6 等相似化学式不敏感的问题
- 标题增强嵌入: 嵌入时拼接论文标题(标题含关键化学式与主题词), 库中存储仍为原文
- 可溯源回答: DeepSeek API 生成, 强制 [编号] 引用 + 文末参考文献列表, temperature=0.2 抑制幻觉
- 化学式归一化: 语料与查询统一为 ASCII 形式(Li₃YCl₆ → Li3YCl6), 避免格式差异导致漏检
- 量化评估: Top-K 检索召回率自动计算, 答案可导出供人工核对引用准确率

## 项目结构

```
pdfs/                文献PDF (版权原因不入库, 见 manifest.csv 清单)
data/chroma_db/      Chroma 向量库 (构建产物, 不入库)
data/test_questions.json  评估测试题集
src/ingest.py        PDF 抽取 + 清洗 + 化学式归一化 + 分块
src/build_db.py      嵌入 + Chroma 建库
src/qa.py            检索 + Prompt 组装 + DeepSeek 生成 (ask 主逻辑)
src/evaluate.py      召回率评估 + 答案导出
notebooks/demo.ipynb 交互式问答演示
```

## 快速开始

```bash
# 1. 环境 (Python 3.10+)
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS

# 2. 配置 DeepSeek API Key
cp .env.example .env   # 然后编辑 .env 填入密钥

# 3. 建库 (PDF 放入 pdfs/ 后)
python src/ingest.py     # -> data/chunks.jsonl
python src/build_db.py   # -> data/chroma_db/

# 4. 提问
python src/qa.py "Li3YCl6的室温离子电导率是多少?"
python src/qa.py         # 交互模式

# 5. 评估
python src/evaluate.py              # 检索召回率
python src/evaluate.py --with-llm   # 生成答案供人工核对引用准确率
```

首次运行会从 hf-mirror.com 下载约 470MB 的嵌入模型。

## 设计说明

- **嵌入模型**: 申报书原定 all-MiniLM-L6-v2 仅支持英文; 因实际使用以中文提问为主, 改用同家族的多语言版本 paraphrase-multilingual-MiniLM-L12-v2 (跨语言句向量, CPU 可运行)。
- **化学式归一化方向**: 统一为 ASCII 数字而非申报书所述下标格式——用户键盘输入天然是 ASCII (Li3YCl6), 且 ASCII 数字在嵌入模型中分词更稳定。语料与查询两端做同一变换, 检索一致性不受影响。
- **图表数据边界**: 系统聚焦文本中明确陈述的理化指标; 仅存在于图表中的数据(如 Nyquist 拟合参数)不在覆盖范围, 系统会提示查阅原文。
- **已知坑(chromadb 1.x)**: Rust 后端对**非 ASCII 绝对路径**有 bug——传入含中文的绝对路径时 HNSW 索引文件写不出来(库静默损坏, 查询报 `Error loading hnsw index`)。代码已统一切换工作目录并使用相对路径规避; 若迁移项目, 注意不要把绝对中文路径直接传给 Chroma。
- **评估说明**: `data/test_questions.json` 当前 12 题, Top-3 召回率 100% (12/12); 题集偏小, 建议后续与导师扩充至 20+ 题再作为正式指标。

## 许可与说明

文献 PDF 仅供课题组内部科研学习使用, 不随仓库分发。
