"""卤化物固态电解质知识问答系统 - 网页界面

启动:
    python app.py                 本地使用 (http://127.0.0.1:7860)
    set HSSE_SHARE_LAN=1 && python app.py    局域网分享 (同学通过你的IP访问)
    或直接双击 "启动问答网页.bat"

功能: 聊天问答(流式) + 用户上传文献PDF(自动入库并可被检索) + 文献库面板
密钥: 服务端统一使用 .env 中的 DEEPSEEK_API_KEY, 用户无需填写。
"""

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from qa import RAGEngine  # noqa: E402
from citations import export_citations  # noqa: E402  (须最先导入: 设置 HF 离线环境变量)
from ingest import load_titles  # noqa: E402
from user_upload import UPLOAD_DIR, list_uploads, process_pdf, save_and_register  # noqa: E402

import gradio as gr  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

SHARE_LAN = os.environ.get("HSSE_SHARE_LAN", "0") == "1"
PORT = int(os.environ.get("HSSE_PORT", "7860"))
# 公网部署保护: 设置 HSSE_ACCESS_PASSWORD 后访问网页需登录(用户名默认 hsse);
# 设置 HSSE_UPLOAD_PASSWORD 后上传文献需单独口令(不设置则上传免费开放)
ACCESS_PASSWORD = os.environ.get("HSSE_ACCESS_PASSWORD", "")
ACCESS_USER = os.environ.get("HSSE_ACCESS_USER", "hsse")
UPLOAD_PASSWORD = os.environ.get("HSSE_UPLOAD_PASSWORD", "")

engine = RAGEngine(top_k=4)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""]
)

EXAMPLES = [
    "Li3YCl6的室温离子电导率是多少? 不同合成方法有差异吗?",
    "氯化物和溴化物电解质的电化学窗口分别是多少?",
    "Li3Ta3O4Cl10氧卤化物的室温离子电导率达到多少?",
    "UCl3型结构与Li3MCl6型在结构上有什么本质区别?",
    "非晶氯化物固态电解质的离子电导率可以达到多少?",
    "卤化物电解质与高电压正极的界面稳定性存在什么问题?",
]


# ---------- 聊天 ----------

def format_hits(hits: list | None) -> str:
    """把检索到的文献片段渲染为可核验的 Markdown。"""
    if not hits:
        return "*暂无*"
    parts = []
    for i, h in enumerate(hits, 1):
        text = h["text"][:350] + ("……" if len(h["text"]) > 350 else "")
        parts.append(
            f"**[{i}]** 《{h['title']}》  \n"
            f"来源: `{h['source']}` | 相关度: {h['similarity']:.2f}\n\n"
            f"> {text}\n"
        )
    return "\n---\n".join(parts)


def respond(message: str, history: list):
    if not message.strip():
        yield "", history, gr.update()
        return
    past_turns = list(history)  # 本轮之前的对话, 用于追问改写
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    try:
        for partial in engine.ask_stream(message, history=past_turns):
            history[-1]["content"] = partial
            yield "", history, "🔍 检索与生成中……"
        yield "", history, format_hits(getattr(engine, "last_hits", None))
    except SystemExit as e:
        history[-1]["content"] = f"系统配置错误: {e}"
        yield "", history, gr.update()
    except Exception as e:
        history[-1]["content"] = f"出错了: {e}\n\n请检查网络连接与 DeepSeek API 额度。"
        yield "", history, gr.update()


# ---------- 文献库面板 ----------

def do_export_citations(fmt: str):
    hits = getattr(engine, "last_hits", None)
    if not hits:
        return gr.update(visible=True, value="请先提问, 再导出本次回答用到的参考文献")
    sources = [h["source"] for h in hits]
    return gr.update(visible=True, value=export_citations(sources, fmt))

def library_stats() -> str:
    total = engine.count_chunks()
    user = engine.count_chunks(lib="user")
    n_uploads = len(list_uploads())
    n_core_pdfs = len(list((PROJECT_ROOT / "pdfs").glob("*.pdf")))
    return (
        f"📚 **核心文献库**: {n_core_pdfs} 篇文献 / {total - user} 个文本块\n\n"
        f"📤 **用户上传库**: {n_uploads} 篇文献 / {user} 个文本块\n\n"
        f"🔢 **向量库总量**: {total} 个文本块"
    )


def core_library_table() -> list[list[str]]:
    pdf_dir = PROJECT_ROOT / "pdfs"
    titles = load_titles(pdf_dir)
    rows = [
        [titles.get(p.name, p.stem), p.name] for p in sorted(pdf_dir.glob("*.pdf"))
    ]
    curated = PROJECT_ROOT / "data" / "curated"
    for t in sorted(curated.glob("*.txt")):
        rows.append([t.read_text(encoding="utf-8").splitlines()[0][:60], f"curated/{t.name}"])
    return rows


def uploads_table() -> list[list[str]]:
    return [
        [u["title"], u["filename"], str(u["n_chunks"]), u["uploaded_at"]]
        for u in list_uploads()
    ]


def refresh_library():
    return library_stats(), core_library_table(), uploads_table()


# ---------- 上传 ----------

def upload_pdf(file, upload_pwd, progress=gr.Progress()):
    if UPLOAD_PASSWORD and upload_pwd != UPLOAD_PASSWORD:
        yield "⚠️ 上传口令错误, 请向管理员索取", *refresh_library()
        return
    if file is None:
        yield "⚠️ 请先选择 PDF 文件", *refresh_library()
        return
    src = Path(file)
    if src.suffix.lower() != ".pdf":
        yield "⚠️ 只支持 PDF 文件", *refresh_library()
        return

    UPLOAD_DIR.mkdir(exist_ok=True)
    dest = UPLOAD_DIR / src.name
    shutil.copy(src, dest)
    yield f"⏳ 已接收 **{src.name}**, 正在抽取文本...", *refresh_library()

    try:
        progress(0.4, desc="文本抽取与分块")
        result = process_pdf(dest, splitter)
        if not result["records"]:
            yield f"⚠️ **{src.name}** 未能抽取出文本(可能是扫描件), 未入库", *refresh_library()
            return
        progress(0.7, desc="向量嵌入与入库")
        engine.add_documents(result["records"])
        save_and_register(dest, len(result["records"]), result["title"])
    except Exception as e:
        yield f"❌ 处理失败: {e}", *refresh_library()
        return

    yield (
        f"✅ **{result['title']}** 已入库: {result['n_pages']} 页 → "
        f"{len(result['records'])} 个文本块, 现在起可被检索",
        *refresh_library(),
    )


# ---------- 界面 ----------

CSS = """
.header { text-align: center; padding: 12px 0 4px; }
.header h1 { margin-bottom: 4px; }
.header p { color: #666; margin-top: 0; }
.stats-box { background: #f0f4ff; border-radius: 10px; padding: 10px 14px; }
"""

theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

GUIDE_MD = """
## 这是什么

**卤化物固态电解质知识问答系统**——北京化工大学大学生创新创业训练计划（大创）项目作品。
它是一个面向卤化物固态电解质（Halide Solid-State Electrolytes, HSSEs）这一全固态锂电池核心材料体系的
**检索增强生成（RAG）问答助手**：你用中文自然语言提问，系统从本地学术文献库中检索最相关的片段，
再由大语言模型（DeepSeek）严格依据这些片段生成答案，并以 **[编号]** 标注引用、在末尾列出参考文献列表。

与通用聊天 AI 的区别：**它只依据文献库里的内容回答**。库里没有的数据，它会明确说"未在文献中找到"，
而不是编造一个看似合理的数字——这是它作为科研工具的核心设计。

## 数据从哪来

**核心文献库**（项目组精选，覆盖卤化物电解质全部主要体系）：

| 体系 | 代表内容 |
|---|---|
| 骨架综述 | Li & Du, *ACS Nano* 2025 系统综述（覆盖氟/氯/溴/碘化物及氧卤化物全类别） |
| 氯化物 | Asano 2018（Li3YCl6 开山之作）、Li3ScCl6、Li3MCl6 合成与结构、三角结构设计（*Science* 2023）等 |
| 溴化物 | Li3HoBr6 合成与性能（*Nano Lett.* 2021）等 |
| 氟化物 | 氟化物筛选与锂金属氟化物离子导通 |
| 氧卤化物 | LiTaOCl4 类（>10 mS/cm）、Li3Ta3O4Cl10（*Science* 2025，13.7 mS/cm 纪录）、非晶氧卤化物 |
| 其他 | LaCl3/UCl3 型非密堆结构、非晶氯化物、多阳离子混合体系、界面稳定性、合成方法学 |

此外还包括一份**人工校准的性能数据表**（整理自上述综述 Table 1，55 种材料的
离子电导率/活化能/电子电导率/电化学窗口），用于弥补 PDF 表格抽取的天然误差。

**用户上传库**：你在右侧上传的 PDF 会自动经过同样的"文本抽取→清洗→分块→向量化"流程进入知识库，
立即参与后续所有问答的检索（上传内容对所有使用者可见，请勿上传涉密文件）。

## 怎么用

1. **提问**：左侧输入框输入问题回车（或点"提问"），答案流式输出；下方示例问题可点击填入。
2. **读懂答案**：论断后的 [1][2] 是引用编号，答案末尾的参考文献列表给出对应文献；
   右侧面板可查看核心库清单和你已上传的文献。
3. **上传文献**：右侧选择 PDF（需上传口令，向管理员索取），等待处理完成提示"已入库"后即可针对它提问。

**提问技巧**：
- 好问题 = 具体材料 + 具体指标，如"Li3InCl6的室温离子电导率是多少？""氯化物和溴化物的电化学窗口分别是多少？"
- 化学式直接用键盘输入 ASCII 形式（Li3YCl6），无需下标格式，系统会自动匹配
- 太大的问题（如"讲讲固态电池"）效果不如具体问题；一次问一个点

## 能力边界（重要）

- 系统只处理**文本信息**：论文中的电化学阻抗谱、XRD 图谱等图表数据无法读取，
  遇到这类数据请按引用去查原文
- 检索和生成仍可能出错：**重要数据写入论文/报告前，请务必按引用编号核对原文**
- 知识范围限于库内文献，新发表的论文可能未收录（可上传补充）

## 版权与免责

本系统仅供科研学习交流使用。文献库内容来自出版商正版渠道（高校图书馆订阅），
请勿将系统内文献内容用于任何再分发。答案由 AI 生成，仅供参考，不构成科研结论。

## 项目信息

- 项目：北京化工大学大学生创新创业训练计划 · 创新训练项目
- 代码开源：GitHub @IceyT520
- 技术支持：遇到异常（长时间无响应/报错）请联系项目组，附上你的问题截图
"""

with gr.Blocks(title="卤化物固态电解质问答系统", theme=theme, css=CSS) as demo:
    gr.Markdown(
        "# ⚗️ 卤化物固态电解质知识问答系统\n"
        "基于 RAG 架构的领域问答助手 — 答案仅依据文献库内容, 带 [编号] 引用, 未收录的数据会明确告知。",
        elem_classes=["header"],
    )

    with gr.Tabs():
        with gr.Tab("💬 问答"):
            with gr.Row():
                # 左侧: 聊天
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=520, label="问答")
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="请输入问题, 如: Li3HoBr6的离子电导率是多少?",
                            scale=8, show_label=False, container=False,
                        )
                        send_btn = gr.Button("提问", variant="primary", scale=1)
                    gr.Examples(EXAMPLES, msg, label="示例问题 (点击填入)")
                    with gr.Accordion("📎 本回答依据的文献片段 (点击展开核验)", open=False):
                        chunks_md = gr.Markdown("*提问后此处显示答案所依据的原文片段*")
                    with gr.Row():
                        cite_fmt = gr.Dropdown(
                            choices=[("GB/T 7714 (国标)", "gbt7714"), ("BibTeX", "bibtex")],
                            value="gbt7714", label="参考文献导出格式", scale=2,
                        )
                        cite_btn = gr.Button("📋 导出本回答的参考文献", scale=2)
                    cite_out = gr.Textbox(label="参考文献 (可直接复制)", lines=4, visible=False)

                # 右侧: 文献库 + 上传
                with gr.Column(scale=2):
                    stats_md = gr.Markdown(elem_classes=["stats-box"])
                    with gr.Accordion("📖 查看核心文献库清单", open=False):
                        core_df = gr.Dataframe(
                            headers=["标题", "文件"], col_count=2,
                            interactive=False, wrap=True,
                        )
                    gr.Markdown("### 📤 上传我的文献 (PDF)")
                    gr.Markdown("上传后自动抽取、分块、入库, 立即可被检索。", elem_classes=["hint"])
                    file_input = gr.File(file_types=[".pdf"], label="选择PDF", type="filepath")
                    if UPLOAD_PASSWORD:
                        upload_pwd = gr.Textbox(
                            label="上传口令", type="password",
                            placeholder="此服务已开启上传保护, 请输入口令",
                        )
                    else:
                        upload_pwd = gr.State("")
                    upload_btn = gr.Button("上传到文献库", variant="secondary")
                    upload_status = gr.Markdown()
                    with gr.Accordion("🗂 我上传的文献", open=True):
                        uploads_df = gr.Dataframe(
                            headers=["标题", "文件名", "块数", "上传时间"], col_count=4,
                            interactive=False, wrap=True,
                        )
                    refresh_btn = gr.Button("🔄 刷新文献库", size="sm")

        with gr.Tab("📘 使用指南"):
            gr.Markdown(GUIDE_MD)

    # 事件
    send_btn.click(respond, [msg, chatbot], [msg, chatbot, chunks_md])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, chunks_md])
    cite_btn.click(do_export_citations, cite_fmt, cite_out)
    upload_btn.click(
        upload_pdf, [file_input, upload_pwd],
        [upload_status, stats_md, core_df, uploads_df],
    )
    refresh_btn.click(refresh_library, None, [stats_md, core_df, uploads_df])
    demo.load(refresh_library, None, [stats_md, core_df, uploads_df])

if __name__ == "__main__":
    server = "0.0.0.0" if SHARE_LAN else "127.0.0.1"
    auth = (ACCESS_USER, ACCESS_PASSWORD) if ACCESS_PASSWORD else None
    print(f"启动网页服务: http://{server}:{PORT}"
          + (" (局域网模式, 同学可通过你的IP访问)" if SHARE_LAN else " (本地模式)"))
    if auth:
        print(f"访问口令已开启: 用户名 {ACCESS_USER}")
    demo.launch(server_name=server, server_port=PORT, auth=auth)
