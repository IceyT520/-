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

from qa import RAGEngine  # noqa: E402  (须最先导入: 设置 HF 离线环境变量)
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

def respond(message: str, history: list):
    if not message.strip():
        yield "", history
        return
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    try:
        for partial in engine.ask_stream(message):
            history[-1]["content"] = partial
            yield "", history
    except SystemExit as e:
        history[-1]["content"] = f"系统配置错误: {e}"
        yield "", history
    except Exception as e:
        history[-1]["content"] = f"出错了: {e}\n\n请检查网络连接与 DeepSeek API 额度。"
        yield "", history


# ---------- 文献库面板 ----------

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

with gr.Blocks(title="卤化物固态电解质问答系统", theme=theme, css=CSS) as demo:
    gr.Markdown(
        "# ⚗️ 卤化物固态电解质知识问答系统\n"
        "基于 RAG 架构的领域问答助手 — 答案仅依据文献库内容, 带 [编号] 引用, 未收录的数据会明确告知。",
        elem_classes=["header"],
    )

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

    # 事件
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
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
