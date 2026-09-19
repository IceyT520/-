"""卤化物固态电解质知识问答系统 - 网页界面

启动:
    python app.py                 本地使用 (http://127.0.0.1:7860)
    set HSSE_SHARE_LAN=1 && python app.py    局域网分享 (同学通过你的IP访问)
    或直接双击 "启动问答网页.bat"

密钥: 服务端统一使用 .env 中的 DEEPSEEK_API_KEY, 用户无需填写。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gradio as gr

from qa import RAGEngine

SHARE_LAN = os.environ.get("HSSE_SHARE_LAN", "0") == "1"
PORT = int(os.environ.get("HSSE_PORT", "7860"))

engine = RAGEngine(top_k=4)


def respond(message: str, history: list):
    if not message.strip():
        yield "请输入问题。"
        return
    try:
        yield from engine.ask_stream(message)
    except SystemExit as e:
        yield f"系统配置错误: {e}"
    except Exception as e:
        yield f"出错了: {e}\n\n请检查网络连接与 DeepSeek API 额度。"


demo = gr.ChatInterface(
    respond,
    title="卤化物固态电解质知识问答系统",
    description=(
        "基于 RAG 架构的领域问答助手: 从卤化物固态电解质文献库检索相关片段, "
        "由 DeepSeek 生成带 [编号] 文献引用的答案。答案仅依据文献内容, "
        "未收录的数据会明确告知, 不会编造。"
    ),
    examples=[
        "Li3YCl6的室温离子电导率是多少? 不同合成方法有差异吗?",
        "氯化物和溴化物电解质的电化学窗口分别是多少?",
        "Li3Ta3O4Cl10氧卤化物的室温离子电导率达到多少?",
        "UCl3型结构与Li3MCl6型在结构上有什么本质区别?",
        "非晶氯化物固态电解质的离子电导率可以达到多少?",
        "卤化物电解质与高电压正极的界面稳定性存在什么问题?",
    ],
    textbox=gr.Textbox(
        placeholder="请输入问题, 如: Li3HoBr6的离子电导率是多少?",
        scale=7,
    ),
    submit_btn="提问",
    stop_btn="停止",
)

if __name__ == "__main__":
    server = "0.0.0.0" if SHARE_LAN else "127.0.0.1"
    print(f"启动网页服务: http://{server}:{PORT}"
          + (" (局域网模式, 同学可通过你的IP访问)" if SHARE_LAN else " (本地模式)"))
    demo.launch(server_name=server, server_port=PORT)
