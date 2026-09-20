#!/usr/bin/env bash
# 卤化物固态电解质RAG问答系统 - 一键部署脚本
# 适用: Ubuntu 22.04 / 24.04 (腾讯云轻量应用服务器)
# 用法: bash deploy/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
echo "工作目录: $(pwd)"

echo "== 1/6 安装系统依赖 =="
sudo apt-get update -qq
sudo apt-get install -y python3 python3-venv python3-pip

echo "== 2/6 创建虚拟环境并安装Python依赖(约5-10分钟) =="
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q -i https://pypi.tuna.tsinghua.edu.cn/simple
# 先装CPU版torch(避免Linux默认PyPI torch附带数GB的CUDA包)
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu \
  || .venv/bin/pip install -q torch -i https://pypi.tuna.tsinghua.edu.cn/simple
# 服务器用最小依赖集(不锁版本), 避免与本机冻结版本的Python版本冲突
.venv/bin/pip install -q -r deploy/requirements-server.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "== 3/6 检查 .env 配置 =="
if [ ! -f .env ]; then
    cp .env.example .env
    echo "!! 已生成 .env 模板, 请先编辑填入 DEEPSEEK_API_KEY:"
    echo "!!   nano .env"
    echo "!! 然后重新运行本脚本: bash deploy/setup_server.sh"
    exit 1
fi
if ! grep -q "sk-" .env; then
    echo "!! .env 中未配置有效的 DEEPSEEK_API_KEY, 请编辑后重新运行本脚本"
    exit 1
fi

echo "== 4/6 构建向量知识库(若尚未构建) =="
if [ ! -d data/chroma_db ]; then
    if [ ! -f data/chunks.jsonl ]; then
        echo "从 pdfs/ 抽取文本(需要先把文献PDF放入 pdfs/ 目录)..."
        .venv/bin/python src/ingest.py
    fi
    echo "嵌入并向量化(首次需下载约470MB模型)..."
    .venv/bin/python src/build_db.py
else
    echo "data/chroma_db 已存在, 跳过"
fi

echo "== 5/6 生成访问口令并配置 systemd 服务(开机自启+崩溃自恢复) =="
ACCESS_PW="${HSSE_ACCESS_PASSWORD:-$(openssl rand -hex 3)}"
sudo tee /etc/systemd/system/hsse-web.service > /dev/null <<EOF
[Unit]
Description=HSSE RAG QA Web Service
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
Environment=HSSE_SHARE_LAN=1
Environment=HSSE_PORT=7860
Environment=HSSE_ACCESS_PASSWORD=${ACCESS_PW}
Environment=HSSE_UPLOAD_PASSWORD=${ACCESS_PW}
ExecStart=$(pwd)/.venv/bin/python app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now hsse-web

echo "== 6/6 完成 =="
echo "==========================================================="
echo " 部署完成! 还差最后一步:"
echo " 到腾讯云控制台 -> 轻量应用服务器 -> 防火墙, 放行 TCP 7860 端口"
echo ""
echo " 访问地址: http://<服务器公网IP>:7860"
echo " 访问账号: hsse"
echo " 访问密码: ${ACCESS_PW}  (上传文献也用此密码)"
echo ""
echo " 常用维护命令:"
echo "   查看状态: sudo systemctl status hsse-web"
echo "   查看日志: sudo journalctl -u hsse-web -f"
echo "   重启服务: sudo systemctl restart hsse-web"
echo "==========================================================="
