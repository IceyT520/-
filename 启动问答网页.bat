@echo off
chcp 65001 >nul
cd /d %~dp0
echo 正在启动卤化物固态电解质知识问答系统...
echo 启动后请在浏览器访问: http://127.0.0.1:7860
echo (关闭本窗口即停止服务)
.venv\Scripts\python.exe app.py
pause
