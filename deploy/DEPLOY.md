# 部署指南: 把问答系统搬到云服务器

部署后同学无需任何操作, 浏览器打开网址即可使用, 你的电脑可以关机。
全程约 30-40 分钟, 只需会复制粘贴命令。

## 前提

- 一台云服务器: **Ubuntu 22.04 或 24.04**, 配置 ≥ 2核4G内存 / 60G硬盘
  (腾讯云轻量应用服务器即可, 选购方法见文末或咨询项目组)
- 服务器的: 公网 IP、登录用户名(通常 `ubuntu` 或 `root`)、密码

## 第一步: 把项目传到服务器

在项目文件夹 (`d:\Desktop\大创`) 里打开 Git Bash:

```bash
# 打包(排除虚拟环境和向量库——服务器上会重建)
tar --exclude=.venv --exclude=.git --exclude=data/chroma_db -czf /tmp/hsse.tar.gz .

# 上传到服务器 (把 IP 换成你的服务器公网IP)
scp /tmp/hsse.tar.gz ubuntu@<服务器IP>:/home/ubuntu/
```

> 注意: 这个包包含 `pdfs/` 里的全部文献和 `.env` 里的密钥, 只传给你自己的服务器, 不要发给他人。

然后在服务器上解压 (用 Git Bash 或 PuTTY 登录):

```bash
ssh ubuntu@<服务器IP>
mkdir -p hsse && tar -xzf hsse.tar.gz -C hsse && cd hsse
```

## 第二步: 一键部署

```bash
bash deploy/setup_server.sh
```

脚本会自动完成: 装 Python 环境 → 装依赖 → 检查密钥 → 构建向量库 → 配置开机自启服务。
中途如果提示 `.env 未配置`, 按提示 `nano .env` 填入 DeepSeek 密钥(`Ctrl+O` 回车保存, `Ctrl+X` 退出), 然后重新运行一次脚本。

脚本最后会输出**访问密码**, 请记下来。

## 第三步: 放行端口(腾讯云控制台)

1. 登录 [腾讯云轻量应用服务器控制台](https://console.cloud.tencent.com/lighthouse)
2. 点进你的服务器 → 「防火墙」→「添加规则」
3. 协议 `TCP`, 端口 `7860`, 来源 `0.0.0.0/0` → 确定

## 第四步: 验证

浏览器访问 `http://<服务器IP>:7860`, 输入用户名 `hsse` 和脚本给出的密码,
能提问、能出带引用的答案即部署成功。把网址和密码发给同学即可。

## 日常维护

```bash
sudo systemctl status hsse-web     # 查看运行状态
sudo journalctl -u hsse-web -f     # 实时日志
sudo systemctl restart hsse-web    # 重启(改代码/改配置后)
```

服务器重启后服务会自动拉起, 无需任何操作。

## 费用说明

- 服务器: 学生价约 26-75 元/年 (2核4G 轻量)
- DeepSeek API: 按 token 计费, 每次问答约几分钱, 同学日常使用一学期预计 <20 元;
  可在 [DeepSeek 控制台](https://platform.deepseek.com) 随时查看用量、设置额度上限

## 常见问题

- **网页打不开**: 检查①防火墙是否放行 7860 ②`sudo systemctl status hsse-web` 是否 active
- **首次启动慢**: 嵌入模型首次下载约 470MB, 属正常, 之后秒开
- **答案报错/额度相关**: 登录 DeepSeek 控制台查余额
- **想改密码**: `sudo nano /etc/systemd/system/hsse-web.service` 改 `HSSE_ACCESS_PASSWORD=` 行,
  然后 `sudo systemctl daemon-reload && sudo systemctl restart hsse-web`
