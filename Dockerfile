FROM python:3.10-slim

WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELSCOPE_CACHE=/app/models \
    FUNASR_CACHE=/app/models

# --- [修改开始] 替换为阿里云 Debian Trixie 源 ---
# 注意：python:3.10-slim 目前基于 Debian 13 (Trixie)
RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|http://deb.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources
# --- [修改结束] ---

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖 - 使用多个国内镜像源提高成功率
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url "https://pypi.mirrors.ustc.edu.cn/simple/ https://mirror.baidu.com/pypi/simple/" && \
    pip install --no-cache-dir torch==2.3.0+cpu torchaudio==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.org/simple

# 复制应用代码
COPY server.py .
COPY model.py .

# 复制本地模型文件（离线模式）
COPY models/ /app/models/

# 暴露端口
EXPOSE 27000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:27000/health')" || exit 1

# 启动命令
CMD ["python", "server.py", "--port", "27000"]