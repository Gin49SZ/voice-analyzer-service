# =============================================================================
# Dockerfile —— voice-analyzer-service 的【唯一】构建文件，构建期全程零网络
#
# 构建所需的全部东西都在本机，构建期一次网络请求都不发：
#   debhouse/*.deb    系统依赖（libsndfile1 / ffmpeg）及其依赖闭包，约 128 MB
#   wheelhouse/*.whl  Python 依赖（111 个 wheel，约 710 MB）
#   models/           离线模型（约 1 GB）
#   基础镜像          python:3.10-slim（可用 PY_BASE 覆盖为内网 registry 地址）
#
# 前两份本地仓库由 compose 的 deps 服务在有外网的机器上下载（容器 + bind mount）：
#   docker compose --profile online run --rm deps    # → ./debhouse + ./wheelhouse
#   docker compose up -d --build                    # 本地构建 + 启动（零网络）
#   等价的一条命令： sh scripts/build.sh
#
# 为什么「下载依赖」不写进 Dockerfile：
#   docker build 的构建上下文是【只读输入】，RUN 生成的文件只留在镜像层、写不回宿主机；
#   而容器的 bind mount 可以 —— 所以下载交给 `docker compose run` 的容器完成。
#
# 为什么本地依赖用 RUN --mount=type=bind 而不是 COPY：
#   COPY 会留下一个约 128 MB 的镜像层，之后即使 rm 掉，体积也照样算进最终镜像。
#   --mount 只把目录挂进这一步，deb / wheel 不落任何层。
#   （runtime 只 COPY --from=pydeps /opt/venv，所以 wheelhouse 永远进不了最终镜像）
#
# 阶段
#   base      系统依赖：从本地 .deb 装入 libsndfile1 / ffmpeg 并自检
#   pydeps    Python 依赖：只用本地 wheelhouse（--no-index，不碰任何索引）
#   runtime   最终镜像（本文件里 runtime 只有这一份定义）
# =============================================================================

# 基础镜像。要严格可复现就换成 digest，或指向内网 registry 地址
ARG PY_BASE=python:3.10-slim


# -----------------------------------------------------------------------------
# Stage: base —— 系统依赖（纯本地，无需 apt 源）
#   libsndfile1 —— soundfile 读写音频的底层库（HTTP 的 audio/wav 路径）
#   ffmpeg      —— torchaudio 解码 webm 需要（HTTP 的 audio/webm 路径）
# -----------------------------------------------------------------------------
FROM ${PY_BASE} AS base

RUN --mount=type=bind,source=debhouse,target=/debs \
    set -eux; \
    N="$(find /debs -maxdepth 1 -name '*.deb' | wc -l)"; \
    echo "[base] 本地 deb 数量: ${N}"; \
    [ "${N}" -gt 0 ] \
      || { echo "[base][FAIL] debhouse/ 里没有 .deb（只有 .gitkeep？）。"; \
           echo "           请先在联网机器执行 sh scripts/build.sh，再把 debhouse/ 整个目录拷到本机。"; exit 1; }; \
    dpkg -i /debs/*.deb > /tmp/dpkg.log 2>&1 \
      || { echo "[base][FAIL] dpkg 安装失败，日志尾部："; tail -25 /tmp/dpkg.log; exit 1; }; \
    ldconfig; \
    ldconfig -p | grep -q 'libsndfile\.so\.1' \
      || { echo "[base][FAIL] 装完后仍找不到 libsndfile.so.1"; exit 1; }; \
    command -v ffmpeg >/dev/null \
      || { echo "[base][FAIL] 装完后仍找不到 ffmpeg"; exit 1; }; \
    echo "[base] 系统依赖就绪：$(ffmpeg -version 2>/dev/null | head -1)"


# -----------------------------------------------------------------------------
# Stage: pydeps —— 只用本地 wheelhouse 装 Python 依赖，全程不联网
# -----------------------------------------------------------------------------
FROM base AS pydeps

COPY requirements.txt /tmp/req/requirements.txt

RUN --mount=type=bind,source=wheelhouse,target=/wheels \
    set -eux; \
    N="$(find /wheels -maxdepth 1 -name '*.whl' | wc -l)"; \
    echo "[pydeps] 本地 wheel 数量: ${N}"; \
    [ "${N}" -gt 0 ] \
      || { echo "[pydeps][FAIL] wheelhouse/ 里没有 .whl（只有 .gitkeep？）。"; \
           echo "               请先在联网机器执行 sh scripts/build.sh，再把 wheelhouse/ 整个目录拷到本机。"; exit 1; }; \
    python -m venv /opt/venv; \
    /opt/venv/bin/pip install --no-cache-dir --no-index \
        --find-links=/wheels \
        -r /tmp/req/requirements.txt; \
    rm -rf /tmp/req; \
    /opt/venv/bin/python -c "import torch, funasr, fastapi, soundfile; print('[pydeps] torch', torch.__version__, '| funasr', funasr.__version__)"


# -----------------------------------------------------------------------------
# Stage: runtime —— 最终镜像（本文件里 runtime 只有这一份定义）
#   FROM base 而非 FROM pydeps：pydeps 那一层只有 /opt/venv 是我们需要的
# -----------------------------------------------------------------------------
FROM base AS runtime

LABEL org.opencontainers.image.title="voice-analyzer-service" \
      org.opencontainers.image.description="FunASR(SenseVoiceSmall) 音频转录服务：文件上传 + WebSocket 流式，含 VAD 与说话人验证，支持离线运行" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.voice-analyzer.build.variant="offline"

# 运行期离线加固：模型全部内置，禁止任何联网尝试
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODELSCOPE_CACHE=/app/models \
    FUNASR_CACHE=/app/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 非 root 运行（uid/gid 1001，带可写 HOME —— 部分库会往 $HOME/.cache 写缓存）
RUN groupadd --gid 1001 app \
 && useradd --uid 1001 --gid app --home-dir /home/app --create-home --shell /usr/sbin/nologin app

COPY --from=pydeps /opt/venv /opt/venv

# 应用代码（timestamps.py 是句子级时间工具，必须一并拷入）
COPY --chown=app:app server.py model.py timestamps.py ./
# 离线模型（约 1 GB），--chown 避免额外的一层 chown 拷贝
COPY --chown=app:app models/ ./models/
# 便于容器内自测
COPY --chown=app:app examples/ ./examples/

USER app

EXPOSE 27000

# start-period 取 300s：实测冷启动「加载 SV + ASR + VAD 三个模型」要 75~110s，
# 但机器繁忙时会到 4 分钟以上（曾实测 263s 才加载完 VAD）。原来写 180s 会让刚起来的
# 服务被判成 unhealthy（healthcheck 3 次失败即置 unhealthy，而容器不会因此重启，
# 于是状态一直停在 unhealthy，运维看着像坏了）。这里留足余量。
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:27000/health', timeout=5).status == 200 else 1)"

# 注意：--gpu true 需要 CUDA 版 torch，本镜像装的是 CPU 版（torch==2.3.0+cpu）
CMD ["python", "server.py", "--port", "27000"]
