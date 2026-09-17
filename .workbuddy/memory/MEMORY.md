# voice-analyzer-service — 项目长期备忘

## 接口契约（务必保持）
- `TranscriptionResponse = {code, info, data, sentences}`。
- `data` 必须**始终是纯文本**（不塞结构化对象）；结构化结果一律放 `sentences`，
  每条为 `{start_ms, end_ms, text}`，时间为相对音频起点的毫秒。
- `sentences` 为附加字段，缺失/为空不得影响老客户端；`code` 语义：0 转写结果、2 说话人、3 flush ack。
- **WS 鉴权失败的「可观测形态」是 HTTP 403 握手拒绝，而非关闭码 1008**（2026-09-17 实测更正）：
  `server.py` 在 `accept()` **之前**调 `websocket.close(code=1008, ...)`，Starlette 会把「未 accept 即关闭」
  表现为 403；客户端只会拿到 `onclose code=1006`（握手阶段没有 WS 关闭帧）。
  websockets 15 抛的是 `InvalidStatus(403)` 而不是 `ConnectionClosed(1008)`。
  → 客户端判鉴权失败必须基于「是否成功 onopen」，只判 1008 是永远走不到的分支。
- `reg_spks` 存在**双重解码**（`parse_qs` 已解码一次，`server.py` 又 `unquote` 一次）。
  对规范编码的 URL 无影响（`encodeURIComponent` 会把 `+`→`%2B`、空格→`%20`）；
  客户端按「整个逗号分隔串编码一次」发送即可，与 `docs/API.md` §6.1 一致。

## 技术约束
- SenseVoice（`model.py` / `remote_code=./model.py`）**不输出时间戳**，时间信息只能来自 VAD 片段边界；
  不要为了时间戳去改模型或更换加载方式（funasr 1.2.7 的内置 `output_timestamp` 时间数值损坏）。
- WS 流式 VAD 输出边界事件（`[beg,-1]` 开始、`[-1,end]` 结束，相对流起点毫秒）；
  不发送 `is_final=True` 的 flush 会丢失尾部片段。
- 本地模型从 `models/` 目录 `local_files_only=True` 加载；`MODELSCOPE_CACHE`/`FUNASR_CACHE` 指向该目录。
- **VAD 调用必须显式写全 `chunk_size` / `is_streaming_input` / `is_final`**（已封装为
  `vad_offline()` / `vad_stream()`，不要绕过）。原因：funasr `AutoModel.inference` 里
  `kwargs = self.kwargs; deep_update(kwargs, cfg)` 会把本次参数**永久**并进模型对象，
  而 HTTP 与 WS 共用同一个 `model_vad`；WS 的 `chunk_size=300/is_final=False` 残留后，
  HTTP 的 `is_streaming_input` 默认成 `True` → 返回事件对而非区间对 → `sentences` 22 段变 44 段、
  大量 `start_ms=0`。（已在容器内真实推理复现；验证脚本已随 `tmp/` 清理，结论保留。）
- 整段 ASR（`data` 字段）在 `use_itn=True` 时，**ITN 标点会跨次抖动**（如 `换得来的，有人`/
  `换得来的有人`）；`use_itn=False` 时逐字节稳定。**与线程数无关**（`OMP_NUM_THREADS=1` 结果同序同值），
  也**不是** VAD 污染。`sentences`（逐段文本 + 时间）始终稳定，抖动只在 `data` 上。

## 工程习惯
- 用户习惯：先给结论/方案，再给可落地的具体实现；输出用 Markdown + 表格。
- 实验与验证脚本放 `tmp/`（临时用，**不入库**，也不要长期堆积；2026-09-17 已整体清空一次）。
- 改动源码前先做隔离实验验证可行性，落地后补协议级回归测试。
- **文档只留两份**（用户 2026-09-17 明确要求）：`README.md`（概览 + 快速开始）与 `docs/API.md`（接口契约）。
  `docs/OFFLINE_DEPLOY.md` 已删除，离线部署说明合并进 README / AGENTS.md。
- 前端测试页：`test_client.html`（HTTP 上传，可逐句试听）、`test_client_wss.html`（WS 流式：
  每句 `mm:ss.mmm` 时间戳、跨会话汇总表、时间轴异常检测、逐句试听、TSV/JSON 导出）。
  上行固定 16 kHz/16 bit/单声道并按 **300ms 整块**（`Int16Array(4800)`）发送，与 `chunk_size_ms` 对齐。

## Docker 构建约定（2026-09-17 精简后的最终形态）
- **只有一个 `Dockerfile`**（`Dockerfile.online` / `Dockerfile.offline` 已删除），
  编排走 `docker-compose.yml` 的 **2 个服务**：
  - `app`（默认 profile）→ 应用镜像的**唯一**构建路径，`build.network: none`；
  - `deps`（profile `online`）→ **只产出物料**，把依赖下到宿主机 `./debhouse` + `./wheelhouse`。
- Dockerfile 三阶段：`base`（本地 deb → `dpkg -i` 装 libsndfile1/ffmpeg）
  → `pydeps`（本地 wheel → `--no-index` 装 Python 依赖）
  → `runtime`（`FROM base` + `COPY --from=pydeps /opt/venv`）。
- **本地依赖一律用 `RUN --mount=type=bind,source=debhouse|wheelhouse,target=/debs|/wheels` 挂载消费，
  绝不 `COPY`**：`COPY` 会把 128MB deb + 710MB wheel 永久留在镜像层（mount 不写层，镜像体积 6.41GB 未涨即为证）。
- **不需要预烘焙基础镜像**了。用户的核心质疑：「既然系统依赖也必须联网才能拿到，为什么不放在本地、
  构建时安装，而非要再启用一个镜像？」→ 正解：apt `--download-only` 把 deb 闭包落到本地，
  构建时 `dpkg -i`。原 `base` 阶段 + 208MB tar 因此整体删除。
- **deb 下载顺序是硬坑**：apt 不会重复下载**已安装**的包。必须先 `--download-only` 抓 deb（干净镜像下 205 个），
  **之后**再装 gcc/python3-dev；顺序反了只下到 202 个，断网 `dpkg -i` 必失败（差的就是工具链那 3 个依赖）。
- 脚本只有两个：`scripts/fetch_deps.sh`（容器内抓 deb+wheel，幂等，`CLEAN=0|deb|wheel|1`）、
  `scripts/build.sh`（物料齐 → 直接构建启动；缺 → 先抓再构建；`NO_START` / `FORCE_FETCH` / `PY_BASE` / `PY_TAR`）。
- **依赖怎么写回宿主机**：`docker build` 上下文只读写不回来；用 `docker compose --profile online run --rm deps`
  在**容器运行时**经 bind mount 落盘。
- 编译工具只进 `base` 阶段的临时安装（同层装完即用）；新增 `.py` 必须同步 `COPY` 进 `runtime`
  （曾漏拷 `timestamps.py`，容器一启动就 ModuleNotFoundError）。
- **`ffmpeg` 必须装**：唯一理由是 `torchaudio 2.3.0+cpu` 的 wheel **没编 ffmpeg 后端**
  （`list_audio_backends()==['soundfile']`，`backend='ffmpeg'` 抛 ValueError，`libtorchaudio.so` 内无 `libav*`）
  → webm 只能走 `ffmpeg` 命令行解码（`server.py: decode_with_ffmpeg()`）。
- **换源铁律：必须用大文件验证源**——小包能下不代表可用（曾用 10KB 的 `addict` 验证，
  误判 pypi.org 可用，首次构建在 `pip --upgrade pip` 步骤读超时失败）。实测：
  PyPI 侧仅 aliyun 能下 46MB 包（pypi.org/ustc 小包 OK 大包超时、tuna 不通）；
  torch 走 `mirror.sjtu.edu.cn/pytorch-wheels/cpu`（18.5MB/s，官方源大响应超时，
  aliyun 的 pytorch-wheels 无 PEP503 索引）；apt 用 ustc、且**必须 https**（http 被拦 502）。
- 抓 wheel 必须**两遍**：非 `+cpu` 包走 PyPI，`torch`/`torchaudio` 再单独走 PyTorch 索引并 `--no-deps`。
  否则 `mkl` 会被解析到 SJTU 里一条指向 tuna 的 403 链接，整个抓取失败。
  另：离线 wheelhouse 必须 `pip wheel` 预构建，不能用 `pip download --only-binary`
  （`aliyun-python-sdk-core==2.16.0` 只有 sdist）。
- 基线：应用镜像 6.41 GB、冷启动 75–110 s（繁忙时 **263 s**）、debhouse 205 deb/128 MB、
  wheelhouse 111 wheel/710 MB、`pip freeze` 111 包。
- `HEALTHCHECK --start-period=300s`：冷启动要加载 SV+ASR+VAD 三个模型，180s 会让刚起来的服务
  被误判 `unhealthy`（容器不会因此重启，状态就永久停在 unhealthy）。
- 验证离线启动用 `--network=none` + `docker exec`（该模式 `-p` 无效，镜像内无 curl）。
- 清理坑：Docker Desktop 上 `docker images -f dangling=true` 会把**带标签**的镜像也列出来，
  按 ID `rmi` 会误删（曾误删 `voice-analyzer-base:3.10`，可从缓存秒级重建，先按 ID 反查标签）。
- 遗留可清理：`voice-analyzer-wheels:3.10`(1.31GB)、`voice-analyzer-base:3.10`(822MB) 属于旧方案产物。
