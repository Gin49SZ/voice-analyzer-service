#!/usr/bin/env sh
# =============================================================================
# build.sh —— 一条命令：有依赖就直接构建启动，没依赖就先下到本地
#
#   sh scripts/build.sh
#
# 自动判断走哪条路：
#   ① 本地已有 ./debhouse + ./wheelhouse
#      → 直接构建 + 启动。构建期零网络（compose 给 app 设了 build.network: none）
#   ② 本地没有（或不全）
#      → 先联网把两样依赖下到本地，再构建 + 启动
#
# 常用变量
#   CLEAN=deb|wheel|1   重抓依赖前先清空本地旧文件（默认 0 不动）
#   NO_START=1     只构建，不启动
#   FORCE_FETCH=1  强制走「联网抓依赖」（即使本地物料看起来是齐的）
#   IMAGE_TAG / PY_BASE / PY_TAR
#   PIP_INDEX_URL / PYTORCH_INDEX_URL / APT_DEBIAN_URL / APT_SECURITY_URL
#
# 换源（本机实测：aliyun 的 PyPI 镜像可下 40 MB 级大包；pypi.org 与 ustc 小包能下、
#       大包读超时；tuna 不通；download.pytorch.org 大响应超时，故默认 aliyun + SJTU；
#       apt 必须 https，默认 ustc）：
#   PIP_INDEX_URL=... PYTORCH_INDEX_URL=... sh scripts/build.sh
# =============================================================================
set -eu

IMAGE_TAG="${IMAGE_TAG:-${IMAGE:-voice-analyzer:1.0}}"
PY_BASE="${PY_BASE:-python:3.10-slim}"
PY_TAR="${PY_TAR:-dist/python-3.10-slim.tar}"
NO_START="${NO_START:-0}"
CLEAN="${CLEAN:-0}"
FORCE_FETCH="${FORCE_FETCH:-0}"
export IMAGE_TAG CLEAN PY_BASE

cd "$(dirname "$0")/.."
echo "[build] 项目根目录: $(pwd)"

# ---------------------------------------------------------------------------
echo ""
echo "==> [1/4] 前置检查"
docker compose version >/dev/null 2>&1 \
    || { echo "    [FAIL] docker compose 不可用（需要 Docker Compose v2）"; exit 1; }
echo "    [OK] Docker $(docker version --format '{{.Server.Version}}') / compose $(docker compose version --short)"

for f in requirements.txt server.py model.py timestamps.py Dockerfile docker-compose.yml scripts/fetch_deps.sh; do
    [ -f "$f" ] || { echo "    [FAIL] 缺少 ${f}"; exit 1; }
done
echo "    [OK] 源码 / 唯一 Dockerfile / compose / 抓取脚本 齐备"

for d in SenseVoiceSmall speech_fsmn_vad_zh-cn-16k-common-pytorch speech_eres2net_large_sv_zh-cn_3dspeaker_16k; do
    [ -d "models/$d" ] || { echo "    [FAIL] 缺少 models/$d"; exit 1; }
done
echo "    [OK] models/ 三个模型齐备（$(du -sh models 2>/dev/null | cut -f1)）"

mkdir -p wheelhouse debhouse dist

# ---------------------------------------------------------------------------
echo ""
echo "==> [2/4] 判断本地物料"

N_WHEEL="$(find wheelhouse -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
N_DEB="$(find debhouse -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l | tr -d ' ')"
echo "    wheelhouse: ${N_WHEEL} 个 .whl"
echo "    debhouse  : ${N_DEB} 个 .deb"

NEED_FETCH=0
if [ "${FORCE_FETCH}" = "1" ]; then
    NEED_FETCH=1
elif [ "${N_WHEEL}" -eq 0 ] || [ "${N_DEB}" -eq 0 ]; then
    NEED_FETCH=1
fi

if [ "${NEED_FETCH}" = "1" ]; then
    # ---- 路 ②：联网抓依赖 -------------------------------------------------
    echo ""
    echo "==> [3/4] 本地依赖不全 → 联网抓取到本地"
    echo "    APT: ${APT_DEBIAN_URL:-https://mirrors.ustc.edu.cn/debian}"
    echo "    PIP: ${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
    docker compose --profile online run --rm deps \
        || { echo "    [FAIL] 抓依赖失败。请确认本机能访问外网与镜像源；"; \
             echo "           若本机不通外网，请把联网机器上的 debhouse/ 与 wheelhouse/ 拷过来。"; exit 1; }
    N_WHEEL="$(find wheelhouse -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
    N_DEB="$(find debhouse -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l | tr -d ' ')"
    [ "${N_WHEEL}" -gt 0 ] || { echo "    [FAIL] wheelhouse/ 仍然没有 .whl"; exit 1; }
    [ "${N_DEB}" -gt 0 ] || { echo "    [FAIL] debhouse/ 仍然没有 .deb"; exit 1; }

    # 顺手把基础镜像导出成 tar，方便拷给不通外网的机器
    if docker image inspect "${PY_BASE}" >/dev/null 2>&1; then
        echo ""
        echo "    -- 导出基础镜像 ${PY_BASE} → ${PY_TAR}（拷给内网机器可省一次 docker pull）"
        docker save "${PY_BASE}" -o "${PY_TAR}"
        echo "    -> ${PY_TAR}（$(ls -lh "${PY_TAR}" | awk '{print $5}')）"
    fi
else
    echo ""
    echo "==> [3/4] 本地依赖齐备 → 跳过下载，直接构建"
fi

# 基础镜像不在本地就从 tar 载入
if ! docker image inspect "${PY_BASE}" >/dev/null 2>&1; then
    if [ -f "${PY_TAR}" ]; then
        echo ""
        echo "    -- 载入基础镜像 ${PY_TAR}"
        docker load -i "${PY_TAR}"
    else
        echo "    [FAIL] 本地没有基础镜像 ${PY_BASE}，也没有 ${PY_TAR}"
        echo "           请 docker pull ${PY_BASE}，或从其它机器 docker save 后拷过来"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
if [ "${NO_START}" = "1" ]; then
    echo ""
    echo "==> [4/4] 构建应用镜像: ${IMAGE_TAG}（NO_START=1，不启动）"
else
    echo ""
    echo "==> [4/4] 构建并启动应用镜像: ${IMAGE_TAG}"
fi
echo "    target=runtime + build.network: none（构建期零网络，依赖只读本地目录）"
docker compose build app

if [ "${NO_START}" = "1" ]; then
    echo ""
    echo "已按要求只构建不启动。启动： docker compose up -d"
    exit 0
fi

docker compose up -d app
echo ""
docker compose ps

cat <<TIP

完成。常用命令（在项目目录下）：
  docker compose logs -f app          # 看日志
  docker compose restart app          # 重启
  docker compose down                 # 停止并删除容器

验证：
  curl http://127.0.0.1:27000/health
  curl -X POST "http://127.0.0.1:27000/transcribe?apiKey=sk-7f3a9b2c1e5d8f4a6b0c9e2d1a5f8b3c" \\
       -F "file=@examples/test.wav;type=audio/wav"

交付给其它机器：
  重新构建  → 拷 debhouse/ + wheelhouse/ + models/ + ${PY_TAR} + 仓库源码
  不再构建  → docker save ${IMAGE_TAG} -o dist/voice-analyzer-image.tar（只需带这一个 tar）
TIP
