#!/bin/sh
# =============================================================================
# fetch_deps.sh —— 在「抓依赖」容器内执行（由 compose 调起）
#     docker compose --profile online run --rm deps
#
# 做什么：把构建镜像需要的两样东西下到【宿主机】本地目录（bind mount）
#   $DEB_DEST/*.deb    系统依赖：libsndfile1 / ffmpeg 及其【依赖闭包】（约 128 MB）
#                      用 apt 的 --download-only 下载，不安装
#   $WHEEL_DEST/*.whl  Python 依赖：含传递依赖；只有 sdist 的包
#                      （aliyun-python-sdk-core==2.16.0）现场编成 wheel（约 710 MB）
#
# 为什么是「容器 + bind mount」而不是写进 Dockerfile：
#   docker build 只能【读】构建上下文，RUN 生成的文件留在镜像层、写不回宿主机；
#   而挂载能真实写回宿主机。这是本方案的核心。
#
# 为什么要「两遍抓 wheel」：
#   依赖里只有 torch / torchaudio 带 `+cpu` 本地版本号，只存在于 PyTorch 系索引；
#   其余（含 mkl / intel-openmp / tbb）在 PyPI 上都有。
#   若把 PyTorch 索引当成全局 --extra-index-url，pip 会去选
#   mirror.sjtu.edu.cn/pytorch-wheels/cpu/mkl/ 里那条指向 tuna 的链接，实测 403，
#   整个抓取失败。所以：第一遍只用 PyPI，第二遍只用 PyTorch 索引且加 --no-deps
#   （依赖已在第一遍备齐，最后由自检兜底校验完整依赖树）。
#
# 幂等：deb 与 wheelhouse 都已满足时直接跳过（约 2 秒）。
#       CLEAN=deb   只清空并重抓系统依赖
#       CLEAN=wheel 只清空并重抓 Python 依赖
#       CLEAN=1     两者都重抓
#
# 环境变量
#   DEB_DEST          默认 /debhouse（compose 挂到宿主机 ./debhouse）
#   WHEEL_DEST        默认 /wheelhouse（compose 挂到宿主机 ./wheelhouse）
#   REQ               默认 /opt/req/requirements.txt（compose 只读挂载的仓库文件）
#   APT_DEBIAN_URL    默认 https://mirrors.ustc.edu.cn/debian
#   APT_SECURITY_URL  默认 https://mirrors.ustc.edu.cn/debian-security
#   PIP_INDEX_URL     默认 https://mirrors.aliyun.com/pypi/simple/
#   PYTORCH_INDEX_URL 默认 https://mirror.sjtu.edu.cn/pytorch-wheels/cpu
#   PIP_TRUSTED_HOST  内网 http 源时填域名
#   CLEAN             0 / 1 / deb / wheel，见上
# =============================================================================
set -eu

DEB_DEST="${DEB_DEST:-/debhouse}"
WHEEL_DEST="${WHEEL_DEST:-/wheelhouse}"
REQ="${REQ:-/opt/req/requirements.txt}"
APT_DEBIAN_URL="${APT_DEBIAN_URL:-https://mirrors.ustc.edu.cn/debian}"
APT_SECURITY_URL="${APT_SECURITY_URL:-https://mirrors.ustc.edu.cn/debian-security}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://mirror.sjtu.edu.cn/pytorch-wheels/cpu}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"
CLEAN="${CLEAN:-0}"

echo "[fetch_deps] DEB_DEST   = ${DEB_DEST}   （= 宿主机 ./debhouse）"
echo "[fetch_deps] WHEEL_DEST = ${WHEEL_DEST} （= 宿主机 ./wheelhouse）"
echo "[fetch_deps] APT  = ${APT_DEBIAN_URL}"
echo "[fetch_deps] PIP  = ${PIP_INDEX_URL}"
echo "[fetch_deps] TORCH= ${PYTORCH_INDEX_URL}"

[ -f "${REQ}" ] || { echo "[fetch_deps][FAIL] 找不到 ${REQ}"; exit 1; }
mkdir -p "${DEB_DEST}/partial" "${WHEEL_DEST}"

# ---------------------------------------------------------------------------
# 1) 净化 requirements.txt
#
# requirements.txt 是 UTF-16 编码，且首行自带
#   `--extra-index-url https://download.pytorch.org/whl/cpu`
# 实测该源在本网络下「小请求 200、大响应（190 MB 的 torch）读超时」，必须剥掉，
# 统一改用 PYTORCH_INDEX_URL 控制。剥掉的只是索引声明，不影响任何钉版。
#
# ⚠️ 绝不能用 grep 过滤：grep 会把 UTF-16 文件判定为 binary 并吞掉全部输出，
#    实测产出 0 字节文件；而 `pip install -r <空文件>` 的退出码是 0，
#    会让「是否已满足」判断永远为真 —— 静默跳过下载、什么都不下。
#    所以用 python 显式解码（镜像里必有 python3），并在最后加非空断言。
# ---------------------------------------------------------------------------
SANITIZED=/tmp/requirements.sanitized.txt
export REQ SANITIZED
python3 - <<'PY'
import os, pathlib, sys

src = pathlib.Path(os.environ["REQ"])
raw = src.read_bytes()

text, used = None, None
for enc in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gbk", "latin-1"):
    try:
        candidate = raw.decode(enc)
    except UnicodeDecodeError:
        continue
    # 真实内容换行多且用 == 钉版；这条校验挡住「用 latin-1 把 UTF-16 解成乱码」
    if candidate.count("\n") > 2 and "==" in candidate:
        text, used = candidate, enc
        break
if text is None:
    sys.exit("[fetch_deps][FAIL] 无法按已知编码解码 %s" % src)
print("[fetch_deps] requirements 编码识别为 %s" % used)

INDEX_PREFIXES = ("--extra-index-url", "--index-url", "--trusted-host")
all_lines = text.splitlines()
lines = [ln for ln in all_lines
         if ln.strip() and not ln.strip().lower().startswith(INDEX_PREFIXES)]
if not lines:
    sys.exit("[fetch_deps][FAIL] 剥离索引声明后没有任何依赖行，拒绝继续")

out = pathlib.Path(os.environ["SANITIZED"])
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[fetch_deps] 已剥离索引声明 -> %s（%d 行依赖 / 原 %d 行）"
      % (out, len(lines), len(all_lines)))
PY
[ -s "${SANITIZED}" ] || { echo "[fetch_deps][FAIL] 净化后的 requirements 为空"; exit 1; }

# ---------------------------------------------------------------------------
# 2) 判断哪些还需要抓
# ---------------------------------------------------------------------------
CLEAN_DEB=0
CLEAN_WHEEL=0
case "${CLEAN}" in
    0|no|"")   ;;
    1|all|yes) CLEAN_DEB=1; CLEAN_WHEEL=1 ;;
    deb*)      CLEAN_DEB=1 ;;
    wheel*)    CLEAN_WHEEL=1 ;;
    *) echo "[fetch_deps][FAIL] CLEAN 只接受 1 / all / deb / wheel，收到 '${CLEAN}'"; exit 1 ;;
esac

if [ "${CLEAN_DEB}" = "1" ]; then
    echo "[fetch_deps] CLEAN=${CLEAN}：清空 ${DEB_DEST} 下的旧 deb"
    find "${DEB_DEST}" -maxdepth 1 -name '*.deb' -delete 2>/dev/null || true
fi
if [ "${CLEAN_WHEEL}" = "1" ]; then
    echo "[fetch_deps] CLEAN=${CLEAN}：清空 ${WHEEL_DEST} 下的旧 wheel"
    find "${WHEEL_DEST}" -maxdepth 1 -name '*.whl' -delete 2>/dev/null || true
fi

N_DEB="$(find "${DEB_DEST}" -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l | tr -d ' ')"
N_WHL="$(find "${WHEEL_DEST}" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
echo "[fetch_deps] 现有 deb: ${N_DEB} 个 / wheel: ${N_WHL} 个"

# --dry-run 需要 pip >= 22.2
if ! pip install --help 2>/dev/null | grep -q -- '--dry-run'; then
    echo "[fetch_deps] 当前 pip 不支持 --dry-run，先升级 pip"
    pip install --no-cache-dir -q --upgrade pip --index-url "${PIP_INDEX_URL}"
fi

NEED_DEB=1
if [ "${CLEAN_DEB}" != "1" ] && [ "${N_DEB}" -gt 0 ]; then NEED_DEB=0; fi

NEED_WHEEL=1
if [ "${CLEAN_WHEEL}" != "1" ] && [ "${N_WHL}" -gt 0 ] \
   && pip install --no-index --find-links="${WHEEL_DEST}" --dry-run -q -r "${SANITIZED}" >/dev/null 2>&1; then
    NEED_WHEEL=0
fi

if [ "${NEED_DEB}" = "0" ] && [ "${NEED_WHEEL}" = "0" ]; then
    echo "[fetch_deps] 本地 debhouse + wheelhouse 均已满足，跳过下载。"
    echo "[fetch_deps] OK"
    exit 0
fi
echo "[fetch_deps] 需要抓取: deb=${NEED_DEB} wheel=${NEED_WHEEL}"

# ---------------------------------------------------------------------------
# 3) apt 源改写 + 索引更新
#
#   apt 必须走 https：默认源是 http://deb.debian.org，在本网络实测被拦成 502。
#   ⚠️ 这一步之后【立刻】下载 deb（见 4），中间不要安装任何东西 —— 原因见 4 的说明。
# ---------------------------------------------------------------------------
echo "[fetch_deps] [apt] 改写源为 https 镜像"
for f in /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources; do
    [ -f "$f" ] || continue
    sed -i \
        -e "s|https*://deb.debian.org/debian-security|${APT_SECURITY_URL}|g" \
        -e "s|https*://security.debian.org/debian-security|${APT_SECURITY_URL}|g" \
        -e "s|https*://deb.debian.org/debian|${APT_DEBIAN_URL}|g" \
        "$f"
done
apt-get update -qq

# ---------------------------------------------------------------------------
# 4) 下载系统依赖的完整 deb 闭包
#    --download-only：只下不装（容器是一次性的，我们只要文件）
#    依赖闭包由 apt 自己算，缺一个都会在离线安装时直接报错
#
#    ⚠️ 必须在「干净的基础镜像」上做，也就是要排在装编译工具之前：
#       apt 不会重复下载容器里【已经装好】的包，而 gcc / python3-dev 会带进一批库。
#       若先装了工具链，这批库就被算作「已安装」而不下载 —— 可构建镜像里并没有它们，
#       于是 dpkg -i 会直接报一堆依赖缺失。
#       实测：先装工具链只下到 202 个 deb，干净镜像下是 205 个，差的就是那 3 个。
# ---------------------------------------------------------------------------
if [ "${NEED_DEB}" = "1" ]; then
    echo "[fetch_deps] [deb] 下载 libsndfile1 + ffmpeg 及其依赖闭包"
    # shellcheck disable=SC2086
    apt-get install -y -q --download-only --no-install-recommends \
        -o Dir::Cache::archives="${DEB_DEST}" \
        libsndfile1 ffmpeg >/dev/null
    N_DEB="$(find "${DEB_DEST}" -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l | tr -d ' ')"
    echo "[fetch_deps] [deb] 下载完成: ${N_DEB} 个 / $(du -sh "${DEB_DEST}" 2>/dev/null | cut -f1)"
    [ "${N_DEB}" -gt 0 ] || { echo "[fetch_deps][FAIL] 一个 deb 都没下到"; exit 1; }
else
    echo "[fetch_deps] [deb] 本地已有 ${N_DEB} 个 deb，跳过"
fi

# ---------------------------------------------------------------------------
# 5) 装编译工具（只为把 sdist 编成 wheel，不进任何交付镜像）
#    刻意排在 deb 下载之后，理由见 4)
# ---------------------------------------------------------------------------
if [ "${NEED_WHEEL}" = "1" ]; then
    echo "[fetch_deps] [apt] 安装编译工具（gcc / g++ / python3-dev）"
    apt-get install -y -q --no-install-recommends \
        gcc g++ python3-dev ca-certificates >/dev/null
fi

# ---------------------------------------------------------------------------
# 6) 抓 wheel：按「是否带本地版本号（+cpu 之类）」分流
#    本地版本号只在 PyTorch 系索引上存在 —— 这批走第二遍。
# ---------------------------------------------------------------------------
if [ "${NEED_WHEEL}" = "1" ]; then
    PYPI_REQ=/tmp/requirements.pypi.txt
    TORCH_REQ=/tmp/requirements.torchidx.txt
    grep -E '\+' "${SANITIZED}" > "${TORCH_REQ}" || true
    grep -vE '\+' "${SANITIZED}" > "${PYPI_REQ}" || true
    echo "[fetch_deps] 分流: PyPI 源 $(grep -c . "${PYPI_REQ}") 行 / PyTorch 索引 $(grep -c . "${TORCH_REQ}") 行"
    [ -s "${PYPI_REQ}" ] || { echo "[fetch_deps][FAIL] PyPI 侧依赖为空"; exit 1; }

    TRUSTED=""
    if [ -n "${PIP_TRUSTED_HOST}" ]; then
        TRUSTED="--trusted-host ${PIP_TRUSTED_HOST}"
    fi

    # --- 第一遍：PyPI 源（绝不能带上 PyTorch 索引，理由见文件头）-------------
    echo "[fetch_deps] [1/2] PyPI 源: ${PIP_INDEX_URL}（首次约 5-10 分钟）"
    # shellcheck disable=SC2086
    pip wheel --no-cache-dir ${TRUSTED} \
        --index-url "${PIP_INDEX_URL}" \
        --find-links="${WHEEL_DEST}" \
        -r "${PYPI_REQ}" \
        -w "${WHEEL_DEST}"

    # --- 第二遍：PyTorch 索引，只抓 torch / torchaudio ----------------------
    if [ -s "${TORCH_REQ}" ]; then
        echo "[fetch_deps] [2/2] PyTorch 索引: ${PYTORCH_INDEX_URL}"
        # --no-deps：torch 的依赖（mkl / tbb / sympy / ...）已在第一遍备齐；
        #            带上依赖解析会又把 pip 引到 PyTorch 索引里那些坏链接上。
        # shellcheck disable=SC2086
        pip wheel --no-cache-dir --no-deps ${TRUSTED} \
            --index-url "${PYTORCH_INDEX_URL}" \
            --find-links="${WHEEL_DEST}" \
            -r "${TORCH_REQ}" \
            -w "${WHEEL_DEST}"
    else
        echo "[fetch_deps] [2/2] 没有需要 PyTorch 索引的包，跳过"
    fi
else
    echo "[fetch_deps] [wheel] 本地 wheelhouse 已满足，跳过"
fi

# ---------------------------------------------------------------------------
# 7) 自检
#    完整依赖树必须能纯 --no-index 解析出来，兜住「第二遍用了 --no-deps」的风险
# ---------------------------------------------------------------------------
echo "[fetch_deps] 自检"
pip install --no-index --find-links="${WHEEL_DEST}" --dry-run -q -r "${SANITIZED}" >/dev/null \
    || { echo "[fetch_deps][FAIL] wheelhouse 无法离线解析完整依赖树"; exit 1; }
N_WHL="$(find "${WHEEL_DEST}" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l | tr -d ' ')"
N_DEB="$(find "${DEB_DEST}" -maxdepth 1 -name '*.deb' 2>/dev/null | wc -l | tr -d ' ')"
echo "[fetch_deps] debhouse:   ${N_DEB} 个 .deb / $(du -sh "${DEB_DEST}" 2>/dev/null | cut -f1)"
echo "[fetch_deps] wheelhouse: ${N_WHL} 个 .whl / $(du -sh "${WHEEL_DEST}" 2>/dev/null | cut -f1)"
echo "[fetch_deps] OK —— 已写入宿主机 ./debhouse 与 ./wheelhouse"
