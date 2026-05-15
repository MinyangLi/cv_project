#!/usr/bin/env bash

set -euo pipefail

# MASK_pipline 项目的统一 Bash 入口。
#
# 这个脚本本质上是在调用：
#   python scripts/run_mask_video.py
#
# 它会负责：
# 1. 激活 `sam3` conda 环境
# 2. 组织默认参数
# 3. 把命令行参数转发给 Python 主流程
#
# 推荐示例 1：跟踪一个具体的黄色时钟
#
#   bash scripts/run_mask_pipeline.sh \
#     --input "Input_video/Unedited/example-4.mp4" \
#     --target-text "yellow clock" \
#     --prompt-box "790,330,1170,770" \
#     --output "outputs/example-4_masked.mp4"
#
# 推荐示例 2：跟踪一个具体的人
#
#   bash scripts/run_mask_pipeline.sh \
#     --input "Input_video/Unedited/example-7.mp4" \
#     --target-text "person" \
#     --yolo-class "person" \
#     --prompt-box "520,180,780,719" \
#     --output "outputs/example-7_masked.mp4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python / environment
CONDA_SH="${CONDA_SH:-/home/czy/anaconda3_2/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sam3}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Core I/O
INPUT_VIDEO=""
OUTPUT_VIDEO=""
CONFIG_PATH="${PROJECT_ROOT}/configs/mask_video.yaml"

# Prompting
TARGET_TEXT=""
YOLO_CLASS=""
TARGET_CLASS=""
PROMPT_BOX=""
ALL_INSTANCES=0

# Runtime / debug
DEVICE=""
MAX_FRAMES=""
SAVE_FRAMES=0
SAVE_DEBUG=0
ENABLE_OPTICAL_FLOW=0
DISABLE_OPTICAL_FLOW=0
ALLOW_YOLO_ONLY_FALLBACK=0

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_mask_pipeline.sh [options]

脚本说明：
  这是 `python scripts/run_mask_video.py` 的 Bash 封装。
  它会先激活 `sam3` conda 环境，再运行视频 mask pipeline。

必填参数：
  --input PATH
      输入视频路径。
      例如：Input_video/Unedited/example-4.mp4

  --target-text TEXT
      给 SAM3 的主文本提示词。
      例如：
        --target-text "yellow clock"
        --target-text "person"
        --target-text "the man in white shirt"

推荐的单实例参数：
  --prompt-box X1,Y1,X2,Y2
      推荐用它来指定“我要跟踪的那个具体物体”。
      只要给了这个框，pipeline 就会把它作为 SAM3 tracker 的实例提示。
      例如：
        --prompt-box "790,330,1170,770"

可选的类别参数：
  --yolo-class NAME
      可选的 YOLO fallback 类别。
      只在 YOLO 回退或类别过滤时使用。
      常见可用值：
        person car bus bicycle motorcycle truck

  --target-class NAME
      兼容旧接口的别名参数。
      内部会同时映射成：
        --target-text NAME
        --yolo-class NAME
      新的用法建议直接写 `--target-text`。

输出参数：
  --output PATH
      输出 mask 视频路径。
      例如：
        --output outputs/example-4_masked.mp4

  --config PATH
      YAML 配置文件路径。
      默认值：
        configs/mask_video.yaml

跟踪 / 实例选择：
  --all-instances
      输出所有匹配实例的并集 mask。
      默认行为不是这个；默认只跟踪一个实例。

运行参数：
  --device NAME
      推理设备。
      例如：
        --device cpu
        --device cuda
        --device cuda:0
      如果不写，就使用 Python 配置文件里的默认值。

  --max-frames N
      只处理前 N 帧。
      适合快速测试和调试。
      如果不写，就处理完整视频。

调试 / 保存相关：
  --save-frames
      保存逐帧二值 mask 图片到：
        outputs/<stem>/masks/*.png

  --save-debug
      保存逐帧调试 overlay 图片到：
        outputs/<stem>/frames/*.png

光流 refinement：
  --enable-optical-flow
      开启可选的光流 mask refinement。
      注意：这不是主分割信号，只是后处理增强。

  --disable-optical-flow
      强制关闭光流 refinement。

Fallback 模式：
  --allow-yolo-only-fallback
      即使 SAM3 不可用，也允许只用 YOLO fallback 跑通。
      适合 SAM3 环境有问题时做调试。

环境相关参数：
  --conda-sh PATH
      conda.sh 的路径。
      默认值：
        /home/czy/anaconda3_2/etc/profile.d/conda.sh

  --conda-env NAME
      conda 环境名。
      默认值：
        sam3

  --python-bin CMD
      激活环境后的 Python 可执行命令名。
      默认值：
        python

示例：
  1. 跟踪一个具体的时钟：
     bash scripts/run_mask_pipeline.sh \
       --input "Input_video/Unedited/example-4.mp4" \
       --target-text "yellow clock" \
       --prompt-box "790,330,1170,770" \
       --output "outputs/example-4_masked.mp4"

  2. 跟踪一个具体的人：
     bash scripts/run_mask_pipeline.sh \
       --input "Input_video/Unedited/example-7.mp4" \
       --target-text "person" \
       --yolo-class "person" \
       --prompt-box "520,180,780,719" \
       --output "outputs/example-7_masked.mp4"

  3. 跑完整视频并额外保存逐帧 mask 图片：
     bash scripts/run_mask_pipeline.sh \
       --input "Input_video/Unedited/example-5.mp4" \
       --target-text "pillow" \
       --prompt-box "60,60,719,620" \
       --save-frames \
       --output "outputs/example-5_masked.mp4"

EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_VIDEO="$2"
      shift 2
      ;;
    --output)
      OUTPUT_VIDEO="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --target-text)
      TARGET_TEXT="$2"
      shift 2
      ;;
    --yolo-class)
      YOLO_CLASS="$2"
      shift 2
      ;;
    --target-class)
      TARGET_CLASS="$2"
      shift 2
      ;;
    --prompt-box)
      PROMPT_BOX="$2"
      shift 2
      ;;
    --all-instances)
      ALL_INSTANCES=1
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --max-frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --save-frames)
      SAVE_FRAMES=1
      shift
      ;;
    --save-debug)
      SAVE_DEBUG=1
      shift
      ;;
    --enable-optical-flow)
      ENABLE_OPTICAL_FLOW=1
      shift
      ;;
    --disable-optical-flow)
      DISABLE_OPTICAL_FLOW=1
      shift
      ;;
    --allow-yolo-only-fallback)
      ALLOW_YOLO_ONLY_FALLBACK=1
      shift
      ;;
    --conda-sh)
      CONDA_SH="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_VIDEO}" ]]; then
  echo "缺少必填参数: --input" >&2
  usage
  exit 1
fi

if [[ -n "${TARGET_CLASS}" ]]; then
  if [[ -z "${TARGET_TEXT}" ]]; then
    TARGET_TEXT="${TARGET_CLASS}"
  fi
  if [[ -z "${YOLO_CLASS}" ]]; then
    YOLO_CLASS="${TARGET_CLASS}"
  fi
fi

if [[ -z "${TARGET_TEXT}" ]]; then
  echo "缺少必填参数: --target-text" >&2
  usage
  exit 1
fi

if [[ "${ENABLE_OPTICAL_FLOW}" -eq 1 && "${DISABLE_OPTICAL_FLOW}" -eq 1 ]]; then
  echo "不能同时设置 --enable-optical-flow 和 --disable-optical-flow" >&2
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "找不到 conda.sh: ${CONDA_SH}" >&2
  exit 1
fi

PY_ARGS=(
  scripts/run_mask_video.py
  --input "${INPUT_VIDEO}"
  --config "${CONFIG_PATH}"
  --target-text "${TARGET_TEXT}"
)

if [[ -n "${OUTPUT_VIDEO}" ]]; then
  PY_ARGS+=(--output "${OUTPUT_VIDEO}")
fi

if [[ -n "${YOLO_CLASS}" ]]; then
  PY_ARGS+=(--yolo-class "${YOLO_CLASS}")
fi

if [[ -n "${PROMPT_BOX}" ]]; then
  PY_ARGS+=(--prompt-box "${PROMPT_BOX}")
fi

if [[ "${ALL_INSTANCES}" -eq 1 ]]; then
  PY_ARGS+=(--all-instances)
fi

if [[ -n "${DEVICE}" ]]; then
  PY_ARGS+=(--device "${DEVICE}")
fi

if [[ -n "${MAX_FRAMES}" ]]; then
  PY_ARGS+=(--max-frames "${MAX_FRAMES}")
fi

if [[ "${SAVE_FRAMES}" -eq 1 ]]; then
  PY_ARGS+=(--save-frames)
fi

if [[ "${SAVE_DEBUG}" -eq 1 ]]; then
  PY_ARGS+=(--save-debug)
fi

if [[ "${ENABLE_OPTICAL_FLOW}" -eq 1 ]]; then
  PY_ARGS+=(--enable-optical-flow)
fi

if [[ "${DISABLE_OPTICAL_FLOW}" -eq 1 ]]; then
  PY_ARGS+=(--disable-optical-flow)
fi

if [[ "${ALLOW_YOLO_ONLY_FALLBACK}" -eq 1 ]]; then
  PY_ARGS+=(--allow-yolo-only-fallback)
fi

echo "项目根目录: ${PROJECT_ROOT}"
echo "Conda 环境: ${CONDA_ENV}"
echo "输入视频: ${INPUT_VIDEO}"
echo "目标文本: ${TARGET_TEXT}"
if [[ -n "${YOLO_CLASS}" ]]; then
  echo "YOLO 类别: ${YOLO_CLASS}"
fi
if [[ -n "${PROMPT_BOX}" ]]; then
  echo "Prompt box: ${PROMPT_BOX}"
fi
if [[ -n "${OUTPUT_VIDEO}" ]]; then
  echo "输出视频: ${OUTPUT_VIDEO}"
fi
if [[ -n "${MAX_FRAMES}" ]]; then
  echo "最大帧数: ${MAX_FRAMES}"
else
  echo "最大帧数: 完整视频"
fi
echo "光流开关: $([[ "${ENABLE_OPTICAL_FLOW}" -eq 1 ]] && echo 已开启 || ([[ "${DISABLE_OPTICAL_FLOW}" -eq 1 ]] && echo 已关闭 || echo 使用配置默认值))"

cd "${PROJECT_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

exec "${PYTHON_BIN}" "${PY_ARGS[@]}"
