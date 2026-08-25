#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/home/acmex/Code/learning/isaacsim}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
MINKOWSKI_DIR="${SCRIPT_DIR}/dependencies/MinkowskiEngine"
POINTNET2_DIR="${SCRIPT_DIR}/sdk/pointnet2"
POINTNET2_INSTALL_DIR="${SCRIPT_DIR}/dependencies/python"
CUDA_TOOLKIT_VIEW="${SCRIPT_DIR}/dependencies/.cuda-toolkit"
MINKOWSKI_COMPAT_PATCH="${SCRIPT_DIR}/minkowski_cuda128_compat.patch"

find_nvcc() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}/bin/nvcc"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi

  # Isaac Sim commonly runs inside a Conda environment where nvcc is shipped
  # as a package but is intentionally not added to PATH.
  local candidate
  for candidate in \
    "${CONDA_PREFIX:-}/pkgs"/cuda-nvcc-tools-*/bin/nvcc \
    "/home/acmex/miniconda3/pkgs"/cuda-nvcc-tools-*/bin/nvcc \
    "/usr/local/cuda"*/bin/nvcc; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "ERROR: Isaac Python launcher not found: ${ISAAC_PYTHON}" >&2
  exit 1
fi

# Keep host-user packages (including other Python minor versions) out of
# Isaac's embedded interpreter and the pip subprocess spawned by setup.py.
export PYTHONNOUSERSITE=1

NVCC_PATH="$(find_nvcc || true)"
if [[ -z "${NVCC_PATH}" ]]; then
  echo "ERROR: nvcc is required to build AnyGrasp's CUDA MinkowskiEngine." >&2
  echo "Install a CUDA Toolkit compatible with Isaac Sim/PyTorch ${TORCH_CUDA_VERSION:-12.8}, then rerun." >&2
  exit 1
fi

NVCC_ROOT="$(dirname "$(dirname "${NVCC_PATH}")")"
CUDA_NVVM_DIR="${ANYGRASP_CUDA_NVVM_DIR:-}"
CUDA_DEV_ROOT="${ANYGRASP_CUDA_DEV_ROOT:-}"
CUDA_RUNTIME_ROOT="${ANYGRASP_CUDA_RUNTIME_ROOT:-}"
CUDA_CRT_INCLUDE_DIR="${ANYGRASP_CUDA_CRT_INCLUDE_DIR:-}"
if [[ -z "${CUDA_DEV_ROOT}" ]]; then
  CUDA_DEV_ROOT="$(find /home/acmex/miniconda3/pkgs -path '*/targets/x86_64-linux/include/cuda_runtime.h' -printf '%h\n' 2>/dev/null | head -1 || true)"
  CUDA_DEV_ROOT="${CUDA_DEV_ROOT%/include}"
fi
if [[ -z "${CUDA_NVVM_DIR}" ]]; then
  CUDA_NVVM_DIR="$(find /home/acmex/miniconda3/pkgs -path '*/nvvm/bin/cicc' -printf '%h\n' 2>/dev/null | head -1 || true)"
  CUDA_NVVM_DIR="${CUDA_NVVM_DIR%/bin}"
fi
if [[ -z "${CUDA_RUNTIME_ROOT}" ]]; then
  CUDA_RUNTIME_ROOT="$(find /home/acmex/miniconda3/pkgs -path '*/targets/x86_64-linux/lib/libcudart.so.12' -printf '%h\n' 2>/dev/null | head -1 || true)"
  CUDA_RUNTIME_ROOT="${CUDA_RUNTIME_ROOT%/lib}"
fi
if [[ -z "${CUDA_CRT_INCLUDE_DIR}" ]]; then
  CUDA_CRT_INCLUDE_DIR="$(find /home/acmex/miniconda3/pkgs -path '*/include/crt/host_config.h' -printf '%h\n' 2>/dev/null | head -1 || true)"
  CUDA_CRT_INCLUDE_DIR="${CUDA_CRT_INCLUDE_DIR%/crt}"
fi
if [[ ! -d "${CUDA_DEV_ROOT}/include" || ! -d "${CUDA_RUNTIME_ROOT}/lib" || ! -d "${CUDA_CRT_INCLUDE_DIR}/crt" || ! -x "${CUDA_NVVM_DIR}/bin/cicc" ]]; then
  echo "ERROR: CUDA development headers or runtime libraries are unavailable." >&2
  echo "Set ANYGRASP_CUDA_DEV_ROOT, ANYGRASP_CUDA_RUNTIME_ROOT, ANYGRASP_CUDA_CRT_INCLUDE_DIR, and ANYGRASP_CUDA_NVVM_DIR to valid CUDA toolkit paths." >&2
  exit 1
fi

# Conda distributes nvcc, CUDA headers, and CUDA libraries as separate
# packages. PyTorch's extension build expects one CUDA_HOME, so expose a
# private merged view without modifying Conda or Isaac Sim installation.
mkdir -p "${CUDA_TOOLKIT_VIEW}/bin" "${CUDA_TOOLKIT_VIEW}/lib"
for cuda_tool in "${NVCC_ROOT}/bin"/*; do
  [[ -e "${cuda_tool}" ]] || continue
  ln -sfn "${cuda_tool}" "${CUDA_TOOLKIT_VIEW}/bin/$(basename "${cuda_tool}")"
done
ln -sfn "${CUDA_NVVM_DIR}" "${CUDA_TOOLKIT_VIEW}/nvvm"
printf '#!/usr/bin/env bash\nexec "%s" -m pip "$@"\n' "${ISAAC_PYTHON}" \
  > "${CUDA_TOOLKIT_VIEW}/bin/pip"
chmod +x "${CUDA_TOOLKIT_VIEW}/bin/pip"
if [[ -L "${CUDA_TOOLKIT_VIEW}/include" ]]; then
  rm "${CUDA_TOOLKIT_VIEW}/include"
fi
mkdir -p "${CUDA_TOOLKIT_VIEW}/include"

# CUDA 12 Conda packages split runtime, CRT, CCCL/Thrust, and compiler
# headers. Merge their top-level entries into the private include directory.
while IFS= read -r cuda_include_dir; do
  [[ -n "${cuda_include_dir}" ]] || continue
  for header in "${cuda_include_dir}"/*; do
    [[ -e "${header}" ]] || continue
    ln -sfn "${header}" "${CUDA_TOOLKIT_VIEW}/include/$(basename "${header}")"
  done
done < <(
  find /home/acmex/miniconda3/pkgs -path '*/targets/x86_64-linux/include' \
    -type d -printf '%p\n' 2>/dev/null | sort -u
)
while IFS= read -r cuda_include_dir; do
  [[ -n "${cuda_include_dir}" ]] || continue
  for header in "${cuda_include_dir}"/*; do
    [[ -e "${header}" ]] || continue
    ln -sfn "${header}" "${CUDA_TOOLKIT_VIEW}/include/$(basename "${header}")"
  done
done < <(
  find "${ISAAC_SIM_ROOT}/exts" -path '*/pip_prebundle/nvidia/*/include' \
    -type d -printf '%p\n' 2>/dev/null | sort -u
)
ln -sfn "${CUDA_RUNTIME_ROOT}/lib" "${CUDA_TOOLKIT_VIEW}/lib64"
for library in "${CUDA_RUNTIME_ROOT}/lib"/*; do
  [[ -e "${library}" ]] || continue
  ln -sfn "${library}" "${CUDA_TOOLKIT_VIEW}/lib/$(basename "${library}")"
done

while IFS= read -r cuda_library_dir; do
  [[ -n "${cuda_library_dir}" ]] || continue
  for library in "${cuda_library_dir}"/*; do
    [[ -e "${library}" ]] || continue
    ln -sfn "${library}" "${CUDA_TOOLKIT_VIEW}/lib/$(basename "${library}")"
  done
done < <(
  find "${ISAAC_SIM_ROOT}/exts" -path '*/pip_prebundle/nvidia/*/lib/libcusparse.so*' \
    -printf '%h\n' 2>/dev/null | sort -u
)

# Conda and Isaac's bundled CUDA libraries commonly expose only versioned
# SONAMEs. The extension linker still asks for the conventional unversioned
# names, so provide those aliases inside the private toolkit view.
if [[ ! -e "${CUDA_TOOLKIT_VIEW}/lib/libcudart.so" && -e "${CUDA_TOOLKIT_VIEW}/lib/libcudart.so.12" ]]; then
  ln -sfn libcudart.so.12 "${CUDA_TOOLKIT_VIEW}/lib/libcudart.so"
fi
if [[ ! -e "${CUDA_TOOLKIT_VIEW}/lib/libcusparse.so" && -e "${CUDA_TOOLKIT_VIEW}/lib/libcusparse.so.12" ]]; then
  ln -sfn libcusparse.so.12 "${CUDA_TOOLKIT_VIEW}/lib/libcusparse.so"
fi

CUDA_HOME="${CUDA_TOOLKIT_VIEW}"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
echo "Using CUDA compiler: ${NVCC_PATH}"
echo "Using CUDA root: ${CUDA_HOME}"

if [[ ! -d "${MINKOWSKI_DIR}" ]]; then
  mkdir -p "${SCRIPT_DIR}/dependencies"
  git clone --depth 1 --branch cuda-12-1 \
    https://github.com/chenxi-wang/MinkowskiEngine.git "${MINKOWSKI_DIR}"
fi
if ! grep -q 'return std::shared_ptr<Self>' "${MINKOWSKI_DIR}/src/3rdparty/concurrent_unordered_map.cuh"; then
  if [[ ! -f "${MINKOWSKI_COMPAT_PATCH}" ]]; then
    echo "ERROR: Missing MinkowskiEngine CUDA 12.8 compatibility patch: ${MINKOWSKI_COMPAT_PATCH}" >&2
    exit 1
  fi
  git -C "${MINKOWSKI_DIR}" apply --check "${MINKOWSKI_COMPAT_PATCH}"
  git -C "${MINKOWSKI_DIR}" apply "${MINKOWSKI_COMPAT_PATCH}"
  echo "Applied MinkowskiEngine CUDA 12.8 compatibility patch."
fi

BLAS_NAME="blas"
BLAS_INCLUDE_DIR="/usr/include"
BLAS_LIBRARY_DIR="/usr/lib/x86_64-linux-gnu/blas"
if [[ -f "${CONDA_PREFIX:-}/include/openblas_config.h" ]]; then
  BLAS_NAME="openblas"
  BLAS_INCLUDE_DIR="${CONDA_PREFIX}/include"
  BLAS_LIBRARY_DIR="${CONDA_PREFIX}/lib"
elif [[ -f "/usr/include/openblas_config.h" ]]; then
  BLAS_NAME="openblas"
  BLAS_INCLUDE_DIR="/usr/include"
  BLAS_LIBRARY_DIR="/usr/lib/x86_64-linux-gnu"
fi

echo "Using BLAS: ${BLAS_NAME}"
cd "${MINKOWSKI_DIR}"
"${ISAAC_PYTHON}" setup.py install \
  --force_cuda \
  --blas="${BLAS_NAME}" \
  --cuda_home="${CUDA_HOME}" \
  --blas_include_dirs="${BLAS_INCLUDE_DIR}" \
  --blas_library_dirs="${BLAS_LIBRARY_DIR}"

"${ISAAC_PYTHON}" -c "import MinkowskiEngine; print('MinkowskiEngine installation verified')"

if [[ ! -d "${POINTNET2_DIR}" ]]; then
  echo "ERROR: AnyGrasp PointNet++ sources are missing: ${POINTNET2_DIR}" >&2
  exit 1
fi
cd "${POINTNET2_DIR}"
mkdir -p "${POINTNET2_INSTALL_DIR}"
"${ISAAC_PYTHON}" setup.py install --install-lib "${POINTNET2_INSTALL_DIR}"
POINTNET2_EGG="$(find "${POINTNET2_INSTALL_DIR}" -maxdepth 1 -type d -name 'pointnet2-*.egg' -print -quit)"
if [[ -z "${POINTNET2_EGG}" ]]; then
  echo "ERROR: PointNet++ installation did not produce a private egg." >&2
  exit 1
fi
PYTHONPATH="${POINTNET2_EGG}:${PYTHONPATH:-}" \
  "${ISAAC_PYTHON}" -c "import pointnet2._ext; print('AnyGrasp PointNet++ extension verified')"
echo "AnyGrasp MinkowskiEngine dependency is ready for Isaac Python."
