# AnyGrasp Runtime Assets

This directory is the local runtime boundary for the licensed AnyGrasp SDK.
It is intentionally excluded from version control except for this document.

Expected local layout:

```text
thirdparty/anygrasp/
|-- checkpoint_detection.tar
`-- sdk/
    `-- grasp_detection/
        |-- gsnet_versions/
        |   `-- license/
        `-- license/
```

The official SDK also needs its bundled `pointnet2/` sources. Install the
private CUDA extensions with:

```bash
TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=1 \
  bash aura_hardware/aura_isaac_bridge/thirdparty/anygrasp/install_minkowski_engine.sh
```

The installer builds MinkowskiEngine and PointNet++ against Isaac Sim's
Python 3.11/PyTorch CUDA 12.8 environment and stores the resulting egg under
`thirdparty/anygrasp/dependencies/`. It does not modify the system CUDA or
Isaac Sim installation.

The SDK, checkpoint, license files, machine feature IDs, and generated
diagnostics are private or license-controlled runtime assets. Do not commit or
publish them. Configure the paths in `aura_bringup/config/config.yaml` when
using a different local installation. The SDK and runtime assets are kept
inside this Aura hardware package so the ROS workspace remains self-contained.

AuraVLA uses the official `gsnet.create_detector()` API and keeps AnyGrasp
mandatory for grasp actions. When the SDK, checkpoint, or license is missing,
the task fails closed with `ANYGRASP_UNAVAILABLE`.
