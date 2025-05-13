# Import TorchX APIs. This works because this file is run in the context of
# our internally built TorchX CLI.
from typing import Dict, List, Optional

import torchx.components.fb.conda as conda
import torchx.specs as specs
from torchx.components.fb import parse_j


# Environment variables to set for the job
env_vars = {
    "DISABLE_NFS": "1",
    "DISABLE_OILFS": "1",
    "MANIFOLDFS_BUCKET": "ondevice_ai_writedata",
    "LD_PRELOAD": "/usr/local/fbcode/platform010/lib/libcuda.so:/usr/local/fbcode/platform010/lib/libnvidia-ml.so",
    "TRITON_LIBCUDA_PATH": "/usr/local/fbcode/platform010/lib/libcuda.so",
}

# Packages to add to the job
additional_packages = [
    "torchx_conda_mount:stable",
    "manifold.manifoldfs:prod",
]


# Defines a custom TorchX component, mast.py:train, along with defaults
# This function constructs and returns a job spec (AppDef)
def ddp(
    *main_args: str,
    m: str = "train.py",
    name: str = "nanogpt",
    h: str = "tc_any",
    j: str = "1x2",
    run_as_root: bool = True,
    env: Optional[Dict[str, str]] = None,
    # out_dir: str = "/mnt/mffuse/out/${app_id}",
    # tb_log: bool = True,
) -> specs.AppDef:
    # Component arguments for fb.conda.torchrun
    kwargs = {
        "name": name,
        "h": h,
        "run_as_root": run_as_root,
        "env": {**env_vars, **env} if env else env_vars,
    }
    nnodes, nproc_per_node = parse_j(j)

    # Application arguments
    args = [
        "--nnodes",
        str(nnodes),
        "--nproc-per-node",
        str(nproc_per_node),
        m,
        *main_args,
        # f"--out_dir={out_dir}",
        # f"--tb_log={tb_log}",
    ]

    # Call the base component to construct the job spec
    job_spec = conda.torchrun(*args, **kwargs)

    # Add additional packages
    packages = [job_spec.roles[0].image, *additional_packages]
    job_spec.roles[0].image = ";".join(packages)

    # Add the mount script to the entrypoint
    if run_as_root:
        job_spec.roles[0].entrypoint = (
            f"/packages/torchx_conda_mount/mount.sh && {job_spec.roles[0].entrypoint}"
        )

    # Return the job spec
    return job_spec
