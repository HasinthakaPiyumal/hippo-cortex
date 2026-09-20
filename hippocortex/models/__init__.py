from hippocortex.models.swr_generator import SWRGenerator

try:
    from hippocortex.models.backbone import MambaBackbone
except ModuleNotFoundError as exc:
    if exc.name != "mamba_ssm":
        raise

    MambaBackbone = None