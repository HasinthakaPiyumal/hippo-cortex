# Expose models subpackage exports
from hippocortex.models.swr_generator import SWRGenerator

try:
    from hippocortex.models.backbone import MambaBackbone
except ImportError:
    pass

