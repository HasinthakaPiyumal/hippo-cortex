import os
import torch
from baselines.inf_ssm.utils.config import build_model, get_config
from baselines.inf_ssm.utils.inf_ssm_loss import InfSSMLoss

def test_inf_ssm_integration():
    # Resolve config file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    config_path = os.path.join(project_root, "baselines/inf-ssm/utils/defocus_mamba_large_22k.yaml")
    
    # Load config and build model
    config = get_config(config_path)
    
    # Build model (on CPU for fast unit testing)
    model = build_model(config)
    
    # Create fake image batch (B=1, C=3, H=192, W=192) matching Mamba-CL resolution
    B = 1
    dummy_images = torch.randn(B, 3, 192, 192)
    
    # Run forward pass
    _ = model(dummy_images)
    
    # Collect extracted states
    extracted_states = []
    for name, module in model.named_modules():
        if hasattr(module, "extracted_A"):
            extracted_states.append((module.extracted_A, module.extracted_C))
            
    # Check that states were extracted from the MambaMixer blocks
    assert len(extracted_states) > 0, "No states extracted!"
    
    # Validate the shapes of extracted state transitions (A) and output mapping (C)
    for A, C in extracted_states:
        # A should be (B, L, O, N) where L is sequence length, O is channel dimension, N is state size
        # C should be (B, L, N)
        assert A.dim() == 4
        assert C.dim() == 3
        assert A.shape[0] == B
        assert C.shape[0] == B
        assert C.shape[1] == A.shape[1]  # Sequence length must match
        
        # Test loss forward with self
        loss_fn = InfSSMLoss()
        loss = loss_fn(A, C, A, C)
        assert loss.item() < 1e-4  # Distance to self should be near 0
