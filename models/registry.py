from __future__ import annotations
from typing import Any, Dict

# Import integrated models
# Each model should be imported from its subdirectory
from models.tgcn import TGCNWrapper
from models.tgcn_att.wrapper import TGCNAttWrapper
from models.tgcn_direct.wrapper import TGCNDirectWrapper
from models.ktgcn.wrapper import KTGCNWrapper

def create_model(name: str, cfg: Dict[str, Any]):
    """
    Factory for models.

    Contract:
      - forward(x, adj=None, **kwargs) -> y_hat
      - x: [B, Tin, N, F]
      - y_hat: [B, Tout, N, 1]
    """
    name = name.lower()

    if name == "tgcn":
        return TGCNWrapper(**cfg)
    if name == "tgcn_att":
        return TGCNAttWrapper(**cfg)
    if name == "tgcn_direct":
        return TGCNDirectWrapper(**cfg)
    if name == "ktgcn":
        return KTGCNWrapper(**cfg)
    # Unknown model
    raise NotImplementedError(
        f"Model '{name}' not integrated yet. "
        "Provide source code and I will wrap it into the unified interface."
    )
