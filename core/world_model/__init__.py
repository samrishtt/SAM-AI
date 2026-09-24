"""World Modeling Subsystem.

Exposes:
- WorldState: Snapshot of latent and symbolic state.
- PredictedTransition: Outcome of simulated counterfactual step.
- WorldModel: Mental sandbox for predictive rollouts and constraint verification.
"""

from core.world_model.latent_simulator import WorldState, PredictedTransition, WorldModel

__all__ = ["WorldState", "PredictedTransition", "WorldModel"]
