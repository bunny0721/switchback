"""Data generating processes for switchback experiments.

A DGP maps an assignment path of length T to an outcome path of length T.
The contract is intentionally narrow so any source of outcomes — closed-form
models, full simulators, or learned replay engines — can be plugged into the
rest of the pipeline.
"""

from switchback.dgp.base import BaseDGP
from switchback.dgp.carryover import CarryoverDGP
from switchback.dgp.simple import SimpleDGP
from switchback.dgp.state_space import LatentStateDGP

__all__ = ["BaseDGP", "SimpleDGP", "CarryoverDGP", "LatentStateDGP"]
