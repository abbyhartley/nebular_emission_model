# normflow is a python package that will contain all the functions I frequenctly reuse in this modeling project
# I'll import these functions as a package and call them in Sherlock
from .selection import DESISelectionConfig, desi_bgs_training_mask

__all__ = ["DESISelectionConfig", "desi_bgs_training_mask"]
