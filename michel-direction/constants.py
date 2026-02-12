# Authored by Hilary Utaegbulam 

"""Constants and class mappings."""
from __future__ import annotations
import argparse

Config = argparse.Namespace


# PDG codes

PDG_ELECTRON_POS = 11
PDG_ELECTRON_NEG = -11
PDG_MUON_POS = 13
PDG_MUON_NEG = -13
PDG_MICHEL = 9999
PDG_BREMS = 10000


# Segmentation class IDs

CLASS_BG = 0
CLASS_ELECTRON = 1
CLASS_MUON = 2
CLASS_MICHEL = 3
CLASS_BREMS = 4

CLASS_ID_TO_NAME = {
    CLASS_BG: "Background",
    CLASS_ELECTRON: "Electron",
    CLASS_MUON: "Muon",
    CLASS_MICHEL: "Michel",
    CLASS_BREMS: "Brems",
}

IGNORE_INDEX = 255


# Detector geometry

PARTITION_SEED = 12345
PARTITION_FRAC = 1.0
TRAIN_FRACTION = 1.0

# Drift / pitch / angle constants for ICEBERG geometry
DRIFT_VELOCITY_CM_PER_US = 0.16
TICK_PERIOD_US = 0.4
DRIFT_CM_PER_TICK = DRIFT_VELOCITY_CM_PER_US * TICK_PERIOD_US

WIRE_PITCH_U_CM = 0.4669
WIRE_PITCH_V_CM = 0.4669
WIRE_PITCH_Z_CM = 0.4792

WIRE_ANGLE_U_DEG = -35.71
WIRE_ANGLE_V_DEG = 35.71
WIRE_ANGLE_Z_DEG = 0.0
