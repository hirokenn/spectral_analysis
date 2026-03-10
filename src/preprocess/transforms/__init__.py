"""Preprocessors that transform full spectra."""

from src.preprocess.transforms.identity import IdentityPreprocessor
from src.preprocess.transforms.savgol import SavitzkyGolayPreprocessor
from src.preprocess.transforms.snv import SNVPreprocessor

__all__ = [
    "IdentityPreprocessor",
    "SavitzkyGolayPreprocessor",
    "SNVPreprocessor",
]
