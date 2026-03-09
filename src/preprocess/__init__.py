"""Preprocessing components."""

from src.preprocess.base import ComposablePreprocessor
from src.preprocess.dwt_features import DWTFeatureExtractor
from src.preprocess.feature_union import FeatureUnion
from src.preprocess.identity import IdentityPreprocessor
from src.preprocess.interval_features import (
    IntervalMeanFeatureExtractor,
    IntervalSlopeFeatureExtractor,
)
from src.preprocess.pipeline import PreprocessingPipeline
from src.preprocess.preprocessor_builder import PreprocessorBuilder
from src.preprocess.snv import SNVPreprocessor
from src.preprocess.water_band_summary import WaterBandSummaryFeatureExtractor

__all__ = [
    "ComposablePreprocessor",
    "DWTFeatureExtractor",
    "FeatureUnion",
    "IdentityPreprocessor",
    "IntervalMeanFeatureExtractor",
    "IntervalSlopeFeatureExtractor",
    "PreprocessingPipeline",
    "PreprocessorBuilder",
    "SNVPreprocessor",
    "WaterBandSummaryFeatureExtractor",
]
