"""Real learner components for teachability-compiler."""

from teachability_compiler.real.model import DecoderConfig, TinyDecoder, count_parameters
from teachability_compiler.real.oracle import RealLearnerOracle
from teachability_compiler.real.tasks import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    PROBE_CLUSTERS,
    VOCAB_SIZE,
    all_cluster_names,
    probe_batch,
    sample_batch,
)

__all__ = [
    "BOS_ID",
    "EOS_ID",
    "PAD_ID",
    "PROBE_CLUSTERS",
    "VOCAB_SIZE",
    "DecoderConfig",
    "RealLearnerOracle",
    "TinyDecoder",
    "all_cluster_names",
    "count_parameters",
    "probe_batch",
    "sample_batch",
]
