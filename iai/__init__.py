from . import data  # register all new datasets
from . import model
from . import utils
from . import evaluator
# config
from .config import add_iai_config

# dataset loading
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.iai_gaze_dataset_mapper import iaiGazeDatasetMapper
from .data.dataset_mappers.Tiai_gaze_dataset_mapper import TiaiGazeDatasetMapper
from .data.dataset_mappers.Tiai_gaze_dataset_mapper_e2e import TiaiGazeDatasetMapper_e2e
from .data.dataset_mappers.Tiai_gaze_dataset_mapper_e2e_v2 import TiaiGazeDatasetMapper_e2e_v2
# models
from .test_time_augmentation import SemanticSegmentorWithTTA
