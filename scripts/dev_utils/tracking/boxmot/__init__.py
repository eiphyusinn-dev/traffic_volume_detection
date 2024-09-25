# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '10.0.74'

# from tracking.boxmot.postprocessing.gsi import gsi
from tracking.boxmot.tracker_zoo import create_tracker, get_tracker_config
# from tracking.boxmot.trackers.botsort.bot_sort import BoTSORT
# from tracking.boxmot.trackers.bytetrack.byte_tracker import BYTETracker
# from tracking.boxmot.trackers.deepocsort.deep_ocsort import DeepOCSort as DeepOCSORT
# from tracking.boxmot.trackers.hybridsort.hybridsort import HybridSORT
from tracking.boxmot.trackers.ocsort.ocsort import OCSort as OCSORT
# from tracking.boxmot.trackers.strongsort.strong_sort import StrongSORT

TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort']

__all__ = ("__version__",
           "StrongSORT", "OCSORT", "BYTETracker", "BoTSORT", "DeepOCSORT", "HybridSORT",
           "create_tracker", "get_tracker_config", "gsi")
