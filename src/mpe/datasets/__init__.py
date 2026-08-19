from mpe.datasets.base import DatasetLoader
from mpe.datasets.belebele import BelebeleKoLoader
from mpe.datasets.polyguard import PolyGuardPromptsLoader
from mpe.datasets.schema import BenchmarkItem, TaskType
from mpe.datasets.xstest import XSTestLoader

LOADERS: dict[str, type[DatasetLoader]] = {
    PolyGuardPromptsLoader.benchmark_name: PolyGuardPromptsLoader,
    XSTestLoader.benchmark_name: XSTestLoader,
    BelebeleKoLoader.benchmark_name: BelebeleKoLoader,
}

__all__ = [
    "LOADERS",
    "BelebeleKoLoader",
    "BenchmarkItem",
    "DatasetLoader",
    "PolyGuardPromptsLoader",
    "TaskType",
    "XSTestLoader",
]
