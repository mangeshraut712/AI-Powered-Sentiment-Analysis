"""Regression tests for the module-level bootstrap of scripts/run_analysis.py."""

import importlib.util
import sys
import types
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_analysis.py"

# Heavy project modules are stubbed so this smoke test only exercises the
# script's own top-level code, not the scientific stack or the dataset.
STUBBED_MODULES = {
    "src.data_loader": ["load_dataset", "get_dataset_info", "save_processed_data"],
    "src.feature_engineering": [
        "create_tfidf_features",
        "extract_keywords",
        "add_text_statistics",
    ],
    "src.models": [
        "EmotionClassifier",
        "train_test_split_data",
        "compare_models",
    ],
    "src.visualization": [
        "plot_sentiment_distribution",
        "plot_word_cloud",
        "plot_confusion_matrix",
        "plot_model_comparison",
    ],
}


def test_run_analysis_module_level_code_executes(monkeypatch):
    # Regression test: the script used ``Path`` in its sys.path bootstrap
    # (scripts/run_analysis.py line 15) before ``from pathlib import Path``
    # ran, so every invocation died immediately with
    # ``NameError: name 'Path' is not defined`` and the emotion model could
    # never be trained (src/app.py tells users to run this script).
    for module_name, attributes in STUBBED_MODULES.items():
        stub = types.ModuleType(module_name)
        for attribute in attributes:
            setattr(stub, attribute, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, stub)
    monkeypatch.setattr(sys, "path", list(sys.path))

    spec = importlib.util.spec_from_file_location(
        "run_analysis_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)

    assert callable(module.main)
    assert str(REPO_ROOT) in sys.path


def test_ensure_output_dirs_creates_every_save_target(monkeypatch, tmp_path):
    # A fresh checkout has no data/processed or results/models|metrics, and the
    # save step used to fail on the first to_csv before any model was written.
    for module_name, attributes in STUBBED_MODULES.items():
        stub = types.ModuleType(module_name)
        for attribute in attributes:
            setattr(stub, attribute, lambda *args, **kwargs: None)
        monkeypatch.setitem(sys.modules, module_name, stub)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("run_analysis_dirs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.ensure_output_dirs(tmp_path)
    module.ensure_output_dirs(tmp_path)  # idempotent
    for rel in ("data/processed", "results/models", "results/metrics"):
        assert (tmp_path / rel).is_dir()
