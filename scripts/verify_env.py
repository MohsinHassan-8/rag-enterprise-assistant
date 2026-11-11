"""Verify the environment: GPU availability and core imports.

Exits with code 1 if any check fails.
"""
import importlib
import sys

REQUIRED_IMPORTS = [
    "pandas",
    "numpy",
    "pyarrow",
    "sentence_transformers",
    "chromadb",
    "spacy",
    "bs4",
    "langdetect",
    "datasets",
]


def check_imports() -> list[str]:
    """Import every required package, print PASS or FAIL for each, and return the failed names."""
    failures = []
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "n/a")
            print(f"[PASS] import {name} ({version})")
        except Exception as exc:
            print(f"[FAIL] import {name}: {exc}")
            failures.append(name)
    return failures


def check_spacy_model() -> bool:
    """True if the spaCy model en_core_web_sm loads."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("[PASS] spaCy model en_core_web_sm loads")
        return True
    except Exception as exc:
        print(f"[FAIL] spaCy model en_core_web_sm: {exc}")
        return False


def check_cuda() -> bool:
    """True if torch can see a CUDA GPU. Prints the device name and memory."""
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"[FAIL] CUDA not available (torch {torch.__version__})")
            return False
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[PASS] CUDA available: {name} ({mem_gb:.1f} GB), torch {torch.__version__}")
        return True
    except Exception as exc:
        print(f"[FAIL] CUDA check: {exc}")
        return False


def main() -> int:
    """Run all checks. Returns 0 if everything passed and 1 otherwise."""
    print(f"Python {sys.version.split()[0]}")
    failed_imports = check_imports()
    spacy_ok = check_spacy_model()
    cuda_ok = check_cuda()

    if failed_imports or not spacy_ok or not cuda_ok:
        print("\nEnvironment check FAILED")
        return 1
    print("\nEnvironment check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())