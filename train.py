import os
import runpy
from pathlib import Path

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    runpy.run_module("src.training.train", run_name="__main__")
