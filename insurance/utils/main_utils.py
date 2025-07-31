# utils/path_utils.py
import os

def from_root():
    return os.path.dirname(os.path.abspath(__file__))