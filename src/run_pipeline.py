"""
Standalone runner for the VOG-MCI detection pipeline.
Sets Agg backend so plt.show() is non-blocking in headless mode.
Run from the src/ directory.
"""
import matplotlib
matplotlib.use('Agg')   # must come before any other matplotlib import

import os
import sys

# Ensure relative paths (../data, ../xai_difference_map.png, etc.) resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- Execute the converted notebook script ----
exec(open("wavelet_transform_detection.txt").read())
