"""
generate_research_physics_figures.py
Generates publication-grade semiconductor failure physics schematics:
1. fig15_cmp_scratch_dilution_mechanics.png
2. fig16_circular_aliasing_geometry.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge, Arc, PathPatch, ConnectionPatch
from matplotlib.path import Path
import matplotlib.gridspec as gridspec
from PIL import Image

OUT_DIR = 'research_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Global Publication-Grade Matplotlib Configurations
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10.0
plt.rcParams['axes.edgecolor'] = '#94A3B8'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['text.color'] = '#0F172A'
plt.rcParams['axes.labelcolor'] = '#0F172A'
plt.rcParams['xtick.color'] = '#334155'
plt.rcParams['ytick.color'] = '#334155'

# Publication Palette
NAVY_DEEP   = '#0A192F'
NAVY_MID    = '#1E293B'
SLATE_MUTED = '#64748B'
SLATE_LIGHT = '#F1F5F9'
BORDER_GRAY = '#CBD5E1'
WHITE       = '#FFFFFF'

# Physical Materials Palette
COPPER_ORANGE = '#EA580C'
COPPER_LIGHT  = '#FDBA74'
DIELECTRIC_CYAN = '#0284C7'
DIELECTRIC_LIGHT = '#E0F2FE'
SILICON_GRAY  = '#334155'
SILICON_LIGHT = '#94A3B8'
PAD_AMBER     = '#D97706'
PAD_LIGHT     = '#FEF3C7'
SLURRY_BLUE   = '#BAE6FD'
BARRIER_DARK  = '#475569'

# Defect Palette
SCRATCH_RED   = '#DC2626'
SCRATCH_LIGHT = '#FEE2E2'
CLUSTER_PURP  = '#6366F1'
CLUSTER_LIGHT = '#E0E7FF'
RING_CRIMSON  = '#B91C1C'
LOCAL_AMBER   = '#F59E0B'
NORMAL_TEAL   = '#10B981'
NORMAL_LIGHT  = '#ECFDF5'

print("Script template ready.")
