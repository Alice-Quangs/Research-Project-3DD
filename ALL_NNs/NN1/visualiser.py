"""
3D Bin Packing Visualizer
Compatible with the EMS-based BinPacking system.
Items are stored as (x_start, y_start, z_start, x_end, y_end, z_end) tuples.
"""

import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ITEM_COLORS = [
    "#5B8DB8", "#7DBE8E", "#C17A5A", "#9B6B9B", "#4A9B8E",
    "#C4A84F", "#7B8DC4", "#C47B7B", "#6BAF8D", "#B87B5B",
    "#8B7BC4", "#7BB8C4", "#C4956B", "#6B8BC4", "#B8845B",
]


def _cuboid_faces(x0, y0, z0, x1, y1, z1):
    return [
        [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
        [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
        [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
        [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
        [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
    ]


def _draw_box(ax, x0, y0, z0, x1, y1, z1, face_color, alpha=0.82):
    """Draw item with a slightly darker edge derived from the face color."""
    import matplotlib.colors as mc
    rgb = mc.to_rgb(face_color)
    dark = tuple(max(0, c * 0.55) for c in rgb)

    poly = Poly3DCollection(
        _cuboid_faces(x0, y0, z0, x1, y1, z1),
        facecolor=face_color,
        edgecolor=dark,
        linewidth=0.8,
        alpha=alpha,
    )
    ax.add_collection3d(poly)


def _draw_bin_outline(ax, length, width, height):
    """Red wireframe bin outline, matching image 2."""
    L, W, H = length, width, height
    edges = [
        ([0,L,L,0,0],[0,0,W,W,0],[0,0,0,0,0]),
        ([0,L,L,0,0],[0,0,W,W,0],[H,H,H,H,H]),
        ([0,0],[0,0],[0,H]),
        ([L,L],[0,0],[0,H]),
        ([L,L],[W,W],[0,H]),
        ([0,0],[W,W],[0,H]),
    ]
    for xs, ys, zs in edges:
        ax.plot(xs, ys, zs, color="red", linewidth=1.6, zorder=10)


def _style_ax(ax, L, W, H):
    """Apply the clean, minimal axis style from image 2."""
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)

    # Tick marks only, no axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    # Sparse ticks so the grid reads cleanly
    n_ticks = 5
    ax.set_xticks(np.linspace(0, L, n_ticks))
    ax.set_yticks(np.linspace(0, W, n_ticks))
    ax.set_zticks(np.linspace(0, H, n_ticks))

    tick_fmt = "{:.0f}"
    ax.set_xticklabels([tick_fmt.format(v) for v in np.linspace(0, L, n_ticks)], fontsize=6)
    ax.set_yticklabels([tick_fmt.format(v) for v in np.linspace(0, W, n_ticks)], fontsize=6)
    ax.set_zticklabels([tick_fmt.format(v) for v in np.linspace(0, H, n_ticks)], fontsize=6)

    # Light grey pane backgrounds — matches image 2
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor("#F0F0F0")
    ax.yaxis.pane.set_facecolor("#F0F0F0")
    ax.zaxis.pane.set_facecolor("#F0F0F0")
    ax.xaxis.pane.set_edgecolor("#CCCCCC")
    ax.yaxis.pane.set_edgecolor("#CCCCCC")
    ax.zaxis.pane.set_edgecolor("#CCCCCC")

    # Fine grid lines
    ax.grid(True, color="#CCCCCC", linewidth=0.4, linestyle="-")

    # Viewing angle close to image 2
    ax.view_init(elev=22, azim=-55)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {os.path.abspath(path)}")


# Single bin 

def visualise_bin(bin_obj, title=None, ax=None, save_path=None, colors=None):
    """Visualize one Bin. Pass ax= to embed in a larger figure (skips save)."""
    if colors is None:
        colors = ITEM_COLORS

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(6, 5))
        ax  = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    L, W, H = bin_obj.length, bin_obj.width, bin_obj.height

    for idx, item in enumerate(bin_obj.placed_items):
        x0, y0, z0, x1, y1, z1 = item
        _draw_box(ax, x0, y0, z0, x1, y1, z1, face_color=colors[idx % len(colors)])

    _draw_bin_outline(ax, L, W, H)
    _style_ax(ax, L, W, H)

    used_vol  = sum((x1-x0)*(y1-y0)*(z1-z0) for x0,y0,z0,x1,y1,z1 in bin_obj.placed_items)
    total_vol = L * W * H
    util_pct  = used_vol / total_vol * 100 if total_vol > 0 else 0

    ax.set_title(
        f"{title or bin_obj.name}\n{len(bin_obj.placed_items)} items  |  {util_pct:.1f}%",
        fontsize=9, pad=6, color="#222222",
    )

    if standalone:
        plt.tight_layout()
        path = save_path or f"{bin_obj.name.replace(' ', '_')}.png"
        _save(fig, path)
        plt.close(fig)

    return fig, ax


# Multi-bin grid 

def visualise_all_bins(open_bins, cols=3, figsize_per_bin=(4.5, 4),
                       suptitle="Bin Packing Result",
                       save_path="bin_packing_result.png"):
    """Render all bins in a tidy grid and save to PNG."""
    n = len(open_bins)
    if n == 0:
        print("No bins to visualize.")
        return None

    rows = (n + cols - 1) // cols
    fig  = plt.figure(
        figsize=(figsize_per_bin[0] * cols, figsize_per_bin[1] * rows),
        facecolor="white",
    )

    for i, bin_obj in enumerate(open_bins):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        visualise_bin(bin_obj, title=f"Bin {i+1} — {bin_obj.name}", ax=ax)

    for j in range(n, rows * cols):
        fig.add_subplot(rows, cols, j + 1).set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.005, color="#111111")

    plt.tight_layout(pad=1.2)
    _save(fig, save_path)
    plt.close(fig)
    return fig


# EMS overlay

def visualise_ems(bin_obj, alpha=0.10, save_path=None):
    """Overlay EMS free-space on top of packed items for debugging."""
    fig, ax = visualise_bin(bin_obj, save_path=None)

    for ems in bin_obj.ems_list:
        x0, y0, z0, x1, y1, z1 = ems
        poly = Poly3DCollection(
            _cuboid_faces(x0, y0, z0, x1, y1, z1),
            facecolor="#00BFFF", edgecolor="#007FBF",
            linewidth=0.4, alpha=alpha,
        )
        ax.add_collection3d(poly)

    packed_patch = mpatches.Patch(color=ITEM_COLORS[0], alpha=0.82, label="Packed items")
    ems_patch    = mpatches.Patch(color="#00BFFF",       alpha=0.4,  label="Free EMS")
    ax.legend(handles=[packed_patch, ems_patch], fontsize=8,
              loc="upper left", framealpha=0.7)

    plt.tight_layout()
    path = save_path or f"{bin_obj.name.replace(' ', '_')}_ems.png"
    _save(fig, path)
    plt.close(fig)
    return fig, ax


# Show the step by step packing process

def visualise_packing_steps(bin_obj, save_dir="./packing_steps", prefix="step"):
    """Save one PNG per placement step for debugging placement order."""
    os.makedirs(save_dir, exist_ok=True)
    L, W, H = bin_obj.length, bin_obj.width, bin_obj.height

    for step in range(1, len(bin_obj.placed_items) + 1):
        fig = plt.figure(figsize=(5, 4.5), facecolor="white")
        ax  = fig.add_subplot(111, projection="3d")

        for idx, item in enumerate(bin_obj.placed_items[:step]):
            x0, y0, z0, x1, y1, z1 = item
            _draw_box(ax, x0, y0, z0, x1, y1, z1,
                      face_color=ITEM_COLORS[idx % len(ITEM_COLORS)])

        _draw_bin_outline(ax, L, W, H)
        _style_ax(ax, L, W, H)
        ax.set_title(f"Step {step} / {len(bin_obj.placed_items)}", fontsize=10)

        path = os.path.join(save_dir, f"{prefix}_{step:03d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"  Saved {len(bin_obj.placed_items)} step images → {os.path.abspath(save_dir)}/")


# Method to be used with the other files
def visualise_result(open_bins, save_path="bin_packing_result.png", show_ems=False):
    """One-call visualization after run_f53 completes."""
    if not open_bins:
        print("Nothing to visualize — no bins were opened.")
        return

    visualise_all_bins(open_bins, save_path=save_path)

    if show_ems:
        ems_path = save_path.replace(".png", "_ems.png")
        print("EMS overlay for bin 1:")
        visualise_ems(open_bins[0], save_path=ems_path)
