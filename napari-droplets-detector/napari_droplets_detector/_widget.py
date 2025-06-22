"""
A napari plugin for interactive detection and analysis of water-in-oil droplets.
"""

import os
import numpy as np
from skimage import (
        filters, measure, morphology, segmentation,
        color, util, feature
)
import napari
from napari.layers import Image
import pandas as pd
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from magicgui import magic_factory
from napari.utils.notifications import show_info
import re
from pathlib import Path
import csv

# HEX conversion and csv import helper tools

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

def _hex_to_rgb_float(hexstr: str):
    m = _HEX_RE.match(hexstr)
    if not m:
        raise ValueError(f"'{hexstr}' is not a valid #RRGGBB colour")
    h = m.group(1)
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _load_hex_palette(csv_path: Path, *, col=0):
    """Return a list[str] of #RRGGBB hex colours from *csv_path*."""
    colours = []
    try:
        with csv_path.open(newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) > col and row[col].strip():
                    colours.append(row[col].strip())
    except Exception as e:
        show_info(f"Could not read palette file: {e}")
        return []
    return colours


def _droplets_detection_alorythm(
    image,
    min_diameter,
    max_diameter,
    *,
    min_circularity=0.70     # tweak if your droplets are very irregular
):
    """
    Segment droplets and discard regions that are

    - too small / too large **by area**
    - or not round enough (circularity < `min_circularity`).

    Returns
    -------
    pruned_mask : bool ndarray
        True only on accepted droplets.
    props : list[RegionProperties]
        Region properties for accepted droplets.
    """

    # Convert image to grayscale if needed
    gray = color.rgb2gray(image) if image.ndim == 3 else util.img_as_float32(image)

    # Do quick denoise + Otsu threshold
    smoothed = filters.gaussian(gray, sigma=1.0)
    mask = smoothed > filters.threshold_otsu(smoothed)

    mask = morphology.binary_closing(mask, morphology.disk(1))
    base_area = np.pi * (min_diameter / 2) ** 2
    mask = morphology.remove_small_objects(mask, min_size=int(base_area * 0.2))
    mask = morphology.remove_small_holes(mask, area_threshold=int(base_area * 0.2))

    # Do watershed split
    dist = ndi.distance_transform_edt(mask)
    coords = feature.peak_local_max(
        dist,
        min_distance=max(1, int(min_diameter * 0.05)),
        labels=mask
    )
    markers = np.zeros_like(dist, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # Prune unwanted regions based on area & circularity
    area_min_ref = np.pi * (min_diameter / 2) ** 2
    area_max_ref = np.pi * (max_diameter / 2) ** 2
    area_lo = area_min_ref / (1.5 ** 2)      # 2/3 of min_diameter
    area_hi = area_max_ref * (1.5 ** 2)      # 1.5 of max_diameter

    keep = np.zeros(labels.max() + 1, dtype=bool)

    props_all = measure.regionprops(labels, intensity_image=gray)
    for rp in props_all:
        A = rp.area
        P = rp.perimeter or 1           # avoid /0
        circ = 4 * np.pi * A / (P * P)  # circularity

        if (area_lo <= A <= area_hi) and (circ >= min_circularity):
            keep[rp.label] = True

    pruned_mask = keep[labels]
    labels_pruned = measure.label(pruned_mask, connectivity=2)
    props = measure.regionprops(labels_pruned, intensity_image=gray)

    return pruned_mask, props

@magic_factory(
    call_button="Process / Refresh Image",
    layout="vertical",
    palette_csv=dict(
        widget_type="FileEdit",
        mode="r",
        filter="CSV (*.csv)",
        label="Palette CSV (optional)"
    ),
    min_diameter=dict(widget_type="FloatSpinBox", min=1.0,  max=1000.0, step=1.0,  value=60.0,  label="Min Diameter (px)"),
    max_diameter=dict(widget_type="FloatSpinBox", min=1.0,  max=5000.0, step=10.0, value=200.0, label="Max Diameter (px)"),
    min_circularity=dict(widget_type="FloatSpinBox", min=0.0, max=1.0,    step=0.05, value=0.5,  label="Min Circularity"),
    min_border_dist=dict(widget_type="FloatSpinBox", min=0.0, max=1000.0, step=10.0, value=50.0,  label="Border Dist. (px)"),
    conversion_factor=dict(widget_type="FloatSpinBox", min=0.01, max=100.0, step=0.0001, value=1.0, label="px → μm"),
)
def droplets_detector_widget(                              # noqa: C901
    viewer: "napari.Viewer",
    image_layer: Image,
    min_diameter: float,
    max_diameter: float,
    min_circularity: float,
    min_border_dist: float,
    conversion_factor: float,
    palette_csv: Path | None = None,
) -> None:
    """Detect droplets and tint them with a palette CSV chosen by the user."""

    # Check image
    if image_layer is None:
        show_info("Please select an image layer")
        return
    image = image_layer.data

    # Palette prep
    if palette_csv and palette_csv.exists():
        hex_palette = _load_hex_palette(palette_csv)
    else:
        hex_palette = []

    # validate / convert
    try:
        rgb_palette = np.array(
            [_hex_to_rgb_float(c) for c in hex_palette],
            dtype=np.float32,
        )
    except ValueError as e:
        show_info(str(e))
        return

    # Caching heavy compute data
    layer_name = "Binary Mask"
    compute = True
    props = labels = None

    if layer_name in viewer.layers:
        lyr = viewer.layers[layer_name]
        meta = lyr.metadata
        if (
            meta.get("min_diameter") == min_diameter
            and meta.get("max_diameter") == max_diameter
            and meta.get("min_circularity") == min_circularity):
            labels = meta.get("labels")
            props = meta.get("props")
            compute = labels is None

    # Do segmentation if not cached
    if compute:
        bin_mask, props = _droplets_detection_alorythm(
            image,
            min_diameter,
            max_diameter,
            min_circularity=min_circularity,
        )
        labels = measure.label(bin_mask, connectivity=2)

    # Apply colouring for droplets
    n_labels = int(labels.max())
    colour_lut = np.zeros((n_labels + 1, 3), dtype=np.float32)  # 0 = black
    if rgb_palette.size:
        # repeat palette cyclically if too short
        reps = int(np.ceil(n_labels / len(rgb_palette)))
        colour_lut[1:] = np.tile(rgb_palette, (reps, 1))[:n_labels]
    else:  # fallback deterministic-random
        rng = np.random.default_rng(42)
        colour_lut[1:] = rng.random((n_labels, 3), dtype=np.float32)

    rgb_mask = colour_lut[labels]

    # Add metadata to napari
    metadata = dict(
        min_diameter=min_diameter,
        max_diameter=max_diameter,
        min_circularity=min_circularity,
        props=props,
        labels=labels,
    )

    if layer_name in viewer.layers:
        viewer.layers[layer_name].data = rgb_mask
        viewer.layers[layer_name].metadata.update(metadata)
    else:
        viewer.add_image(
            rgb_mask,
            name=layer_name,
            rgb=True,
            opacity=0.9,
            metadata=metadata,
        )    # Filter regions based on criteria
    results = []
    shapes_data = []

    for region in props:
        # Calculate perimeter and area
        perimeter = region.perimeter
        area = region.area

        # Calculate circularity
        # A perfect circle has circularity of 1
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0

        # Calculate diameter
        diameter = region.equivalent_diameter

        # Check boundary conditions
        y, x = region.centroid
        image_shape = image.shape

        within_border = (
            x >= min_border_dist and
            x <= (image_shape[1] - min_border_dist) and
            y >= min_border_dist and
            y <= (image_shape[0] - min_border_dist)
        )

        # Check criteria
        if (circularity >= min_circularity and
            diameter >= min_diameter and
            diameter <= max_diameter and
            within_border):

            # Store result
            result = {
                "droplet_id": region.label,
                "diameter_px": diameter,
                "diameter_um": diameter * conversion_factor,
                "circularity": circularity,
                "area_px": area,
                "area_um2": area * conversion_factor**2,
                "centroid_y": y,
                "centroid_x": x,
                "image_name": image_layer.name
            }
            results.append(result)

            # Create shape for visualization
            # Napari expects consistent data types for shapes
            # Convert to float64 for consistency
            radius = float(diameter / 2)
            # Create circle points
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = x + radius * np.cos(theta)
            circle_y = y + radius * np.sin(theta)
            circle_points = np.column_stack([circle_y, circle_x])
            shapes_data.append(circle_points)

    # Add or update shapes layer
    if "Detected Droplets" in viewer.layers:
        viewer.layers.remove("Detected Droplets")

    if shapes_data:
        shapes_layer = viewer.add_shapes(
            shapes_data,
            shape_type='polygon',
            edge_color='red',
            face_color='transparent',
            name="Detected Droplets",
            metadata={
                "results": results,
                "min_circularity": min_circularity,
                "conversion_factor": conversion_factor
            }
        )

        show_info(f"Detected {len(results)} droplets")
    else:
        show_info("No droplets detected with current parameters")


@magic_factory(call_button="Export to CSV")
def export_csv_widget(
    viewer: napari.Viewer,
    path: str = ""
) -> None:
    """Export detected droplets to CSV."""
    # Check if shapes layer exists
    if "Detected Droplets" not in viewer.layers:
        show_info("No detected droplets. Run detection first.")
        return

    # Get results from shapes layer metadata
    shapes_layer = viewer.layers["Detected Droplets"]
    if not hasattr(shapes_layer, "metadata") or "results" not in shapes_layer.metadata:
        show_info("No droplet data found. Run detection first.")
        return

    results = shapes_layer.metadata["results"]

    # Get file path if not provided
    if not path:
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            None, "Save CSV File", "", "CSV Files (*.csv)"
        )
        if not path:
            show_info("Export canceled")
            return

    # Create dataframe
    df = pd.DataFrame(results)

    # Save to file
    try:
        df.to_csv(path, index=False)
        show_info(f"Exported {len(df)} droplets to {path}")
    except Exception as e:
        show_info(f"Error exporting CSV: {str(e)}")


@magic_factory(call_button="Analyze Droplets")
def droplets_analysis_widget(
    viewer: napari.Viewer
) -> None:
    """Carry out simple droplets analysis."""
    # Check if shapes layer exists
    if "Detected Droplets" not in viewer.layers:
        show_info("No detected droplets. Run detection first.")
        return

    # Get results from shapes layer metadata
    shapes_layer = viewer.layers["Detected Droplets"]
    if not hasattr(shapes_layer, "metadata") or "results" not in shapes_layer.metadata:
        show_info("No droplet data found. Run detection first.")
        return

    results = shapes_layer.metadata["results"]
    min_circularity = shapes_layer.metadata.get("min_circularity", 0.8)

    # Get directory path
    from qtpy.QtWidgets import QFileDialog
    path = QFileDialog.getExistingDirectory(None, "Select Directory for Visualization")
    if not path:
        show_info("Export canceled")
        return

    # Create output directory if it doesn't exist
    try:
        os.makedirs(path, exist_ok=True)

        # Create dataframe from results
        df = pd.DataFrame(results)

        # Create histogram of droplet diameters
        plt.figure(figsize=(10, 6))
        plt.hist(df['diameter_um'], bins=30, alpha=0.7, color='blue')
        plt.xlabel('Droplet Diameter (μm)')
        plt.ylabel('Count')
        plt.title('Distribution of Droplet Diameters')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(path, 'diameter_histogram.png'), dpi=300)
        plt.close()

        # Create scatter plot of diameter vs circularity
        plt.figure(figsize=(10, 6))
        plt.scatter(df['diameter_um'], df['circularity'], alpha=0.5)
        plt.xlabel('Droplet Diameter (μm)')
        plt.ylabel('Circularity')
        plt.title('Droplet Diameter vs Circularity')
        plt.axhline(y=min_circularity, color='r', linestyle='--',
                  label=f'Min Circularity: {min_circularity}')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(path, 'diameter_vs_circularity.png'), dpi=300)
        plt.close()

        # Save CSV of results
        csv_path = os.path.join(path, 'droplet_results.csv')
        df.to_csv(csv_path, index=False)

        # Save summary statistics
        stats_path = os.path.join(path, 'summary_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Total droplets detected: {len(df)}\n")
            f.write("Diameter statistics (μm):\n")
            f.write(f"  Min: {df['diameter_um'].min():.2f}\n")
            f.write(f"  Max: {df['diameter_um'].max():.2f}\n")
            f.write(f"  Mean: {df['diameter_um'].mean():.2f}\n")
            f.write(f"  Median: {df['diameter_um'].median():.2f}\n")
            f.write(f"  StdDev: {df['diameter_um'].std():.2f}\n\n")
            f.write("Circularity statistics:\n")
            f.write(f"  Min: {df['circularity'].min():.2f}\n")
            f.write(f"  Max: {df['circularity'].max():.2f}\n")
            f.write(f"  Mean: {df['circularity'].mean():.2f}\n")
            f.write(f"  Median: {df['circularity'].median():.2f}\n")

        show_info(f"Visualization exported to {path}")
    except Exception as e:
        show_info(f"Error exporting visualization: {str(e)}")
