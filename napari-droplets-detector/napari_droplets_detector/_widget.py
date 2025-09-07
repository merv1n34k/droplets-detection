"""
A napari plugin for interactive detection and analysis of water-in-oil droplets.
Enhanced with eccentricity filtering, inclusion detection, visualization, and fluorescence cross-referencing.
"""

import os
import numpy as np
from skimage import filters, measure, morphology, segmentation, color, util, feature
import napari
from napari.layers import Image
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from magicgui import magicgui
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


def _droplets_detection_algorithm(
    image,
    min_diameter,
    max_diameter,
    *,
    min_circularity=0.70,
    max_eccentricity=0.9,  # NEW: eccentricity filter
):
    """
    Segment droplets and discard regions that are:
    - too small / too large by area
    - not round enough (circularity < min_circularity)
    - too elongated (eccentricity > max_eccentricity)

    Returns
    -------
    pruned_mask : bool ndarray
        True only on accepted droplets.
    props : list[RegionProperties]
        Region properties for accepted droplets.
    labels_pruned : ndarray
        Labeled array of accepted droplets.
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
        dist, min_distance=max(1, int(min_diameter * 0.05)), labels=mask
    )
    markers = np.zeros_like(dist, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # Prune unwanted regions based on area, circularity, and eccentricity
    area_min_ref = np.pi * (min_diameter / 2) ** 2
    area_max_ref = np.pi * (max_diameter / 2) ** 2
    area_lo = area_min_ref / (1.5**2)  # 2/3 of min_diameter
    area_hi = area_max_ref * (1.5**2)  # 1.5 of max_diameter

    keep = np.zeros(labels.max() + 1, dtype=bool)

    props_all = measure.regionprops(labels, intensity_image=gray)
    for rp in props_all:
        A = rp.area
        P = rp.perimeter or 1  # avoid /0
        circ = 4 * np.pi * A / (P * P)  # circularity
        ecc = rp.eccentricity  # NEW: check eccentricity

        # Apply all filters including eccentricity
        if (
            (area_lo <= A <= area_hi)
            and (circ >= min_circularity)
            and (ecc <= max_eccentricity)
        ):
            keep[rp.label] = True

    pruned_mask = keep[labels]
    labels_pruned = measure.label(pruned_mask, connectivity=2)
    props = measure.regionprops(labels_pruned, intensity_image=gray)

    return pruned_mask, props, labels_pruned


def _find_inclusions_detailed(region_image, region_mask, min_inclusion_size=5):
    """
    Find inclusions (smaller objects) within a droplet region and return their properties.

    Parameters
    ----------
    region_image : ndarray
        Grayscale image of the droplet region
    region_mask : ndarray
        Binary mask of the droplet region
    min_inclusion_size : int
        Minimum size for inclusions in pixels

    Returns
    -------
    inclusion_props : list
        List of inclusion properties (centroid, area, etc.)
    inclusion_count : int
        Number of inclusions detected
    """
    if region_image.size == 0 or not np.any(region_mask):
        return [], 0

    inclusion_props = []

    try:
        # Apply intensity thresholding to find candidate inclusions
        local_thresh = filters.threshold_otsu(region_image[region_mask])

        # Bright inclusions
        bright_mask = (region_image > local_thresh * 1.2) & region_mask
        bright_mask = morphology.remove_small_objects(
            bright_mask, min_size=min_inclusion_size
        )

        # Dark inclusions
        dark_mask = (region_image < local_thresh * 0.8) & region_mask
        dark_mask = morphology.remove_small_objects(
            dark_mask, min_size=min_inclusion_size
        )

        # Combine and label
        inclusion_mask = bright_mask | dark_mask
        labeled_inclusions = measure.label(inclusion_mask)

        # Get properties and filter
        props = measure.regionprops(labeled_inclusions, intensity_image=region_image)

        for prop in props:
            if prop.perimeter > 0:
                circ = 4 * np.pi * prop.area / (prop.perimeter**2)
                if circ >= 0.3:  # Very lenient circularity for inclusions
                    inclusion_props.append(
                        {
                            "centroid": prop.centroid,
                            "area": prop.area,
                            "circularity": circ,
                            "intensity": prop.mean_intensity,
                            "type": "bright"
                            if prop.mean_intensity > local_thresh
                            else "dark",
                        }
                    )
    except:
        pass

    return inclusion_props, len(inclusion_props)


def _find_fluorescence_inclusions(
    fluorescence_region,
    fluorescence_mask,
    min_inclusion_size=5,
    intensity_threshold=0.5,
):
    """
    Find bright inclusions in fluorescence channel using manual threshold.

    Parameters
    ----------
    fluorescence_region : ndarray
        Fluorescence image of the droplet region
    fluorescence_mask : ndarray
        Binary mask of the droplet region
    min_inclusion_size : int
        Minimum size for inclusions
    intensity_threshold : float
        Absolute intensity threshold for detection

    Returns
    -------
    fluor_props : list
        List of fluorescent inclusion properties
    fluor_count : int
        Number of fluorescent inclusions
    """
    if fluorescence_region.size == 0 or not np.any(fluorescence_mask):
        return [], 0

    fluor_props = []

    try:
        # Use manual threshold - only detect bright spots above threshold
        fluor_inclusion_mask = (
            fluorescence_region > intensity_threshold
        ) & fluorescence_mask

        # Remove small objects
        fluor_inclusion_mask = morphology.remove_small_objects(
            fluor_inclusion_mask, min_size=min_inclusion_size
        )

        # Label and get properties
        fluor_labeled = measure.label(fluor_inclusion_mask)
        props = measure.regionprops(fluor_labeled, intensity_image=fluorescence_region)

        for prop in props:
            fluor_props.append(
                {
                    "centroid": prop.centroid,
                    "area": prop.area,
                    "intensity": prop.mean_intensity,
                }
            )
    except:
        pass

    return fluor_props, len(fluor_props)


def _match_inclusions_spatial(bf_props, fluor_props, proximity_threshold=5.0):
    """
    Match inclusions between channels based on spatial proximity.

    Parameters
    ----------
    bf_props : list
        Brightfield inclusion properties
    fluor_props : list
        Fluorescence inclusion properties
    proximity_threshold : float
        Maximum distance in pixels to consider inclusions matched

    Returns
    -------
    matched_count : int
        Number of matched inclusions
    bf_matched : list
        Indices of matched brightfield inclusions
    fluor_matched : list
        Indices of matched fluorescence inclusions
    """
    if not bf_props or not fluor_props:
        return 0, [], []

    # Extract centroids
    bf_centroids = np.array([p["centroid"] for p in bf_props])
    fluor_centroids = np.array([p["centroid"] for p in fluor_props])

    # Calculate pairwise distances
    distances = cdist(bf_centroids, fluor_centroids)

    # Find matches within proximity threshold
    bf_matched = []
    fluor_matched = []

    # Greedy matching: find closest pairs within threshold
    while True:
        if distances.size == 0:
            break
        min_idx = np.unravel_index(distances.argmin(), distances.shape)
        min_dist = distances[min_idx]

        if min_dist <= proximity_threshold:
            bf_matched.append(min_idx[0])
            fluor_matched.append(min_idx[1])
            # Remove matched pairs from consideration
            distances[min_idx[0], :] = np.inf
            distances[:, min_idx[1]] = np.inf
        else:
            break

    return len(bf_matched), bf_matched, fluor_matched


def _update_fluor_threshold_range(widget):
    """Update fluorescence threshold slider range based on selected image."""
    fluor_layer = widget.fluorescence_layer.value
    slider = widget.fluor_intensity_threshold

    if fluor_layer is not None:
        try:
            # Get the fluorescence image and calculate its intensity range
            fluor_image = fluor_layer.data
            gray_fluor = (
                color.rgb2gray(fluor_image)
                if fluor_image.ndim == 3
                else util.img_as_float32(fluor_image)
            )

            min_val = float(gray_fluor.min())
            max_val = float(gray_fluor.max())

            # Update slider range
            slider.min = min_val
            slider.max = max_val
            # Set to midpoint if current value is outside range
            if slider.value < min_val or slider.value > max_val:
                slider.value = (min_val + max_val) / 2

            # Update label to show range
            slider.label = f"Fluor Threshold ({min_val:.3f}-{max_val:.3f})"
        except Exception:
            # Reset on error
            slider.min = 0.0
            slider.max = 1.0
            slider.value = 0.5
            slider.label = "Fluorescence Intensity Threshold"
    else:
        # Reset if no layer selected
        slider.min = 0.0
        slider.max = 1.0
        slider.value = 0.5
        slider.label = "Fluorescence Intensity Threshold"


def droplets_detector_widget():
    """Create and return the droplets detector widget with callbacks."""

    @magicgui(
        call_button="Process / Refresh Image",
        layout="vertical",
        palette_csv=dict(
            widget_type="FileEdit",
            mode="r",
            filter="CSV (*.csv)",
            label="Palette CSV (optional)",
        ),
        image_layer=dict(label="Brightfield"),
        min_diameter=dict(
            widget_type="FloatSpinBox",
            min=1.0,
            max=1000.0,
            step=1.0,
            value=60.0,
            label="Min Diameter (px)",
        ),
        max_diameter=dict(
            widget_type="FloatSpinBox",
            min=1.0,
            max=5000.0,
            step=10.0,
            value=200.0,
            label="Max Diameter (px)",
        ),
        min_circularity=dict(
            widget_type="FloatSpinBox",
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.5,
            label="Min Circularity",
        ),
        max_eccentricity=dict(
            widget_type="FloatSpinBox",
            min=0.0,
            max=1.0,
            step=0.05,
            value=0.9,
            label="Max Eccentricity",
        ),
        min_border_dist=dict(
            widget_type="FloatSpinBox",
            min=0.0,
            max=1000.0,
            step=10.0,
            value=50.0,
            label="Border Dist. (px)",
        ),
        conversion_factor=dict(
            widget_type="FloatSpinBox",
            min=0.01,
            max=100.0,
            step=0.0001,
            value=1.14,
            label="px → μm",
        ),
        detect_inclusions=dict(
            widget_type="CheckBox", value=False, label="Detect Inclusions"
        ),
        fluorescence_layer=dict(label="Fluorescence (optional)"),
        min_inclusion_size=dict(
            widget_type="SpinBox",
            min=1,
            max=100,
            value=5,
            label="Min Inclusion Size (px)",
        ),
        inclusion_proximity=dict(
            widget_type="FloatSpinBox",
            min=0.1,
            max=50.0,
            step=0.5,
            value=5.0,
            label="Inclusion Match Distance (px)",
        ),
        fluor_intensity_threshold=dict(
            widget_type="FloatSlider",
            min=0.0,
            max=1.0,
            value=0.5,
            label="Fluorescence Intensity Threshold",
        ),
        visualize_inclusions=dict(
            widget_type="CheckBox", value=True, label="Visualize Inclusions"
        ),
    )
    def widget_impl(
        viewer: "napari.Viewer",
        image_layer: Image,
        min_diameter: float,
        max_diameter: float,
        min_circularity: float,
        max_eccentricity: float,
        min_border_dist: float,
        conversion_factor: float,
        palette_csv: Path | None = None,
        detect_inclusions: bool = False,
        fluorescence_layer: Image = None,
        min_inclusion_size: int = 5,
        inclusion_proximity: float = 5.0,
        fluor_intensity_threshold: float = 0.5,
        visualize_inclusions: bool = True,
    ) -> None:
        """Detect droplets with enhanced inclusion detection and visualization.

        The fluorescence threshold is an absolute intensity value that automatically
        adjusts its range based on the selected fluorescence image."""

        # Check image
        if image_layer is None:
            show_info("Please select an image layer")
            return
        image = image_layer.data

        # Get fluorescence image if provided
        fluorescence_image = None
        if fluorescence_layer is not None and detect_inclusions:
            fluorescence_image = fluorescence_layer.data
            # Check dimensions match
            if fluorescence_image.shape[:2] != image.shape[:2]:
                show_info(
                    "Warning: Fluorescence image dimensions don't match - skipping cross-channel analysis"
                )
                fluorescence_image = None

        # Convert to grayscale for processing
        gray = color.rgb2gray(image) if image.ndim == 3 else util.img_as_float32(image)
        gray_fluor = None

        if fluorescence_image is not None:
            gray_fluor = (
                color.rgb2gray(fluorescence_image)
                if fluorescence_image.ndim == 3
                else util.img_as_float32(fluorescence_image)
            )
            # Display threshold info
            if detect_inclusions:
                min_intensity = float(gray_fluor.min())
                max_intensity = float(gray_fluor.max())
                show_info(
                    f"Using fluorescence threshold: {fluor_intensity_threshold:.3f} "
                    f"(image range: {min_intensity:.3f}-{max_intensity:.3f})"
                )

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

        # Detect droplets with eccentricity filter
        bin_mask, props, labels = _droplets_detection_algorithm(
            image,
            min_diameter,
            max_diameter,
            min_circularity=min_circularity,
            max_eccentricity=max_eccentricity,
        )

        # Apply colouring for droplets
        n_labels = int(labels.max())
        colour_lut = np.zeros((n_labels + 1, 3), dtype=np.float32)
        if rgb_palette.size:
            reps = int(np.ceil(n_labels / len(rgb_palette)))
            colour_lut[1:] = np.tile(rgb_palette, (reps, 1))[:n_labels]
        else:
            rng = np.random.default_rng(42)
            colour_lut[1:] = rng.random((n_labels, 3), dtype=np.float32)

        rgb_mask = colour_lut[labels]

        # Process each droplet for filtering and inclusion detection
        results = []
        shapes_data = []

        # For inclusion visualization
        all_bf_inclusions = []  # Store all brightfield inclusions with global coordinates
        all_fluor_inclusions = []  # Store all fluorescence inclusions
        all_matched_bf = []  # Store matched brightfield inclusions
        all_matched_fluor = []  # Store matched fluorescence inclusions

        for region in props:
            # Calculate metrics
            perimeter = region.perimeter
            area = region.area
            circularity = (
                4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            )
            diameter = region.equivalent_diameter
            eccentricity = region.eccentricity

            # Check boundary conditions
            y, x = region.centroid
            image_shape = image.shape

            within_border = (
                x >= min_border_dist
                and x <= (image_shape[1] - min_border_dist)
                and y >= min_border_dist
                and y <= (image_shape[0] - min_border_dist)
            )

            # Check all criteria including eccentricity
            if (
                circularity >= min_circularity
                and eccentricity <= max_eccentricity
                and diameter >= min_diameter
                and diameter <= max_diameter
                and within_border
            ):
                # Base result
                result = {
                    "droplet_id": region.label,
                    "diameter_px": diameter,
                    "diameter_um": diameter * conversion_factor,
                    "circularity": circularity,
                    "eccentricity": eccentricity,
                    "area_px": area,
                    "area_um2": area * conversion_factor**2,
                    "centroid_y": y,
                    "centroid_x": x,
                    "image_name": image_layer.name,
                }

                # Detect inclusions if enabled
                if detect_inclusions:
                    # Extract region from image
                    minr, minc, maxr, maxc = region.bbox
                    region_gray = gray[minr:maxr, minc:maxc]
                    region_mask = labels[minr:maxr, minc:maxc] == region.label

                    # Find brightfield inclusions
                    bf_props, bf_count = _find_inclusions_detailed(
                        region_gray, region_mask, min_inclusion_size
                    )

                    # Convert local coordinates to global
                    for prop in bf_props:
                        global_centroid = (
                            prop["centroid"][0] + minr,
                            prop["centroid"][1] + minc,
                        )
                        all_bf_inclusions.append(
                            {
                                "centroid": global_centroid,
                                "area": prop["area"],
                                "droplet_id": region.label,
                            }
                        )

                    result["inclusion_count_bf"] = bf_count

                    # Process fluorescence if available
                    if fluorescence_image is not None and gray_fluor is not None:
                        region_fluor = gray_fluor[minr:maxr, minc:maxc]

                        # Find fluorescence inclusions with absolute intensity threshold
                        fluor_props, fluor_count = _find_fluorescence_inclusions(
                            region_fluor,
                            region_mask,
                            min_inclusion_size,
                            fluor_intensity_threshold,
                        )

                        # Match inclusions between channels
                        matched_count, bf_matched_idx, fluor_matched_idx = (
                            _match_inclusions_spatial(
                                bf_props, fluor_props, inclusion_proximity
                            )
                        )

                        # Store global coordinates for fluorescence inclusions
                        for i, prop in enumerate(fluor_props):
                            global_centroid = (
                                prop["centroid"][0] + minr,
                                prop["centroid"][1] + minc,
                            )
                            fluor_data = {
                                "centroid": global_centroid,
                                "area": prop["area"],
                                "droplet_id": region.label,
                            }
                            all_fluor_inclusions.append(fluor_data)

                            # Mark if matched
                            if i in fluor_matched_idx:
                                all_matched_fluor.append(fluor_data)

                        # Mark matched brightfield inclusions
                        for idx in bf_matched_idx:
                            if idx < len(bf_props):
                                global_centroid = (
                                    bf_props[idx]["centroid"][0] + minr,
                                    bf_props[idx]["centroid"][1] + minc,
                                )
                                all_matched_bf.append(
                                    {
                                        "centroid": global_centroid,
                                        "area": bf_props[idx]["area"],
                                        "droplet_id": region.label,
                                    }
                                )

                        result["inclusion_count_fluor"] = fluor_count
                        result["inclusion_count_matched"] = matched_count

                results.append(result)

                # Create visualization shape for droplet
                radius = float(diameter / 2)
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = x + radius * np.cos(theta)
                circle_y = y + radius * np.sin(theta)
                circle_points = np.column_stack([circle_y, circle_x])
                shapes_data.append(circle_points)

        # Add layers to viewer
        layer_name = "Binary Mask"
        if layer_name in viewer.layers:
            viewer.layers[layer_name].data = rgb_mask
        else:
            viewer.add_image(rgb_mask, name=layer_name, rgb=True, opacity=0.9)

        # Add or update droplet shapes layer
        if "Detected Droplets" in viewer.layers:
            viewer.layers.remove("Detected Droplets")

        if shapes_data:
            shapes_layer = viewer.add_shapes(
                shapes_data,
                shape_type="polygon",
                edge_color="red",
                face_color="transparent",
                edge_width=2,
                name="Detected Droplets",
                metadata={
                    "results": results,
                    "min_circularity": min_circularity,
                    "max_eccentricity": max_eccentricity,
                    "conversion_factor": conversion_factor,
                    "detect_inclusions": detect_inclusions,
                },
            )

        # Visualize inclusions if enabled
        if detect_inclusions and visualize_inclusions:
            # Remove old inclusion layers
            for layer_name in [
                "BF Inclusions",
                "Fluor Inclusions",
                "Matched Inclusions",
            ]:
                if layer_name in viewer.layers:
                    viewer.layers.remove(layer_name)

            # Add brightfield inclusions as small filled circles
            if all_bf_inclusions:
                bf_circles = []
                for inc in all_bf_inclusions:
                    y, x = inc["centroid"]
                    # Create small circle (3 pixel radius)
                    radius = 3
                    theta = np.linspace(0, 2 * np.pi, 20)
                    circle_x = x + radius * np.cos(theta)
                    circle_y = y + radius * np.sin(theta)
                    circle_points = np.column_stack([circle_y, circle_x])
                    bf_circles.append(circle_points)

                viewer.add_shapes(
                    bf_circles,
                    shape_type="polygon",
                    edge_color="cyan",
                    face_color="cyan",
                    edge_width=1,
                    opacity=0.7,
                    name="BF Inclusions",
                )

            # Add fluorescence inclusions if available
            if all_fluor_inclusions:
                fluor_circles = []
                for inc in all_fluor_inclusions:
                    y, x = inc["centroid"]
                    radius = 3
                    theta = np.linspace(0, 2 * np.pi, 20)
                    circle_x = x + radius * np.cos(theta)
                    circle_y = y + radius * np.sin(theta)
                    circle_points = np.column_stack([circle_y, circle_x])
                    fluor_circles.append(circle_points)

                viewer.add_shapes(
                    fluor_circles,
                    shape_type="polygon",
                    edge_color="lime",
                    face_color="lime",
                    edge_width=1,
                    opacity=0.7,
                    name="Fluor Inclusions",
                )

            # Highlight matched inclusions
            if all_matched_bf:
                matched_circles = []
                for inc in all_matched_bf:
                    y, x = inc["centroid"]
                    # Slightly larger circle for matched
                    radius = 4
                    theta = np.linspace(0, 2 * np.pi, 20)
                    circle_x = x + radius * np.cos(theta)
                    circle_y = y + radius * np.sin(theta)
                    circle_points = np.column_stack([circle_y, circle_x])
                    matched_circles.append(circle_points)

                viewer.add_shapes(
                    matched_circles,
                    shape_type="polygon",
                    edge_color="yellow",
                    face_color="yellow",
                    edge_width=2,
                    opacity=0.5,
                    name="Matched Inclusions",
                )

        # Show summary
        if results:
            msg = f"Detected {len(results)} droplets"
            if detect_inclusions:
                total_bf = sum(r.get("inclusion_count_bf", 0) for r in results)
                msg += f"\n{total_bf} brightfield inclusions"
                if fluorescence_image is not None:
                    total_fluor = sum(
                        r.get("inclusion_count_fluor", 0) for r in results
                    )
                    total_matched = sum(
                        r.get("inclusion_count_matched", 0) for r in results
                    )
                    msg += f"\n{total_fluor} fluorescence inclusions"
                    msg += f"\n{total_matched} matched inclusions"
            show_info(msg)
        else:
            show_info("No droplets detected with current parameters")

    # Connect the callback for fluorescence layer changes
    widget_impl.fluorescence_layer.changed.connect(
        lambda: _update_fluor_threshold_range(widget_impl)
    )

    # Return the widget instance
    return widget_impl


def export_csv_widget():
    """Create and return the CSV export widget."""

    @magicgui(call_button="Export to CSV")
    def widget(viewer: napari.Viewer, path: str = "") -> None:
        """Export detected droplets to CSV."""
        # Check if shapes layer exists
        if "Detected Droplets" not in viewer.layers:
            show_info("No detected droplets. Run detection first.")
            return

        # Get results from shapes layer metadata
        shapes_layer = viewer.layers["Detected Droplets"]
        if (
            not hasattr(shapes_layer, "metadata")
            or "results" not in shapes_layer.metadata
        ):
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

    return widget


def droplets_analysis_widget():
    """Create and return the droplets analysis widget."""

    @magicgui(call_button="Analyze Droplets")
    def widget(viewer: napari.Viewer) -> None:
        """Analyze detected droplets with enhanced metrics."""
        # Check if shapes layer exists
        if "Detected Droplets" not in viewer.layers:
            show_info("No detected droplets. Run detection first.")
            return

        # Get results from shapes layer metadata
        shapes_layer = viewer.layers["Detected Droplets"]
        if (
            not hasattr(shapes_layer, "metadata")
            or "results" not in shapes_layer.metadata
        ):
            show_info("No droplet data found. Run detection first.")
            return

        results = shapes_layer.metadata["results"]
        min_circularity = shapes_layer.metadata.get("min_circularity", 0.8)
        max_eccentricity = shapes_layer.metadata.get("max_eccentricity", 1.0)
        detect_inclusions = shapes_layer.metadata.get("detect_inclusions", False)

        # Get directory path
        from qtpy.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(
            None, "Select Directory for Analysis Results"
        )
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
            plt.hist(df["diameter_um"], bins=30, alpha=0.7, color="blue")
            plt.xlabel("Droplet Diameter (μm)")
            plt.ylabel("Count")
            plt.title("Distribution of Droplet Diameters")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(path, "diameter_histogram.png"), dpi=300)
            plt.close()

            # Create scatter plot of diameter vs circularity
            plt.figure(figsize=(10, 6))
            plt.scatter(df["diameter_um"], df["circularity"], alpha=0.5)
            plt.xlabel("Droplet Diameter (μm)")
            plt.ylabel("Circularity")
            plt.title("Droplet Diameter vs Circularity")
            plt.axhline(
                y=min_circularity,
                color="r",
                linestyle="--",
                label=f"Min Circularity: {min_circularity}",
            )
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(path, "diameter_vs_circularity.png"), dpi=300)
            plt.close()

            # Create eccentricity plot if available
            if "eccentricity" in df.columns:
                plt.figure(figsize=(10, 6))
                plt.scatter(
                    df["diameter_um"], df["eccentricity"], alpha=0.5, color="green"
                )
                plt.xlabel("Droplet Diameter (μm)")
                plt.ylabel("Eccentricity")
                plt.title("Droplet Diameter vs Eccentricity")
                plt.axhline(
                    y=max_eccentricity,
                    color="r",
                    linestyle="--",
                    label=f"Max Eccentricity: {max_eccentricity}",
                )
                plt.grid(alpha=0.3)
                plt.legend()
                plt.savefig(os.path.join(path, "diameter_vs_eccentricity.png"), dpi=300)
                plt.close()

            # Create inclusion analysis if available
            if detect_inclusions and "inclusion_count_bf" in df.columns:
                # Histogram of inclusions per droplet
                plt.figure(figsize=(10, 6))
                max_val = (
                    int(df["inclusion_count_bf"].max())
                    if df["inclusion_count_bf"].max() > 0
                    else 1
                )
                plt.hist(
                    df["inclusion_count_bf"],
                    bins=range(max_val + 2),
                    alpha=0.7,
                    color="orange",
                )
                plt.xlabel("Number of Inclusions per Droplet")
                plt.ylabel("Count")
                plt.title("Distribution of Inclusions in Droplets")
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(path, "inclusion_distribution.png"), dpi=300)
                plt.close()

                # Scatter plot of droplet size vs inclusion count
                plt.figure(figsize=(10, 6))
                plt.scatter(
                    df["diameter_um"],
                    df["inclusion_count_bf"],
                    alpha=0.5,
                    color="purple",
                )
                plt.xlabel("Droplet Diameter (μm)")
                plt.ylabel("Number of Inclusions")
                plt.title("Droplet Size vs Inclusion Count")
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(path, "diameter_vs_inclusions.png"), dpi=300)
                plt.close()

                # If fluorescence data available, create comparison plot
                if "inclusion_count_fluor" in df.columns:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                    # BF vs Fluor inclusions
                    ax1.scatter(
                        df["inclusion_count_bf"], df["inclusion_count_fluor"], alpha=0.5
                    )
                    ax1.set_xlabel("Brightfield Inclusions")
                    ax1.set_ylabel("Fluorescence Inclusions")
                    ax1.set_title("Brightfield vs Fluorescence Inclusion Counts")
                    ax1.grid(alpha=0.3)
                    # Add diagonal line for reference
                    max_count = max(
                        df["inclusion_count_bf"].max(),
                        df["inclusion_count_fluor"].max(),
                    )
                    ax1.plot(
                        [0, max_count],
                        [0, max_count],
                        "r--",
                        alpha=0.5,
                        label="1:1 line",
                    )
                    ax1.legend()

                    # Matching efficiency
                    if "inclusion_count_matched" in df.columns:
                        matching_efficiency = df["inclusion_count_matched"] / df[
                            "inclusion_count_bf"
                        ].replace(0, np.nan)
                        ax2.hist(
                            matching_efficiency.dropna(),
                            bins=20,
                            alpha=0.7,
                            color="green",
                        )
                        ax2.set_xlabel("Matching Efficiency (Matched/BF)")
                        ax2.set_ylabel("Count")
                        ax2.set_title("Distribution of Inclusion Matching Efficiency")
                        ax2.grid(alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(os.path.join(path, "inclusion_comparison.png"), dpi=300)
                    plt.close()

            # Save CSV of results
            csv_path = os.path.join(path, "droplet_results.csv")
            df.to_csv(csv_path, index=False)

            # Save summary statistics
            stats_path = os.path.join(path, "summary_statistics.txt")
            with open(stats_path, "w") as f:
                f.write(f"Total droplets detected: {len(df)}\n\n")

                f.write("Diameter statistics (μm):\n")
                f.write(f"  Min: {df['diameter_um'].min():.2f}\n")
                f.write(f"  Max: {df['diameter_um'].max():.2f}\n")
                f.write(f"  Mean: {df['diameter_um'].mean():.2f}\n")
                f.write(f"  Median: {df['diameter_um'].median():.2f}\n")
                f.write(f"  StdDev: {df['diameter_um'].std():.2f}\n\n")

                f.write("Circularity statistics:\n")
                f.write(f"  Min: {df['circularity'].min():.3f}\n")
                f.write(f"  Max: {df['circularity'].max():.3f}\n")
                f.write(f"  Mean: {df['circularity'].mean():.3f}\n")
                f.write(f"  Median: {df['circularity'].median():.3f}\n\n")

                if "eccentricity" in df.columns:
                    f.write("Eccentricity statistics:\n")
                    f.write(f"  Min: {df['eccentricity'].min():.3f}\n")
                    f.write(f"  Max: {df['eccentricity'].max():.3f}\n")
                    f.write(f"  Mean: {df['eccentricity'].mean():.3f}\n")
                    f.write(f"  Median: {df['eccentricity'].median():.3f}\n\n")

                if detect_inclusions and "inclusion_count_bf" in df.columns:
                    f.write("Inclusion statistics:\n")
                    f.write(
                        f"  Total droplets with inclusions: {(df['inclusion_count_bf'] > 0).sum()}\n"
                    )
                    f.write(
                        f"  Mean inclusions per droplet: {df['inclusion_count_bf'].mean():.2f}\n"
                    )
                    f.write(
                        f"  Max inclusions in a droplet: {df['inclusion_count_bf'].max()}\n"
                    )

                    if "inclusion_count_fluor" in df.columns:
                        f.write(f"\nFluorescence channel:\n")
                        f.write(
                            f"  Mean fluorescent inclusions: {df['inclusion_count_fluor'].mean():.2f}\n"
                        )
                        if "inclusion_count_matched" in df.columns:
                            f.write(
                                f"  Mean matched inclusions: {df['inclusion_count_matched'].mean():.2f}\n"
                            )
                            avg_efficiency = (
                                df["inclusion_count_matched"]
                                / df["inclusion_count_bf"].replace(0, np.nan)
                            ).mean()
                            f.write(
                                f"  Average matching efficiency: {avg_efficiency:.2%}\n"
                            )

            show_info(f"Analysis results exported to {path}")
        except Exception as e:
            show_info(f"Error exporting analysis: {str(e)}")

    return widget
