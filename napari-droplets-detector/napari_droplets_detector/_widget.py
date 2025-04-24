"""
A napari plugin for interactive detection and analysis of water-in-oil droplets.
"""

import os
import numpy as np
from skimage import io, filters, measure, morphology
import napari
from napari.layers import Image, Shapes
import pandas as pd
import matplotlib.pyplot as plt
from magicgui import magic_factory
from napari.utils.notifications import show_info

def _droplets_detection_alorythm(image, min_diameter):
    # Apply Gaussian blur to reduce noise
    smoothed = filters.gaussian(image, sigma=1.0)

    # Threshold the image using Otsu's method
    threshold_value = filters.threshold_otsu(smoothed)
    binary = smoothed > threshold_value

    # Clean up binary image
    # Remove small objects
    min_size = int(np.pi * (min_diameter/2)**2 * 0.5)
    cleaned = morphology.remove_small_objects(binary, min_size=min_size)

    # Fill holes in the binary image
    binary_mask = morphology.remove_small_holes(cleaned)

    # Detect droplets
    # Label connected components
    labeled_mask = measure.label(binary_mask, return_num=True)[0]

    # Measure properties of the detected regions
    props = measure.regionprops(labeled_mask, image)

    return (binary_mask, props)

@magic_factory(
    call_button="Process Image",
    layout="vertical",
    min_diameter={"widget_type": "FloatSpinBox", "min": 1.0, "max": 1000.0, "step": 1.0, "value": 60.0, "label": "Min Diameter (px)"},
    max_diameter={"widget_type": "FloatSpinBox", "min": 1.0, "max": 5000.0, "step": 10.0, "value": 200.0, "label": "Max Diameter (px)"},
    min_circularity={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.5, "label": "Min Circularity"},
    min_border_dist={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 10.0, "value": 50.0, "label": "Border Distance (px)"},
    conversion_factor={"widget_type": "FloatSpinBox", "min": 0.01, "max": 100.0, "step": 0.0001, "value": 1.0, "label": "Conversion Factor (px to μm)"},
)
def droplets_detector_widget(
    viewer: napari.Viewer,
    image_layer: Image,
    min_diameter: float,
    max_diameter: float,
    min_circularity: float,
    min_border_dist: float,
    conversion_factor: float,
) -> None:
    """
    Detect and analyze water-in-oil droplets in images.
    """
    if image_layer is None:
        show_info("Please select an image layer")
        return

    # Get image data
    image = image_layer.data

    # Convert RGB to grayscale if needed
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = np.mean(image[:,:,:3], axis=2).astype(np.uint8)

    processed_image = _droplets_detection_alorythm(image, min_diameter)

    # Add or update binary mask layer
    if "Binary Mask" in viewer.layers:
        viewer.layers["Binary Mask"].data = processed_image[0]
    else:
        viewer.add_image(
            processed_image[0],
            name="Binary Mask",
            opacity=0.5,
            colormap="red"
        )

    # Filter regions based on criteria
    results = []
    shapes_data = []

    for region in processed_image[1]:
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
