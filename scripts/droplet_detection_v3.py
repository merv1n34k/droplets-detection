# Water-in-Oil Droplet Detection Script

import os
import numpy as np
import pandas as pd
from skimage import (
        io, filters, measure, morphology, exposure,
        segmentation, restoration, color
)
from pathlib import Path
import napari
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


class DropletDetector:
    """Detect water-in-oil droplets and analyze their properties."""

    def __init__(
        self,
        min_diameter=10,
        max_diameter=500,
        min_circularity=0.5,
        min_border_dist=200,
        conversion_factor=1
    ):
        """
        Initialize the droplet detector with thresholds.

        Parameters:
        -----------
        min_diameter : float
            Minimum diameter of droplets to detect (in pixels)
        max_diameter : float
            Maximum diameter of droplets to detect (in pixels)
        min_circularity : float
            Minimum circularity threshold (0-1 range, where 1
            is a perfect circle)
        min_border_dist : float
            Minimum distance of centroid origin to image border
        conversion_factor : float
            Factor for conversion pixels to micrometers (um)
        """
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.min_circularity = min_circularity
        self.min_border_dist = min_border_dist
        self.conversion_factor = conversion_factor
        self.viewer = None
        print(f"Initialized DropletDetector with:\n"
              f" - Min diameter: {min_diameter} pixels\n"
              f" - Max diameter: {max_diameter} pixels\n"
              f" - Min circularity: {min_circularity:.2f}\n"
              f" - Min border distance: {min_border_dist} pixels\n"
              f" - Conversion factor (px to um): {conversion_factor:.2f}"
              )

    def process_directory(
        self,
        input_dir,
        output_csv,
        show_viewer=False,
        grid_view=True,
        layers_view="all"
    ):
        """
        Process all images in a directory and save results to CSV.

        Parameters:
        -----------
        input_dir : str
            Path to directory containing images
        output_csv : str
            Path to save the results CSV
        show_viewer : bool
            Whether to show napari viewer during processing
        grid_view : bool
            If True, show all images in a grid; if False, show each
            image individually
        layers_view : str
            Which layers view in napari, available options are:
            "mask", "image", "droplets", and "all"
        """
        input_path = Path(input_dir)
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = [f for f in input_path.iterdir()
                       if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {input_dir}")
            return

        all_results = []
        image_collection = []

        # Initialize viewer at the beginning if needed
        if show_viewer:
            self.viewer = napari.Viewer()

        for img_file in tqdm(image_files, desc="Processing images"):
            try:
                image = io.imread(img_file)
                # Process the image
                results = self.process_image(image, img_file.name)
                # Use only results, as `process_image` include binary mask
                all_results.extend(results[0])

                # Display or store image based on view mode
                if show_viewer:
                    processed_data = self._prepare_visualization(results)

                    # Store for collection of images
                    image_collection.append({
                        'name': img_file.name,
                        'image': image,
                        'binary_mask': processed_data['binary_mask'],
                        'droplets': processed_data['droplets'],
                        'properties': processed_data['properties']
                    })

            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        # Save results to CSV
        df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        # Display all images in the viewer if requested
        if show_viewer and self.viewer is not None and image_collection:
            self._display_image_collection(image_collection, grid_view, layers_view)
            napari.run()



    def process_image(self, image, image_name=""):
        """
        Detect oil droplets in `image` (numpy array) and return the same
        output structure as before: (results_list, binary_mask).

        Parameters
        ----------
        image : ndarray
            Raw image (already loaded via skimage.io.imread or similar).
        image_name : str
            Optional label passed through to each result dict.
        """

        # ------------------------------------------------------------------
        # STEP 0 – prepare a single-channel image in [0,1] for the pipeline
        # ------------------------------------------------------------------
        if image.ndim == 3:
            gray = color.rgb2gray(image)           # converts & scales to float
        else:
            gray = image.astype(float)
            if gray.max() > 1.0:                   # scale if 8-bit / 16-bit
                gray /= gray.max()

        # ------------------------------------------------------------------
        # STEP 1 – boost local contrast (CLAHE)
        # clip_limit tuned to keep noise low; feel free to expose as argument
        # ------------------------------------------------------------------
        gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)

        # ------------------------------------------------------------------
        # STEP 2 – denoise while keeping edges (bilateral)
        # ------------------------------------------------------------------
        gray_smooth = restoration.denoise_bilateral(
            gray_eq,
            sigma_color=0.05,      # colour variance
            sigma_spatial=3,       # spatial radius in px
            channel_axis=None      # skimage>=0.22
        )

        # ------------------------------------------------------------------
        # STEP 3 – adaptive threshold (Sauvola)
        # window size ~¾ of the minimum droplet diameter so that each window
        # “sees” mostly background + a bit of droplet.
        # ------------------------------------------------------------------
        block_size = max(15, int(self.min_diameter * 0.75))
        local_thresh = filters.threshold_sauvola(gray_smooth, window_size=block_size)
        binary = gray_smooth > local_thresh

        # ------------------------------------------------------------------
        # STEP 4 – close small gaps along edges
        # disk radius ≈ min_diameter / 4 – tweak if over/under closing
        # ------------------------------------------------------------------
        selem = morphology.disk(max(1, int(self.min_diameter / 4)))
        binary_closed = morphology.binary_closing(binary, selem)

        # ------------------------------------------------------------------
        # STEP 5 – remove tiny specks and fill holes
        # ------------------------------------------------------------------
        min_size = int(np.pi * (self.min_diameter/2)**2 * 0.5)
        cleaned = morphology.remove_small_objects(binary_closed, min_size=min_size)
        filled = morphology.remove_small_holes(cleaned, area_threshold=min_size)

        # ------------------------------------------------------------------
        # STEP 6 – optional watershed split (helps if droplets touch)
        # comment out this block if you *never* have merged droplets
        # ------------------------------------------------------------------
        distance = ndi.distance_transform_edt(filled)
        # peaks where distance is at least 50 % of the local max
        markers = measure.label(distance > 0.5 * distance.max())
        labeled_mask = segmentation.watershed(-distance, markers, mask=filled)

        # ------------------------------------------------------------------
        # Measure & filter exactly as before
        # ------------------------------------------------------------------
        props = measure.regionprops(labeled_mask, intensity_image=gray)

        # Make sure conversion factor is numeric
        conversion_factor = self.conversion_factor if isinstance(
            self.conversion_factor, (int, float)) else 1

        results = []
        for region in props:
            perimeter = region.perimeter
            area = region.area
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            diameter = region.equivalent_diameter

            y, x = region.centroid
            h, w = gray.shape
            within_border = (
                x >= self.min_border_dist and x <= (w - self.min_border_dist) and
                y >= self.min_border_dist and y <= (h - self.min_border_dist)
            )

            if (circularity >= self.min_circularity and
                self.min_diameter <= diameter <= self.max_diameter and
                within_border):

                results.append({
                    "droplet_id": region.label,
                    "diameter_px": diameter,
                    "diameter_um": diameter * conversion_factor,
                    "circularity": circularity,
                    "area_px": area,
                    "area_um2": area * conversion_factor**2,
                    "centroid_y": y,
                    "centroid_x": x,
                    "image_name": image_name
                })

        # Return API-compatible outputs
        binary_mask = filled          # for napari overlay etc.
        return results, binary_mask

    def _prepare_visualization(self, results):
        """
        Prepare visualization data for an image.

        Parameters:
        -----------
        image : ndarray
            Original image
        results : list
            List of droplet detection results

        Returns:
        --------
        dict
            Dictionary with visualization data
        """
        shapes_data = []
        shape_properties = {'diameter': [], 'circularity': []}

        for result in results[0]:
            # Create circle at centroid with appropriate radius
            y, x = result['centroid_y'], result['centroid_x']
            radius = result['diameter_px'] / 2

            # Create circle points
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = x + radius * np.cos(theta)
            circle_y = y + radius * np.sin(theta)
            circle_points = np.column_stack([circle_y, circle_x])

            shapes_data.append(circle_points)
            shape_properties['diameter'].append(result['diameter_px'])
            shape_properties['circularity'].append(result['circularity'])

        return {
            'binary_mask': results[1],
            'droplets': shapes_data,
            'properties': shape_properties
        }

    def _display_image_collection(self, image_collection, grid=True, layers="all"):
        """
        Display a grid of images in the napari viewer.

        Parameters:
        -----------
        image_collection : list
            List of dictionaries with image data
        """
        if not self.viewer:
            self.viewer = napari.Viewer()

        # Clear existing layers
        self.viewer.layers.clear()

        # Create a grid of images
        rows = int(np.ceil(np.sqrt(len(image_collection))))
        cols = int(np.ceil(len(image_collection) / rows))

        for i, item in enumerate(image_collection):
            # Calculate grid position
            if grid:
                row = i // cols
                col = i % cols
            else:
                row = 0
                col = 0

            # Add image to viewer with offset
            offset = [row * image_collection[0]['image'].shape[0],
                      col * image_collection[0]['image'].shape[1]]

            # Add layers
            # Add original image
            if layers == "all" or layers == "image":
                self.viewer.add_image(
                    item['image'],
                    name=f"Original_{item['name']}",
                    translate=offset
                )

            # Add binary mask
            if layers == "all" or layers == "mask":
                self.viewer.add_image(
                    item['binary_mask'],
                    name=f"Mask_{item['name']}",
                    opacity=0.5,
                    translate=offset
                )

            # Add shape layers for droplets
            if layers == "all" or layers == "droplets":
                if item['droplets']:
                    # Offset each droplet shape
                    offset_shapes = []
                    for shape in item['droplets']:
                        offset_shape = shape.copy()
                        offset_shape[:, 0] += offset[0]
                        offset_shape[:, 1] += offset[1]
                        offset_shapes.append(offset_shape)

                    self.viewer.add_shapes(
                        offset_shapes,
                        shape_type='polygon',
                        edge_color='red',
                        face_color='transparent',
                        name=f"Droplets_{item['name']}",
                        properties=item['properties']
                    )

        # Reset view to fit all images
        self.viewer.reset_view()

    def analyze_results(self, csv_path, output_dir=None):
        """
        Analyze droplet detection results and generate visualizations.

        Parameters:
        -----------
        csv_path : str
            Path to the CSV file with droplet data
        output_dir : str, optional
            Directory to save visualization files
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Load the CSV data with explicit path handling
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)

            # Check if required columns exist
            required_cols = ['image_name', 'diameter_um', 'circularity']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Error: Missing columns in CSV: {missing_cols}")
                return

            # Print summary statistics
            print(f"Total droplets detected: {len(df)}")
            print(f"Number of images: {df['image_name'].nunique()}")
            print(f"Droplets per image: {len(df) / df['image_name'].nunique():.1f}")
            print("Diameter statistics:")
            print(f"  Min: {df['diameter_um'].min():.2f}")
            print(f"  Max: {df['diameter_um'].max():.2f}")
            print(f"  Mean: {df['diameter_um'].mean():.2f}")
            print(f"  Median: {df['diameter_um'].median():.2f}")

            # Create histogram of droplet diameters
            plt.figure(figsize=(10, 6))
            plt.hist(df['diameter_um'], bins=30, alpha=0.7, color='blue')
            plt.xlabel('Droplet Diameter (pixels)')
            plt.ylabel('Count')
            plt.title('Distribution of Droplet Diameters')
            plt.grid(alpha=0.3)

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'diameter_histogram.png'), dpi=300)
                plt.close()
            else:
                plt.show()

            # Create scatter plot of diameter vs circularity
            plt.figure(figsize=(10, 6))
            plt.scatter(df['diameter_um'], df['circularity'], alpha=0.5)
            plt.xlabel('Droplet Diameter (pixels)')
            plt.ylabel('Circularity')
            plt.title('Droplet Diameter vs Circularity')
            plt.axhline(y=self.min_circularity, color='r', linestyle='--',
                        label=f'Min Circularity: {self.min_circularity}')
            plt.grid(alpha=0.3)
            plt.legend()

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'diameter_vs_circularity.png'), dpi=300)
                plt.close()
            else:
                plt.show()

        except Exception as e:
            print(f"Error analyzing results: {e}")


def main():
    """Main function to run the droplet detection."""
    import argparse

    parser = argparse.ArgumentParser(description='Detect and analyze water-in-oil droplets')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path to save results CSV')
    parser.add_argument('--min_diameter', type=float, default=10,
                        help='Minimum droplet diameter in pixels')
    parser.add_argument('--max_diameter', type=float, default=500,
                        help='Maximum droplet diameter in pixels')
    parser.add_argument('--min_circularity', type=float, default=0.8,
                        help='Minimum circularity (0-1)')
    parser.add_argument('--show_viewer', action='store_true',
                        help='Show napari viewer during processing')
    parser.add_argument('--no_grid_view', action='store_false', default=True,
                        help='Show all images in a grid layout (default)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze results after processing')
    parser.add_argument('--viz_dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--layers_view', type=str, default='all',
                        help='Show selected layers')
    parser.add_argument('--min_border_dist', type=float, default=0,
                        help='Maximum centroid distance to image borders')
    parser.add_argument('--conversion_factor', type=float, default=1,
                        help='Conversion factor from px to um (not used)')

    args = parser.parse_args()

    # Initialize the detector
    detector = DropletDetector(
        min_diameter=args.min_diameter,
        max_diameter=args.max_diameter,
        min_circularity=args.min_circularity,
        min_border_dist=args.min_border_dist,
        conversion_factor=args.conversion_factor
    )

    # Process images
    detector.process_directory(
        args.input_dir,
        args.output_csv,
        show_viewer=args.show_viewer,
        grid_view=args.no_grid_view,
        layers_view=args.layers_view
    )

    # Analyze results if requested
    if args.analyze:
        detector.analyze_results(args.output_csv, args.viz_dir)


if __name__ == "__main__":
    main()
