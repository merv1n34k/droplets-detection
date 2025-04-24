# Water-in-Oil Droplet Detection Script

import os
import numpy as np
import pandas as pd
from skimage import io, filters, measure, morphology, feature
from pathlib import Path
import napari
from tqdm import tqdm
import matplotlib.pyplot as plt


class DropletDetector:
    """Detect water-in-oil droplets and analyze their properties."""

    def __init__(self, min_diameter=10, max_diameter=500, min_circularity=0.8):
        """
        Initialize the droplet detector with thresholds.

        Parameters:
        -----------
        min_diameter : float
            Minimum diameter of droplets to detect (in pixels)
        max_diameter : float
            Maximum diameter of droplets to detect (in pixels)
        min_circularity : float
            Minimum circularity threshold (0-1 range, where 1 is a perfect circle)
        """
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.min_circularity = min_circularity
        self.viewer = None

    def process_directory(self, input_dir, output_csv, show_viewer=False):
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
        """
        input_path = Path(input_dir)
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_files = [f for f in input_path.iterdir()
                      if f.suffix.lower() in image_extensions]

        all_results = []

        if show_viewer:
            self.viewer = napari.Viewer()

        for img_file in tqdm(image_files, desc="Processing images"):
            try:
                image = io.imread(img_file)
                if len(image.shape) > 2 and image.shape[2] > 1:
                    # Convert RGB to grayscale if needed
                    image = np.mean(image[:,:,:3], axis=2).astype(np.uint8)

                results = self.process_image(image, img_file.name, show_viewer)
                all_results.extend(results)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        # Save results to CSV
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        if show_viewer and self.viewer is not None:
            napari.run()

    def process_image(self, image, image_name="", show_viewer=False):
        """
        Process a single image to detect droplets.

        Parameters:
        -----------
        image : ndarray
            Input image as numpy array
        image_name : str
            Name of the image file for reference
        show_viewer : bool
            Whether to display the image in napari viewer

        Returns:
        --------
        list of dicts
            List of dictionaries containing properties of detected droplets
        """
        # Pre-process image
        # Apply Gaussian blur to reduce noise
        smoothed = filters.gaussian(image, sigma=1.0)

        # Threshold the image using Otsu's method
        threshold_value = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold_value

        # Clean up binary image
        # Remove small objects
        min_size = int(np.pi * (self.min_diameter/2)**2 * 0.5)  # Minimum area in pixels
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)

        # Fill holes in the binary image
        filled = morphology.remove_small_holes(cleaned)

        # Label connected components
        labeled_mask, num_labels = measure.label(filled, return_num=True)

        # Measure properties of the detected regions
        props = measure.regionprops(labeled_mask, image)

        # Filter regions based on circularity and diameter
        results = []
        valid_regions = []

        for region in props:
            # Calculate perimeter and area
            perimeter = region.perimeter
            area = region.area

            # Calculate circularity: 4*pi*area/(perimeter^2)
            # A perfect circle has circularity of 1
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # Calculate diameter (average of major and minor axis lengths)
            # Some versions use equivalent_diameter
            if hasattr(region, 'equivalent_diameter_area'):
                diameter = region.equivalent_diameter_area
            else:
                diameter = region.equivalent_diameter

            # Alternative diameter calculation
            # diameter = 2 * np.sqrt(area / np.pi)

            # Check if the droplet meets our criteria
            if (circularity >= self.min_circularity and
                diameter >= self.min_diameter and
                diameter <= self.max_diameter):

                result = {
                    'image_name': image_name,
                    'droplet_id': region.label,
                    'diameter': diameter,
                    'circularity': circularity,
                    'area': area,
                    'centroid_y': region.centroid[0],
                    'centroid_x': region.centroid[1]
                }
                results.append(result)
                valid_regions.append(region)

        # Visualization using napari if requested
        if show_viewer and self.viewer is not None:
            # Clear previous layers
            self.viewer.layers.clear()

            # Add original image
            self.viewer.add_image(image, name='Original')

            # Add binary mask
            self.viewer.add_image(filled, name='Binary Mask', opacity=0.5)

            # Create circles for detected droplets
            shapes_data = []
            shape_properties = {'diameter': [], 'circularity': []}

            for region in valid_regions:
                # Create circle at centroid with appropriate radius
                y, x = region.centroid
                radius = region.equivalent_diameter / 2

                # Create circle points (Napari expects shapes as polygon vertices)
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = x + radius * np.cos(theta)
                circle_y = y + radius * np.sin(theta)
                circle_points = np.column_stack([circle_y, circle_x])

                shapes_data.append(circle_points)
                shape_properties['diameter'].append(region.equivalent_diameter)
                shape_properties['circularity'].append(
                    4 * np.pi * region.area / (region.perimeter * region.perimeter)
                )

            if shapes_data:
                self.viewer.add_shapes(
                    shapes_data,
                    shape_type='polygon',
                    edge_color='red',
                    face_color='transparent',
                    name='Detected Droplets',
                    properties=shape_properties
                )

            # Refresh viewer
            self.viewer.reset_view()

        return results

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

        # Load the CSV data
        df = pd.read_csv(csv_path)

        # Print summary statistics
        print(f"Total droplets detected: {len(df)}")
        print(f"Droplets per image: {len(df) / df['image_name'].nunique():.1f}")
        print(f"Diameter statistics:")
        print(f"  Min: {df['diameter'].min():.2f}")
        print(f"  Max: {df['diameter'].max():.2f}")
        print(f"  Mean: {df['diameter'].mean():.2f}")
        print(f"  Median: {df['diameter'].median():.2f}")

        # Create histogram of droplet diameters
        plt.figure(figsize=(10, 6))
        plt.hist(df['diameter'], bins=30, alpha=0.7, color='blue')
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
        plt.scatter(df['diameter'], df['circularity'], alpha=0.5)
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
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze results after processing')
    parser.add_argument('--viz_dir', type=str, default=None,
                        help='Directory to save visualizations')

    args = parser.parse_args()

    # Initialize the detector
    detector = DropletDetector(
        min_diameter=args.min_diameter,
        max_diameter=args.max_diameter,
        min_circularity=args.min_circularity
    )

    # Process images
    detector.process_directory(
        args.input_dir,
        args.output_csv,
        show_viewer=args.show_viewer
    )

    # Analyze results if requested
    if args.analyze:
        detector.analyze_results(args.output_csv, args.viz_dir)


if __name__ == "__main__":
    main()
