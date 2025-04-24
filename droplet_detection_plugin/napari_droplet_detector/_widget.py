#!/usr/bin/env python
# Napari Droplet Detector Plugin

import os
import numpy as np
from skimage import filters, measure, morphology
import napari
from napari.layers import Image, Shapes
from napari.qt.threading import thread_worker
from magicgui import magic_factory
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QDoubleSpinBox, QComboBox, QGroupBox, QCheckBox, QLineEdit,
    QColorDialog, QFileDialog, QTabWidget, QMessageBox
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
import pandas as pd
import matplotlib.pyplot as plt


class DropletDetectorWidget(QWidget):
    """Widget for interactive droplet detection in Napari."""
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.setMinimumWidth(350)
        
        # Initialize settings
        self.settings = {
            'min_diameter': 10,
            'max_diameter': 500,
            'min_circularity': 0.8,
            'max_border_dist': 0,
            'conversion_factor': 1.0,
            'edge_color': 'red',
            'edge_width': 2,
            'line_style': 'solid'
        }
        
        # State variables
        self.binary_mask = None
        self.original_image = None
        self.processed_results = []
        self.droplet_shapes = None
        
        # Create the UI
        self.init_ui()
        
        # Connect to layer selection change event
        self.viewer.layers.selection.events.changed.connect(self.on_layer_selection_change)
    
    def init_ui(self):
        """Initialize the user interface components."""
        # Create tabs
        tabs = QTabWidget()
        self.layout().addWidget(tabs)
        
        # Parameters Tab
        param_tab = QWidget()
        param_layout = QVBoxLayout()
        param_tab.setLayout(param_layout)
        
        # Image Selection Group
        image_group = QGroupBox("Image Selection")
        image_layout = QVBoxLayout()
        
        self.image_selector = QComboBox()
        self.update_image_selector()
        image_layout.addWidget(QLabel("Image Layer:"))
        image_layout.addWidget(self.image_selector)
        
        image_group.setLayout(image_layout)
        param_layout.addWidget(image_group)
        
        # Detection Parameters Group
        params_group = QGroupBox("Detection Parameters")
        params_layout = QVBoxLayout()
        
        # Min diameter
        min_diam_layout = QHBoxLayout()
        min_diam_layout.addWidget(QLabel("Min Diameter (px):"))
        self.min_diameter_spin = QDoubleSpinBox()
        self.min_diameter_spin.setRange(1, 1000)
        self.min_diameter_spin.setValue(self.settings['min_diameter'])
        self.min_diameter_spin.valueChanged.connect(self.update_parameters)
        min_diam_layout.addWidget(self.min_diameter_spin)
        params_layout.addLayout(min_diam_layout)
        
        # Max diameter
        max_diam_layout = QHBoxLayout()
        max_diam_layout.addWidget(QLabel("Max Diameter (px):"))
        self.max_diameter_spin = QDoubleSpinBox()
        self.max_diameter_spin.setRange(1, 5000)
        self.max_diameter_spin.setValue(self.settings['max_diameter'])
        self.max_diameter_spin.valueChanged.connect(self.update_parameters)
        max_diam_layout.addWidget(self.max_diameter_spin)
        params_layout.addLayout(max_diam_layout)
        
        # Min circularity
        min_circ_layout = QHBoxLayout()
        min_circ_layout.addWidget(QLabel("Min Circularity (0-1):"))
        self.min_circularity_spin = QDoubleSpinBox()
        self.min_circularity_spin.setRange(0, 1)
        self.min_circularity_spin.setSingleStep(0.05)
        self.min_circularity_spin.setValue(self.settings['min_circularity'])
        self.min_circularity_spin.valueChanged.connect(self.update_parameters)
        min_circ_layout.addWidget(self.min_circularity_spin)
        params_layout.addLayout(min_circ_layout)
        
        # Max border distance
        border_dist_layout = QHBoxLayout()
        border_dist_layout.addWidget(QLabel("Border Distance (px):"))
        self.border_dist_spin = QDoubleSpinBox()
        self.border_dist_spin.setRange(0, 1000)
        self.border_dist_spin.setValue(self.settings['max_border_dist'])
        self.border_dist_spin.valueChanged.connect(self.update_parameters)
        border_dist_layout.addWidget(self.border_dist_spin)
        params_layout.addLayout(border_dist_layout)
        
        # Conversion factor
        conv_factor_layout = QHBoxLayout()
        conv_factor_layout.addWidget(QLabel("Conversion Factor (px to μm):"))
        self.conv_factor_spin = QDoubleSpinBox()
        self.conv_factor_spin.setRange(0.01, 100)
        self.conv_factor_spin.setDecimals(3)
        self.conv_factor_spin.setSingleStep(0.1)
        self.conv_factor_spin.setValue(self.settings['conversion_factor'])
        self.conv_factor_spin.valueChanged.connect(self.update_parameters)
        conv_factor_layout.addWidget(self.conv_factor_spin)
        params_layout.addLayout(conv_factor_layout)
        
        params_group.setLayout(params_layout)
        param_layout.addWidget(params_group)
        
        # Styling Tab
        style_tab = QWidget()
        style_layout = QVBoxLayout()
        style_tab.setLayout(style_layout)
        
        # Droplet Styling Group
        style_group = QGroupBox("Droplet Styling")
        style_inner_layout = QVBoxLayout()
        
        # Edge color
        edge_color_layout = QHBoxLayout()
        edge_color_layout.addWidget(QLabel("Edge Color:"))
        self.edge_color_btn = QPushButton()
        self.edge_color_btn.setStyleSheet(f"background-color: {self.settings['edge_color']}")
        self.edge_color_btn.clicked.connect(self.select_edge_color)
        edge_color_layout.addWidget(self.edge_color_btn)
        style_inner_layout.addLayout(edge_color_layout)
        
        # Line width
        line_width_layout = QHBoxLayout()
        line_width_layout.addWidget(QLabel("Line Width:"))
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.5, 10)
        self.line_width_spin.setSingleStep(0.5)
        self.line_width_spin.setValue(self.settings['edge_width'])
        self.line_width_spin.valueChanged.connect(self.update_style)
        line_width_layout.addWidget(self.line_width_spin)
        style_inner_layout.addLayout(line_width_layout)
        
        # Line style
        line_style_layout = QHBoxLayout()
        line_style_layout.addWidget(QLabel("Line Style:"))
        self.line_style_combo = QComboBox()
        self.line_style_combo.addItems(["solid", "dashed", "dotted"])
        self.line_style_combo.setCurrentText(self.settings['line_style'])
        self.line_style_combo.currentTextChanged.connect(self.update_style)
        line_style_layout.addWidget(self.line_style_combo)
        style_inner_layout.addLayout(line_style_layout)
        
        # Layer visibility options
        vis_layout = QVBoxLayout()
        self.show_mask_check = QCheckBox("Show Binary Mask")
        self.show_mask_check.setChecked(True)
        self.show_mask_check.stateChanged.connect(self.update_visibility)
        vis_layout.addWidget(self.show_mask_check)
        
        self.show_droplets_check = QCheckBox("Show Detected Droplets")
        self.show_droplets_check.setChecked(True)
        self.show_droplets_check.stateChanged.connect(self.update_visibility)
        vis_layout.addWidget(self.show_droplets_check)
        
        style_inner_layout.addLayout(vis_layout)
        style_group.setLayout(style_inner_layout)
        style_layout.addWidget(style_group)
        
        # Add tabs to the tabwidget
        tabs.addTab(param_tab, "Parameters")
        tabs.addTab(style_tab, "Styling")
        
        # Export Group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        # Bake mask button
        self.bake_mask_btn = QPushButton("Bake Binary Mask")
        self.bake_mask_btn.clicked.connect(self.bake_mask)
        export_layout.addWidget(self.bake_mask_btn)
        
        # Export to CSV button
        self.export_csv_btn = QPushButton("Export Droplets to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        export_layout.addWidget(self.export_csv_btn)
        
        # Export visualization button
        self.export_viz_btn = QPushButton("Export Visualization")
        self.export_viz_btn.clicked.connect(self.export_visualization)
        export_layout.addWidget(self.export_viz_btn)
        
        export_group.setLayout(export_layout)
        self.layout().addWidget(export_group)
        
        # Status label
        self.status_label = QLabel("Select an image to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.status_label)
        
        # Process button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_current_image)
        self.layout().addWidget(self.process_btn)
    
    def update_image_selector(self):
        """Update the image layer selector with available image layers."""
        self.image_selector.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self.image_selector.addItem(layer.name)
    
    def on_layer_selection_change(self, event=None):
        """Handle changes to the selected layer in napari viewer."""
        # Update image selector when layers change
        self.update_image_selector()
    
    def update_parameters(self):
        """Update detection parameters from UI controls."""
        self.settings['min_diameter'] = self.min_diameter_spin.value()
        self.settings['max_diameter'] = self.max_diameter_spin.value()
        self.settings['min_circularity'] = self.min_circularity_spin.value()
        self.settings['max_border_dist'] = self.border_dist_spin.value()
        self.settings['conversion_factor'] = self.conv_factor_spin.value()
        
        # Update detected droplets if we have an image
        if self.original_image is not None:
            self.process_image(self.original_image)
    
    def select_edge_color(self):
        """Open a color dialog to select the edge color."""
        color = QColorDialog.getColor(QColor(self.settings['edge_color']), self)
        if color.isValid():
            self.settings['edge_color'] = color.name()
            self.edge_color_btn.setStyleSheet(f"background-color: {color.name()}")
            self.update_style()
    
    def update_style(self):
        """Update the styling of droplet outlines."""
        self.settings['edge_width'] = self.line_width_spin.value()
        self.settings['line_style'] = self.line_style_combo.currentText()
        
        # Apply styling to existing droplet layer if it exists
        if self.droplet_shapes is not None and self.droplet_shapes in self.viewer.layers:
            self.droplet_shapes.edge_width = self.settings['edge_width']
            self.droplet_shapes.edge_color = self.settings['edge_color']
            
            # Apply line style
            if self.settings['line_style'] == 'dashed':
                # Napari uses linestyle (-, --, :, -.)
                self.droplet_shapes.edge_width_is_relative = False
                self.droplet_shapes.edge_style = '--'
            elif self.settings['line_style'] == 'dotted':
                self.droplet_shapes.edge_width_is_relative = False
                self.droplet_shapes.edge_style = ':'
            else:  # solid
                self.droplet_shapes.edge_width_is_relative = False
                self.droplet_shapes.edge_style = '-'
    
    def update_visibility(self):
        """Update layer visibility based on checkboxes."""
        # Update mask visibility
        mask_layer = None
        for layer in self.viewer.layers:
            if layer.name == "Binary Mask":
                mask_layer = layer
                break
        
        if mask_layer is not None:
            mask_layer.visible = self.show_mask_check.isChecked()
        
        # Update droplet shapes visibility
        if self.droplet_shapes is not None and self.droplet_shapes in self.viewer.layers:
            self.droplet_shapes.visible = self.show_droplets_check.isChecked()
    
    def process_current_image(self):
        """Process the currently selected image."""
        # Get the selected image name
        if self.image_selector.count() == 0:
            self.status_label.setText("No image layers available")
            return
        
        selected_image_name = self.image_selector.currentText()
        
        # Get the layer from the viewer
        selected_layer = None
        for layer in self.viewer.layers:
            if layer.name == selected_image_name:
                selected_layer = layer
                break
        
        if selected_layer is None:
            self.status_label.setText("Selected layer not found")
            return
        
        # Get the image data
        image_data = selected_layer.data
        
        # Convert RGB to grayscale if needed
        if len(image_data.shape) > 2 and image_data.shape[2] > 1:
            image_data = np.mean(image_data[:,:,:3], axis=2).astype(np.uint8)
        
        # Store the original image
        self.original_image = image_data
        
        # Process the image
        self.status_label.setText("Processing image...")
        self.process_image(image_data)
    
    def process_image(self, image):
        """
        Process the image to detect droplets.
        
        Parameters:
        -----------
        image : ndarray
            Input image as numpy array
        """
        # Apply Gaussian blur to reduce noise
        smoothed = filters.gaussian(image, sigma=1.0)
        
        # Threshold the image using Otsu's method
        threshold_value = filters.threshold_otsu(smoothed)
        binary = smoothed > threshold_value
        
        # Clean up binary image
        # Remove small objects
        min_size = int(np.pi * (self.settings['min_diameter']/2)**2 * 0.5)
        cleaned = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Fill holes in the binary image
        binary_mask = morphology.remove_small_holes(cleaned)
        
        # Store the binary mask
        self.binary_mask = binary_mask
        
        # Display the binary mask in the viewer
        mask_name = "Binary Mask"
        
        # Check if mask layer already exists
        mask_exists = False
        for layer in self.viewer.layers:
            if layer.name == mask_name:
                # Update existing layer
                layer.data = binary_mask
                mask_exists = True
                break
        
        # Create new mask layer if it doesn't exist
        if not mask_exists:
            self.viewer.add_image(
                binary_mask,
                name=mask_name,
                opacity=0.5,
                visible=self.show_mask_check.isChecked()
            )
        
        # Detect droplets based on binary mask
        self.detect_droplets(image, binary_mask)
    
    def detect_droplets(self, image, binary_mask):
        """
        Detect droplets from binary mask and display them.
        
        Parameters:
        -----------
        image : ndarray
            Original image
        binary_mask : ndarray
            Binary mask of potential droplets
        """
        # Label connected components
        labeled_mask, num_labels = measure.label(binary_mask, return_num=True)
        
        # Measure properties of the detected regions
        props = measure.regionprops(labeled_mask, image)
        
        # Filter regions based on criteria and prepare visualization
        results = []
        shapes_data = []
        shape_properties = {
            'diameter_px': [], 
            'diameter_um': [],
            'circularity': [],
            'area_px': [],
            'area_um2': []
        }
        
        for region in props:
            # Calculate perimeter and area
            perimeter = region.perimeter
            area = region.area
            
            # Calculate circularity: 4*pi*area/(perimeter^2)
            circularity = 0
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
            # Calculate diameter
            diameter = region.equivalent_diameter
            
            # Check boundary conditions
            image_shape = image.shape
            y, x = region.centroid
            
            border_dist = self.settings['max_border_dist']
            within_border = (
                x >= border_dist and 
                x <= (image_shape[1] - border_dist) and
                y >= border_dist and 
                y <= (image_shape[0] - border_dist)
            )
            
            # Check criteria
            if (circularity >= self.settings['min_circularity'] and
                diameter >= self.settings['min_diameter'] and
                diameter <= self.settings['max_diameter'] and
                within_border):
                
                # Store result
                result = {
                    'droplet_id': region.label,
                    'diameter_px': diameter,
                    'diameter_um': diameter * self.settings['conversion_factor'],
                    'circularity': circularity,
                    'area_px': area,
                    'area_um2': area * self.settings['conversion_factor']**2,
                    'centroid_y': y,
                    'centroid_x': x
                }
                results.append(result)
                
                # Create circle points for visualization
                radius = diameter / 2
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = x + radius * np.cos(theta)
                circle_y = y + radius * np.sin(theta)
                circle_points = np.column_stack([circle_y, circle_x])
                
                shapes_data.append(circle_points)
                shape_properties['diameter_px'].append(diameter)
                shape_properties['diameter_um'].append(diameter * self.settings['conversion_factor'])
                shape_properties['circularity'].append(circularity)
                shape_properties['area_px'].append(area)
                shape_properties['area_um2'].append(area * self.settings['conversion_factor']**2)
        
        # Store results
        self.processed_results = results
        
        # Update status
        self.status_label.setText(f"Detected {len(results)} droplets")
        
        # Add or update shapes layer
        if self.droplet_shapes is not None and self.droplet_shapes in self.viewer.layers:
            self.viewer.layers.remove(self.droplet_shapes)
        
        if shapes_data:
            self.droplet_shapes = self.viewer.add_shapes(
                shapes_data,
                shape_type='polygon',
                edge_color=self.settings['edge_color'],
                face_color='transparent',
                edge_width=self.settings['edge_width'],
                name="Detected Droplets",
                properties=shape_properties,
                visible=self.show_droplets_check.isChecked()
            )
            
            # Apply line style
            if self.settings['line_style'] == 'dashed':
                self.droplet_shapes.edge_style = '--'
            elif self.settings['line_style'] == 'dotted':
                self.droplet_shapes.edge_style = ':'
            else:  # solid
                self.droplet_shapes.edge_style = '-'
        else:
            self.droplet_shapes = None
    
    def bake_mask(self):
        """Add the binary mask as a permanent layer in the viewer."""
        if self.binary_mask is None:
            self.status_label.setText("No binary mask available")
            return
        
        # Create a new layer for the baked mask
        baked_name = f"Baked Mask {len([l for l in self.viewer.layers if 'Baked Mask' in l.name]) + 1}"
        self.viewer.add_image(
            self.binary_mask,
            name=baked_name,
            opacity=0.5
        )
        
        self.status_label.setText(f"Baked mask as '{baked_name}'")
    
    def export_to_csv(self):
        """Export detected droplets to a CSV file."""
        if not self.processed_results:
            self.status_label.setText("No results to export")
            return
        
        # Get the save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Ensure the file has .csv extension
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        
        # Create dataframe and save to CSV
        df = pd.DataFrame(self.processed_results)
        
        # Add image name if available
        if self.image_selector.currentText():
            df['image_name'] = self.image_selector.currentText()
        
        # Save to file
        try:
            df.to_csv(file_path, index=False)
            self.status_label.setText(f"Exported {len(df)} droplets to {file_path}")
        except Exception as e:
            self.status_label.setText(f"Error exporting CSV: {str(e)}")
    
    def export_visualization(self):
        """Export visualization of droplet detection and graphs."""
        if not self.processed_results or self.original_image is None:
            self.status_label.setText("No results to visualize")
            return
        
        # Get the save directory
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory for Visualization"
        )
        
        if not dir_path:
            return
        
        try:
            # Create dataframe from results
            df = pd.DataFrame(self.processed_results)
            
            # Create histogram of droplet diameters
            plt.figure(figsize=(10, 6))
            plt.hist(df['diameter_um'], bins=30, alpha=0.7, color='blue')
            plt.xlabel('Droplet Diameter (μm)')
            plt.ylabel('Count')
            plt.title('Distribution of Droplet Diameters')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(dir_path, 'diameter_histogram.png'), dpi=300)
            plt.close()
            
            # Create scatter plot of diameter vs circularity
            plt.figure(figsize=(10, 6))
            plt.scatter(df['diameter_um'], df['circularity'], alpha=0.5)
            plt.xlabel('Droplet Diameter (μm)')
            plt.ylabel('Circularity')
            plt.title('Droplet Diameter vs Circularity')
            plt.axhline(y=self.settings['min_circularity'], color='r', linestyle='--',
                      label=f'Min Circularity: {self.settings["min_circularity"]}')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(dir_path, 'diameter_vs_circularity.png'), dpi=300)
            plt.close()
            
            # Save CSV of results
            csv_path = os.path.join(dir_path, 'droplet_results.csv')
            if self.image_selector.currentText():
                df['image_name'] = self.image_selector.currentText()
            df.to_csv(csv_path, index=False)
            
            # Save summary statistics
            stats_path = os.path.join(dir_path, 'summary_statistics.txt')
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
            
            self.status_label.setText(f"Exported visualization to {dir_path}")
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Complete",
                f"Visualization files saved to:\n{dir_path}"
            )
            
        except Exception as e:
            self.status_label.setText(f"Error exporting visualization: {str(e)}")
            QMessageBox.warning(
                self,
                "Export Error",
                f"An error occurred during export:\n{str(e)}"
            )


# This function will be called by napari to create the widget
@napari.qt.plugin_dock_widget
def create_droplet_detector():
    return DropletDetectorWidget

# For testing without installing as a plugin
if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = DropletDetectorWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Droplet Detector")
    napari.run()