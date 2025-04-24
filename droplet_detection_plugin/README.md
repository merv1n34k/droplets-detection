# napari-droplet-detector

A napari plugin for detecting and analyzing water-in-oil droplets with real-time parameter adjustment.

## Installation

```bash
git clone https://github.com/merv1n34k/napari-droplet-detection.git
cd napari-droplet-detection
pip install -e napari-droplet-detector
```

## Usage

1. Launch napari and load your images
2. Activate: `Plugins > Droplet Detector`
3. Select an image layer from the dropdown
4. Adjust parameters and click "Process Image"
5. Use tabs to configure styling and export options

## Parameters

```
min_diameter      # Minimum diameter in pixels
max_diameter      # Maximum diameter in pixels
min_circularity   # 0-1 value (1 = perfect circle)
max_border_dist   # Minimum distance from image borders
conversion_factor # Multiply pixels to get micrometers
```

## Features

- Real-time droplet detection with adjustable thresholds
- Customizable visualization (line color, width, style)
- Baking binary masks as permanent layers
- CSV export with comprehensive measurements
- Statistical analysis and visualization export

