# Napari droplets detector

A `napari` plugin for detecting and analyzing water-in-oil droplets with parameter adjustment.

## Installation

```bash
git clone https://github.com/merv1n34k/droplets-detection.git
cd droplets-detection
pip install -e napari-droplets-detector
```

## Usage

1. Launch `napari` and load your images
2. Activate: `Plugins > Droplets Detector`
3. Select an image layer from the drop-down menu
4. Adjust parameters and click "Process Image"
5. Use widgets to export and analyze detected droplets

## Parameters

```
min_diameter      # Minimum diameter in pixels
max_diameter      # Maximum diameter in pixels
min_circularity   # 0-1 value (1 = perfect circle)
min_border_dist   # Minimum distance from image borders
conversion_factor # Factor to convert pixels to micrometers
```

## Features

- Droplet detection with adjustable parameters
- Custom palette import for visually beautiful droplets
- CSV export with comprehensive measurements
- Statistical analysis and visualization export

