# Droplets detection

Note: currently it is partially implemented inclusions detection, but the
algorithm is inaccurate, use `detect.py` for advanced droplets/inclusions
detection.

This is a collection of python scripts and a napari plugin for detecting droplets from microscopy images.

## How to use
The repo contains automated pipelines (see `scripts` directory), napari plugin for playing with droplet-detection parameter, and new script for visualizing each image processing step. See `README` in plugin directory for more details. Use the latest script version *(v4)*, other versions are legacy scripts, example:

```bash

python droplet_detection_v4.py --input_dir ~/path/to/images/ --output_csv ~/path/to/output.csv --min_diameter 60 --max_diameter 200 --min_circularity 0.7 --min_border_dist 50 --conversion_factor 1.1376

```

For more detailed options use `python droplet_detection_v4 -h`.

## License

Distributed under MIT License, see `LICENSE`
