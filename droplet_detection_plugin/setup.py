from setuptools import setup, find_packages

setup(
    name="napari-droplet-detector",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "napari>=0.4.12",
        "numpy>=1.20.0",
        "scikit-image>=0.18.0",
        "pandas>=1.2.0",
        "matplotlib>=3.4.0",
        "magicgui>=0.3.0",
    ],
    entry_points={
        "napari.plugin": [
            "drop_detect = napari_droplet_detector",
        ],
    },
)
