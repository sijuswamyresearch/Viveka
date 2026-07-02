from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="viveka-refiner",
    version="1.0.0",
    author="Siju K S, Vipin Venugopal, and Soman KP",
    author_email="sijuswamy@gmail.com",
    description="Test-time refinement for chest X-ray denoising using physics-based constraints",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sijuswamyresearch/Viveka",
    project_urls={
        "Live Demo": "https://viveka-module.streamlit.app/",
        "Bug Tracker": "https://github.com/sijuswamyresearch/Viveka/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.19.0",
    ],
    keywords="chest-xray, denoising, post-processing, medical-imaging, test-time, radiology",
)