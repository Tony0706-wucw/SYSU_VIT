from setuptools import setup, find_packages

setup(
    name="vit-training",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "tqdm>=4.50.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Vision Transformer Training Framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vit-training",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
