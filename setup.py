from setuptools import setup, find_packages

setup(
    name="hyurl",
    version="0.1.0",
    description="A reinforcement learning library",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add your dependencies here
        # Example: "numpy>=1.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)