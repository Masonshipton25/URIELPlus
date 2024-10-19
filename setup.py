from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="urielplus",
    version="1.0",
    author="Mason Shipton",
    author_email="masonshipton25@gmail.com",
    description="URIEL+: Knowledge base for natural language processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/Masonshipton25/URIELPlus",
    project_urls={
        "Bug Tracker": "https://github.com/Masonshipton25/URIELPlus/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.26.1",
        "tensorflow>=2.11.1",
        "tensorflow-addons>=0.19",
        "scikit-learn>=1.5.1",
        "scipy>=1.14.1",
        "pandas>=2.2.2",
        "contexttimer>=0.3.3",
        "fancyimpute>=0.7.0",
        "joblib>=1.4.2",
        "setuptools>=68.2.0",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
