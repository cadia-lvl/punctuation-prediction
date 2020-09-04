import setuptools

with open("README.md", "r") as readme:
    long_description = readme.read()

setuptools.setup(
    name="punctuator-ice",
    version="1.1.2",
    author="Helga Svala Sigurðardóttir",
    author_email="helgas@ru.is",
    url="https://github.com/cadia-lvl/punctuation-prediction/tree/master/punctuation_package",
    description="A frontend to punctuation prediction for Icelandic text",
    license="MIT",
    python_requires=">=3.7.*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA :: 10.2",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Utilities",
        "Topic :: Text Processing :: Linguistic",
    ],
    install_requires=[
        "tensorflow >= 2.1.0",
        "torch >= 1.4.0",
        "transformers >= 2.11.0",
    ],
    entry_points={"console_scripts": ["punctuate=punctuator.main:main",],},
)

