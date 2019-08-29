import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="piscola",
    version="0.0.4",
    author="Tomás Enrique Müller Bravo",
    author_email="t.e.muller-bravo@soton.ac.uk",
    description="A Type Ia Supernova Light-curve Fitter",
    long_description="Too lazy right now...",
    long_description_content_type="text/markdown",
    url="https://github.com/temuller/piscola",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
