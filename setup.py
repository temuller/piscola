import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="piscola",
    version="1.0",
    author="Tomás Enrique Müller Bravo",
    author_email="t.e.muller-bravo@soton.ac.uk",
    license="MIT",
    description="Type Ia Supernova Light-curve fitting code ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/temuller/piscola",
    #packages=setuptools.find_packages(),
    packages=['piscola'],
    package_dir={'': 'src'},
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['matplotlib', 'pandas', 'lmfit', 'peakutils', 'george', 'emcee', 'extinction', 'sfdmap', 'astropy', 'numpy'],
    package_data={'piscola' : ["filters/*", "filters/*/*", "templates/*", "templates/*/*", "sfddata-master/*", "README.md", "LICENSE"]},
    include_package_data=True,
)
