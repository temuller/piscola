import setuptools
import piscola

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="piscola",
    version=piscola.__version__,
    author="Tomás Enrique Müller Bravo",
    author_email="t.e.muller-bravo@soton.ac.uk",
    license="MIT",
    description="Type Ia Supernova Light-curve fitting code ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/temuller/piscola",
    packages=['piscola'],
    #package_dir={'': 'piscola'},
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    #dependency_links=['https://github.com/kbarbary/extinction/tarball/master#egg=extinction'],
    #package_data={'piscola' : ["filters/*", "filters/*/*", "templates/*", "templates/*/*", "sfddata-master/*", "README.md", "LICENSE"]},
    package_data={'' : ["README.md", "LICENSE"]},
    include_package_data=True,
    zip_safe=True
)
