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
    #packages=setuptools.find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    #dependency_links=['https://github.com/kbarbary/extinction/tarball/master#egg=extinction'],
    package_data={'' : ["filters/*", "filters/*/*", "templates/*", "templates/*/*"]},
    #package_data={'piscola' : ["templates/*.dat"]},
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    zip_safe=True,
)
