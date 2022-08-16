import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

with open("src/piscola/_version.py") as version_file:
    for line in version_file:
        if "__version__" in line:
            __version__ = line.split()[-1].replace('"', "")

setuptools.setup(
    name="piscola",
    version=__version__,
    author="Tomás Enrique Müller Bravo",
    author_email="t.e.muller-bravo@soton.ac.uk",
    license="MIT",
    description="Type Ia Supernova Light-curve fitting code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/temuller/piscola",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    scripts=["bin/piscola"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    package_data={
        "piscola": [
            "filters/*",
            "filters/*/*",
            "templates/*",
            "templates/*/*",
            "standards/*",
        ]
    },
    include_package_data=True,
)
