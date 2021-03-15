import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

setuptools.setup(
    name="piscola",
    version="0.1.0",
    author="Tomás Enrique Müller Bravo",
    author_email="t.e.muller-bravo@soton.ac.uk",
    license="MIT",
    description="Type Ia Supernova Light-curve fitting code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/temuller/piscola",
    packages=['piscola'],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    #package_data={'' : ["filters/*", "filters/*/*", "templates/*", "templates/*/*"]},
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    zip_safe=True,
)
