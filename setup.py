from setuptools import setup, find_packages

import ccsd


NAME = "ccsd"
VERSION = ccsd.__version__
DESCRIPTION = (
    "CCSD (Combinatorial Complex Score-based Diffusion) "
    "is a sophisticated score-based diffusion model "
    "designed to generate Combinatorial Complexes using "
    "Stochastic Differential Equations. This cutting-edge "
    "approach enables the generation of complex objects "
    "with higher-order structures and relations, thereby "
    "enhancing our ability to learn underlying "
    "distributions and produce more realistic objects."
)
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

LONG_DESCRIPTION = ""  # README + CHANGELOG
with open("README.md", "r+", encoding="UTF-8") as f:
    LONG_DESCRIPTION += f.read()
LONG_DESCRIPTION += "\n\n"
with open("CHANGELOG.rst", "r+") as f:
    LONG_DESCRIPTION += f.read()
URL = "https://ccsd.readthedocs.io/en/latest/"
DOWNLOAD_URL = "https://pypi.org/project/ccsd/"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/AdrienC21/CCSD/issues",
    "Documentation": URL,
    "Source Code": "https://github.com/AdrienC21/CCSD",
}
AUTHOR = "Adrien Carrel"
AUTHOR_EMAIL = "a.carrel@hotmail.fr"
LICENSE = "MIT"
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Unix",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
KEYWORDS = [
    "ccsd",
    "python",
    "diffusion-models",
    "combinatorial-complex",
    "score-based-generative-models",
    "score-based-generative-modeling" "topological-deep-learning",
    "higher-order-models",
    "tdl",
    "molecule",
    "molecule-generation",
    "machine-learning",
    "deep-learning",
    "graph-neural-networks",
    "graph",
    "topology",
    "diffusion",
    "artificial-intelligence",
    "topological-neural-networks",
    "topological-data-analysis",
]
PACKAGES = find_packages(include=["ccsd"])
PYTHON_REQUIRES = ">=3.10"
with open("requirements.txt", "r+") as f:
    INSTALL_REQUIRES = [x.replace("\n", "") for x in f.readlines()]
SETUP_REQUIRES = (["pytest-runner"],)
TESTS_REQUIRE = (["pytest==7.4.0"],)
TEST_SUITE = "tests"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=PACKAGES,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    test_suite=TEST_SUITE,
)
