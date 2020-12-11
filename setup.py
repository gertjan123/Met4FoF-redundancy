from os import path

from setuptools import find_packages, setup


def get_readme():
    """Get README.md's content"""
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name="Met4FoF-redundancy",
    version="0.0.1",
    description="Met4FoF-redundancy is a software package containing software tools "
    "that can be used to analyze measurement data which contain redundancy",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/gertjan123/Met4FoF-redundancy",
    author="Gertjan (G.J.P.) Kok",
    author_email="GKok@vsl.nl",
    packages=find_packages(exclude=["test_redundancy"]),
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "agentMET4FOF",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or "
        "later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
