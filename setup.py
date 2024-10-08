from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os


class CustomInstallCommand(install):
    """Custom command to initialize submodules before installation."""

    def run(self):
        # Ensure we are in the root directory of the project
        project_root = os.path.abspath(os.path.dirname(__file__))
        # Run git submodule update --init --recursive
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "--recursive"], cwd=project_root
        )
        # Proceed with the standard installation
        install.run(self)


# TODO improve with https://python-poetry.org/
setup(
    name="text_metrics",
    version="1.1.2",
    packages=[
        "text_metrics",
        "text_metrics.surprisal_extractors",
        # pimentel_word_prob is a submodule of text_metrics
        "text_metrics.pimentel_word_prob",
        # "text_metrics.pimentel_word_prob.wordsprobability",
        # "text_metrics.pimentel_word_prob.wordsprobability.models",
        # "text_metrics.pimentel_word_prob.wordsprobability.utils",
    ],
    cmdclass={
        "install": CustomInstallCommand,
    },
    url="https://github.com/lacclab/text-metrics",
    license="",
    author="Omer Shubi",
    author_email="omer.shubi@gmail.com",
    description="Utils for Eye Tracking Measurements",
    include_package_data=True,
    package_data={"": ["data/*.tsv"]},
    extras_require={"lm_zoo": ["lm-zoo"]},
    install_requires=[
        "pandas>2.1.0",
        "transformers>=4.40.1",
        "wordfreq>=3.0.3",
        "numpy>=1.20.3",
        "torch",
        "spacy",
        "accelerate",
        "sentence-splitter",
    ],
)
