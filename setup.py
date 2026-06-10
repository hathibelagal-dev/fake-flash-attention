from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fake-flash-attention",
    version="2.6.3.post1",
    packages=find_packages(),
    description="Fake flash-attn package for T4 / CPU / compatibility",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
