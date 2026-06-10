from setuptools import setup, find_packages

setup(
    name="flash-attn",
    version="2.6.3.post1",
    packages=find_packages(),
    description="Fake flash-attn package for T4 / CPU / compatibility",
    python_requires=">=3.8",
)
