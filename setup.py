import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = [
    "torch==1.9.*", "pytorch-lightning==1.5.*", "shapely==1.8.*", "omegaconf==2.1.*", "matplotlib==3.2.*",
    "wandb==0.12.*", "opencv-python==4.4.*"
]

extras_require = {
    "docs": ["mkdocs-material==7.3.*", "mkdocstrings==0.16.*"],
    "tests": ["pytest==6.2.*"],
    "lint": ["isort==5.10.*", "yapf==0.31.*", "pylint==2.11.*"],
    "graph": ["torch-geometric==2.0.*"],
    "se3": ["se3-transformer-pytorch==0.8.*"],
}
extras_require["all"] = sum(extras_require.values(), [])
extras_require["dev"] = extras_require["docs"] + extras_require["tests"] + extras_require["lint"]

setuptools.setup(name="cartesius",
                 version="0.1.dev0",
                 author="Nicolas REMOND",
                 author_email="remondn@spacewalk.tech",
                 description="Benchmark & Pretraining for Cartesian coordinates feature extraction",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/TeamSPWK/cartesius",
                 packages=setuptools.find_packages(),
                 classifiers=[
                     "Programming Language :: Python :: 3.6",
                     "Operating System :: OS Independent",
                 ],
                 python_requires=">=3.6",
                 install_requires=reqs,
                 extras_require=extras_require,
                 entry_points={"console_scripts": ["cartesius=cartesius.train:main"]},
                 package_data={"cartesius": ["config/*.yaml", "data/*.json"]})
