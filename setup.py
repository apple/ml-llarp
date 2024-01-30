import setuptools

setuptools.setup(
    name="llarp",
    packages=setuptools.find_packages(),
    version="0.1",
    install_requires=[
        "torch==2.0.1",
        "transformers==4.31.0",
        "einops==0.7.0",
        "gym==0.23.0",
        "wandb==0.13.1",
        "flamingo-pytorch==0.1.2",
        "peft==0.4.0",
        "sentencepiece",
    ],
)
