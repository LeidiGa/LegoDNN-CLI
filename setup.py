import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="legodnn",
    version="0.1.0",
    author="BITLinc",
    author_email="1730395941@qq.com",
    description="A block extration tool",
    long_description=long_description,
    py_modules=['legodnn'],
    install_requires=[
        'click',
        'torch',
        'numpy',
        'nni', 
        'tensorboard',
        'matplotlib',
        'networkx',
        'tqdm',
        'thop',
        'onnx',
        'colorama',
    ],
    long_description_content_type="text/markdown",
    url="https://github.com/LeidiGa/LegoDNN-CLI.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts':[
            'legodnn=cli:legodnn'
        ]
    }
)
