import setuptools

# Reads the content of your README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ManniGrad",
    version="0.1.0",
    author="Vainavi Nair",
    # Your email address.
    author_email="vainavinair2009558@gmail.com",
    description="A tiny autograd engine and neural network library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The URL for your project's homepage.
    url="https://github.com/vainavinair/ManniGrad", 
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],
)
