import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tsnmf", # Replace with your own username
    version="1.0",
    author="Victor Navarro & Eduardo Graells",
    author_email="victor.navarro@ug.uchile.cl",
    description="Implementation of Topic-Supervised Non-Negative Matrix Factorization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vokturz/tsnmf-sparse",
    packages=setuptools.find_packages(),
    license='BSD 3-clause "New" or "Revised License"',
    install_requires=['numpy','sklearn','scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords='tsnmf nmf sparse',
)