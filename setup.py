import setuptools

with open("README.md", "r") as reamde:
    description = reamde.read()

setuptools.setup(
    name="thapl",
    version="0.0.1",
    author="Matan Peled",
    author_email="mip@cs.technion.ac.il",
    description="Thapl -- a THeAterical Programming Language",
    long_description=description,
    long_description_content_type="text/markdown",
    url="http://github.com/chaosite",
    packages=setuptools.find_packages(),
    setup_requires=['wheel'],
    install_requires=[
      'lark-parser',
      'numpy',
      'sympy',
      'recordclass',
      'numeral',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
