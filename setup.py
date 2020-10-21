from setuptools import setup, find_namespace_packages

setup(
    name='diskchef',
    version='v0.1',
    packages=find_namespace_packages(include=["diskchef*"], exclude=["diskchef.tests*"]),
    package_data={'diskchef.lamda': ['*.dat']},
    url='https://gitlab.com/SmirnGreg/diskchef',
    license='(c) authors',
    author='Grigorii V. Smirnov-Pinchukov',
    author_email='smirnov@mpia.de',
    description='Tool to retrieve chemical and physical parameters out of submm observations',
    install_requires=[
        'numpy >= 1.19.0',
        'matplotlib >= 3.3.0',
        'astropy >= 4.0',
        'scipy >= 1.5',
        'named_constants',
        'tqdm',
        'divan @ git+https://gitlab.com/SmirnGreg/divan.git',
        'PyYAML >= 5.0',
    ],
    python_requires=">=3.8",
)
