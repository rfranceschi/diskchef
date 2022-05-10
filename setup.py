from setuptools import setup, find_namespace_packages

setup(
    name='diskchef',
    packages=find_namespace_packages(include=["diskchef*"], exclude=["diskchef.tests*"]),
    package_data={
        'diskchef.lamda.files': ['*.dat'],
        'diskchef.chemistry.scikit_estimators': ['*.pkl'],
        'diskchef.dust_opacity.files': ['*.inp'],
        'diskchef.physics.data': ['*.dat'],
    },
    url='https://gitlab.com/SmirnGreg/diskchef',
    license='(c) authors',
    author='Grigorii V. Smirnov-Pinchukov',
    author_email='smirnov@mpia.de',
    description='Tool to retrieve chemical and physical parameters out of submm observations',
    install_requires=[
        'numpy >= 1.18.0',
        'matplotlib >= 3.3.0',
        'astropy >= 4.1.1',
        'scipy >= 1.5',
        'tqdm',
        'chemical_names @ git+https://gitlab.com/SmirnGreg/chemical_names.git',
        'uvplot',
        'spectral_cube >= 0.6.0',
        'radmc3dPy @ git+https://github.com/dullemond/radmc3d-2.0.git#subdirectory=python/radmc3dPy',
        'scikit-learn == 1.0.1',
        'emcee',
        'corner',
        'ultranest == 3.3.3',
        'h5py',
        'mpi4py',
        'regions'
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning', 'cython'],
    python_requires=">=3.8",
)

try:
    import galario
except ImportError as err1:
    print(err1)
    print("Install galario:")
    print("$ conda install -c conda-forge galario")
