from setuptools import setup

setup(
    name='diskchef',
    version='v0.1',
    packages=['diskchef', 'diskchef.engine', 'diskchef.physics', 'diskchef.chemistry'],
    url='https://gitlab.com/SmirnGreg/diskchef',
    license='(c) authors',
    author='Grigorii V. Smirnov-Pinchukov',
    author_email='smirnov@mpia.de',
    description='Tool to retrieve chemical and physical parameters out of submm observations'
)
