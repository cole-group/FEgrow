from setuptools import setup, find_packages
from fegrow.version import __version__

setup(
    name='fegrow',
    version=__version__,
    description='FEgrow: generate congeneric ligands for FEP by growing a template molecule. ',
    long_description='See https://cole-group.github.io/FEgrow/',
    url='https://github.com/cole-group/FEgrow',
    author='Mateusz K. Bieniek, Ben Cree, Rachael Pirie, Josh Horton, Daniel Cole',
    author_email='bieniekmat@gmail.com',
    install_requires=['parmed', 'tqdm', 'typing-extensions'], #  FIXME: see env.yml for others
    license_files = ('LICENSE.txt',),
    packages=find_packages(),
    include_package_data=True,
)
