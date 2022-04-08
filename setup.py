from setuptools import setup, find_packages

setup(
    name='fegrow',
    version='1.0.0',
    description='RGroup: generate congeneric ligands for FEP by growing a template molecule. ',
    long_description='Copy from README file',
    url='https://blogs.ncl.ac.uk/danielcole/',
    author='Mateusz K. Bieniek, Ben Cree, Rachael Pirie, Josh Horton, Daniel Cole',
    author_email='bieniekmat@gmail.com',
    install_requires=['parmed', 'tqdm', 'mdanalysis', 'typing-extensions'], #  FIXME: see env.yml for others
    license_files = ('LICENSE.txt',),
    packages=find_packages(),
    include_package_data=True,
)
