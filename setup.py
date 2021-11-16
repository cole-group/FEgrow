from setuptools import setup, find_packages

setup(
    name='rgroup',
    version='0.0.1.dev1',
    description='RGroup: generate congeneric ligands by modifying groups. ',
    long_description='Copy from README file',
    url='https://blogs.ncl.ac.uk/danielcole/',
    author='Mateusz K. Bieniek, Ben Cree, Rachael Pirie, Josh Horton, Daniel Cole',
    author_email='bieniekmat@gmail.com',
    install_requires=['parmed', 'tqdm', 'mdanalysis', 'typing-extensions'], #  FIXME: see env.yml for others
    packages=find_packages(),
    include_package_data=True,
)
