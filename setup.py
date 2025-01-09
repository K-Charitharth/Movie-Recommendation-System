from setuptools import find_packages, setup
from typing import List

# Navigates us to setup.py
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file=file_path) as file_obj:
        requirements = file_obj.readlines()
        # When we use Readlines we will get \n alongside. we use below line to avoid it
        requirements = [i.replace("\n", "") for i in requirements]

        # Not required for installation so we are ignoring it
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements


setup(

    name = 'Movie-Recommendation-System',
    version = '0.0.1',
    author = 'Kothakota Charitharth',
    packages = find_packages(),
    install_requires = get_requirements('./requirements.txt')
)