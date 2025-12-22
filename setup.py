from setuptools import find_packages,setup
from typing import List

hyphen_escape = '-e .'
def get_requirements(file_path: str) -> List[str]:
  '''This function will return the list of requirements'''

  requirements = []
  with open(file_path) as file:
    requirements = file.readlines()
    requirements = [req.replace("\n","") for req in requirements] 

    if hyphen_escape in requirements:
      requirements.remove(hyphen_escape)
  return requirements

setup ( name="ML Project",
      version='0.0.1',
      author='Kevin',
      author_email='kevin.jesu11@gmail.com',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')

)



     