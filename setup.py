from setuptools import find_packages,setup

from typing import List

hypen_e_dot="-e ."


def get_requirement(file_path :str)-> List[str]:
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)

    return requirements





setup(
    name="Delivey Time Prediction",
    version="v1.0.0",
    author="shahabas",
    author_email="mohdshahabasm@gmail.com",
    install_requires=get_requirement("requirements.txt"),
    packages=find_packages()
)

