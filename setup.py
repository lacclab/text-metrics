from setuptools import setup

setup(
    name='eyeutils',
    version='0.4.3',
    packages=['eyeutils'],
    url='',
    license='',
    author='Omer Shubi',
    author_email='omer.shubi@gmail.com',
    description='Utils for Eye Tracking Measurements',
    include_package_data=True,
    package_data={'': ['data/*.tsv']},
)
