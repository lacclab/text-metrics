from setuptools import setup
# TODO improve with https://python-poetry.org/
setup(
    name='eyeutils',
    version='0.7.1',
    packages=['eyeutils'],
    url='',
    license='',
    author='Omer Shubi',
    author_email='omer.shubi@gmail.com',
    description='Utils for Eye Tracking Measurements',
    include_package_data=True,
    package_data={'': ['data/*.tsv']},
    extras_require = {
            'lm_zoo': ['lm-zoo']
        },
    install_requires=
        'wordfreq==3.0.3'
)
