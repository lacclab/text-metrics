from setuptools import setup
# TODO improve with https://python-poetry.org/
setup(
    name='text_metrics',
    version='0.8.0',
    packages=['text_metrics'],
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
