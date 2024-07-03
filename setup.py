from setuptools import setup
# TODO improve with https://python-poetry.org/
setup(
    name='text_metrics',
    version='1.1.1',
    packages=['text_metrics'],
    url='https://github.com/lacclab/text-metrics',
    license='',
    author='Omer Shubi',
    author_email='omer.shubi@gmail.com',
    description='Utils for Eye Tracking Measurements',
    include_package_data=True,
    package_data={'': ['data/*.tsv']},
    extras_require = {
            'lm_zoo': ['lm-zoo']
        },
    install_requires=[
        'pandas>2.1.0',
        'transformers>=4.40.1',
        'wordfreq>=3.0.3',
        'numpy>=1.20.3',
        'torch',
        'spacy'
    ]
)
