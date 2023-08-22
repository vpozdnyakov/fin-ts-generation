from setuptools import find_packages, setup

setup(
    name='fints_generation',
    packages=find_packages(),
    version='0.0.1',
    description='Generation of financial time series',
    url='https://github.com/vpozdnyakov/fin-ts-generation',
    author='Vitaliy Pozdnyakov, Ramil Chermanteev',
    install_requires=[
        'pandas',
        'numpy',
        'torch',
        'statsmodels',
        'scipy',
        'lightning',
    ],
)
