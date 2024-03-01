from setuptools import setup, find_packages

setup(
    name='binary-encoding',
    version='1.0.0',
    description='Train neural networks to observe emergence of binary encoding',
    author='Luigi Sbailò',
    author_email='luigi.sbailo@gmail.com',
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=[
        'torch>=2.2',
        'numpy>=1.26',
        'scikit-learn>=1.2',
        'scipy>=1.11',
    ],
)
