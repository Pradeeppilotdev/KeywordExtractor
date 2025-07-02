from setuptools import setup, find_packages

setup(
    name="keyword_extraction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'spacy==3.5.3',
        'sentence-transformers==2.2.2',
        'keybert==0.7.0',
        'yake==0.4.8',
        'torch>=1.9.0',
        'transformers>=4.11.0'
    ],
    python_requires='>=3.7',
) 