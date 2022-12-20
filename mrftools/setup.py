from setuptools import setup, find_packages

setup(
    name='mrftools',
    version='0.0.1',
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    url='http://learning.cs.vt.edu',
    license='MIT',
    author='Virginia Tech Machine Learning Laboratory',
    author_email='bhuang@vt.edu',
    description='Learning and inference with Markov random fields'
)
