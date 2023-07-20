from setuptools import find_packages, setup


setup(
    name='armored',
    packages=find_packages(include=['armored']),
    version='0.1.0',
    description='Automated Recommendation for Microbiome Optimization using Rational Experimental Design',
    author='Jaron Thompson',
    license='MIT',
    install_requires=['numpy',
                      'pandas',
                      'jax',
                      'jaxlib'],
)
