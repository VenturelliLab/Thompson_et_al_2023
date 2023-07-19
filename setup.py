from setuptools import find_packages, setup


setup(
    name='armored',
    packages=find_packages(include=['armored']),
    version='0.1.0',
    description='Automated Recommendation for Microbiome Optimization using Rational Experimental Design',
    author='Jaron Thompson',
    license='MIT',
    install_requires=['numpy==1.22.3',
                      'pandas==1.4.1',
                      'jax==0.3.4',
                      'jaxlib==0.3.2'],
)
