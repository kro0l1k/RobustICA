from setuptools import setup, find_packages

def read_file(filename):
    with open(filename, encoding='utf-8') as f:
        return f.read().strip()

version = '1.0.0'
readme = read_file('README.md')

setup(
    name='RobustICA',
    version=version,
    author='Jan Zajac',
    author_email='janek.k.zajac@gmail.com',
    description='Robust Independent Component Analysis using Signature Coordinates',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    keywords='machine-learning, signature, time-series, unsupervised machine-learning, robust machine-learning',
    url='https://github.com/kro0l1k/RICA',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.26.4', 
        'scikit-learn>=1.3.2', 
        'signax==0.2.1', 
        'matplotlib>=3.8.0', 
        'pandas>=2.1.3',
        'jax>=0.4.25', 
        'jaxlib>=0.4.25', 
        'statsmodels'
    ],
    extras_require={
        'cuda': ['jax[cuda12_pip]>=0.4.25', 'jaxlib[cuda12_pip]>=0.4.25'],
        'metal': ['jax-metal>=0.0.6'],
        'cpu': ['jax>=0.4.25', 'jaxlib>=0.4.25'],
    },
    python_requires='>=3.9',
    classifiers=[
        'Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)