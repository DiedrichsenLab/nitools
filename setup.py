from distutils.core import setup

with open('README.rst') as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name = 'neuroimagingtools',
    packages = ['nitools'],
    version = '1.1.3',
    license='MIT',
    description = 'Neuroimaging analysis tools',
    author = 'Jörn Diedrichsen',
    author_email = 'joern.diedrichsen@googlemail.com',
    url = 'https://github.com/DiedrichsenLab/nitools',
    download_url = 'https://github.com/DiedrichsenLab/nitools/archive/refs/tags/v1.1.3.tar.gz',
    long_description=LONG_DESCRIPTION,
    keywords = ['imaging analysis', 'nifti', 'gifti','cifti'],
    install_requires=[
        'bezier >= 2021.2.12',
        'trimesh >= 3.22.1',
        'numpy >= 1.20.1',
        'matplotlib >= 3.4.3',
        'nibabel >= 4.0.2',
        'pandas >= 1.3.2'],
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
    ],
    python_requires='>=3.6'
)