from distutils.core import setup

with open('README.md') as fp:
    LONG_DESCRIPTION = fp.read()

setup(
    name = 'neuroimagingtools',
    packages = ['nitools'],
    version = '1.1.2',
    license='MIT',
    description = 'Neuroimaging analysis tools',
    author = 'Jörn Diedrichsen',
    author_email = 'joern.diedrichsen@googlemail.com',
    url = 'https://github.com/DiedrichsenLab/nitools',
    download_url = 'https://github.com/DiedrichsenLab/nitools/archive/refs/tags/v1.1.2.tar.gz',
    long_description=LONG_DESCRIPTION,
    keywords = ['imaging analysis', 'nifti', 'gifti','cifti'],
    install_requires=[
          'bezier',
          'trimesh',
          'numpy',
          'matplotlib',
          'nibabel',
          'pandas'],
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