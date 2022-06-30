from distutils.core import setup
setup(
  name = 'neuroimagingtools',
  packages = ['nitools'],
  version = '0.5.0',
  license='MIT',
  description = 'Neuroimaging analysis tools',
  author = 'Jörn Diedrichsen',
  author_email = 'joern.diedrichsen@googlemail.com',
  url = 'https://github.com/DiedrichsenLab/nitools',
  download_url = 'https://github.com/DiedrichsenLab/nitools/archive/refs/tags/v0.5.0.tar.gz',
  keywords = ['imaging analysis', 'nifti', 'gifti','cifti'],
  install_requires=[
          'numpy',
          'matplotlib',
          'nibabel'],
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