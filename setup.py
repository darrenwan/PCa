from setuptools import setup
from setuptools import find_packages

long_description = '''
CIE platform is designed to make clinical machine learning more convenient and normalizable. 
'''

setup(name='CIE',
      version='0.0.1',
      description='CIE platform',
      long_description=long_description,
      author='Peter Zhao',
      author_email='wenhuai.zhao@gmail.com',
      url='https://github.com/hitales/CIE',
      download_url='https://github.com/hitales/CIE/cie',
      license='Hitales',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'pyyaml',
                        'h5py'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'flaky',
                    'pytest-cov',
                    'pandas'],
      },
      classifiers=[
          'Development Status :: 1 - Production/Stable',
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3.6',
      ],
      packages=find_packages())
