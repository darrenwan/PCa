import os
from setuptools import setup, find_packages


long_description = '''
CIE platform is designed to make clinical machine learning more convenient and normalizable. 
'''

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]
setup(name='CIE',
      version='0.0.1',
      description='CIE platform',
      long_description=long_description,
      author='Peter Zhao',
      author_email='wenhuai.zhao@gmail.com',
      url='https://github.com/hitales/CIE',
      download_url='https://github.com/hitales/CIE/cie',
      license='Hitales',
      install_requires=install_reqs,
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
