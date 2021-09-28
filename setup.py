from setuptools import setup

setup(name='ML',
      version='0.1',
      description='Machine learning lib',
      url='https://github.dev/simbarras/ML',
      author='Simon Barras',
      author_email='simon.barras02@gmail.com',
      license='MIT',
      packages=['ml'],
      install_requires=[
          'numpy',
          'matplotlib.pyplot',
          'sklearn.dataset',
      ],
      zip_safe=False)