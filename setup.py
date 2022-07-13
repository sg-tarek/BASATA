from xml.etree.ElementInclude import include
from setuptools import setup, find_packages

def readme():
    with open("README.md", encoding="utf8") as f:
        README = f.read()
    return README

setup(
    name='basata',
    version='1.0.12',
    description='BASATA - Code less while gaining more',
    long_description=readme(),
    long_description_content_type="text/markdown",
    license='MIT',
    author="Tarek K. Ghanoum",
    author_email='ta.ghanoum@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    project_urls={
    'Documentation': 'https://sg-tarek.github.io/'
    },
    packages=find_packages(include=['basata']),
    url='https://github.com/sg-tarek/BASATA',
    keywords='ML AI Supervised TimeSeries Unsupservised NLP',
    install_requires=[
          'scikit-learn',
          'catboost',
          'xgboost',
          'lightgbm',
          'scikit-learn',
          'seaborn',
          'pandas',
          'matplotlib',
          'numpy',
          'tqdm',
          'lime',
          'sweetviz'
          'shap'
      ]
)