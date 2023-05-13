from setuptools import setup

setup(
    name='minimel',
    version='0.1',
    description='Minimal Entity Linking',
    url='http://github.com/bennokr/minimel',
    author='Benno Kruit',
    author_email='b.b.kruit@vu.nl',
    license='MIT',
    packages=['minimel'],
    zip_safe=False,
    entry_points={ 'console_scripts': ['minimel = minimel.__main__:main'] },
    install_requires = [
        'vowpalwabbit',
        'tqdm',
        'defopt',
        'DAWG-Python',
        'pandas',
    ],
    extras_require = {
        'train': ['mwparserfromhell','dask[distributed]','DAWG', 'seaborn'],
        'stem': ['icu_tokenizer'],
        'ja': ['mecab-python3','unidic-lite'],
        'fa': ['PersianStemmer'],
        'demo': ['Flask'],
        'docs': ['sphinxcontrib-apidoc','sphinxcontrib-ansi', 'myst-parser'],
    },
)