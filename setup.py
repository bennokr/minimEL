from setuptools import setup

setup(
    name="minimel",
    version="0.1",
    description="Minimal Entity Linking",
    url="http://github.com/bennokr/minimel",
    author="Benno Kruit",
    author_email="b.b.kruit@vu.nl",
    license="MIT",
    packages=["minimel"],
    zip_safe=False,
    entry_points={"console_scripts": ["minimel = minimel.__main__:main"]},
    install_requires=["vowpalwabbit", "DAWG-Python", "tqdm", "defopt", "pandas"],
    extras_require={
        "train": ["wikimapper", "mwparserfromhell", "dask[distributed]", "dawg2", "seaborn", "dask_jobqueue", "dask[dataframe]", "scikit-learn"],
        "mentions": ["ahocorasick-rs"],
        "stem": ["icu_tokenizer"],
        "ja": ["mecab-python3", "unidic-lite"],
        "fa": ["PersianStemmer"],
        "demo": ["Flask", "gitpython", "ahocorasick-rs"],
        "docs": [
            "sphinxcontrib-apidoc",
            "sphinxcontrib-ansi",
            "sphinx-mdinclude",
            "sphinx-autodoc-typehints",
            "nbsphinx",
            "pandoc",
        ],
    },
)
