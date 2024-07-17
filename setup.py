from setuptools import setup, find_packages

setup(
    name='your_package_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'matplotlib',
        'numpy',
        'pulp',
        'scipy',
        'seaborn',
    ],
    entry_points={
        'console_scripts': [
            'your_command=your_module:main_function',
        ],
    },
)
