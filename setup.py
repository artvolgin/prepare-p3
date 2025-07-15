from setuptools import setup, find_packages

setup(
    name='prepare_phase_3',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'catboost==1.2',
        'joblib==1.2.0',
        'lightgbm==4.6.0',
        'optuna==3.6.1',
        'pandas==2.3.1',
        'scikit_learn==1.7.0',
        'scipy==1.16.0',
        'shap==0.46.0',
        'xgboost==3.0.2',
        'numpy==1.26.4',
        'statsmodels==0.14.1',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)