# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

### Bug Fixes

- Fix typo in tutorial.rst (#193)



### Features

- Publish using PyPi Trusted Providers (#183)

## [0.3.0] - 2023-08-01

## [sprint-1.28] - 2023-05-22

## [0.2.12] - 2023-04-25

### Bug Fixes

- Fixed broken gridding after bokeh 3.0.x update (#153)

## [0.2.11] - 2023-04-24

### Bug Fixes

- Fixed null pointer error for Numpy inputs to Tyrus (#149)

## [0.2.10] - 2023-04-13

## [0.2.9] - 2023-02-23

### FAI-825

- Add feature and output name specification to models (#130)

### FAI-841

- Python benchmarks failing since namespace migration (#82)

### FAI-849

- Tyrus dashboard (#97)

### FAI-855

- Move arrow converters into exp-core (#101)

### FAI-863

- XAI-Bench (#109)
- XAIbench submodule tracking main branch, fork of xai-bench (#111)

### FAI-864

- Fix failing RtD build after to pip migration (#100)

### FAI-867

- Add Java 9+ Arrow compatibility flags (#103)

### FAI-879

- Sync version with exp-core (#107)

### FAI-880

- Move explainers into separate files (#110)

### FAI-881

- Upgrade Pylint version (#112)

### FAI-882

- Add kwargs to explainers (#113)

### FAI-884

- Add SHAP background generators to bindings (#117)

### FAI-886

- Unified input/output types, conversion functions, and docstrings (#116)

### FAI-887

- Generalize ExplanationResults (#115)

### FAI-889

- Allow non-string categorical feature domains (#118)

### FAI-892

- Tyrus TempDirectory (#121)

### FAI-893

- Make test plots non-blocking by default (#120)

### FAI-896

- Doc standardization and cleanup (#126)

### FAI-897

- Add bokeh dependency (#123)

### FAI-900

- Added feature domain argument to counterfactuals (#128)

### FAI-901

- Increased input/ouput conversion flexibility (#127)

### FAI-903

- Add group fairness Python bindings (#129)

### FAI-911

- Improve API to declare selections in Python fairness (#132)

### FAI-917

- Add error message capturing within Python models (#137)

### FAI-926

- Implement custom counterfactual goal criteria (#140)

### FAI-952

- Change dependencies to align with ODH workbench images (#143)

## [test-pipeline-1] - 2022-10-12

## [0.2.5] - 2022-09-14

### Bug Fixes

- Fixed linting, adding doc build reqs to requirements-dev

- Fixed line-too-long linting errors

- Fixed distutils import formatting

- Fixed conflicts with FAI-806, added build output

- Fixed incorrect LIME typing

- Fixed typo in Output field

- Fixed rtd.yaml python version inconsistency

- Fixed path in rtd install config

- Fixed custom.css path

- Fixed failing tests

- Fixed black-induced line-too-long

- Fixed failing shapresults.getSaliencies call

- Fixed linting errors re. import order

- Fixed failing initializer tests

- Fixed unrelated explainers change

- Fixed incorrect ShapResults.get_saliency output type

- Fixed broken favicon

- Fixed linting issue

- Fixed enumeration tuple issue

- Fixed failing SHAP tests, improved output name imputation in Model

- Fixed overzealous find and replace of trustyai

- Fixed overzealous find and replace of trustyai part 2

- Fixed broken output casting for 1d output arrays


### FAI-797b

- Include doc link in README

## [0.2.4] - 2022-06-16

## [0.2.3] - 2022-05-12

### Bug Fixes

- Fixed missing final newline


## [0.2.2] - 2022-04-25

## [0.2.1] - 2022-04-22

## [0.2.0] - 2022-04-22

### Bug Fixes

- Fixed linting issues

- Fixed linting issues re. the glob checker for wildcard paths

- Fixed line-too-long linting issue in model/__init__


### Refactor

- Refactored arrow inclusion; mvn pulls dependencies, then includes precompiled arrowconverters jar


## [0.1.1] - 2022-03-09

## [0.0.9] - 2022-02-17

## [0.0.8] - 2022-02-17

## [0.0.7] - 2022-02-17

## [0.0.6] - 2022-02-17

## [0.0.5] - 2022-01-22

## [0.0.4] - 2022-01-22

## [0.0.3] - 2021-11-04

## [0.0.2] - 2021-08-09

## [0.0.1] - 2021-07-21

