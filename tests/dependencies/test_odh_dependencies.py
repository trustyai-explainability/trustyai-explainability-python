import os
import urllib.request
from dparse import parse, filetypes

DEFAULT_PIPFILE = "https://raw.githubusercontent.com/opendatahub-io/notebooks/main/jupyter/datascience/ubi9-python-3.9/Pipfile"
INPUT_PIPFILE = os.getenv("INPUT_PIPFILE", DEFAULT_PIPFILE)


def test_odh_dependencies():
    '''Tests whether TrustyAI's dependencies are compatible with ODH workbench-image's https://github.com/opendatahub-io/notebooks/tree/main/jupyter/datascience/ubi9-python-3.9'''
    # download the pipfile
    urllib.request.urlretrieve(INPUT_PIPFILE, "/tmp/Pipfile")

    with open('./requirements.txt', 'r') as file:
        reqtxt = parse(file.read(), file_type=filetypes.requirements_txt)

    with open('./requirements-dev.txt', 'r') as file:
        reqdevtxt = parse(file.read(), file_type=filetypes.requirements_txt)

    with open('/tmp/Pipfile', 'r') as file:
        pipfile = parse(file.read(), file_type=filetypes.pipfile)

    reqtxt_names = {dependency.name: dependency for dependency in reqtxt.dependencies}
    reqdevtxt_names = {dependency.name: dependency for dependency in reqdevtxt.dependencies}

    mismatched_specs = []

    for dependency in pipfile.dependencies:
        if dependency.name in reqtxt_names.keys():
            print(f"{dependency} found")
            trusty_specs = reqtxt_names[dependency.name].specs
            if dependency.specs == trusty_specs:
                print(f"\tSpecs match ({dependency.specs})")
            else:
                print(
                    f"\tSpecs do not match ({dependency.specs} ODH vs. {reqtxt_names[dependency.name].specs} TrustyAI)")
                mismatched_specs.append(dependency)
        if dependency.name in reqdevtxt_names.keys():
            print(f"{dependency} found")
            trusty_specs = reqdevtxt_names[dependency.name].specs
            if dependency.specs == trusty_specs:
                print(f"\tSpecs match ({dependency.specs})")
            else:
                print(
                    f"\tSpecs do not match ({dependency.specs} ODH vs. {reqtxt_names[dependency.name].specs} TrustyAI)")
                mismatched_specs.append(dependency)

    assert len(mismatched_specs) == 0
