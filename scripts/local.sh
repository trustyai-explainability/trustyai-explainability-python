#!/bin/bash
# Copyright 2022 Red Hat, Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

EXP_CORE_DEST=$1

if [[ "$EXP_CORE_DEST" == "" ]]
then
  EXP_CORE_DEST="../trustyai-explainability"
else
  echo "Building trustyai-explainability from ${EXP_CORE_DEST}"
fi

echo "Copying JARs from ${EXP_CORE_DEST} into ${ROOT_DIR}/dep/org/trustyai/"
mvn install package -DskipTests -f "${EXP_CORE_DEST}"/pom.xml -Pshaded
mv "${EXP_CORE_DEST}"/explainability-arrow/target/explainability-arrow-*.jar "${ROOT_DIR}"/src/trustyai/dep/org/trustyai/


if [[ "$VIRTUAL_ENV" != "" ]]
then
  pip install "${ROOT_DIR}" --force
else
    echo "Not in a virtualenv. Installation not recommended."
    exit 1
fi
