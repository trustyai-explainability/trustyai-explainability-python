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
TMP_DIR=$(mktemp -d)

EXP_CORE="trustyai-explainability"

EXP_CORE_DEST="${TMP_DIR}/${EXP_CORE}"
if [ ! -d "${EXP_CORE_DEST}" ]
then
  echo "Cloning trustyai-explainability into ${EXP_CORE_DEST}"
  git clone --branch main https://github.com/${EXP_CORE}/${EXP_CORE}.git "${EXP_CORE_DEST}"
  echo "Copying JARs from ${EXP_CORE_DEST} into ${ROOT_DIR}/dep/org/trustyai/"
  mvn install package -DskipTests -f "${EXP_CORE_DEST}"/explainability-core/pom.xml -Pshaded
  mv "${EXP_CORE_DEST}"/explainability-core/target/explainability-core-*-SNAPSHOT.jar ../src/trustyai/dep/org/trustyai/
else
    echo "Directory ${EXP_CORE_DEST} already exists. Please delete it or move it."
    exit 1
fi

ARROW="trusty-ai-arrow"
ARROW_DEST="${TMP_DIR}/${ARROW}"
if [ ! -d "${ARROW_DEST}" ]
then
  echo "Cloning trusty-ai-arrow into ${ARROW_DEST}"
  mvn compile package -DskipTests -f "${ROOT_DIR}"/java_sources/trusty-ai-arrow/pom.xml
  echo "Copying JARs from ${ARROW_DEST} into ${ROOT_DIR}/dep/org/trustyai/"
  mv "${ROOT_DIR}"/java_sources/trusty-ai-arrow/target/arrow-converters-*.jar "${ROOT_DIR}"/src/trustyai/dep/org/trustyai/
else
    echo "Directory ${ARROW_DEST} already exists. Please delete it or move it."
    exit 1
fi

if [[ "$VIRTUAL_ENV" != "" ]]
then
  pip install "${ROOT_DIR}" --force
else
    echo "Not in a virtualenv. Installation not recommended."
    exit 1
fi
