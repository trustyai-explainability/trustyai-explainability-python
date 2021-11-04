#!/usr/bin/env sh

TRUSTY_VERSION="1.12.0.Final"

mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get \
    -DremoteRepositories=https://repository.sonatype.org/content/repositories/central  \
    -Dartifact=org.kie.kogito:explainability-core:$TRUSTY_VERSION \
    -Dmaven.repo.local=dep -q

# We also need the test JARs in order to get the test models
wget -O ./dep/org/kie/kogito/explainability-core/$TRUSTY_VERSION/explainability-core-$TRUSTY_VERSION-tests.jar \
      https://repo1.maven.org/maven2/org/kie/kogito/explainability-core/$TRUSTY_VERSION/explainability-core-$TRUSTY_VERSION-tests.jar