#!/usr/bin/env sh

mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get \
    -DremoteRepositories=https://repository.sonatype.org/content/repositories/central  \
    -Dartifact=org.kie.kogito:explainability-core:1.8.0.Final \
    -Dmaven.repo.local=dep -q

# We also need the test JARs in order to get the test models
wget -O ./dep/org/kie/kogito/explainability-core/1.8.0.Final/explainability-core-1.8.0.Final-tests.jar https://repo1.maven.org/maven2/org/kie/kogito/explainability-core/1.8.0.Final/explainability-core-1.8.0.Final-tests.jar