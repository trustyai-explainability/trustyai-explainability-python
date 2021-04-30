#!/usr/bin/env sh

mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get \
     -DremoteRepositories=https://repository.sonatype.org/content/repositories/central  \
     -Dartifact=org.kie.kogito:explainability-core:1.5.0.Final \
     -Dmaven.repo.local=dep -q