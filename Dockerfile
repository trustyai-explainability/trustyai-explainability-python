FROM python:3.9.6-slim-buster

# work-around on Debian JDK install bug
RUN mkdir -p /usr/share/man/man1
RUN apt-get update
RUN apt-get install -y cmake openjdk-11-jdk-headless maven build-essential wget

COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -r requirements-dev.txt

# Install the python-trustyai bindings
RUN python3 setup.py install

USER root

ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/$NB_USER

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid $NB_UID \
    $NB_USER

COPY . $HOME
RUN chown -R $NB_UID $HOME

USER $NB_USER

WORKDIR $HOME

RUN mvn org.apache.maven.plugins:maven-dependency-plugin:2.10:get \
    -DremoteRepositories=https://repository.sonatype.org/content/repositories/central  \
    -Dartifact=org.kie.kogito:explainability-core:1.8.0.Final \
    -Dmaven.repo.local=dep -q && \
    wget -O ./dep/org/kie/kogito/explainability-core/1.8.0.Final/explainability-core-1.8.0.Final-tests.jar \
    https://repo1.maven.org/maven2/org/kie/kogito/explainability-core/1.8.0.Final/explainability-core-1.8.0.Final-tests.jar

# Launch the notebook server
WORKDIR $HOME/examples
CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]