rm src/trustyai/dep/org/trustyai/*
cd ../trustyai-explainability
mvn install package -DskipTests -f explainability-core/pom.xml -Pshaded
cd ../trustyai-explainability-python
mv ../trustyai-explainability/explainability-core/target/explainability-core-2.0.0-SNAPSHOT.jar src/trustyai/dep/org/trustyai/
mv ../trustyai-explainability/explainability-core/target/explainability-core-2.0.0-SNAPSHOT-tests.jar src/trustyai/dep/org/trustyai/
# mvn compile package -DskipTests -f java_sources/trusty-ai-arrow/pom.xml
# mv java_sources/trusty-ai-arrow/target/arrow-converters-0.0.1.jar src/trustyai/dep/org/trustyai/
# python3 -m pip install --upgrade pip
# pip3 install -r requirements-dev.txt
pip3 install . --force --no-binary :trustyai: