# trusty-ai-arrow

### Build
```
cd trusty-ai-arrow
mvn clean install dependency:copy-dependencies -DincludeScope=runtime -DoutputDirectory=target/lib -DskipTests=true
```

### Add to Python bindings
In `trusty-ai-python-module/trustyai/__init__.py`, add the following to `CORE_DEPS`:
```
    "[path]/trusty-ai-arrow/trusty-ai-arrow/target/arrow-converters*"
    "[path]/trusty-ai-arrow/trusty-ai-arrow/target/lib/arrow*"
    "[path]/trusty-ai-arrow/trusty-ai-arrow/target/lib/flatbuffers*"
    "[path]/trusty-ai-arrow/trusty-ai-arrow/target/lib/netty*
```