echo off
copy %cd%\%1 %cd%\%3\%~n1%~x1
copy %cd%\%2 %cd%\%3\%~n2%~x2
docker run --rm -v %cd%\%3:/app/output -e ENV_CSV_FILE=%~n1%~x1 -e ENV_YAML_FILE=%~n2%~x2 lambdata_brit228:latest
