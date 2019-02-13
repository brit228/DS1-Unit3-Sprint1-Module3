#!/bin/bash
echo cp $PWD/$1 $PWD/$3/${1%.*}.${1##*.}
echo cp $PWD/$2 $PWD/$3/${2%.*}.${2##*.}
echo docker run --rm -v $PWD/$3:/app/output -e ENV_CSV_FILE=${1%.*}.${1##*.} -e ENV_YAML_FILE=${2%.*}.${2##*.} lambdata_brit228:latest
