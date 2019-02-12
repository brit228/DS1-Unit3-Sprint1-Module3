import yaml
import docker_lambdata as ld
import pandas as pd
import sys

if __name__ == "__main__":
    if len (sys.argv) != 3:
        sys.exit(1)
    
    csv_file = sys.argv[1]
    yaml_file = sys.argv[2]

    df = pd.read_csv(csv_file)