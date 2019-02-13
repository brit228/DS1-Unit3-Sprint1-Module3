import yaml
import docker_lambdata as ld
import pandas as pd
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    csv_file = "/app/output/" + sys.argv[1]
    yaml_file = yaml.load(open("/app/output/" + sys.argv[2], "r"))
    output = open("/app/output/log.txt", "w")
    output.write(str(yaml_file))
    df_helper = ld.DataFrameHelper(pd.read_csv(csv_file),
                                   "/app/output/output.txt")
    if "invalid_dict" in yaml_file:
        output.write("invalid_dict\n")
        df_helper.invalid_data_values(yaml_file["invalid_dict"])
    df_helper.check_data()
    if "extend_data" in yaml_file:
        if "nrows" in yaml_file["extend_data"] and\
             "nsamples" in yaml_file["extend_data"]:
            df_helper.generate_data(yaml_file["extend_data"]["nrows"],
                                    yaml_file["extend_data"]["nsamples"])
    if "chi_cols" in yaml_file:
        cols = [c for c in list(df_helper.data_frame)
                if c in yaml_file["chi_cols"]]
        if len(cols) > 1:
            for i, c in enumerate(cols[:-1]):
                for j, d in enumerate(cols[i+1:]):
                    df_helper.chi_viz(c, d)
    if "split_data" in yaml_file:
        if "ycols" in yaml_file["split_data"] and\
            "train_ratio" in yaml_file["split_data"] and\
                "val_ratio" in yaml_file["split_data"]:
            ycols = yaml_file["split_data"]["ycols"]
            tr_ratio = yaml_file["split_data"]["train_ratio"]
            te_ratio = yaml_file["split_data"]["val_ratio"]
            out = df_helper.data_split(ycols, tr_ratio, te_ratio)
            (X1, X2, X3, y1, y2, y3) = out
            if X1.count()[0] > 0:
                X1.to_csv("/app/output/ind_train.csv")
                y1.to_csv("/app/output/dep_train.csv")
            if X2.count()[0] > 0:
                X2.to_csv("/app/output/ind_val.csv")
                y2.to_csv("/app/output/dep_val.csv")
            if X3.count()[0] > 0:
                X3.to_csv("/app/output/ind_test.csv")
                y3.to_csv("/app/output/dep_test.csv")
    df_helper.data_frame.to_csv("/app/output/formatted_data.csv")
