import pandas as pd
import os


def read_and_split_data():
    df = pd.read_csv("household_power_consumption.txt", sep=";", header=None)
    chunks = 5
    step = len(df) // chunks
    df_i = df.iloc[:step, :]
    df_i.to_csv("data/household_power_consumption_0.csv", index=False)
    for i in range(0, len(df), step):
        df_i = df.iloc[i : i + step, :]
        df_i.to_csv(f"data/household_power_consumption_{i}.csv", index=False)


def read_household_power_consumption():
    flist = []
    rootdir = os.path.join(os.getcwd(), "data")

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            df = pd.read_csv(os.path.join(subdir, file), low_memory=False)
            flist.append(df)

    df_out = pd.concat(flist, axis=0, ignore_index=False)
    return df_out


if __name__ == "__main__":
    # read_and_split_data()
    df = read_household_power_consumption()
    print(df.shape)
    print(df.head())
