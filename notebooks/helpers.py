import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy(deep=True)

    # Remove the leading `fraud_` from the merchant name.
    merchant_name_pattern = re.compile("^fraud_")
    clean["merchant"] = clean["merchant"].replace(merchant_name_pattern, "")

    # Convert the `trans_date_trans_time` to a Datetime. Get the year, month, day, hour, min.
    clean["timestamp"] = pd.to_datetime(clean["trans_date_trans_time"])
    clean["year"] = clean["timestamp"].dt.year
    clean["month"] = clean["timestamp"].dt.month
    clean["day"] = clean["timestamp"].dt.day
    clean["day_name"] = clean["timestamp"].dt.day_name()

    # Get the date of birth year.
    clean["dob"] = pd.to_datetime(clean["dob"])
    clean["dob_year"] = clean["dob"].dt.year

    # Convert city and merchant lat/long into distances.
    clean["lat_delta"] = abs(clean["lat"] - clean["merch_lat"])
    clean["long_delta"] = abs(clean["long"] - clean["merch_long"])

    # Drop unwanted columns.
    clean = clean.drop(columns=[
        "cc_num",
        "city",
        "dob",
        "first",
        "last",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "state",
        "street",
        "timestamp",
        "trans_date_trans_time",
        "trans_num",
        "unix_time",
        "zip",
    ])

    # Return the clean data set.
    return clean


def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy(deep=True)

    label_encoder = LabelEncoder()

    columns_to_encode = [
        "category",
        "day_name",
        "gender",
        "job",
        "merchant"
    ]

    for col in columns_to_encode:
        encoded[col] = label_encoder.fit_transform(encoded[col])

    return encoded


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = read_source_csv("../data/fraudTrain.csv")
    test = read_source_csv("../data/fraudTest.csv")
    return (train, test)


def read_source_csv(path_to_source: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_source, index_col=0)
    return df


def xy_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return (x, y)
