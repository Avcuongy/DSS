import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
)


def preprocessing_numerical(
    df: pd.DataFrame,
    numerical_cols: list[str] = [],
    numerical_method: str = "StandardScaler",
    is_return_scaler: bool = False,
):
    if not numerical_cols:
        raise ValueError("numerical_cols is empty")

    if numerical_method == "StandardScaler":
        scaler = StandardScaler()
    elif numerical_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler: {numerical_method}")

    scaler.fit(df[numerical_cols])

    df_scaled = pd.DataFrame(
        scaler.transform(df[numerical_cols]),
        columns=numerical_cols,
        index=df.index,
    )

    return (scaler, df_scaled) if is_return_scaler else df_scaled


def preprocessing_categorical(
    df: pd.DataFrame, nominal_cols: list[str] = [], ordinal_cols: list[str] = []
):
    nominal_cols = nominal_cols or []
    ordinal_cols = ordinal_cols or []

    if not nominal_cols and not ordinal_cols:
        return df.copy()

    df = df.copy()

    non_cat_cols = [col for col in df.columns if col not in nominal_cols + ordinal_cols]
    df_result = df[non_cat_cols].copy()

    if nominal_cols:
        onehot_encoder = OneHotEncoder(
            handle_unknown="ignore", drop="first", sparse_output=False
        )
        encoded_array = onehot_encoder.fit_transform(df[nominal_cols])
        encoded_cols = onehot_encoder.get_feature_names_out(nominal_cols)
        df_nominal = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        df_result = pd.concat([df_result, df_nominal], axis=1)

    if ordinal_cols:
        df_ordinal = pd.DataFrame(index=df.index)
        for col in ordinal_cols:
            le = LabelEncoder()
            df_ordinal[col] = le.fit_transform(df[col].astype(str))
        df_result = pd.concat([df_result, df_ordinal], axis=1)

    return df_result
