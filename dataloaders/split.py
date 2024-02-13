from sklearn.model_selection import train_test_split


def data_split(df):
    # eng_train, eng_test, spn_train, spn_test = train_test_split(eng_clean, spn_clean, test_size=0.2, stratify=eng_length, random_state=12)
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['eng_length'], random_state=12)
    print(len(df_train), len(df_test))

    df_train = df_train.sort_values('eng_length').reset_index(drop=True)
    df_test = df_test.sort_values('spn_length').reset_index(drop=True)

    return df_train, df_test