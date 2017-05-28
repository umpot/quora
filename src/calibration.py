def calibrate_df(df, cols):
    train = 0.3691985666274004
    test=0.17426487864605453

    a = test / train
    b = (1 - test) / (1 - train)

    def calibrate(x):
        if x is None:
            return None
        return (a * x) / ((a * x) + (b * (1 - x)))

    for col in cols:
        df[col] = df[col].apply(calibrate)
