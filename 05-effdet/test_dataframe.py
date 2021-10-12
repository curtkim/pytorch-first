import pandas as pd


def test_isin():
    df = pd.DataFrame({'name': ['a', 'b', 'c']})
    cond = df['name'].isin(['c', 'a'])

    df_in = df.loc[cond]
    assert len(df_in) == 2
    df_in_not = df.loc[~cond]
    assert len(df_in_not) == 1
