import polars

def update(df: polars.DataFrame, df_other: polars.DataFrame, join_columns: list[str]) -> polars.DataFrame:
    """
    Updates a polars DataFrame using another DataFrame based on a list of join columns. Updated rows are replaced with the new
    ones, rows that are absent from the original DataFrame are added. No row is removed from the original DataFrame since doing so
    would actually be simply replacing the old data with the new one...

    :param df: the original DataFrame to be updated
    :param df_other: the DataFrame with new data
    :param join_columns: the list of columns used to join the DataFrames
    :return: the updated DataFrame
    """
    # The columns that will be updated
    columns = [c for c in df_other.columns if c not in join_columns]
    updated_columns = (df.join(df_other, how='left', on=join_columns, suffix='_NEW')
                       .with_columns(**{c: polars.coalesce([polars.col(c + '_NEW'), polars.col(c)]) for c in columns})
                       .select(polars.all().exclude('^.*_NEW$')))  # <- this drops the temporary '*_NEW' columns
    new_columns = df_other.join(df, left_on='name', right_on='name', how='anti')
    return updated_columns.merge_sorted(new_columns, key='name')
