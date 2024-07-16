import os
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from awswrangler.redshift import read_sql_query
from bizdays import Calendar
from dotenv import load_dotenv
from redshift_connector import Connection
from plotnine import (
    ggplot,
    geom_bar,
    aes,
    geom_line,
    geom_histogram,
    facet_grid,
    theme,
    element_text,
    scale_x_datetime,
    geom_hline,
    annotate,
    scale_x_continuous,
    coord_flip,
    scale_y_continuous,
    scale_x_discrete,
)


def load_data_from_redshift() -> pl.DataFrame:
    """Load data from Redshift into a Pandas DataFrame. Assumes you have a .env file in root of the repository with the following variables:
    REDSHIFT_USER, REDSHIFT_PASSWORD, REDSHIFT_HOST, REDSHIFT_PORT, REDSHIFT_DATABASE"""
    load_dotenv()
    df = read_sql_query(
        sql="select * from dbt_rnd_oleks.demand",
        con=Connection(
            user=os.getenv("REDSHIFT_USER"),
            password=os.getenv("REDSHIFT_PASSWORD"),
            host=os.getenv("REDSHIFT_HOST"),
            port=os.getenv("REDSHIFT_PORT"),
            database=os.getenv("REDSHIFT_DATABASE"),
        ),
    )
    return pl.DataFrame(df)


def get_analysis_data() -> pl.DataFrame:
    file_path = "/home/rnd/projects/demand-data-exploration/data/analysis_data.parquet"

    if os.path.exists(file_path):
        print("Clean data found, loading...")
        df = pl.read_parquet(file_path)
    else:
        print("No clean data found, generating from raw data...")
        df = load_data_from_redshift()

        df.write_parquet(file_path)

    return df


def get_post_test_email_data() -> pl.DataFrame:
    file_path = (
        "/home/rnd/projects/demand-data-exploration/data/df_post_email_filter.parquet"
    )

    if os.path.exists(file_path):
        print("Data found, loading...")
        df = pl.read_parquet(file_path)
    else:
        raise (
            NotImplementedError,
            "No data found, please re-run the explore_test_emails.ipynb notebook",
        )

    return df


def get_post_outliers_data() -> pl.DataFrame:
    file_path = "/home/rnd/projects/demand-data-exploration/data/df_post_outlier_removal.parquet"

    if os.path.exists(file_path):
        print("Data found, loading...")
        df = pl.read_parquet(file_path)
    else:
        raise (
            NotImplementedError,
            "No data found, please re-run the explore_outliers.ipynb notebook",
        )

    return df


def drop_nas_in_cols(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    return transform_and_cmp_heights(
        df, df.with_columns(pl.col(cols).drop_nans()).drop_nulls(subset=cols)
    )


def filter_bad_orders(df: pl.DataFrame) -> pl.DataFrame:
    cnc_ext = [
        "step",
        "stp",
        "iges",
        "igs",
        "sldprt",
        "3dm",
        "sat",
        "x_t",
        "ipt",
    ]
    is_part_with_bad_extension = (pl.col("type") == "part") & (
        ~(pl.col("title").str.to_lowercase().str.split(".").list.get(-1).is_in(cnc_ext))
    )
    # 5% removed, rob still thinks this is a lot, revisit! stl can be on platform but cant be ordered
    # stl orders might be converted to step and then ordered which can be causing this to have too many filtered out orders
    # add .ipt to the list of good extensions

    is_row_with_multibody = pl.col("multibody")
    is_part_tiny = (pl.col("type") == "part") & (pl.col("volume") < 1)

    df = df.with_columns(
        order_has_bad_extension=is_part_with_bad_extension.any().over("order_uuid"),
        order_has_multibody=is_row_with_multibody.any().over("order_uuid"),
        order_has_tiny_part=is_part_tiny.any().over("order_uuid"),
    )

    return transform_and_cmp_heights(
        df,
        df.filter(
            ~pl.col("order_has_multibody"),
            ~pl.col("order_has_tiny_part"),
            ~pl.col("order_has_bad_extension"),
        ),
    )


def pick_one_revision(df: pl.DataFrame, debug: bool, level: List[str]) -> pl.DataFrame:
    df_level = transform_and_cmp_heights(
        df,
        df.with_columns(
            keep=(pl.col("quote_revision").is_first_distinct().over(level))
        ).filter(pl.col("keep")),
    )

    if debug:
        print(
            df_level.filter(
                pl.col("order_uuid") == "ca238a13-a822-451a-b3e6-d8461fdccd3d"
            )[
                [
                    "order_uuid",
                    "quote_uuid",
                    "line_item_uuid",
                    "quote_revision",
                    "quote_finalized_at",
                    "quote_status",
                    "quote_is_internal",
                ]
            ].sort(
                "quote_revision"
            )
        )

    df_level = df_level.with_columns(
        quote_status_priority=pl.col("quote_status").replace_strict(
            {
                "paid": 0,
                "refunded": 1,
                "payment": 2,
                "confirmation_in_progress": 3,
                "processing": 4,
            },
            default=999,  # confirmation_in_progress? confirm what this is
        ),
        # Quote gets locked by admin (internal) and then is shared with customer (external)
        # We want to prioritize external quotes
        quote_is_internal_priority=pl.col("quote_is_internal").replace_strict(
            {"false": 0, "true": 1}
        ),
    )

    if debug:
        print(
            df_level.sort(
                "order_uuid",
                "quote_status_priority",
                "quote_revision",
                descending=[False, False, True],
            ).filter(pl.col("order_uuid") == "ca238a13-a822-451a-b3e6-d8461fdccd3d")[
                [
                    "order_uuid",
                    "quote_uuid",
                    "line_item_uuid",
                    "quote_revision",
                    "quote_finalized_at",
                    "quote_status",
                    "quote_status_priority",
                    "quote_is_internal_priority",
                ]
            ]
        )

    # If order level, we want to keep only one quote from the picked revision
    df_picked_revision = transform_and_cmp_heights(
        df_level,
        df_level.sort(
            "order_uuid",
            "quote_status_priority",
            "quote_is_internal_priority",
            "quote_revision",
            descending=[False, False, False, True],
        ).unique("order_uuid", keep="first"),
    )

    # If quote level, we want to keep all line items from the picked revision
    if len(level) > 2:
        df_picked_revision = transform_and_cmp_heights(
            df_level,
            df_level.join(
                df_picked_revision,
                on=["order_uuid", "quote_uuid", "quote_revision"],
                how="inner",
            ),
        )

    if debug:
        print(
            df_picked_revision.filter(
                pl.col("order_uuid") == "ca238a13-a822-451a-b3e6-d8461fdccd3d"
            )[
                [
                    "order_uuid",
                    "quote_uuid",
                    "line_item_uuid",
                    "quote_revision",
                    "quote_finalized_at",
                    "quote_status",
                    "quote_status_priority",
                    "quote_is_internal_priority",
                ]
            ]
        )

    return df_picked_revision


def plot_bar(df: pl.DataFrame, x: str, y: str, is_grouped: bool = False) -> ggplot:
    return (
        ggplot(
            df if is_grouped else df.group_by(x).agg(pl.len()),
            aes(x, y, fill=x),
        )
        + geom_bar(stat="identity", show_legend=True)
        + theme(axis_text_x=element_text(angle=10))
    )


def transform_and_cmp_heights(
    df_old: pl.DataFrame, df_new: pl.DataFrame
) -> pl.DataFrame:
    print(
        f"Height comparison \nBefore: {df_old.height} \nAfter:  {df_new.height} \nDiff:  {df_new.height - df_old.height} ({(df_new.height - df_old.height) / df_old.height * 100:.2f}%)"
    )
    return df_new


def get_percentages_and_concat(
    dfs: Dict[pl.DataFrame, Tuple], count_col: str, group_col: str
) -> pl.DataFrame:
    df = (
        pl.concat(
            [
                dfs[x]
                .get_column(count_col)
                .value_counts()
                .with_columns(
                    (pl.col("count") / pl.col("count").sum()).alias(
                        f"percentage_{count_col}"
                    ),
                    group=x,
                )
                for x in dfs.keys()
            ]
        )
        .with_columns(pl.col("group").list.get(0).alias(group_col))
        .drop("group")
    )

    return df


def plot_percentage_bars(
    df: pl.DataFrame,
    x: str,
    y: str,
    facet_col: str,
    plot_extras: List[Any] = [],
) -> ggplot:
    return (
        ggplot(
            df,
            aes(x, y, fill=facet_col),
        )
        + geom_bar(stat="identity", show_legend=True)
        + facet_grid(f"{facet_col}~", space="free")
        + coord_flip()
        + plot_extras
    )


if __name__ == "__main__":
    df = get_analysis_data()
    print(df)
