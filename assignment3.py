import numpy as np
import pandas as pd

from helper import read_dataset, create_plot
from sim_parameters import TRASITION_PROBS, HOLDING_TIMES


def create_individual_simulation(date_range, age_group):
    individual_time_series = pd.Series(dtype=str, index=date_range)
    i = 0
    follower_state = "H"
    while i < len(date_range):
        follower_i = i
        i += max(HOLDING_TIMES[age_group][follower_state], 1)
        individual_time_series.iloc[follower_i:i] = follower_state
        follower_state = np.random.choice(list(TRASITION_PROBS[age_group][follower_state].keys()),
                                          p=list(TRASITION_PROBS[age_group][follower_state].values()))
    return individual_time_series


def run(countries_csv_name: str = "a3-countries.csv",
        countries: list = [],
        sample_ratio: int = 1e6,
        start_date: str = "2021-04-01",
        end_date: str = "2022-04-30"):
    df = read_dataset(countries_csv_name)

    fixed_date_range = pd.date_range(start_date, end_date)
    gather_country_dfs = []
    for country in countries:
        n = int(df.query(f"country == '{country}'").squeeze()["population"] / sample_ratio)
        country_distribution = (df.query(f"country == '{country}'").loc[:,
                                ['less_5', '5_to_14', '15_to_24', '25_to_64', 'over_65']]
                                .div(100).squeeze().to_dict()
                                )
        country_sample = np.random.choice(list(country_distribution.keys()), n, list(country_distribution.values()))
        gather_dfs = []
        for i in range(len(country_sample)):
            gather_dfs.append(create_individual_simulation(fixed_date_range, country_sample[i]))
        time_df = pd.concat(gather_dfs, axis=1)

        time_df.index.name = "date"
        time_df.columns = pd.MultiIndex.from_tuples(
            [(country, i, country_sample[i]) for i in range(len(time_df.columns))],
            names=["country", "person_id", "age_group_name"]
        )
        time_df = (
            pd.concat([
                time_df.stack(level=[0, 1, 2]).to_frame("state"),
                time_df.shift().stack(level=[0, 1, 2]).to_frame("prev_state"),
                time_df.ne(time_df.shift()).cumsum().stack(level=[0, 1, 2]).to_frame("streak")
            ], axis=1)
        )
        time_df.reset_index(inplace=True)
        time_df = (time_df
                   .assign(staying_days=time_df.groupby(["person_id", "streak"]).cumcount())
                   .drop("streak", axis=1))
        gather_country_dfs.append(time_df)
    final_df = pd.concat(gather_country_dfs)
    final_df.loc[:, [
        "person_id", "age_group_name", "country", "date", "state", "staying_days", "prev_state"
                   ]].to_csv("a3-covid-simulated-timeseries.csv")
    summary_df = final_df.groupby(["date", "country"])["state"]\
                         .value_counts().unstack().fillna(0)\
                         .astype(int)
    summary_df.to_csv("a3-covid-summary-timeseries.csv")
    create_plot("a3-covid-summary-timeseries.csv", countries)
    return final_df
