import argparse
import pandas as pd
from LMFD import LatentMonotonicFeatureDiscovery
import scipy.stats
import logging

def get_sorted_sensors(df):
    sensors = df.columns[1:].tolist()
    sensor_spearmans_rho = {}
    for s in sensors:
        sensor_data = df[s].values.tolist()
        sensor_spearmans_rho[s] = scipy.stats.spearmanr(range(len(sensor_data)), sensor_data)[0]
    
    # Sort the sensors by their absolute spearmans rho
    sorted_sensors = {k: v for k, v in sorted(sensor_spearmans_rho.items(), key=lambda item: abs(item[1]), reverse=True)}
    return sorted_sensors

def get_climate_data():
    # Climate dataset
    df = pd.read_csv('data/climate.txt', delimiter='\t')
    time = df['Year'].values.tolist()
    feature_names = [key for key, value in get_sorted_sensors(df).items() if abs(value) < 0.8]

    # FIXME: This is a hack to get the right features
    # feature_names = ['90S-24S', '64N-90N']

    return df, feature_names, time

def get_artificial_data():
    # Artificial data
    df = pd.read_csv('data/artificial.txt', delimiter='\t')
    time = df['time'].values.tolist()
    feature_names = ['1', '2']

    return df, feature_names, time

def get_infra_watch_data():
    # InfraWatch data
    df = pd.read_csv('data/InfraWatch.csv')
    timestamps = df['Timestamp'].values.tolist()
    time = pd.to_datetime(timestamps)
    feature_names = [key for key, value in get_sorted_sensors(df).items() if abs(value) < 0.2]

    # FIXME: This is a hack to get the right features
    # feature_names = ['159', '165']

    return df, feature_names, time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="LMFD experiments"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["artificial", "climate", "InfraWatch"],
        default="artifical",
        help="Experiment to run"
    )

    args = parser.parse_args()

    if args.experiment == "artificial":
        print("Running artifical experiment")
        df, feature_names, time = get_artificial_data()
    elif args.experiment == "climate":
        print("Running climate experiment")
        df, feature_names, time = get_climate_data()
    elif args.experiment == "InfraWatch":
        print("Running InfraWatch experiment")
        df, feature_names, time = get_infra_watch_data()
    else:
        raise ValueError("Unknown experiment")


    grammar_string = """
        V -> \
            A1 |\
            A2 Add Const B1 | \
            A2 Mult B1 | \
            A2 Div B2 
        Const -> \
            'c1*'
        A1 -> \
            's1' 
        A2 -> \
            's1' | \
            's2' | \
            'ewma(s1, c4)' | \
            'sigmoid(s1)' | \
            'exp(c2*s1)'
        B1 -> \
            's2' | \
            'ewma(s2, c5)' | \
            'sigmoid(s2)' | \
            'exp(c3*s2)' 
        B2 -> \
            's2' | \
            'ewma(s2, c5)' | \
            'sigmoid(s2)'
        Add -> \
            '+' 
        Mult -> \
            '*' 
        Div -> \
            '/' 
    """

    logging.basicConfig(level=logging.INFO)

    monotonic_discovery = LatentMonotonicFeatureDiscovery(df, feature_names, grammar_string, time)
    monotonic_discovery.start()


