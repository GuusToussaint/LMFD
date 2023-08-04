import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from nltk import CFG, ChartParser
from nltk.parse.generate import generate
import pandas as pd
import sympy
from sympy import sympify, lambdify, symbols, Function
import scipy
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter
)
from smac import Scenario, HyperparameterOptimizationFacade
import logging
import itertools
from tqdm import tqdm
import copy
import functools
import concurrent.futures
from functions import emwa, sigmoid, exp
from optimize import smac_optimize, grid_search


def parse_tree_to_equation(tree):
    """
    Convert a parse tree to an equation which can be evaluated
    """
    def recurse(t):
        if isinstance(t, str):
            # If the node is a leaf, return the value
            return t
        
        # Otherwise, recurse on the children and add brackets
        leafs = []
        for s in t:
            leafs.append(recurse(s))
        if len(leafs) == 1:
            # No need to add brackets if there is only one child
            return leafs[0]
        # Add brackets around the children
        return f"({' '.join(leafs)})"
    
    # Recurse on the root node
    tree_string = recurse(tree)
    return tree_string

def normalize(sensor_data):
    x = np.array(sensor_data)
    return (x - x.mean()) / x.std()

class LMFD:
    def __init__(self, df: pd.DataFrame, feature_names: list[str], grammar_string: str, x_labels):
        self.df = df
        self.feature_names = feature_names
        self.grammar_string = grammar_string
        self.top_n = 10
        self.x_labels = x_labels
        self.n_trials = 100
        self.run_concurent = False
        self.bounds = {
            'c1': (-1, 1),
            'c2': (-1, 1),
            'c3': (-1, 1),
        }


    def start(self):

        if self.run_concurent:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)
            futures = []

        results = []

        possible_monotonic_features = self.generate_equation()
        unqiue_combinations_of_features = self.generate_combinations_of_features()
        with tqdm(total=len(unqiue_combinations_of_features) * 2, bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
            def done_callback(arg):
                pbar.update(1)

            for index, (feature_1, feature_2) in enumerate(unqiue_combinations_of_features):
                if self.run_concurent:
                    f = executor.submit(self.find_best_equation, feature_1, feature_2, possible_monotonic_features)
                    f.add_done_callback(done_callback)                
                    futures.append(f)

                    f = executor.submit(self.find_best_equation, feature_2, feature_1, possible_monotonic_features)
                    f.add_done_callback(done_callback)                
                    futures.append(f)
                else:
                    print(f"Creating monotonic proxy for {feature_1} and {feature_2}")
                    result = self.find_best_equation(feature_1, feature_2, possible_monotonic_features)
                    results.extend(result)

                    print(f"Creating monotonic proxy for {feature_2} and {feature_1}")
                    result = self.find_best_equation(feature_2, feature_1, possible_monotonic_features)
                    results.extend(result)


            if self.run_concurent:
                concurrent.futures.wait(futures)

        if self.run_concurent:
            for future in futures:
                current_result = future.result()
                results.extend(current_result)

        results = sorted(results, key=lambda x: x[1][0], reverse=True)
        for index, (equation, (rho, incumbent, feature_1, feature_2)) in enumerate(results):
            equation = sympify(equation)
            print(f"${self.generate_title('{%s}' % feature_1, '{%s}' % feature_2, equation, incumbent)}$ & ${rho:.4f}$ \\\\ ")
    
            if index <= self.top_n:
                title = self.generate_title(feature_1, feature_2, equation, incumbent)
                self.plot_equation(equation, incumbent, feature_1, feature_2, rho, title)

            if index > 100:
                break
    

    def generate_title(self, feature_1, feature_2, possible_monotonic_feature, incumbent):
        c1,c2,c3,c4,c5,s1,s2 = symbols('c1,c2,c3,c4,c5,s1,s2')
        equation_string = str(possible_monotonic_feature)
        if possible_monotonic_feature.has(c1):
            equation_string = equation_string.replace('c1', f"{incumbent['c1']:.3f}")
        if possible_monotonic_feature.has(c2):
            equation_string = equation_string.replace('c2', f"{incumbent['c2']:.3f}")
        if possible_monotonic_feature.has(c3):
            equation_string = equation_string.replace('c3', f"{incumbent['c3']:.3f}")
        if possible_monotonic_feature.has(c4):
            equation_string = equation_string.replace('c4', f"{incumbent['c4']:.3f}")
        if possible_monotonic_feature.has(c5):
            equation_string = equation_string.replace('c5', f"{incumbent['c5']:.3f}")

        if possible_monotonic_feature.has(s1):
            equation_string = equation_string.replace('s1', f's_{feature_1}')
        if possible_monotonic_feature.has(s2):
            equation_string = equation_string.replace('s2', f's_{feature_2}')

        return equation_string

    def plot_equation(self, equation, incumbent, feature_1, feature_2, rho, title):
        # Create the plot
        # plt.figure(figsize=(20, 10))
        plt.clf()
        ax = plt.axes()

        s_1_norm = normalize(self.df[feature_1].values.tolist())
        s_2_norm = normalize(self.df[feature_2].values.tolist())

        proxy = self.evaluate_lambda(equation, feature_1, feature_2, incumbent)

        # Plot the data
        ax.plot(self.x_labels, s_1_norm, color='green', linewidth=0.5, label=feature_1)
        ax.plot(self.x_labels, s_2_norm, color='orange', linewidth=0.5, label=feature_2)
        ax.plot(self.x_labels, proxy, color='red', linewidth=1, label='Proxy')

        # Settings for the plot
        ax.set(xlabel='Index', ylabel=f"Sensor value", title=f"{title} (rho: {rho:.4f})")
        # ax.grid()
        ax.legend()
        # xfmt = md.DateFormatter('%Y-%m-%d')
        # ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(f"plots/{feature_1}_{feature_2}_{rho:.4f}.pdf")

    def generate_combinations_of_features(self):
        unique_combinations = list(itertools.combinations(self.feature_names, 2))
        return unique_combinations

    def find_best_equation(self, feature_1: str, feature_2: str, possible_monotonic_features_string):
        optimized_equations = {}
        for possible_monotonic_feature_string in possible_monotonic_features_string:
            possible_monotonic_feature = sympify(possible_monotonic_feature_string)
            incumbent, rho = self.optimize(feature_1, feature_2, possible_monotonic_feature)
            optimized_equations[str(possible_monotonic_feature)] = (rho, incumbent, feature_1, feature_2)
        sorted_optimized_equations = sorted(optimized_equations.items(), key=lambda x: x[1][0], reverse=True)
        return sorted_optimized_equations

    def contains_constants(self, equation):
        # Define functions and symbols used in the equations
        c1,c2,c3,c4,c5 = symbols('c1,c2,c3,c4,c5')
        return equation.has(c1) or equation.has(c2) or equation.has(c3) or equation.has(c4) or equation.has(c5)

    def optimize(self, feature_1, feature_2, possible_monotonic_feature):

        if not self.contains_constants(possible_monotonic_feature):
            # If the equation does not contain constants we can just evaluate it
            proxy = self.evaluate_lambda(possible_monotonic_feature, feature_1, feature_2, {})

            # Calculate the spearmans rho
            rho = scipy.stats.spearmanr(range(len(proxy)), proxy)[0]
            return {}, abs(rho)

        grid_search_result = grid_search(
            possible_monotonic_feature,
            self.df,
            self.evaluate_lambda,
            feature_1,
            feature_2,
            self.bounds,
            self.n_trials,
        )

        # smac_result = smac_optimize(
        #     possible_monotonic_feature,
        #     self.df,
        #     self.evaluate_lambda,
        #     feature_1,
        #     feature_2,
        #     self.bounds,
        #     self.n_trials,
        # )

        # print(f"SMAC: {smac_result[1]}, Grid search: {grid_search_result[1]}")

        return grid_search_result

    
    def evaluate_lambda(self, equation, feature_1, feature_2, constants):
        # Define functions and symbols used in the equations
        ewma_symbol, sigmoid_symbol, exp_symbol = Function('ewma'), Function('sigmoid'), Function('exp')
        c1,c2,c3,c4,c5,s1,s2 = symbols('c1,c2,c3,c4,c5,s1,s2')

        # Convert the equation to a lambda function
        equation_lambda = lambdify([
            c1,
            c2,
            c3,
            c4,
            c5,
            s1,
            s2,
            ewma_symbol,
            sigmoid_symbol,
            exp_symbol,
        ], equation, 'numpy')

        # Get the current values of the features
        c1, c2, c3, c4, c5 = 0, 0, 0, 0, 0
        if 'c1' in constants.keys():
            c1 = constants['c1']
        if 'c2' in constants.keys():
            c2 = constants['c2']
        if 'c3' in constants.keys():
            c3 = constants['c3']
        if 'c4' in constants.keys():
            c4 = constants['c4']    
        if 'c5' in constants.keys():
            c5 = constants['c5']

        # TODO: This should be made more efficient
        s1, s2 = normalize(self.df[feature_1].values.tolist()), normalize(self.df[feature_2].values.tolist())

        # Calculate the values for the current proxy
        proxy = equation_lambda(c1, c2, c3, c4, c5, s1, s2, emwa, sigmoid, exp)

        # Apply a penalty for nan values
        proxy_numpy = np.array(proxy)
        proxy_mean = proxy_numpy[~np.isnan(proxy_numpy)].mean()
        clean_proxy = [proxy_mean if np.isnan(x) else x for x in proxy]

        return clean_proxy

    def create_configspace(self, equation):
        # TODO: This should be made more generic, now it only works for the current grammar
        constants = {}
        c1,c2,c3,c4,c5,s1,s2 = symbols('c1,c2,c3,c4,c5,s1,s2')

        if equation.has(c1):
            constants['c1'] = UniformFloatHyperparameter(
                'c1',
                lower=-1.0,
                upper=1.0,
            )
        if equation.has(c2):
            constants['c2'] = UniformFloatHyperparameter(
                'c2',
                lower=-1.0,
                upper=1.0,
            )
        if equation.has(c3):
            constants['c3'] = UniformFloatHyperparameter(
                'c3',
                lower=-1.0,
                upper=1.0,
            )
        if equation.has(c4):
            constants['c4'] = UniformIntegerHyperparameter(
                'c4',
                lower=1,
                upper=self.df.shape[0]-1,
            )
        if equation.has(c5):
            constants['c5'] = UniformIntegerHyperparameter(
                'c5',
                lower=1,
                upper=self.df.shape[0]-1,
            )

        configspace = ConfigurationSpace(constants)
        return configspace

    def generate_equation(self) -> list:

        # Construct a context free grammar from the grammar string
        grammar = CFG.fromstring(self.grammar_string)

        # Generate all possible equations from the grammar
        all_equations = list(generate(grammar))

        # Create a parser from the grammar this parser will be used to 
        # parse the generated sentences to trees. Thus we can the same 
        # equation for different order of operations.
        parser = ChartParser(grammar)

        # Parse all the generated sentences to mathematically unique equations
        results = []
        for equation in all_equations:
            equation_trees = list(parser.parse(equation))
            for equation_tree in equation_trees:
                # Parse the current tree to a string that can be used as an equation
                equation_string = parse_tree_to_equation(equation_tree)

                # Simplify the equation using sympy to remove semantically different but
                # mathematically equivalent equations
                simplified_equation = sympify(equation_string)


                # Add the simplified equation to the list of equations
                # if it is not already in the list
                if isinstance(simplified_equation, sympy.core.numbers.Number):
                    print(equation_string, simplified_equation)
                    continue
                results.append(str(simplified_equation)) if simplified_equation not in results else None
        
        # c1,c2,c3,c4,c5,s1,s2 = symbols('c1,c2,c3,c4,c5,s1,s2')
        # for r in results:
        #     r = sympify(r)
        #     constants_in_functions = sum(
        #         [
        #             r.has(c1),
        #             r.has(c2),
        #             r.has(c3),
        #             r.has(c4),
        #             r.has(c5),
        #         ]
        #     )
        #     if constants_in_functions > 2:
        #         print(r, constants_in_functions)
        # exit(1)
        return results

def get_sorted_sensors(df):
    sensors = df.columns[1:].tolist()
    sensor_spearmans_rho = {}
    for s in sensors:
        sensor_data = df[s].values.tolist()
        sensor_spearmans_rho[s] = scipy.stats.spearmanr(range(len(sensor_data)), sensor_data)[0]
    
    # Sort the sensors by their absolute spearmans rho
    sorted_sensors = {k: v for k, v in sorted(sensor_spearmans_rho.items(), key=lambda item: abs(item[1]), reverse=True)}
    return sorted_sensors

if __name__ == "__main__":
    # As input we want a dataframe, and list of feature names

    # # Artificial data
    # df = pd.read_csv('artificial 1.txt', delimiter='\t')
    # time = df['time'].values.tolist()
    # print(get_sorted_sensors(df))
    # # feature_names = ['s1', 's2']
    # feature_names = ['1', '2']

    # # Climate dataset
    # df = pd.read_csv('combined.txt', delimiter='\t')
    # time = df['Year'].values.tolist()
    # # feature_names = [key for key, value in get_sorted_sensors(df).items() if abs(value) < 0.8]
    # feature_names = ['90S-24S', '64N-90N']

    # InfraWatch data
    df = pd.read_csv('all.csv')
    timestamps = df['Timestamp'].values.tolist()
    time = pd.to_datetime(timestamps)
    # Create a list of feature names
    # feature_names = [key for key, value in get_sorted_sensors(df).items() if abs(value) < 0.2]
    # # feature_names = ['159', '165']
    feature_names = ['165', '159']


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

    print(len(feature_names))
    monotonic_discovery = LMFD(df, feature_names, grammar_string, time)
    monotonic_discovery.start()
