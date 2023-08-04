import scipy
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter
)
from smac import Scenario, HyperparameterOptimizationFacade
from sympy import symbols
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import ParameterGrid


def construct_config_space(equation, bounds: dict, dataframe: pd.DataFrame):
    # Using the bounds construct the configuration space
    constants = {}
    c1,c2,c3,c4,c5,s1,s2 = symbols('c1,c2,c3,c4,c5,s1,s2')

    if equation.has(c1):
        constants['c1'] = UniformFloatHyperparameter(
            'c1',
            lower=bounds['c1'][0],
            upper=bounds['c1'][1],
        )
    if equation.has(c2):
        constants['c2'] = UniformFloatHyperparameter(
            'c2',
            lower=bounds['c2'][0],
            upper=bounds['c2'][1],
        )
    if equation.has(c3):
        constants['c3'] = UniformFloatHyperparameter(
            'c3',
            lower=bounds['c3'][0],
            upper=bounds['c3'][1],
        )
    if equation.has(c4):
        constants['c4'] = UniformIntegerHyperparameter(
            'c4',
            lower=1,
            upper=dataframe.shape[0]-1,
        )
    if equation.has(c5):
        constants['c5'] = UniformIntegerHyperparameter(
            'c5',
            lower=1,
            upper=dataframe.shape[0]-1,
        )

    configspace = ConfigurationSpace(constants)
    return configspace


def smac_optimize(
        equation,
        dataframe: pd.DataFrame,
        evaluate_lambda,
        feature_1,
        feature_2,
        bounds: dict,
        iterations: int,
) -> (Configuration, float):
    '''
    Returns the incumbent and the best value found by SMAC

        Parameters:
            equation (function): The function to be optimized
            dataframe (pd.DataFrame): The dataframe containing the data
            evaluate_lambda (function): A function that evaluates the function to be optimized
            bounds (dict): The bounds of the parameters
            iterations (int): The number of iterations
            feature_1 (list): The feature vectors of the first feature
            feature_2 (list): The feature vectors of the second feature

        Returns:
            incumbent (Configuration): The incumbent, which contains the best parameters found
            best_rho (float): The best value found
    '''

    configspace = construct_config_space(equation, bounds, dataframe)

    def train(config: Configuration, seed: int = 0) -> float:
        proxy = evaluate_lambda(equation, feature_1, feature_2, config)

        # This can happen 
        if np.array(proxy).std() == 0:
            return 1

        # Calculate the spearmans rho
        rho = scipy.stats.spearmanr(range(len(proxy)), proxy)[0]
        abs_rho = abs(rho)
        return 1 - abs_rho
    
    # TODO: We might want to make this paralizable 
    scenario = Scenario(
        configspace=configspace,
        n_trials=iterations,
    )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train,
        overwrite=True,
        logging_level=logging.ERROR,
    )

    incumbent = smac.optimize()
    best_rho = 1 - train(incumbent)

    return incumbent, best_rho

# 2. Grid search
def grid_search(
        equation,
        dataframe: pd.DataFrame,
        evaluate_lambda,
        feature_1,
        feature_2,
        bounds: dict,
        iterations: int,
    ) -> (Configuration, float):
    '''
    Returns the incumbent and the best value found by the Grid Search approach

        Parameters:
            equation (function): The function to be optimized
            dataframe (pd.DataFrame): The dataframe containing the data
            evaluate_lambda (function): A function that evaluates the function to be optimized
            bounds (dict): The bounds of the parameters
            iterations (int): The number of iterations
            feature_1 (list): The feature vectors of the first feature
            feature_2 (list): The feature vectors of the second feature

        Returns:
            incumbent (Configuration): The incumbent, which contains the best parameters found
            best_rho (float): The best value found
    '''

    # Calculate the resolution
    # It is important, for a fair comparison, that the number of final evaluations is not larger than the number of iterations
    num_of_parameters = np.sum([equation.has(symbols('c1')), equation.has(symbols('c2')), equation.has(symbols('c3')), equation.has(symbols('c4')), equation.has(symbols('c5'))])
    resolution = np.floor(np.exp(np.log(iterations) / num_of_parameters)).astype(int)

    # Construct the grid
    parameter_options = {
        'c1': np.linspace(bounds['c1'][0], bounds['c1'][1], resolution) if equation.has(symbols('c1')) else None,
        'c2': np.linspace(bounds['c2'][0], bounds['c2'][1], resolution) if equation.has(symbols('c2')) else None,
        'c3': np.linspace(bounds['c3'][0], bounds['c3'][1], resolution) if equation.has(symbols('c3')) else None,
        'c4': np.linspace(1, dataframe.shape[0]-1, resolution, dtype=int) if equation.has(symbols('c4')) else None,
        'c5': np.linspace(1, dataframe.shape[0]-1, resolution, dtype=int) if equation.has(symbols('c5')) else None,
    }
    trimmed_parameter_options = {k: v for k, v in parameter_options.items() if v is not None}

    incumbent, best_rho = None, 0.0
    configspace = construct_config_space(equation, bounds, dataframe)

    parameter_grid = ParameterGrid(trimmed_parameter_options)
    for parameters in parameter_grid:
        current_config = Configuration(configspace, values=parameters)
        proxy = evaluate_lambda(equation, feature_1, feature_2, current_config)

        # This can happen 
        if np.array(proxy).std() == 0:
            abs_rho = 0
        else:
            rho = scipy.stats.spearmanr(range(len(proxy)), proxy)[0]
            abs_rho = abs(rho)

        # Calculate the spearmans rho
        if abs_rho > best_rho:
            incumbent = current_config
            best_rho = abs_rho
    
    return incumbent, best_rho

# 3. Random search
def random_search():

    pass
