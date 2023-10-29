import math
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch
import pickle
import numpy as np
import pandas as pd

from pandas import DataFrame, concat
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from torch import Tensor, tensor, float32
from typing import Callable, Union, List

def as_torch_tensor(func: Callable[..., DataFrame]):
    def wrapper(*args, **kwargs):
        df: DataFrame = func(*args, **kwargs)
        return tensor(df, dtype=float32)
    return wrapper

class Preprocessor():
    __input_transformer: ColumnTransformer
    __target_transformer: ColumnTransformer

    def __init__(self):
        self.__input_transformer = ColumnTransformer(
            transformers=[
                ('std', StandardScaler(), [
                    'longitude',
                    'latitude',
                ]),
                ('pwr', PowerTransformer(method="box-cox", standardize=True), [
                    'housing_median_age',
                    'total_rooms',
                    'total_bedrooms',
                    'population',
                    'households',
                    'median_income',
                ]),
                ('one_hot', OneHotEncoder(), ['ocean_proximity']),
            ],
            remainder='drop',
        )
        self.__target_transformer = ColumnTransformer(
            transformers=[
                ('std', StandardScaler(), ['median_house_value'])
            ],
            remainder='drop',
        )

    def fill_missing(self, x: DataFrame):
        x = x.copy()

        total_bedrooms_mean = x['total_bedrooms'].mean()
        x['total_bedrooms'] = x['total_bedrooms'].fillna(total_bedrooms_mean)

        return x

    def fit_input(self, input_df: DataFrame):
        self.__input_transformer.fit(input_df)

    def fit_target(self, target_df: DataFrame):
        self.__target_transformer.fit(target_df)

    @as_torch_tensor
    def transform_input(self, input_df: DataFrame) -> Tensor:
        return self.__input_transformer.transform(input_df)

    @as_torch_tensor
    def transform_target(self, target_df: DataFrame) -> Tensor:
        return self.__target_transformer.transform(target_df)

    def inverse_transform_target(self, target_tensor: Tensor):
        target_np = target_tensor.detach().numpy()

        return self.__target_transformer \
            .named_transformers_['std'] \
            .inverse_transform(target_np)


class Regressor():
    __preprocessor: Preprocessor
    __loss_history: List[float]
    __loss_history_eval: List[float]

    def __init__(self, x, nb_epoch=400, model=None, batch_size=16, learning_rate=0.001, loss_fn=nn.MSELoss()):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn

        # Initialise Loss History
        self.__loss_history = []
        self.__loss_history_eval = []

        # Construct Preprocessor
        self.__preprocessor = Preprocessor()

        # Initialise Preprocessor
        X, _ = self._preprocessor(x, training=True)

        self.input_size = X.shape[1]
        self.output_size = 1

        # default configuration
        if model is None:
            self.model = nn.Sequential(
                nn.Linear(self.input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.output_size),
            )
        else:
            self.model = model

    def _preprocessor(self, x, y=None, training=False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        # clean data
        x = self.__preprocessor.fill_missing(x)

        
        # If training flag set, fit preprocessor to training data.
        if training:
            # Remove rows who's age are at maximum threshold.
            # age_filter = x['housing_median_age'] <= 50
            # x = x[age_filter]
            # if y is not None: y = y[age_filter]

            # Remove rows who's price is at maximum threshold.
            # if y is not None:
            #     price_filter = y['median_house_value'] <= 500000
            #     x = x[price_filter]
            #     y = y[price_filter]
            
            self.__preprocessor.fit_input(x)
            if y is not None: self.__preprocessor.fit_target(y)

        return (
            self.__preprocessor.transform_input(x), 
            self.__preprocessor.transform_target(y) if y is not None else None
        )

    def fit(self, x, y, x_eval = None, y_eval = None):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        input_data, target_data = self._preprocessor(
            x, y=y, training=True)  # Do not forget

        for epoch_n in range(self.nb_epoch):
            # Adam optimiser is sick
            optimiser = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
            )

            # self.model.parameters(), lr=self.learning_rate)

            # shuffle data and split into batches using DataLoader
            rand_perm = np.random.permutation(len(input_data))
            shuffled_input_data = input_data[rand_perm]
            shuffled_target_data = target_data[rand_perm]

            # split into self.batch_size sized batches
            no_batches = int(x.shape[0] / self.batch_size)

            if no_batches == 0:
                return self.model

            input_batches = np.array_split(shuffled_input_data, no_batches)
            target_batches = np.array_split(shuffled_target_data, no_batches)

            for (input_batch, target_batch) in zip(input_batches, target_batches):
                optimiser.zero_grad()

                # Perform a forward pass through the model
                outputs = self.model(input_batch)

                # Compute loss
                # Compute gradient of loss via backwards pass
                loss = self.loss_fn(outputs, target_batch)
                loss.backward()

                # Change the weights via gradient decent
                optimiser.step()

            error_train = self.score(x, y)
            self.__loss_history.append(error_train)


            if x_eval is not None and y_eval is not None:
                error_eval = self.score(x_eval, y_eval)
                self.__loss_history_eval.append(error_eval)


        return self.model

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def loss_history(self) -> List[float]:
        return self.__loss_history

    def loss_history_eval(self) -> List[float]:
        return self.__loss_history_eval

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False)  # Do not forget

        O = self.model.forward(X)

        return self.__preprocessor.inverse_transform_target(O)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        X_norm, Y_norm = self._preprocessor(x, y=y, training=False)  # Do not forget

        Y_pred_norm = self.model.forward(X_norm)

        Y_pred = self.__preprocessor.inverse_transform_target(Y_pred_norm)
        Y = self.__preprocessor.inverse_transform_target(Y_norm)

        return mean_squared_error(Y_pred, Y, squared=False)


def save_regressor(trained_model, model_name: Union[str, None] = None):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    model_pickle_path = 'part2_model.pickle' if model_name is None \
        else f'assets/{model_name}-lr-{trained_model.learning_rate}-epch-{trained_model.nb_epoch}.pickle'

    with open(model_pickle_path, 'wb') as target:
        pickle.dump(trained_model, target)
    print(f"\nSaved model in {model_pickle_path}")


def load_regressor(model_name: Union[str, None] = None):
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    model_pickle_path = 'part2_model.pickle' if model_name is None \
        else f'assets/{model_name}.pickle'

    # If you alter this, make sure it works in tandem with save_regressor
    with open(model_pickle_path, 'rb') as target:
        trained_model = pickle.load(target)
    print(f"\nLoaded model in {model_pickle_path}\n")
    return trained_model


def RegressorHyperParameterSearch():
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs

    train_data = pd.read_csv("housing.csv")
    eval_data = pd.read_csv("housing_eval.csv")

    data_main(train_data, eval_data)

def k_fold_main(k):
    data = pd.read_csv("housing.csv")

    chunk_size = data.shape[0] // k
    data_split = [data[i : i + chunk_size] for i in range(0, data.shape[0], chunk_size)]

    total = 0

    for i in range(0, k):
        # The eval data is this kth of the data.
        eval_data = data_split[i]

        # The training data is all but the eval data
        train_data = data_split[:]
        del train_data[i]
        train_data = pd.concat(train_data)

        total += data_main(train_data, eval_data)

    print(f"Average score over {k} splits = {total / k}")


# Trains model with train_data,
# Evaluates with eval_data,
# Returns score
def data_main(train_data, eval_data) -> float:
    output_label = "median_house_value"

    # Splitting input and output
    x_train = train_data.loc[:, train_data.columns != output_label]
    y_train = train_data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = load_regressor("../part2_model")
    # regressor = load_regressor()
    regressor.fit(x_train, y_train)
    #save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}".format(error))

    # Eval Error
    eval_x_train = eval_data.loc[:, eval_data.columns != output_label]
    eval_y_train = eval_data.loc[:, [output_label]]
    eval_error = regressor.score(eval_x_train, eval_y_train)
    print("\nRegressor error vs eval: {}\n".format(eval_error))

    return eval_error


if __name__ == "__main__":
    example_main()
    # k_fold_main(10)
