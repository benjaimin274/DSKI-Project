import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random
from typing import List, Dict, Union

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import seaborn as sns
from sklearn.metrics import confusion_matrix#
import shap

# Declare global variable for type hints:
ModelType = Union[DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier]
EnsembleModelType = Union[RandomForestClassifier, GradientBoostingClassifier]

# Abbrevations:
# DecisionTreeClassifier = dtc
# RandomForestClassifier = rfc
# GradientBoostingClassifier = gbc

class LoadData:
    def __init__(self):
        pass

    def normal_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(r"data\Titanic-Dataset.csv")
        return df
    
    def synthetic_data(self, num_rows: int) -> pd.DataFrame:
        df = self.normal_dataset()
        relevant_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived", "Embarked"]
        df = df[relevant_columns]

        metadata = Metadata.detect_from_dataframe(df)

        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(df)
        synthetic_data = synthesizer.sample(num_rows= num_rows)
        return synthetic_data
    
    def random_data(self, num_rows: int) -> pd.DataFrame:
        random.seed(42)
        
        df_structure = {
            "Survived": [],
            "Pclass": [],
            "Sex": [],
            "Age": [],
            "SibSp": [],
            "Parch": [],
            "Fare": [],
            "Embarked": [],
        }

        # Fully random data, no distribution preserved:
        survived_range = [0,1] # binary: survived or not survived
        pclass_range = [1, 2, 3] # there are exactly three passenger classes
        sex_range = [0, 1] # binary: female = 0, male = 1
        age_range = [age for age in range(0, 81)] # min max range, int steps
        sip_sp_range = [num_sib for num_sib in range(0, 9)] # number of siblings/spouses
        parch_range = [parch for parch in range(0, 7)] #number of parents/ children on board
        fare_range = [fare for fare in range(0, 513)] # out of simplicity, reduce to integer values (ranges preserved)
        embarked_range = ["S", "C", "Q"]

        for i in range(num_rows):
            df_structure["Survived"].append(random.choice(survived_range))
            df_structure["Pclass"].append(random.choice(pclass_range))
            df_structure["Sex"].append(random.choice(sex_range))
            df_structure["Age"].append(random.choice(age_range))
            df_structure["SibSp"].append(random.choice(sip_sp_range))
            df_structure["Parch"].append(random.choice(parch_range))
            df_structure["Fare"].append(random.choice(fare_range))
            df_structure["Embarked"].append(random.choice(embarked_range))
        return pd.DataFrame(df_structure)


class DataPreparation:
    def __init__(self, data: pd.DataFrame):
        self.target_col = "Survived"
        self.df = data

        self.default_preparation()
    
    # Maybe still allow irrelevant features to show the effects of bad feature selection?! 
    def default_preparation(self)-> None:
        # Try statement allows to also process false data
        try:
            irrelevant_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
            self.df = self.df.drop(irrelevant_columns, axis= 1)

            # Convert categorical data (male/female) to binary:
            mapping = {'female': 0, 'male': 1}
            self.df["Sex"] = self.df["Sex"].map(mapping)
        except:
            pass

        # One hot encode "Embarked":
        embarked_one_hot_encoded = pd.get_dummies(self.df["Embarked"], prefix= "Embarked", dtype= int)
        self.df.drop("Embarked", axis=1, inplace= True)

        self.df = pd.concat([self.df, embarked_one_hot_encoded], axis=1)
    
    def available_cols(self) -> List[str]:
        """
        Returns all columns that are available as features.
        """
        cols = list(self.df.columns)
        cols.remove(self.target_col)
        return cols
    
    def choose_columns(self, selected_cols: List[str]):
        columns_to_keep = selected_cols + ["Survived"]
        self.df = self.df[columns_to_keep]
    
    def default_features(self) -> List[str]:
        return ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    # Maybe include this part into the default data preparation... Not sure (could be too much to select.)
    def check_for_missing_values_in_age(self) -> bool:
        if "Age" in self.df.columns:
            return self.df["Age"].isna().any()
        else:
            return False

    def impute_missing_values(self) -> None:
        median_age = self.df["Age"].median()
        self.df = self.df.fillna(value= median_age)

    def drop_missing_values(self) -> None:
        self.df.dropna(inplace = True, axis= 0)
    
    def X_y_dataset(self):
        """
        The raw X and y dataset is only needed for 5-fold validation
        """
        y = self.df[self.target_col]
        X = self.df.drop([self.target_col], axis= 1)
        return X, y

    def train_test_datset(self) -> Dict[str, List]:
        y = self.df[self.target_col]
        X = self.df.drop([self.target_col], axis= 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify= y)
        
        prepared_data = {
            "train": [X_train, y_train],
            "test": [X_test, y_test]
        }
        return prepared_data

    def train_val_test_dataset(self) -> Dict[str, List]:
        y = self.df[self.target_col]
        X = self.df([self.target_col], axis= 1)
        X_training, X_test, y_training, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify= y)

        X_train, X_val, y_train, y_val = train_test_split(
            X_training, y_training, test_size=0.2, random_state=42, stratify= y_training)
        
        prepared_data = {
            "train": [X_train, y_train],
            "val": [X_val, y_val],
            "test": [X_test, y_test]
        }
        return prepared_data

# Maybe include SHAP ?! (to get the feature importances)
class ModelTrainingAndEvaluation:
    def __init__(self, prepared_data: Dict):
        self.prepared_data = prepared_data

    def train_basic_decision_tree(self, depth: int = None, metric: str = "gini"):
        X_train, y_train = self.prepared_data["train"]
        clf_basic_tree = DecisionTreeClassifier(max_depth= depth, criterion= metric)
        clf_basic_tree.fit(X_train, y_train)
        return clf_basic_tree
    
    def train_random_forest(self, n_estimators: int = 100, depth: int = None, metric: str = "gini"):
        X_train, y_train = self.prepared_data["train"]
        clf_random_forest = RandomForestClassifier(n_estimators= n_estimators, 
                                                   max_depth= depth, 
                                                   criterion= metric, 
                                                   random_state=42)
        clf_random_forest.fit(X_train, y_train)
        return clf_random_forest
    
    def train_boosted_tree(self, loss: str = "log_loss", learning_rate: float = 0.1, n_estimators: int = 100, depth: int = None, metric: str = "squared_error"):
        X_train, y_train = self.prepared_data["train"]
        clf_boosted_tree = GradientBoostingClassifier(loss= loss, learning_rate= learning_rate, n_estimators= n_estimators, criterion= metric, max_depth= depth)
        clf_boosted_tree.fit(X_train, y_train)
        return clf_boosted_tree
    
    def evaluate_with_test_dataset(self, model: ModelType):
        X_test, y_test = self.prepared_data["test"]
        accuracy = model.score(X_test, y_test)
        return accuracy
    
    def create_heatmap(self, model: ModelType) -> Figure:
        X_test, y_test = self.prepared_data["test"]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_true= y_test, y_pred= y_pred)
        fig, ax = plt.subplots()  
      
        # Create heatmap on the axes  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
        ax.set_title("Heatmap")  
        ax.set_xlabel("Predicted")  
        ax.set_ylabel("Ground truth")  
        return fig  

    # Let them visualize a normal tree:
    def visualize_tree_dtc(self, model: DecisionTreeClassifier) -> Figure:
        """
        Visualizes the inputed tree model. It only shows the first four layers due to readability.
        """
        X_train, y_train = self.prepared_data["train"]
        X_train: pd.DataFrame
        fig, ax = plt.subplots()
        tree.plot_tree(model, feature_names= X_train.columns, max_depth= 3, fontsize=4, ax = ax)
        return fig

    # Probabely won't be used...
    def visualize_tree_ensemble(self, model: EnsembleModelType, n: int):
        estimator_n = model.estimator_[n]
        X_train, y_train = self.prepared_data["train"]
        X_train: pd.DataFrame
        fig, ax = plt.subplots()
        tree.plot_tree(estimator_n, feature_names= X_train.columns, fontsize=5, ax = ax)
        return fig
    
    # Shap function does not exactly work as intended
    def shap_plot(self, model: ModelType):
        X_train, y_train = self.prepared_data["train"]
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, features = X_train, class_inds = [0])
        plt.show()
    
    #------------------k fold validation------------------------------#
    def k_fold_eval_dtc(self, X, y, depth: int = None, metric: str = "gini") -> pd.DataFrame:
        dtc_model = DecisionTreeClassifier(max_depth= depth, criterion= metric)
        scores = cross_val_score(dtc_model, X, y, cv = 5)
        output = {
            "Bester Wert": [scores.max()],
            "Schlechtester Wert": [scores.min()],
            "Durchschnittlicher Wert": [scores.mean()]
        }
        return pd.DataFrame(output)

    def k_fold_eval_rfc(self, X, y, n_estimators: int, depth: int, metric: str) -> pd.DataFrame:
        rfc_model = RandomForestClassifier(n_estimators= n_estimators, 
                                                   max_depth= depth, 
                                                   criterion= metric, 
                                                   random_state=42)
        scores = cross_val_score(rfc_model, X, y, cv = 5)
        output = {
            "Bester Wert": [scores.max()],
            "Schlechtester Wert": [scores.min()],
            "Durchschnittlicher Wert": [scores.mean()]
        }
        return pd.DataFrame(output)
    
    # Add the boosted tree...

class ModelTuning:
    # Maybe only the most important ones?!
    def __init__(self, model: ModelType, prepared_data: Dict):
        self.model = model
        self.prepared_data = prepared_data
    
    # All tuning methods for DecisionTreeClassifier:
    def calculate_optimal_depth_dtc(self, metric: str = "gini"):
        """
        Find out the optimal depth for a normal decision tree.
        """
        X_train, y_train = self.prepared_data["train"]
        X_val, y_val = self.prepared_data["val"]
        
        pruning = {
            "depth": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        for depth in range(1, 100):
            clf_model = DecisionTreeClassifier(max_depth= depth, criterion= metric)
            clf_model.fit(X_train, y_train)

            if clf_model.get_depth() < depth:
                print(f"The maximum depth is already reached at {depth} !")
                break

            train_score = clf_model.score(X_train, y_train)
            val_score = clf_model.score(X_val, y_val)

            pruning["depth"].append(depth)
            pruning["train_accuracy"].append(train_score)
            pruning["val_accuracy"].append(val_score)
        return pruning
    
    def calculate_optimal_metric_dtc(self):
        pass

    def visualize_optimal_depth_dtc(self):
        pass

    def get_optimal_depth_dtc(self, pruning: Dict) -> int:
        df_pruning = pd.DataFrame(pruning)
        max_val_score = df_pruning["val_accuracy"].max()
        optimal_depth_index = df_pruning[df_pruning["val_accuracy"] == max_val_score].index

        optimal_depth = df_pruning.loc[optimal_depth_index[0], "depth"]
        return optimal_depth
    
    # Tuning methods for RandomForestClassifier: (n_estimators, critereon)


    # Tuning methods for GradientBoostingClassifier: (loss, learning_rate) 

if __name__ == "__main__":
    load = LoadData()
    data = load.normal_dataset()

    data_prep = DataPreparation(data)
    data_prep.impute_missing_values()
    prepared_data = data_prep.train_test_datset()

    training = ModelTrainingAndEvaluation(prepared_data)
    dtc = training.train_basic_decision_tree(depth= 3)
    training.shap_plot(dtc)
    # score = training.evaluate_with_test_dataset(dtc)
    # print("Accuracy of the model:", score)
    # fig = training.create_heatmap(dtc)
    #plt.show()
    #training.visualize_tree_dtc(dtc)
    pass

    