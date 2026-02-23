import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random
from typing import List, Dict, Union

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap
from xgboost import XGBClassifier

# Declare global variable for type hints:
ModelType = Union[DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier]
EnsembleModelType = Union[RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier]

# Abbrevations:
# DecisionTreeClassifier = dtc
# RandomForestClassifier = rfc
# AdaBoostClassifier = gbc 

# Unclear what to do about the randomness factor:
# random.seed(42), maybe fix it by using it in the init part of the function?!

class LoadData:
    def __init__(self):
        pass

    def normal_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(r"data\Titanic-Dataset.csv")
        return df
    
    def normal_and_random_data(self, num_rows: int) -> pd.DataFrame:
        df = self.normal_dataset()
        relevant_columns = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        df = df[relevant_columns]

        random_data = self.random_data(num_rows= num_rows)
        merged_datasets: pd.DataFrame = pd.concat([df, random_data], axis = 0)

        merged_datasets = merged_datasets.sample(frac=1, random_state= 42).reset_index(drop=True)
        return merged_datasets
    
    def random_data(self, num_rows: int) -> pd.DataFrame:
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
        sex_range = ["male", "female"] # binary: female = 0, male = 1
        age_range = [age for age in range(0, 81)] # min max range, int steps
        sip_sp_range = [num_sib for num_sib in range(0, 9)] # number of siblings/spouses
        parch_range = [parch for parch in range(0, 7)] #number of parents/ children on board
        fare_range = [fare for fare in range(0, 513)] # out of simplicity, reduce to integer values (ranges preserved)
        embarked_range = ["S", "C", "Q"]

        for i in range(num_rows):
            random.seed(i)

            df_structure["Survived"].append(random.choice(survived_range))
            df_structure["Pclass"].append(random.choice(pclass_range))
            df_structure["Sex"].append(random.choice(sex_range))
            df_structure["Age"].append(random.choice(age_range))
            df_structure["SibSp"].append(random.choice(sip_sp_range))
            df_structure["Parch"].append(random.choice(parch_range))
            df_structure["Fare"].append(random.choice(fare_range))
            df_structure["Embarked"].append(random.choice(embarked_range))
        
        df = pd.DataFrame(df_structure)
        return df

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
        except:
            pass

        if "Sex" in self.df.columns:
            # Convert categorical data (male/female) to binary:
            mapping = {'female': 0, 'male': 1}
            self.df["Sex"] = self.df["Sex"].map(mapping)

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

    def drop_missing_values(self) -> None: # <-- Somehow, this removes all the rows when data is generated randomely...
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
        X = self.df.drop([self.target_col], axis= 1)
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

class ModelTrainingAndEvaluation:
    def __init__(self, prepared_data: Dict):
        self.prepared_data = prepared_data

    #------------------------Pure model training-----------------------#

    def train_basic_decision_tree(self, depth: int = None, metric: str = "gini"):
        X_train, y_train = self.prepared_data["train"]
        clf_basic_tree = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
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
    
    def train_ada_boost(self, n_estimators: int, learning_rate: float, metric: str, depth: int = None):
        X_train, y_train = self.prepared_data["train"]
        weak_learner = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
        clf_ada_boost = AdaBoostClassifier(estimator= weak_learner, n_estimators= n_estimators, learning_rate= learning_rate, random_state=42)
        clf_ada_boost.fit(X_train, y_train)
        return clf_ada_boost
    
    ### Maybe implement this one; only if it outperforms the other trees notabely...
    def train_xgboost(self):
        X_train, y_train = self.prepared_data["train"]
        xgboost_model = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
        xgboost_model.fit(X_train, y_train)
        return xgboost_model
    
    #-------------------------Evaluation---------------------------# 
    def evaluate_the_training_data(self, model: ModelType):
        X_train, y_train = self.prepared_data["train"]
        train_score = model.score(X_train, y_train)
        return train_score
        
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
    def visualize_tree_ensemble(self, model: EnsembleModelType, n: int) -> Figure:
        estimator_n = model.estimators_[n]
        X_train, y_train = self.prepared_data["train"]
        X_train: pd.DataFrame
        fig, ax = plt.subplots()
        tree.plot_tree(estimator_n, feature_names= X_train.columns, max_depth= 3, fontsize=4, ax = ax)
        return fig
    
    def shap_plot(self, model: ModelType) -> Figure:
        X_train, y_train = self.prepared_data["train"]  
        explainer = shap.Explainer(model, X_train)  
        shap_values = explainer(X_train)

        # Summary plot for class 1 (positive class); but the chart would be the same for the other class due to the binary classification
        plt.close('all') # close all plots that already exist...
        shap.summary_plot(shap_values[:, :, 1], X_train, plot_type = "bar")
        fig = plt.gcf()
        return fig
    
    #------------------k fold validation------------------------------#
    def k_fold_eval_dtc(self, X, y, depth: int = None, metric: str = "gini") -> pd.DataFrame:
        dtc_model = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
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
    
    def k_fold_eval_btc(self, X, y, learning_rate: float, n_estimators: int, depth: int, metric: str):
        weak_learner = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
        clf_ada_boost = AdaBoostClassifier(estimator= weak_learner, n_estimators= n_estimators, learning_rate= learning_rate, random_state=42)
        scores = cross_val_score(clf_ada_boost, X, y, cv = 5)
        output = {
            "Bester Wert": [scores.max()],
            "Schlechtester Wert": [scores.min()],
            "Durchschnittlicher Wert": [scores.mean()]
        }
        return pd.DataFrame(output)
    
class ModelTuning:
    def __init__(self, prepared_data: Dict):
        self.prepared_data = prepared_data
    
    #---------------Calculate Decision Tree Classifier hyperparameter tuning---------------------#
    def calculate_optimal_depth_dtc(self, metric: str) -> Dict[str, List]:
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
            clf_model = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
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
    
    def calculate_optimal_metric_dtc(self, depth: int) -> Dict[str, List]:
        X_train, y_train = self.prepared_data["train"]
        X_val, y_val = self.prepared_data["val"]
        
        pruning = {
            "metric": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        possible_metrics = ['gini', 'entropy', 'log_loss']

        for metric in possible_metrics:
            clf_model = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
            clf_model.fit(X_train, y_train)

            train_score = clf_model.score(X_train, y_train)
            val_score = clf_model.score(X_val, y_val)

            pruning["metric"].append(metric)
            pruning["train_accuracy"].append(train_score)
            pruning["val_accuracy"].append(val_score)
        return pruning
    
    #---------------Calculate Random Forest Classifier hyperparameter tuning---------------------#
    def calculate_optimal_metric_rfc(self, n_estimators: int, depth: int) -> Dict[str, List]:
        X_train, y_train = self.prepared_data["train"]
        X_val, y_val = self.prepared_data["val"]
        
        pruning = {
            "metric": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        possible_metrics = ['gini', 'entropy', 'log_loss']

        for metric in possible_metrics:
            rfc_model = RandomForestClassifier(n_estimators= n_estimators, max_depth= depth, criterion= metric, random_state=42)
            rfc_model.fit(X_train, y_train)

            train_score = rfc_model.score(X_train, y_train)
            val_score = rfc_model.score(X_val, y_val)

            pruning["metric"].append(metric)
            pruning["train_accuracy"].append(train_score)
            pruning["val_accuracy"].append(val_score)
        return pruning
    
    def calculate_optimal_n_estimators_rfc(self, metric: str, depth: int) -> Dict[str, List]:
        X_train, y_train = self.prepared_data["train"]
        X_val, y_val = self.prepared_data["val"]
        
        pruning = {
            "n_estimators": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        estimator_ranges = [1] + [i for i in range(10, 110, 10)]
        for n in estimator_ranges:
            rfc_model = RandomForestClassifier(n_estimators= n, max_depth= depth, criterion= metric, random_state=42)
            rfc_model.fit(X_train, y_train)

            train_score = rfc_model.score(X_train, y_train)
            val_score = rfc_model.score(X_val, y_val)

            pruning["n_estimators"].append(n)
            pruning["train_accuracy"].append(train_score)
            pruning["val_accuracy"].append(val_score)
        return pruning
    
    #---------------Calculate Ada Boost Classifier hyperparameter tuning---------------------#
    def calculate_optimal_estimators_gbc(self, learning_rate: float, metric: str, depth: int) -> Dict[str, List]:
        X_train, y_train = self.prepared_data["train"]
        X_val, y_val = self.prepared_data["val"]
        
        pruning = {
            "n_estimators": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        estimator_ranges = [1] + [i for i in range(10, 110, 10)]
        for n in estimator_ranges:
            weak_learner = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
            clf_ada_boost = AdaBoostClassifier(estimator= weak_learner, n_estimators= n, learning_rate= learning_rate, random_state=42)
            clf_ada_boost.fit(X_train, y_train)

            train_score = clf_ada_boost.score(X_train, y_train)
            val_score = clf_ada_boost.score(X_val, y_val)

            pruning["n_estimators"].append(n)
            pruning["train_accuracy"].append(train_score)
            pruning["val_accuracy"].append(val_score)
        return pruning
    
    def calculate_optimal_learning_rate_gbc(self, n_estimators: int, metric: str, depth: int):
        X_train, y_train = self.prepared_data["train"]
        X_val, y_val = self.prepared_data["val"]
        
        pruning = {
            "learning_rate": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

        learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5]

        for lr in learning_rates:
            weak_learner = DecisionTreeClassifier(max_depth= depth, criterion= metric, random_state= 42)
            clf_ada_boost = AdaBoostClassifier(estimator= weak_learner, n_estimators= n_estimators, learning_rate= lr, random_state=42)
            clf_ada_boost.fit(X_train, y_train)

            train_score = clf_ada_boost.score(X_train, y_train)
            val_score = clf_ada_boost.score(X_val, y_val)

            pruning["learning_rate"].append(lr)
            pruning["train_accuracy"].append(train_score)
            pruning["val_accuracy"].append(val_score)
        return pruning
    
    #-----------------------Define plot structure for different hyperparameter----------------------#
    def optimal_depth_plot_main_frame(self, depth_pruning) -> Figure:
        fig, ax = plt.subplots()
        ax.scatter(x = depth_pruning["depth"], y = depth_pruning["train_accuracy"], c= "r", label="Training Accuracy")
        ax.scatter(x = depth_pruning["depth"], y = depth_pruning["val_accuracy"], c= "b", label = "Validation Accuracy")
        ax.grid(visible= True)
        ax.set_xticks(range(1, depth_pruning["depth"][-1] + 1))

        # Make the plot more descriptive:
        ax.set_xlabel("Tree Depth")  
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy vs Tree Depth")
        ax.legend()
        return fig
    
    def optimal_metric_plot_main_frame(metric_pruning) -> Figure:
        fig, ax = plt.subplots()
        ax.plot(metric_pruning["metric"], metric_pruning["train_accuracy"], c= "r", marker = "o", label="Training Accuracy")
        ax.plot(metric_pruning["metric"], metric_pruning["val_accuracy"], c= "b", marker = "o", label = "Validation Accuracy")

        # Make the plot more descriptive:
        ax.grid(visible= True)
        ax.set_xlabel("Splitting Criterion")  
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy vs Splitting Criterion")
        ax.legend()
        return fig
    
    def optimal_estimators_main_frame(self, n_estimator_pruning) -> Figure:
        fig, ax = plt.subplots()
        ax.plot(n_estimator_pruning["n_estimators"], n_estimator_pruning["train_accuracy"], c= "r", marker = "o", label="Training Accuracy")
        ax.plot(n_estimator_pruning["n_estimators"], n_estimator_pruning["val_accuracy"], c= "b", marker = "o", label = "Validation Accuracy")
        ax.set_xticks([1] + [i for i in range(10, 110, 10)])

        # Make the plot more descriptive:
        ax.grid(visible= True)
        ax.set_xlabel("N estimators")  
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy vs n estimators")
        ax.legend()
        return fig
    
    #------------ dtc final plots-------------#
    def optimal_metric_plot_dtc(self, depth: int) -> Figure:
        metric_pruning = self.calculate_optimal_metric_dtc(depth= depth)
        figure = self.optimal_metric_plot_main_frame(metric_pruning)
        return figure
    
    def optimal_depth_dtc_plot(self, metric: str) -> Figure:
        depth_pruning = self.calculate_optimal_depth_dtc(metric= metric)
        figure = self.optimal_depth_plot_main_frame(depth_pruning)
        return figure
    
    #----------- rfc final plots--------------#
    def optimal_metric_plot_rfc(self, n_estimators: int, depth: int) -> Figure:
        metric_pruning = self.calculate_optimal_metric_rfc(n_estimators, depth)
        figure = self.optimal_metric_plot_main_frame(metric_pruning)
        return figure
    
    def optimal_n_estimators_plot_rfc(self, metric: str, depth: int) -> Figure:
        n_estimators_pruning = self.calculate_optimal_n_estimators_rfc(metric, depth)
        figure = self.optimal_estimators_main_frame(n_estimators_pruning)
        return figure
    
    #-------------gbc final plots--------------#
    def optimal_n_estimators_plot_gbc(self, learning_rate: float, metric: str, depth: int):
        n_estimators_pruning = self.calculate_optimal_estimators_gbc(learning_rate, metric, depth)
        figure = self.optimal_estimators_main_frame(n_estimators_pruning)
        return figure
    
    def optimal_learning_rate_plot_gbc(self, n_estimators: int, metric: str, depth: int):
        lr_pruning = self.calculate_optimal_learning_rate_gbc(n_estimators, metric, depth)
        fig, ax = plt.subplots()
        ax.plot(lr_pruning["learning_rate"], lr_pruning["train_accuracy"], c= "r", marker = "o", label="Training Accuracy")
        ax.plot(lr_pruning["learning_rate"], lr_pruning["val_accuracy"], c= "b", marker = "o", label = "Validation Accuracy")
        ax.set_xticks([0.01, 0.1, 0.5, 1.0, 1.5])

        # Make the plot more descriptive:
        ax.grid(visible= True)
        ax.set_xlabel("Learning Rate")  
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy vs Learning Rate")
        ax.legend()
        return fig

# Implement a loop that increases the number of random samples to see the impact on the tree:
# Important: Take the selected choises of the user (= missing values, selected features, tree parameters into account...)
# --> Is a bit tedious to implement

class ShowOverfitting:
    def __init__(self):
        pass

    def provide_dataset(self, num_rows: int, selected_features: List[str], selected_missing_method: str):
        """
        Returns the dataset with all the choices, the user has made before.
        """
        data_selection = LoadData()
        df = data_selection.normal_and_random_data(num_rows= num_rows)

        preparation = DataPreparation(data= df)

        preparation.choose_columns(selected_features)

        if selected_missing_method == "Fehlende Werte behalten":
            None
        if selected_missing_method == "Fehlende Werte entfernen":
            preparation.drop_missing_values() # <-- bug is here, dataset gets modified somehow...
        if selected_missing_method == "Median einsetzen":
            preparation.impute_missing_values()

        prepared_data = preparation.train_test_datset()

        X_train, y_train = prepared_data["train"]
        X_test, y_test = prepared_data["test"]

        return X_train, y_train, X_test, y_test 

    def calc_for_dtc(self, user_stop: int, selected_features: List[str], 
                           selected_missing_method: str, selected_metric: str, 
                           selected_depth: int | None) -> Dict[str, List]:
                           
        capture_data = {
            "amount_random_data": [],
            "train_accuracy": [],
            "test_accuracy": []
        }

        for num_rows in range(0, user_stop + 500, 500):
            dtc_model = DecisionTreeClassifier(criterion= selected_metric, max_depth= selected_depth)
            X_train, y_train, X_test, y_test = self.provide_dataset(num_rows, selected_features, selected_missing_method)

            dtc_model.fit(X_train, y_train)

            train_score = dtc_model.score(X_train, y_train)
            test_score = dtc_model.score(X_test, y_test)

            capture_data["amount_random_data"].append(num_rows)
            capture_data["train_accuracy"].append(train_score)
            capture_data["test_accuracy"].append(test_score)
        return capture_data
    
    def calc_for_rfc(self, user_stop: int, selected_features: List[str], 
                           selected_missing_method: str, selected_metric: str, 
                           selected_estimators: int, selected_depth: int) -> Dict[str, List]:
        
        captured_data = {
            "amount_random_data": [],
            "train_accuracy": [],
            "test_accuracy": []
        }

        for num_rows in range(0, user_stop, 500):
            rfc_model = RandomForestClassifier(n_estimators= selected_estimators, criterion= selected_metric, max_depth= selected_depth)

            X_train, y_train, X_test, y_test = self.provide_dataset(num_rows, selected_features, selected_features, selected_missing_method)

            train_score = rfc_model.score(X_train, y_train)
            test_score = rfc_model.score(X_test, y_test)

            captured_data["amount_random_data"].append(num_rows)
            captured_data["train_accuracy"].append(train_score)
            captured_data["test_accuracy"].append(test_score)
        return captured_data
    
    def calc_for_gbc(self, user_stop: int, selected_features: List[str], 
                           selected_missing_method: str, selected_learning_rate: float, selected_metric: str, 
                           selected_estimators: int, selected_depth: int):
        captured_data = {
            "amount_random_data": [],
            "train_accuracy": [],
            "test_accuracy": []
        }

        for num_rows in range(0, user_stop, 500):
            weak_learner = DecisionTreeClassifier(max_depth= selected_depth, criterion= selected_metric, random_state= 42)
            clf_ada_boost = AdaBoostClassifier(estimator= weak_learner, n_estimators= selected_estimators, 
                                               learning_rate= selected_learning_rate, random_state=42)

            X_train, y_train, X_test, y_test = self.provide_dataset(num_rows, selected_features, selected_features, selected_missing_method)

            train_score = clf_ada_boost.score(X_train, y_train)
            test_score = clf_ada_boost.score(X_test, y_test)

            captured_data["amount_random_data"].append(num_rows)
            captured_data["train_accuracy"].append(train_score)
            captured_data["test_accuracy"].append(test_score)
        return captured_data
    
    def plot_main_frame(self, captured_data):
        fig, ax = plt.subplots()
        ax.scatter(x = captured_data["amount_random_data"], y = captured_data["train_accuracy"], c= "r", label="Training Accuracy")
        ax.scatter(x = captured_data["amount_random_data"], y = captured_data["test_accuracy"], c= "b", label = "Test Accuracy")
        ax.grid(visible= True)
        ax.set_xticks(range(0, captured_data["amount_random_data"][-1] + 500, 500))

        # Make the plot more descriptive:
        ax.set_xlabel("Additional Random Data")  
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy vs Increasingly Random Data")
        ax.legend()
        return fig

    def create_plot_dtc(self, user_stop: int, selected_features: List[str], 
                           selected_missing_method: str, selected_metric: str, 
                           selected_depth: int):
        
        captured_data = self.calc_for_dtc(user_stop, selected_features, 
                                            selected_missing_method, selected_metric, 
                                            selected_depth)
        
        figure = self.plot_main_frame(captured_data)
        return figure

    def create_plot_rfc(self, user_stop: int, selected_features: List[str], 
                           selected_missing_method: str, selected_metric: str, 
                           selected_estimators: int, selected_depth: int) -> Figure:
        
        captured_data = self.calc_for_rfc(user_stop, selected_features, selected_missing_method,
                                          selected_metric, selected_estimators, selected_depth)
        figure = self.plot_main_frame(captured_data)
        return figure
    
    def create_plot_btc(self, user_stop: int, selected_features: List[str], 
                        selected_missing_method: str, selected_learning_rate: float, selected_metric: str, 
                        selected_estimators: int, selected_depth: int) -> Figure:
        
        captured_data = self.calc_for_gbc(user_stop, selected_features, selected_missing_method, 
                                          selected_learning_rate, selected_metric, selected_estimators, selected_depth)
        figure = self.plot_main_frame(captured_data)
        return figure      

if __name__ == "__main__":
    pass
    # load = LoadData()
    # org_and_random = load.normal_and_random_data(1000)
    # print(org_and_random.shape)
    # print(org_and_random.describe())

    overfit = ShowOverfitting()
    selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    figure = overfit.create_plot_dtc(10000, selected_features, "Fehlende Werte entfernen", "gini", 10)
    plt.show()

    