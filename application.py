import streamlit as st
from backend import LoadData, DataPreparation, ModelTrainingAndEvaluation, ModelTuning, ShowOverfitting

if __name__ == "__main__":
    # Initialize session states:
    if "dataset_is_selected" not in st.session_state:
        st.session_state["dataset_is_selected"] = False

    if "features_confirmed" not in st.session_state:
        st.session_state["features_confirmed"] = False

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = None

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    
    if "disable_data_functionalities" not in st.session_state:
        st.session_state["disable_data_functionalities"] = False
    
    # Deactivate it always...
    st.session_state["activate_additional_feature"] = False
    
    if "missing_values_exist" not in st.session_state:
        st.session_state["missing_values_exist"] = False
    
    # Start application:
    st.title("Entscheidungsbäume Playground")
    st.subheader("Datenauswahl")

    #-------------------User selects the dataset--------------------------#
    selected_dataset = st.selectbox(label = "Wähle die Art der Daten aus, mit denen du experimentieren möchtest.", 
                                    options= ["Original Datensatz", "Zufälliger Datensatz", "Original + Zufällig"], 
                                    index = None, disabled= st.session_state["disable_data_functionalities"])

    #-------Pases if a dataset was selected--------#
    if selected_dataset is not None:
        data_selection = LoadData()
        #---------------Manage the selected datset-------------------------# 
        if selected_dataset == "Original Datensatz":
            df = data_selection.normal_dataset()
            st.success("Der Datensatz wurde erfolgreich ausgewählt!")
            st.dataframe(df)
            st.write(f"Der Datensatz hat {df.shape[0]} Reihen und {df.shape[1]} Spalten.")
            # Flow control:
            st.session_state["dataset_is_selected"] = True
        else:
            # Set this to False to prevent Errors
            st.session_state["dataset_is_selected"] = False

            #------------User can decide how many rows of data he wants-----#
            num_of_rows = st.number_input("Wähle die Anzahl an zufällig generierten Reihen des Datensatzes", 
                                          min_value= 0, value= None, step = 1, disabled= st.session_state["disable_data_functionalities"])

            #------------Only passes if the number of rows was specified----#
            if num_of_rows is not None:
                if selected_dataset == "Original + Zufällig":
                    df = data_selection.normal_and_random_data(num_rows= num_of_rows)
                    st.session_state["activate_additional_feature"] = True # <-- activates a special feature after model training

                if selected_dataset == "Zufälliger Datensatz":
                    df = data_selection.random_data(num_rows= num_of_rows)
                st.success("Der Datensatz wurde erfolgreich ausgewählt!")
                st.dataframe(df)
                st.write(f"Der Datensatz hat {df.shape[0]} Reihen und {df.shape[1]} Spalten.")
                # Flow control:
                st.session_state["dataset_is_selected"] = True
        st.divider()

        #-----------------Pases if a dataset was selected-------------------#
        if st.session_state["dataset_is_selected"]:
            
            #-------------------Data Preparation----------------------------#
            st.subheader("Datenvorbereitung")
            preparation = DataPreparation(data= df)

            available_features = preparation.available_cols()
            optimal_features = preparation.default_features()

            #-------------------Let the user choose the Features------------#
            default_features = [feature for feature in optimal_features if feature in available_features]
            selected_features = st.multiselect("Wähle Features für das Training:", 
                                               options= available_features, default= default_features, 
                                               disabled= st.session_state["disable_data_functionalities"])

            # Save this in the session state
            if st.button("Auswahl bestätigen", disabled= st.session_state["disable_data_functionalities"]):
                st.session_state["features_confirmed"] = True
            
            #--------------Only passes if the features are selected--------#
            if st.session_state["features_confirmed"]:
                preparation.choose_columns(selected_cols= selected_features)

                # Control the entry to the Model Training section:
                data_is_ready = True

                # Manage data if it has missing values.
                if preparation.check_for_missing_values_in_age():
                    data_is_ready = False

                    num_of_missing_values = preparation.num_of_missing_values_in_age()

                    st.info(f"Es existieren {num_of_missing_values} fehlende Werte in der Spalte Age!")
                    selected_missing_method = st.selectbox(label= "Wähle eine Methode mit den fehlenden Werten umzugehen", 
                                                            options= ["Fehlende Werte behalten", "Fehlende Werte entfernen", "Median einsetzen"], 
                                                            index = None, disabled= st.session_state["disable_data_functionalities"])

                    if selected_missing_method is not None:
                        if selected_missing_method == "Fehlende Werte behalten":
                            st.session_state["missing_values_exist"] = True
                        elif selected_missing_method == "Fehlende Werte entfernen":
                            rows_before = preparation.df.shape[0]
                            preparation.drop_missing_values()
                            rows_after = preparation.df.shape[0]
                            notification = f"{rows_before - rows_after} Reihen wurden entfernt."
                            st.info(notification)
                        elif selected_missing_method == "Median einsetzen":
                            preparation.impute_missing_values()
                        
                        data_is_ready = True
                    
                    st.divider()
                
                #----------------------Pases if the features are procesed-------------#
                if data_is_ready:
                    prepared_data = preparation.train_test_datset()

                    #------------------------Model Training----------------------------#
                    st.subheader("Model Training")

                    # Let the user select the Model they want to train their data with.
                    col1, col2, col3 = st.columns(3)

                    # Implement the that the session state variables are reset (= set to false) after pushing the buttons!!!
                    with col1:
                        if st.button("Normaler Decision Tree", width= "stretch"):
                            st.session_state["selected_model"] = "dtc"
                            st.session_state["model_trained"] = False
                    with col2:
                        if st.button("Random Forest", width= "stretch"):
                            st.session_state["selected_model"] = "rfc"
                            st.session_state["model_trained"] = False
                    with col3: 
                        if st.button("Gradient Boosted Tree", width= "stretch", disabled= st.session_state["missing_values_exist"]):
                            st.session_state["selected_model"] = "gbc"
                            st.session_state["model_trained"] = False
                    
                    #------------------------Pases if a model was selected-------------------------#
                    if st.session_state["selected_model"] is not None:
                        # Maybe introduce a new session state vraible here:
                        st.session_state["disable_data_functionalities"] = True

                        # Initialize the class:
                        training = ModelTrainingAndEvaluation(prepared_data)
                        #---------------------Training of the Normal Decision Tree----------------------------#
                        if st.session_state["selected_model"] == "dtc":

                            st.subheader("Normaler Decision Tree")
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ["gini", "entropy", "log_loss"]
                                selected_metric = st.selectbox("Wähle ein Splitkriterium für das Modell", options= options, index= 0)
                            with col2:
                                selected_depth = st.number_input("Wähle die maximale Tiefe des Modells (optional)", value= None, step= 1)
                            
                            #------------------Pases if the model was trained--------------------------------#
                            if st.button("Model trainieren") or st.session_state["model_trained"]:
                                trained_model = training.train_basic_decision_tree(depth= selected_depth, metric= selected_metric)
                                training_accuracy = training.evaluate_the_training_data(model = trained_model)
                                test_accuracy = training.evaluate_with_test_dataset(model = trained_model)

                                st.session_state["model_trained"] = True # <-- should rerun immediately to disable the data selection.
                                
                                st.write(f"Die Genauigkeit des Decision Trees mit den Trainings Daten beträgt: {training_accuracy:.3f}")
                                st.write(f"Die Genauigkeit des Decision Trees mit den Test Daten beträgt: {test_accuracy:.3f}") # round after three digits.

                                #----------------Visualize the Decision Tree------#
                                with st.expander("Zeige den Entscheidungsbaum"):
                                    figure = training.visualize_tree_dtc(trained_model)
                                    st.pyplot(figure)
                                
                                #-----------------SHAP Analysis--------------------#
                                with st.expander("Shap Analysis (Feature Importance)"):
                                    figure = training.shap_plot(trained_model)
                                    st.pyplot(figure)

                                #-----------Heatmap--------------------------------#
                                with st.expander("Zeige die zugehörige Heatmap"):
                                    figure = training.create_heatmap(trained_model)
                                    st.pyplot(figure)

                                #-------------5 Fold Validation---------------------#
                                with st.expander("5-Fold Validation"):
                                    X, y = preparation.X_y_dataset()
                                    output_scores = training.k_fold_eval_dtc(X, y, selected_depth, selected_metric)
                                    st.table(output_scores)
                                
                                #---------------Hyperparameter tuning----------------#
                                with st.expander("Model Hyperparameter Tuning"):
                                    information_text = """  
                                    Um die Hyperparameter zu tunen, werden die Daten folgendermaßen aufgeilt:

                                    - Trainingsdaten: 80% → davon 20% Validierungsdaten  
                                    - Testdaten: 20%  
                                    """  
                                    st.info(information_text)

                                    # Initialize the tuning class
                                    prepared_data_with_val = preparation.train_val_test_dataset()
                                    tuning = ModelTuning(prepared_data_with_val)

                                    selected_hyperparam = st.selectbox(label = "Wähle den Hyperparamter, der getuned werden soll", 
                                                                        options= ["Baum Tiefe", "Splitting Metric"], 
                                                                        index = None)
                                    if selected_hyperparam is not None:
                                        if selected_hyperparam == "Baum Tiefe":
                                            st.info("Die anderen Hyperparameter werden vom Training übernommen.")
                                            fig = tuning.optimal_depth_dtc_plot(metric = selected_metric)
                                            st.pyplot(fig= fig)

                                        if selected_hyperparam == "Splitting Metric":
                                            st.info("Die anderen Hyperparameter werden vom Training übernommen.")
                                            fig = tuning.optimal_metric_plot_dtc(depth = selected_depth)
                                            st.pyplot(fig= fig)

                                if st.session_state["activate_additional_feature"]:
                                    with st.expander("Performance mit Erhöhung des Anteils der zufälligen Daten"):
                                        stop_range = [i for i in range(0, 10500, 500)]
                                        user_stop = st.selectbox("Wähle die Maximale Anzahl an zusätzlichen zufälligen Daten", 
                                                                 options= stop_range, index= None)
                                        if user_stop is not None:
                                            overfitting = ShowOverfitting()
                                            fig = overfitting.create_plot_dtc(user_stop, selected_features, selected_missing_method, 
                                                                              selected_metric, selected_depth)
                                            st.pyplot(fig)
                        
                        #------------------------Training of the Random Forest-----------------------------#
                        if st.session_state["selected_model"] == "rfc":
                            st.subheader("Random Forest")
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ["gini", "entropy", "log_loss"]
                                selected_metric = st.selectbox("Wähle ein Splitkriterium für das Modell", options= options, index= 0)
                            with col2:
                                selected_depth = st.number_input("Wähle die maximale Tiefe (optional)", value= 4, step= 1)
                                selected_estimators = st.number_input("Wähle die Anzahl an Bäumen", value= 100, step= 1)
                            
                            if st.button("Model trainieren") or st.session_state["model_trained"]:
                                trained_model = training.train_random_forest(n_estimators= selected_estimators,
                                                                            depth= selected_depth, 
                                                                            metric= selected_metric)
                                st.session_state["model_trained"] = True

                                training_accuracy = training.evaluate_the_training_data(model = trained_model)
                                test_accuracy = training.evaluate_with_test_dataset(model = trained_model)

                                st.write(f"Die Genauigkeit des Random Forests mit den Trainings Daten beträgt: {training_accuracy:.3f}")
                                st.write(f"Die Genauigkeit des Random Forests mit den Test Daten beträgt: {test_accuracy:.3f}")

                                with st.expander("Visualisiere einzelne Bäume aus dem Ensemble"):
                                    selected_tree_n= st.number_input("Gebe den Index des Baumes an, welchen du visualisieren möchtest", 
                                                                        min_value= 0, 
                                                                        max_value= selected_estimators - 1,
                                                                        value = None)
                                    if selected_tree_n is not None:
                                        figure = training.visualize_tree_ensemble(trained_model, n = selected_tree_n)
                                        st.pyplot(figure)
                                
                                with st.expander("Shap Analysis (Feature Importance)"):
                                    figure = training.shap_plot(trained_model)
                                    st.pyplot(figure)

                                with st.expander("Zeige die zugehörige Heatmap"):
                                    figure = training.create_heatmap(trained_model)
                                    st.pyplot(figure)

                                with st.expander("5-Fold Validation"):
                                    X, y = preparation.X_y_dataset()
                                    output_scores = training.k_fold_eval_rfc(X, y, selected_estimators, selected_depth, selected_metric)
                                    st.table(output_scores)
                                
                                with st.expander("Model Hyperparameter Tuning"):
                                    information_text = """  
                                    Um die Hyperparameter zu tunen, werden die Daten folgendermaßen aufgeilt:

                                    - Trainingsdaten: 80% → davon 20% Validierungsdaten  
                                    - Testdaten: 20%  
                                    """  
                                    st.info(information_text)

                                    # Initialize the tuning class
                                    prepared_data_with_val = preparation.train_val_test_dataset()
                                    tuning = ModelTuning(prepared_data_with_val)

                                    selected_hyperparam = st.selectbox(label = "Wähle den Hyperparamter, der getuned werden soll", 
                                                                        options= ["N Estimators", "Splitting Metric"], 
                                                                        index = None)
                                    if selected_hyperparam is not None:
                                        if selected_hyperparam == "N Estimators":
                                            st.info("Die anderen Hyperparameter werden vom Training übernommen.")
                                            fig = tuning.optimal_n_estimators_plot_rfc(metric = selected_metric, depth= selected_depth)
                                            st.pyplot(fig= fig)

                                        if selected_hyperparam == "Splitting Metric":
                                            st.info("Die anderen Hyperparameter werden vom Training übernommen.")
                                            fig = tuning.optimal_metric_plot_rfc(n_estimators= selected_estimators, depth= selected_depth)
                                            st.pyplot(fig= fig)
                                
                                if st.session_state["activate_additional_feature"]:
                                    with st.expander("Performance mit Erhöhung des Anteils der zufälligen Daten"):
                                        stop_range = [i for i in range(0, 10500, 500)]
                                        user_stop = st.selectbox("Wähle die Maximale Anzahl an zusätzlichen zufälligen Daten", 
                                                                 options= stop_range, index= None)
                                        if user_stop is not None:
                                            overfitting = ShowOverfitting()
                                            fig = overfitting.create_plot_rfc(user_stop, selected_features, selected_missing_method, 
                                                                              selected_metric, selected_estimators, selected_depth)
                                            st.pyplot(fig)

                        #------------------------------------Training of the Gradient Boosted Classifier-------------------------#
                        if st.session_state["selected_model"] == "gbc":
                            st.subheader("Gradient Boosted Classifier")
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ['friedman_mse', 'squared_error']
                                selected_metric = st.selectbox("Wähle ein Splitkriterium für das Modell", options= options, index= 0)

                                selected_learning_rate = st.number_input("Wähle die Lernrate des Modells", 
                                                                         min_value= 0.01, max_value= 2.0, value = 0.1)
                            with col2:
                                selected_depth = st.number_input("Wähle die maximale Tiefe (optional)", value= 4, step= 1)
                                selected_estimators = st.number_input("Wähle die Anzahl an Bäumen", value= 50, step= 1)

                                loss_options = ['log_loss', 'deviance', 'exponential']
                                selected_loss = st.selectbox("Wähle eine Loss function", options= loss_options, index= 0)
                            
                            if st.button("Modell trainieren") or st.session_state["model_trained"]:
                                trained_model = training.train_gradient_boost_tree(loss_function= selected_loss, lr= selected_learning_rate, 
                                                                                   estimators= selected_estimators, metric= selected_metric, depth= selected_depth)
                                st.session_state["model_trained"] = True

                                training_accuracy = training.evaluate_the_training_data(model = trained_model)
                                test_accuracy = training.evaluate_with_test_dataset(model = trained_model)

                                st.write(f"Die Genauigkeit des Gradient Boosted Classifiers mit den Trainings Daten beträgt: {training_accuracy:.3f}")
                                st.write(f"Die Genauigkeit des Gradient Boosted Classifiers mit den Test Daten beträgt: {test_accuracy:.3f}")

                                with st.expander("Visualisiere einzelne Bäume (= weak Learner) aus dem Ensemble"):
                                    selected_tree_n= st.number_input("Gebe den Index des Baumes an, welchen du visualisieren möchtest", 
                                                                        min_value= 0, 
                                                                        max_value= selected_estimators - 1,
                                                                        value = None)
                                    if selected_tree_n is not None:
                                        figure = training.visualize_tree_ensemble(trained_model, n = selected_tree_n)
                                        st.pyplot(figure)
                                
                                with st.expander("Shap Analysis (Feature Importance)"):
                                    figure = training.shap_plot(trained_model)
                                    st.pyplot(figure)

                                with st.expander("Zeige die zugehörige Heatmap"):
                                    figure = training.create_heatmap(trained_model)
                                    st.pyplot(figure)

                                with st.expander("5-Fold Validation"):
                                    X, y = preparation.X_y_dataset()
                                    output_scores = training.k_fold_eval_gbc(X, y, selected_loss, selected_learning_rate, selected_estimators, selected_metric, selected_depth)
                                    st.table(output_scores)
                                
                                with st.expander("Model Hyperparameter Tuning"):
                                    information_text = """  
                                    Um die Hyperparameter zu tunen, werden die Daten folgendermaßen aufgeilt:

                                    - Trainingsdaten: 80% → davon 20% Validierungsdaten  
                                    - Testdaten: 20%  
                                    """  
                                    st.info(information_text)

                                    # Initialize the tuning class
                                    prepared_data_with_val = preparation.train_val_test_dataset()
                                    tuning = ModelTuning(prepared_data_with_val)

                                    selected_hyperparam = st.selectbox(label = "Wähle den Hyperparamter, der getuned werden soll", 
                                                                        options= ["N Estimators", "Learning Rate"], 
                                                                        index = None)
                                    if selected_hyperparam is not None:
                                        if selected_hyperparam == "N Estimators":
                                            st.info("Die anderen Hyperparameter werden vom Training übernommen.")
                                            fig = tuning.optimal_n_estimators_plot_gbc(selected_loss, selected_learning_rate, selected_metric, selected_depth)
                                            st.pyplot(fig= fig)

                                        if selected_hyperparam == "Learning Rate":
                                            st.info("Die anderen Hyperparameter werden vom Training übernommen.")
                                            fig = tuning.optimal_learning_rate_plot_gbc(selected_loss, selected_estimators, selected_metric, selected_depth)
                                            st.pyplot(fig= fig)
                                
                                if st.session_state["activate_additional_feature"]:
                                    with st.expander("Performance mit Erhöhung des Anteils der zufälligen Daten"):
                                        stop_range = [i for i in range(0, 10500, 500)]
                                        user_stop = st.selectbox("Wähle die Maximale Anzahl an zusätzlichen zufälligen Daten", 
                                                                 options= stop_range, index= None)
                                        if user_stop is not None:
                                            overfitting = ShowOverfitting()
                                            fig = overfitting.create_plot_gbc(user_stop, selected_features, selected_missing_method, selected_loss,
                                                                              selected_learning_rate, selected_metric, selected_estimators, selected_depth)
                                            st.pyplot(fig)
                    st.divider()

                    #----------------------Possibility to reset the data configs------------------#
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        if st.button("Daten Auswahl/ Vorbereitung zurücksetzen", width= "stretch"):
                            for key in st.session_state.keys():  
                                del st.session_state[key]  
                            st.rerun()
    