import streamlit as st
from backend import LoadData, DataPreparation, ModelTrainingAndEvaluation, ModelTuning

def initialize_session_states() -> None:
    """
    Initialize variables in st.session state to ensure the correct flow of the application.
    """
    if "dataset_is_selected" not in st.session_state:
        st.session_state["dataset_is_selected"] = False

    if "features_confirmed" not in st.session_state:
        st.session_state["features_confirmed"] = False

    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = None

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

def user_selected_random_forest():
    pass

# Tasks: 
# Use different functions for the layers to keep the code clean and readable
# Maybe implement caching to keep the app fast?!
# Maybe implement, that stuff "resets" if things were changed a few steps before.


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
    

    # Start application:
    st.title("Entscheidungsbäume Playground")
    st.subheader("Datenauswahl")

    #-------------------User selects the dataset--------------------------#
    selected_dataset = st.selectbox(label = "Wähle die Art der Daten aus, mit denen du experimentiren möchtest.", 
                                    options= ["Original Datensatz", "Augmentierter Datensatz","Zufälliger Datensatz"], 
                                    index = None)

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
            #------------User can decide how many rows of data he wants-----#
            num_of_rows = st.number_input("Wähle die Anzahl an Reihen des Datensatzes", min_value= 0, value= None, step = 1)

            #------------Only passes if the number of rows was specified----#
            if num_of_rows is not None:
                if selected_dataset == "Augmentierter Datensatz":
                    df = data_selection.synthetic_data(num_rows= num_of_rows)
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
            selected_features = st.multiselect("Wähle Features für das Training:", options= available_features, default= default_features)

            # Save this in the session state
            if st.button("Auswahl bestätigen"):
                st.session_state["features_confirmed"] = True
            
            #--------------Only passes if the features are selected--------#
            if st.session_state["features_confirmed"]:
                preparation.choose_columns(selected_cols= selected_features)

                # Control the entry to the Model Training section:
                data_is_ready = True # <-- not sure if this needs to be stored in the session state

                # Manage data if it has missing values.
                if preparation.check_for_missing_values_in_age():
                    data_is_ready = False

                    st.info("Es existieren fehlende Werte in der Spalte Age!")
                    selected_method = st.selectbox(label= "Wähle eine Methode mit den fehlenden Wetren umzugehen", 
                                                options= ["Fehlende Werte behalten", "Fehlende Werte entfernen", "Median einsetzen"], 
                                                index = None)

                    if selected_method is not None:
                        if selected_method == "Fehlende Werte behalten":
                            pass # Just do nothing
                        elif selected_method == "Fehlende Werte entfernen":
                            rows_before = preparation.df.shape[0]
                            preparation.drop_missing_values()
                            rows_after = preparation.df.shape[0]
                            notification = f"{rows_before - rows_after} Reihen wurden entfernt."
                            st.info(notification)
                        elif selected_method == "Median einsetzen":
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
                        if st.button("Normaler Decision Tree"):
                            st.session_state["selected_model"] = "dtc"
                            st.session_state["model_trained"] = False
                    with col2:
                        if st.button("Random Forest"):
                            st.session_state["selected_model"] = "rfc"
                            st.session_state["model_trained"] = False
                    with col3:
                        if st.button("Boosted Tree"):
                            st.session_state["selected_model"] = "btc"
                            st.session_state["model_trained"] = False
                    
                    #------------------------Pases if a model was selected-------------------------#
                    if st.session_state["selected_model"] is not None:
                        # Initialize the class:
                        training = ModelTrainingAndEvaluation(prepared_data)

                        #---------------------Training of the Normal Decision Tree----------------------------#
                        if st.session_state["selected_model"] == "dtc":

                            st.subheader("Normaler Decision Tree")
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ["gini", "entropy", "log_loss"]
                                selected_metric = st.selectbox("Wähle eine Metrik für das Model", options= options, index= 0)
                            with col2:
                                selected_depth = st.number_input("Wähle die maximale Tiefe (optional)", value= None, step= 1)
                            
                            #------------------Pases if the model was trained--------------------------------#
                            if st.button("Model trainieren") or st.session_state["model_trained"]:
                                trained_model = training.train_basic_decision_tree(depth= selected_depth, metric= selected_metric)
                                training_accuracy = training.evaluate_the_training_data(model = trained_model)
                                test_accuracy = training.evaluate_with_test_dataset(model = trained_model)

                                st.session_state["model_trained"] = True
                                
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
                                    st.info("Um die Hyperparameter zu tunen, werden die Daten folgendermaßen aufgeilt:")

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
                        
                        #------------------------Training of the Random Forest-----------------------------#
                        if st.session_state["selected_model"] == "rfc":
                            st.subheader("Random Forest")
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ["gini", "entropy", "log_loss"]
                                selected_metric = st.selectbox("Wähle eine Metrik für das Model", options= options, index= 0)
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
                                        print(trained_model)
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
                                
                                # Add hyperparameter tuning here

                            
                        if st.session_state["selected_model"] == "btc":
                            st.write("Not yet developed")