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

    # Let the user choose the type of data he wants to experiment with:
    selected_dataset = st.selectbox(label = "Wähle die Art der Daten aus, mit denen du experimentiren möchtest.", 
                                    options= ["Original Datensatz", "Augmentierter Datensatz","Zufälliger Datensatz"], 
                                    index = None)

    # A specific dataset was selected:
    if selected_dataset is not None:
        data_selection = LoadData()
        # Manage the different cases:
        if selected_dataset == "Original Datensatz":
            df = data_selection.normal_dataset()
            st.success("Der Datensatz wurde erfolgreich ausgewählt!")
            st.dataframe(df)
            st.write(f"Der Datensatz hat {df.shape[0]} Reihen und {df.shape[1]} Spalten.")
            # Flow control:
            st.session_state["dataset_is_selected"] = True
        else:
            # Let the user choose the desired number of rows!
            num_of_rows = st.number_input("Wähle die Anzahl an Reihen des Datensatzes", min_value= 0, value= None, step = 1)

            # Problem: The dataset gets generated again after each interaction... (fixed with a random seed)
            # Add another flow control which is like: "Already generated" (Hopefully no conflicts because the class "resets")
            # Or just use a seed to keep the generated data the same after each rerun
            if num_of_rows is not None: # The col number was selected
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

        # Insert a condition that it is only rendered once the step above is ready.
        if st.session_state["dataset_is_selected"]:
            # Next layer: Data preparation
            st.subheader("Datenvorbereitung")
            preparation = DataPreparation(data= df)
            available_features = preparation.available_cols()
            default_features = preparation.default_features()

            # User can select the features he/she wants:
            selected_features = st.multiselect("Wähle Features für das Training:", options= available_features, default= default_features)

            # Save this in the session state
            if st.button("Auswahl bestätigen"):
                st.session_state["features_confirmed"] = True
            
            # Features are selected and confirmed:
            if st.session_state["features_confirmed"]: # <-- probabely use another variable here to control the flow state...
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
                
                if data_is_ready: # <-- probabely use another variable here to control the flow state...
                    prepared_data = preparation.train_test_datset()

                    # Next Layer: Model training
                    st.subheader("Model Training")

                    # Let the user select the Model they want to train their data with.
                    col1, col2, col3 = st.columns(3)

                    # Not sure wheter to disable them after a model was selected.
                    with col1:
                        if st.button("Normaler Decision Tree"):
                            st.session_state["selected_model"] = "dtc"
                    with col2:
                        if st.button("Random Forest"):
                            st.session_state["selected_model"] = "rfc"
                    with col3:
                        if st.button("Boosted Tree"):
                            st.session_state["selected_model"] = "btc"
                    
                    # Model was selected:
                    if st.session_state["selected_model"] is not None: # <-- probabely use another variable here to control the flow state...
                        # Initialize the class:
                        training = ModelTrainingAndEvaluation(prepared_data)

                        # Training of the "normal" decision tree
                        if st.session_state["selected_model"] == "dtc":
                            st.subheader("Normaler Decision Tree")
                            # Let the user choose the metric: (maybe save it somewhere for the tuning later)
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ["gini", "entropy", "log_loss"]
                                selected_metric = st.selectbox("Wähle eine Metrik für das Model", options= options, index= 0)
                            with col2:
                                selected_depth = st.number_input("Wähle die maximale Tiefe (optional)", value= None, step= 1)
                            
                            if st.button("Model trainieren"):
                                trained_model = training.train_basic_decision_tree(depth= selected_depth, metric= selected_metric)
                                accuracy = training.evaluate_with_test_dataset(model = trained_model)

                                st.write(f"Die Genauigkeit des Decision Trees mit den Test Daten beträgt: {accuracy:.3f}") # round after three digits.

                                with st.expander("Zeige den Entscheidungsbaum"):
                                    figure = training.visualize_tree_dtc(trained_model)
                                    st.pyplot(figure)

                                with st.expander("Zeige die zugehörige Heatmap"):
                                    figure = training.create_heatmap(trained_model)
                                    st.pyplot(figure)
                                st.session_state["model_trained"] = True

                                with st.expander("5-Fold Validation"):
                                    X, y = preparation.X_y_dataset()
                                    output_scores = training.k_fold_eval_dtc(X, y, selected_depth, selected_metric)
                                    st.table(output_scores)
                        
                        # Training of the random forest:
                        if st.session_state["selected_model"] == "rfc":
                            st.subheader("Random Forest")
                            col1, col2 = st.columns(2)
                            with col1:
                                options = ["gini", "entropy", "log_loss"]
                                selected_metric = st.selectbox("Wähle eine Metrik für das Model", options= options, index= 0)
                            with col2:
                                selected_depth = st.number_input("Wähle die maximale Tiefe (optional)", value= None, step= 1)
                                selected_estimators = st.number_input("Wähle die Anzahl an Bäumen", value= 100, step= 1)
                            
                            if st.button("Model trainieren"):
                                trained_model = training.train_random_forest(n_estimators= selected_estimators,
                                                                            depth= selected_depth, 
                                                                            metric= selected_metric)
                                accuracy = training.evaluate_with_test_dataset(model = trained_model)
                                st.write(f"Die Genauigkeit des Random Forests mit den Test Daten beträgt: {accuracy:.3f}") # round after three digits.
                                # Maybe add the heatmap here...
                                with st.expander("Zeige die zugehörige Heatmap"):
                                    figure = training.create_heatmap(trained_model)
                                    st.pyplot(figure)

                                with st.expander("5-Fold Validation"):
                                    X, y = preparation.X_y_dataset()
                                    output_scores = training.k_fold_eval_rfc(X, y, selected_estimators, selected_depth, selected_metric)
                                    st.table(output_scores)

                                st.session_state["model_trained"] = True