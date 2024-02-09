import pandas as pd
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from utils import calculate_kl_divergence
import streamlit as st
import matplotlib.pyplot as plt
from visualization import visualize_distributions

def generate_synthetic_data(df, column, column_type, train_params, model_params, model_name, sample_size):
    synthesizer = RegularSynthesizer(modelname=model_name, model_parameters=model_params)
    synthesizer.fit(data=df, train_arguments=train_params, num_cols=[column] if column_type == 'numerical' else [], cat_cols=[column] if column_type == 'categorical' else [])
    synthetic_data = synthesizer.sample(len(df))
    return synthetic_data

def iterative_synthetic_data_generation(df, column, column_type, train_params, model_params, model_name, sample_size, visualization_mode):
    total_samples = len(df)
    iterations = total_samples // sample_size
    collected_data = pd.DataFrame(columns=df.columns)
    remaining_df = df.copy()
    kl_divergences = []
    for i in range(iterations):
        new_samples = remaining_df.sample(n=sample_size, random_state=42)
        collected_data = pd.concat([collected_data, new_samples])
        remaining_df = remaining_df.drop(new_samples.index)
        synthetic_data = generate_synthetic_data(collected_data, column, column_type, train_params, model_params, model_name, sample_size)
        kl_div = calculate_kl_divergence(df[column], synthetic_data[column])
        kl_divergences.append(kl_div)
        
        # Streamlit to update the plots and KL divergence display
        with st.container():
            fig, ax = plt.subplots(figsize=(10, 6))
            visualize_distributions(df[column], synthetic_data[column], column, i+1, kl_divergences, visualization_mode, ax=ax)
            st.pyplot(fig)
            st.write(f"Iteration {i+1} - KL Divergence: {kl_div}")
