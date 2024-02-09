import streamlit as st
import pandas as pd
from visualization import visualize_distributions
from data_generation import iterative_synthetic_data_generation, generate_synthetic_data
from utils import StreamlitConsole, extract_last_epoch_losses

# Streamlit app starts here
st.title('Synthetic Data Generator Evaluative Visualization Tool')
# Create a placeholder for the plot and KL divergence display
plot_placeholder = st.empty()
kl_sidebar = st.sidebar.empty()
console_log_placeholder = st.sidebar.empty()  # Placeholder for console log
progress_bar_placeholder = st.empty()
percentage_indicator_placeholder = st.empty()

# Inputs for the data generation
uploaded_file = st.file_uploader("Choose an excel file", type="xlsx")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    column = st.text_input("Column name:", 'age')
    column_type = st.selectbox("Column type:", ['numerical', 'categorical'])
    visualization_mode = st.selectbox("Visualization mode:", ['line', 'color', 'both'])
    epochs = st.number_input("Number of epochs:", min_value=1, value=10)
    batch_size = st.number_input("Batch size:", min_value=1, value=500)
    lr = st.number_input("Learning rate:", min_value=0.0001, max_value=None, step=0.0001, format="%.4f")
    beta_1 = st.number_input("Beta 1:", value=0.6)
    beta_2 = st.number_input("Beta 2:", value=0.9)
    model_name = st.text_input("Model name:", value='ctgan')
    sample_size = st.number_input("Sample size:", min_value=1, value=500)
    # Training and model parameters
    train_params = TrainParameters(epochs=epochs)
    model_params = ModelParameters(batch_size=batch_size, lr=lr, betas=(beta_1, beta_2))
    if st.button('Generate Synthetic Data'):
        total_iterations = len(df) // sample_size
        with StreamlitConsole() as console:
            kl_divergences = []
            sidebar_content = ""  # Initialize empty string to store sidebar content
            for i in range(1, (len(df) // sample_size) + 1):
                synthetic_data = generate_synthetic_data(df.sample(n=sample_size*i, random_state=42), column, column_type, train_params, model_params, model_name, sample_size)
                kl_div = calculate_kl_divergence(df[column], synthetic_data[column])
                kl_divergences.append(kl_div)
    
                # Extract last epoch losses from the console output
                last_epoch_loss = extract_last_epoch_losses(console.get_console_output())
                
                # Update the sidebar content with the new iteration's output
                iteration_output = f"Iteration {i}:\nKL Divergence = {kl_div}\nLast epoch losses = {last_epoch_loss if last_epoch_loss else 'N/A'}\n\n"
                sidebar_content += iteration_output
                kl_sidebar.text(sidebar_content)
    
                # Update the plot in the main area
                fig, ax = plt.subplots(figsize=(10, 6))
                visualize_distributions(df[column], synthetic_data[column], column, i, kl_divergences, visualization_mode, ax)
                plot_placeholder.pyplot(fig)

                # Update the progress bar and percentage indicator
                progress_percentage = i / total_iterations
                progress_bar_placeholder.progress(progress_percentage)
                percentage_indicator_placeholder.markdown(f"**Progress: {progress_percentage * 100:.0f}%**")

                # Clear the buffer for the next iteration
                console.buffer.truncate(0)
                console.buffer.seek(0)
