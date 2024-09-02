import streamlit as st
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from accelerate import dispatch_model
import cvxpy as cp
import matplotlib.pyplot as plt
import re

# Model name
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load the model and tokenizer with error handling
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Dispatch the model to use disk offloading
    model = dispatch_model(model, device_map="disk")
except OSError as e:
    st.error(f"Failed to load model: {e}")
    st.stop()  # Stop the app if model loading fails

# Create a pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Configuration for text generation
generation_args = {
    "max_new_tokens": 600,
    "return_full_text": False,
    "temperature": 0.3,
    "do_sample": False
}


def generate_response(user_input):
    """
    Generate a response from the model based on user input.
    """
    output = pipe(user_input, **generation_args)
    return output[0]['generated_text']


def extract_payoffs(description):
    """
    Extract payoffs from the description using regex patterns.
    """
    patterns = {
        'reward': r'reward\s*for\s*mutual\s*cooperation\s*[-]?\d+',
        'temptation': r'temptation\s*to\s*betray\s*[-]?\d+',
        'punishment': r'punishment\s*for\s*mutual\s*betrayal\s*[-]?\d+',
        'sucker': r'sucker\'s\s*payoff\s*[-]?\d+'
    }
    payoffs = {'reward': 3, 'temptation': 5, 'punishment': -1, 'sucker': -5}
    for key, pattern in patterns.items():
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            value = int(re.search(r'[-]?\d+', match.group()).group())
            payoffs[key] = value
    return np.array([
        [payoffs['reward'], payoffs['temptation']],
        [payoffs['sucker'], payoffs['punishment']]
    ])


def compute_nash_equilibrium_simple(payoff_matrix):
    """
    Compute the Nash equilibrium for a simple 2x2 game using CVXPY.
    """
    num_strategies = payoff_matrix.shape[0]
    p = cp.Variable(num_strategies)
    constraints = [p >= 0, cp.sum(p) == 1]
    objective = cp.Maximize(p @ np.mean(payoff_matrix, axis=1))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return p.value


def plot_payoff_matrix(matrix):
    """
    Plot the payoff matrix using seaborn.
    """
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', ax=ax,
                xticklabels=['Cooperate', 'Betray'], yticklabels=['Cooperate', 'Betray'])
    plt.title('Payoff Matrix')
    plt.xlabel('Player 2')
    plt.ylabel('Player 1')
    return fig


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Interactive Dilemma Analyzer with AI")
    st.header("Describe Your Dilemma")
    user_description = st.text_area("Enter a description of your dilemma in natural language:")

    if st.button("Analyze Dilemma"):
        thoughtful_response = generate_response(user_description)
        st.write("AI Generated Response:")
        st.write(thoughtful_response)

        payoff_matrix = extract_payoffs(user_description)
        st.write("Payoff Matrix:")
        st.write(payoff_matrix)

        fig = plot_payoff_matrix(payoff_matrix)
        st.pyplot(fig)

        equilibrium = compute_nash_equilibrium_simple(payoff_matrix)
        if equilibrium is not None:
            st.write("Nash Equilibrium Strategies:")
            st.write(f"Strategy probabilities: {equilibrium}")
        else:
            st.write("No Nash equilibrium found.")

        st.write("""
        The Nash Equilibrium represents the strategy mix where neither player can benefit by unilaterally changing their strategy.
        In this game, each player's strategy probabilities indicate the likelihood of choosing to cooperate or betray.
        - **Player 1**: The probability of cooperating is given by the first value, and betraying is the complement.
        - **Player 2**: Similarly, the probability of cooperating is given by the second value, and betraying is the complement.
        Use this information to understand the strategic balance between cooperation and betrayal in your specific dilemma.
        """)


if __name__ == "__main__":
    main()
