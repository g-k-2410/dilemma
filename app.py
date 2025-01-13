import streamlit as st
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import cvxpy as cp
import matplotlib.pyplot as plt
import re

# Function to load the model, cached for performance
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cpu", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to create a text generation pipeline
def create_pipeline(model, tokenizer):
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate a response from the model based on user input
def generate_response(pipe, user_input):
    generation_args = {
        "max_new_tokens": 600,
        "return_full_text": False,
        "do_sample": True,  # Set do_sample to True to use temperature
        "temperature": 0.3
    }
    try:
        output = pipe(user_input, **generation_args)
        return output[0]['generated_text']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

# Function to extract payoff values from a natural language description using regex
def extract_payoffs(description):
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
    st.write(f"Extracted payoffs: {payoffs}")
    return np.array([
        [payoffs['reward'], payoffs['temptation']],
        [payoffs['sucker'], payoffs['punishment']]
    ])

# Function to compute the Nash equilibrium for a simple 2x2 game using CVXPY
def compute_nash_equilibrium_simple(payoff_matrix):
    try:
        num_strategies = payoff_matrix.shape[0]
        p = cp.Variable(num_strategies)
        constraints = [p >= 0, cp.sum(p) == 1]
        objective = cp.Maximize(p @ np.mean(payoff_matrix, axis=1))
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return p.value
    except Exception as e:
        st.error(f"Error computing Nash equilibrium: {e}")
        return None

# Function to plot the payoff matrix using seaborn
def plot_payoff_matrix(matrix):
    try:
        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', ax=ax,
                    xticklabels=['Cooperate', 'Betray'], yticklabels=['Cooperate', 'Betray'])
        plt.title('Payoff Matrix')
        plt.xlabel('Player 2')
        plt.ylabel('Player 1')
        return fig
    except Exception as e:
        st.error(f"Error plotting payoff matrix: {e}")
        return None

# Function to draw puzzle pieces that either fit together or not based on the fit parameter
def draw_puzzle_pieces(fit: bool):
    fig, ax = plt.subplots()
    if fit:
        ax.plot([0, 1], [0, 1], 'k-', lw=4)
        ax.plot([1, 2], [1, 0], 'k-', lw=4)
        ax.plot([2, 3], [0, 1], 'k-', lw=4)
        ax.plot([0, 2], [2, 2], 'k-', lw=4)
        ax.plot([1, 3], [2, 2], 'k-', lw=4)
    else:
        ax.plot([0, 1], [0, 1], 'r-', lw=4)
        ax.plot([1, 2], [1, 0], 'r-', lw=4)
        ax.plot([0, 2], [1, 2], 'r-', lw=4)
        ax.plot([2, 3], [2, 3], 'r-', lw=4)
    
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    return fig

# Main function to run the Streamlit app
def main():
    st.title("Interactive Dilemma Analyzer with AI")
    st.header("Describe Your Dilemma")
    user_description = st.text_area("Enter a description of your dilemma in natural language:")

    # Display initial puzzle image
    st.pyplot(draw_puzzle_pieces(fit=False))

    if st.button("Analyze Dilemma"):
        with st.spinner("Analyzing dilemma..."):
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer()
            # Create pipeline
            pipe = create_pipeline(model, tokenizer)
            # Generate thoughtful response
            thoughtful_response = generate_response(pipe, user_description)
            st.write("AI Generated Response:")
            st.write(thoughtful_response)

            # Extract payoffs from the description
            payoff_matrix = extract_payoffs(user_description)
            st.write("Payoff Matrix:")
            st.write(payoff_matrix)

            # Plot the payoff matrix
            fig = plot_payoff_matrix(payoff_matrix)
            if fig:
                st.pyplot(fig)

            # Compute Nash equilibrium
            equilibrium = compute_nash_equilibrium_simple(payoff_matrix)
            if equilibrium is not None:
                st.write("Nash Equilibrium Strategies:")
                st.write(f"Strategy probabilities: {equilibrium}")
            else:
                st.write("No Nash equilibrium found.")

            st.write("""
            The Nash Equilibrium represents the strategy mix where neither player can benefit by unilaterally changing their strategy.
            In this game, each player's strategy probabilities indicate the likelihood of choosing to cooperate or betray.
            - Player 1: The probability of cooperating is given by the first value, and betraying is the complement.
            - Player 2: Similarly, the probability of cooperating is given by the second value, and betraying is the complement.
            Use this information to understand the strategic balance between cooperation and betrayal in your specific dilemma.
            """)

        # Display final puzzle image
        st.pyplot(draw_puzzle_pieces(fit=True))

if __name__ == "__main__":
    main()
