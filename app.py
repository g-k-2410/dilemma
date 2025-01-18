import streamlit as st
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import cvxpy as cp
import matplotlib.pyplot as plt
import re
import warnings
import logging

# Suppress warnings and logging from HuggingFace
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ==========================
# AI Model Utility Functions
# ==========================
@st.cache_resource
def load_model_and_tokenizer():
    """Load the causal language model and tokenizer."""
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cpu", torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def create_pipeline(model, tokenizer):
    """Create a text-generation pipeline."""
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(pipe, user_input):
    """Generate a response based on user input."""
    prompt = f"I am facing this dilemma: {user_input}. Can you please suggest a solution?"
    generation_args = {
        "max_new_tokens": 150,
        "return_full_text": False,
        "do_sample": True,
        "temperature": 0.7
    }
    try:
        output = pipe(prompt, **generation_args)
        return output[0]['generated_text']
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

# ================================
# Game Theory and Payoff Utilities
# ================================
def extract_payoffs(description):
    """Extract payoff values from the description using regex."""
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

def compute_nash_equilibrium_simple(payoff_matrix):
    """Compute the Nash equilibrium for a 2x2 game using CVXPY."""
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

def plot_payoff_matrix(matrix):
    """Plot the payoff matrix using seaborn."""
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', ax=ax,
                xticklabels=['Cooperate', 'Betray'], yticklabels=['Cooperate', 'Betray'])
    plt.title('Payoff Matrix')
    plt.xlabel('Player 2')
    plt.ylabel('Player 1')
    return fig

# ====================
# Streamlit Application
# ====================
def main():
    st.title("Interactive Dilemma Analyzer with AI and Game Theory")
    st.header("Describe Your Dilemma")
    user_description = st.text_area(
        "Enter a description of your dilemma in natural language:",
        "I need to decide whether to point out my mother's judgmental behavior or not."
    )

    if st.button("Analyze Dilemma"):
        with st.spinner("Analyzing dilemma..."):
            # Step 1: Generate AI Response
            model, tokenizer = load_model_and_tokenizer()
            pipe = create_pipeline(model, tokenizer)
            ai_response = generate_response(pipe, user_description)
            st.subheader("AI Generated Response:")
            st.write(ai_response)

            # Step 2: Extract Payoffs
            payoff_matrix = extract_payoffs(user_description)
            st.subheader("Payoff Matrix:")
            st.write(payoff_matrix)

            # Step 3: Plot Payoff Matrix
            fig = plot_payoff_matrix(payoff_matrix)
            if fig:
                st.pyplot(fig)

            # Step 4: Compute Nash Equilibrium
            equilibrium = compute_nash_equilibrium_simple(payoff_matrix)
            if equilibrium is not None:
                st.subheader("Nash Equilibrium Strategies:")
                st.write(f"Strategy probabilities: {equilibrium}")
                st.write("""
                - Player 1: Probability of cooperating (first value), probability of betraying (second value).\n
                - Player 2: Same interpretation.\n
                Use these probabilities to balance cooperation and betrayal strategies.
                """)
            else:
                st.error("No Nash equilibrium could be calculated.")

if __name__ == "__main__":
    main()
