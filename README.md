# dilemma
In progress project to answer your Whya Hows and Whats.

# Interactive Dilemma Analyzer with AI

## Overview

The **Interactive Dilemma Analyzer** is a Streamlit application designed to analyze and visualize dilemmas using artificial intelligence. This app integrates a pre-trained language model to generate insights, extracts payoff values from user descriptions, computes Nash equilibrium for a 2x2 game, and visualizes the results with interactive plots.

## Features

- **AI-Driven Response Generation**: Utilizes a pre-trained language model to provide thoughtful responses based on user input.
- **Payoff Extraction**: Extracts game-theoretic payoffs from natural language descriptions using regular expressions.
- **Nash Equilibrium Computation**: Computes Nash equilibrium strategies for the provided payoff matrix using CVXPY.
- **Visualization**: Displays a heatmap of the payoff matrix using Seaborn to illustrate the strategic balance between cooperation and betrayal.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Streamlit
- Numpy
- Seaborn
- PyTorch
- Transformers
- Accelerate
- CVXPY
- Matplotlib

You can install the necessary libraries using pip:

```bash
pip install streamlit numpy seaborn torch transformers accelerate cvxpy matplotlib
