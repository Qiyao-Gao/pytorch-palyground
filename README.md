# PyTorch Playground

A lightweight, Streamlit-based playground inspired by TensorFlow Playground but using a PyTorch MLP. Train on simple 2D datasets and watch the decision boundary evolve.

## Getting started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your browser to experiment with datasets, architecture depth, activations, and optimizer settings.

> Can this run "here"? The playground needs a Streamlit webserver to be reachable in your browser. In this chat environment there is no shared browser window, so please run the above `streamlit run app.py` command on your own machine (or a local container) and open the displayed URL to interact with the UI.

## Features
- Choose among moons, circles, or blobs toy datasets with adjustable noise and sample size.
- Configure number of hidden layers, hidden units, and activation functions.
- Adjust training hyperparameters including learning rate, weight decay, batch size, epochs, and validation split.
- Visualize the live decision boundary, training loss, and validation accuracy.
