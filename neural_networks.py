import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Create results directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation='tanh'):
        np.random.seed(0)
        self.learning_rate = learning_rate
        self.activation_function = activation

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.Z_hidden = X @ self.weights_input_hidden + self.bias_hidden

        if self.activation_function == 'tanh':
            self.hidden_layer_output = np.tanh(self.Z_hidden)
        elif self.activation_function == 'relu':
            self.hidden_layer_output = np.maximum(0, self.Z_hidden)
        elif self.activation_function == 'sigmoid':
            self.hidden_layer_output = 1 / (1 + np.exp(-self.Z_hidden))
        else:
            raise ValueError('Unsupported activation function')

        self.Z_output = self.hidden_layer_output @ self.weights_hidden_output + self.bias_output
        self.output = 1 / (1 + np.exp(-self.Z_output))
        return self.output

    def backward(self, X, y):
        num_samples = y.shape[0]

        error_output = self.output - y
        gradient_weights_hidden_output = (self.hidden_layer_output.T @ error_output) / num_samples
        gradient_bias_output = np.sum(error_output, axis=0, keepdims=True) / num_samples

        error_hidden = error_output @ self.weights_hidden_output.T
        if self.activation_function == 'tanh':
            gradient_hidden = error_hidden * (1 - np.tanh(self.Z_hidden) ** 2)
        elif self.activation_function == 'relu':
            gradient_hidden = error_hidden * (self.Z_hidden > 0).astype(float)
        elif self.activation_function == 'sigmoid':
            sigmoid_Z_hidden = 1 / (1 + np.exp(-self.Z_hidden))
            gradient_hidden = error_hidden * sigmoid_Z_hidden * (1 - sigmoid_Z_hidden)
        else:
            raise ValueError('Unsupported activation function')

        gradient_weights_input_hidden = (X.T @ gradient_hidden) / num_samples
        gradient_bias_hidden = np.sum(gradient_hidden, axis=0, keepdims=True) / num_samples

        self.weights_hidden_output -= self.learning_rate * gradient_weights_hidden_output
        self.bias_output -= self.learning_rate * gradient_bias_output
        self.weights_input_hidden -= self.learning_rate * gradient_weights_input_hidden
        self.bias_hidden -= self.learning_rate * gradient_bias_hidden

        self.gradient_weights_input_hidden = gradient_weights_input_hidden
        self.gradient_weights_hidden_output = gradient_weights_hidden_output

# Generate dataset
def generate_data(num_samples=200):
    np.random.seed(0)
    X = np.random.randn(num_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)
    return X, y

# Update function for animation
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y, hidden_x_min, hidden_x_max, hidden_y_min, hidden_y_max, hidden_z_min, hidden_z_max):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    current_step = frame * 10

    # Hidden Layer Visualization
    hidden_layer_output = mlp.hidden_layer_output
    ax_hidden.scatter(hidden_layer_output[:, 0], hidden_layer_output[:, 1], hidden_layer_output[:, 2],
                      c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title(f'Hidden Space At Step {current_step}')
    ax_hidden.set_xlim(hidden_x_min, hidden_x_max)
    ax_hidden.set_ylim(hidden_y_min, hidden_y_max)
    ax_hidden.set_zlim(hidden_z_min, hidden_z_max)

    # Input Space Decision Boundary
    x_min_input, x_max_input = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min_input, y_max_input = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_input, yy_input = np.meshgrid(np.linspace(x_min_input, x_max_input, 300),
                                     np.linspace(y_min_input, y_max_input, 300))
    grid = np.c_[xx_input.ravel(), yy_input.ravel()]
    probabilities = mlp.forward(grid).reshape(xx_input.shape)
    ax_input.contourf(xx_input, yy_input, probabilities, levels=[0, 0.5, 1], colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f'Input Space At Step {current_step}')

    # Gradient Visualization
    max_gradient = max(np.abs(mlp.gradient_weights_input_hidden).max(), np.abs(mlp.gradient_weights_hidden_output).max(), 1e-6)
    linewidth_scale = 5

    input_neurons = [(-1, 1), (-1, -1)]
    hidden_neurons = [(0, 1), (0, 0), (0, -1)]
    output_neuron = (1, -1)

    for neuron, label in zip(input_neurons, ['x1', 'x2']):
        ax_gradient.add_patch(Circle(neuron, 0.1, color='blue'))
        ax_gradient.text(neuron[0], neuron[1], label, ha='center', va='center', fontsize=10, color='white')

    for neuron, label in zip(hidden_neurons, ['h1', 'h2', 'h3']):
        ax_gradient.add_patch(Circle(neuron, 0.1, color='blue'))
        ax_gradient.text(neuron[0], neuron[1], label, ha='center', va='center', fontsize=10, color='white')

    ax_gradient.add_patch(Circle(output_neuron, 0.1, color='blue'))
    ax_gradient.text(output_neuron[0], output_neuron[1], 'y', ha='center', va='center', fontsize=10, color='white')

    for i, input_neuron in enumerate(input_neurons):
        for j, hidden_neuron in enumerate(hidden_neurons):
            gradient = mlp.gradient_weights_input_hidden[i, j]
            linewidth = np.clip(np.abs(gradient) / max_gradient * linewidth_scale, 0.1, linewidth_scale)
            ax_gradient.plot([input_neuron[0], hidden_neuron[0]], [input_neuron[1], hidden_neuron[1]],
                             color='purple', linewidth=linewidth)

    for j, hidden_neuron in enumerate(hidden_neurons):
        gradient = mlp.gradient_weights_hidden_output[j, 0]
        linewidth = np.clip(np.abs(gradient) / max_gradient * linewidth_scale, 0.1, linewidth_scale)
        ax_gradient.plot([hidden_neuron[0], output_neuron[0]], [hidden_neuron[1], output_neuron[1]],
                         color='purple', linewidth=linewidth)

    ax_gradient.set_title(f'Gradients At Step {current_step}')

# Main visualization function
def visualize(activation, learning_rate, steps):
    X, y = generate_data()
    mlp = MLP(input_size=2, hidden_size=3, output_size=1, learning_rate=learning_rate, activation=activation)

    fig = plt.figure(figsize=(18, 6))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    mlp.forward(X)
    hidden_layer_output = mlp.hidden_layer_output
    hidden_x_min, hidden_x_max = hidden_layer_output[:, 0].min() - 0.5, hidden_layer_output[:, 0].max() + 0.5
    hidden_y_min, hidden_y_max = hidden_layer_output[:, 1].min() - 0.5, hidden_layer_output[:, 1].max() + 0.5
    hidden_z_min, hidden_z_max = hidden_layer_output[:, 2].min() - 0.5, hidden_layer_output[:, 2].max() + 0.5

    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden,
                ax_gradient=ax_gradient, X=X, y=y,
                hidden_x_min=hidden_x_min, hidden_x_max=hidden_x_max,
                hidden_y_min=hidden_y_min, hidden_y_max=hidden_y_max,
                hidden_z_min=hidden_z_min, hidden_z_max=hidden_z_max),
        frames=steps // 10,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(output_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

# Main execution
if __name__ == "__main__":
    activation = "tanh"  # Change to 'relu' or 'sigmoid' to test other activation functions
    learning_rate = 0.1
    steps = 1000
    visualize(activation, learning_rate, steps)