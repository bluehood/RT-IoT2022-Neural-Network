import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_network(ax, layer_sizes):
    v_spacing = 0.2
    h_spacing = 0.5
    ax.axis('off')

    # Layers
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0
        for j in range(layer_size):
            circle = patches.Circle((i * h_spacing, layer_top - j * v_spacing), radius=0.1, color='skyblue', ec='black')
            ax.add_patch(circle)
            ax.annotate(f'Node {j + 1}\nLayer {i + 1}', (i * h_spacing, layer_top - j * v_spacing), xytext=(i * h_spacing - 0.25, layer_top - j * v_spacing + 0.15))

    # Connections
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                line = plt.Line2D([i * h_spacing, (i + 1) * h_spacing], [layer_top - j * v_spacing, layer_top - k * v_spacing], c='black')
                ax.add_line(line)

# Example: a neural network with 3 layers (input, hidden, output)
layer_sizes = [4, 5, 3]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, aspect='equal')

draw_neural_network(ax, layer_sizes)
plt.show()
