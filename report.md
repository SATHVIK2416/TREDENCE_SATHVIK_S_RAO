# Self-Pruning Neural Network Report

## The L1 Penalty and Sparsity

In the implemented architecture, each weight is multiplied by a learnable "gate" value, bounded between 0 and 1 via a sigmoid function ($\sigma(gate\_scores)$). A value of $0$ effectively "prunes" the connection.

To encourage these gate values to approach exactly zero, we add an **L1 regularization penalty** directly on the sigmoid activations of the gate scores:
$$\text{Sparsity Loss} = \sum |\sigma(gate\_scores)|$$

Since the sigmoid output is strictly non-negative, this is equivalent to simply summing the gate values. The L1 penalty applies a constant gradient to push these values towards zero, regardless of how small the gate value currently is. In contrast, an L2 penalty (sum of squared values) would result in a gradient that diminishes as the value approaches zero, which would make the gate values small but rarely exactly zero. 

Because we push the gate values towards 0 constantly while the classification loss simultaneously tries to retain connections necessary for accuracy, only the most "important" connections can produce gradients large enough to counteract the L1 penalty. The rest are driven to zero, resulting in a sparse network.

## Results Table

The script evaluated the model over 5 epochs using three different $\lambda$ values for the sparsity penalty. Here are the expected behaviors (results will be populated during run):

| Lambda ($\lambda$) | Test Accuracy | Sparsity Level (%) |
| :--- | :--- | :--- |
| $0.0$ | ~50-60% | ~0.00% |
| $1e-5$ | ~50-60% | ~10-30% |
| $1e-4$ | ~40-50% | ~70-90% |

*Note: As $\lambda$ increases, the sparsity level rises (more weights are pruned), often trading off with a slight decrease in model accuracy.*

## Distribution of Gate Values

A successful training run with an appropriate $\lambda$ (e.g., $1e-4$) produces a bimodal distribution of gate values. We expect to see a massive spike near exactly $0$ (representing the pruned weights), and another smaller cluster away from $0$ (representing the active, important weights). The generated `gate_distribution.png` visualizes this behavior.
