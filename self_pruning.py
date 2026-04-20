import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import numpy as np


# ---------------------------------------------------------
# Part 1: The "Prunable" Linear Layer
# ---------------------------------------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Standard Kaiming init for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # Initialize gate scores to a positive value so training starts with active connections
        # A value of 2.0 gives a sigmoid output of ~0.88 (mostly on)
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x):
        # 1. Apply Sigmoid to gate_scores
        gates = torch.sigmoid(self.gate_scores)

        # 2. Calculate pruned weights
        pruned_weights = self.weight * gates

        # 3. Perform linear operation
        return F.linear(x, pruned_weights, self.bias)


# ---------------------------------------------------------
# Part 2: Network Definition & Sparsity Logic
# ---------------------------------------------------------
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()
        self.flatten = nn.Flatten()
        # Simple feed-forward network for 32x32x3 images
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_sparsity_loss(self):
        """Calculates the L1 norm of all gate values (after sigmoid)."""
        l1_loss = 0.0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                # Since gates are from sigmoid, they are always positive.
                # L1 norm is just the sum.
                gates = torch.sigmoid(m.gate_scores)
                l1_loss += torch.sum(gates)
        return l1_loss

    def get_sparsity_level(self, threshold=1e-2):
        """Calculates the percentage of weights that are effectively pruned."""
        total_weights = 0
        pruned_weights = 0

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, PrunableLinear):
                    gates = torch.sigmoid(m.gate_scores)
                    total_weights += gates.numel()
                    pruned_weights += torch.sum(gates < threshold).item()

        return (pruned_weights / total_weights) * 100.0 if total_weights > 0 else 0.0

    def get_all_gate_values(self):
        """Returns a flattened numpy array of all gate values for plotting."""
        all_gates = []
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, PrunableLinear):
                    gates = torch.sigmoid(m.gate_scores).cpu().numpy().flatten()
                    all_gates.extend(gates)
        return np.array(all_gates)


# ---------------------------------------------------------
# Part 3: Training and Evaluation
# ---------------------------------------------------------
def train_and_evaluate(lmbda, epochs=5, device="cpu"):
    print(f"\n--- Training with Lambda = {lmbda} ---")

    # Data loaders
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Use smaller subsets to speed up demonstration if needed, but standard size here
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=0
    )

    model = PrunableNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            cls_loss = criterion(outputs, labels)
            sparsity_loss = model.get_sparsity_loss()

            # Total Loss
            loss = cls_loss + lmbda * sparsity_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(trainloader):.4f}"
        )

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity_level = model.get_sparsity_level()

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Sparsity Level: {sparsity_level:.2f}%")

    return model, accuracy, sparsity_level


if __name__ == "__main__":
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lambdas = [0.0, 1e-5, 1e-4]
    results = []
    best_model = None
    best_gates = None

    for l in lambdas:
        # Run for a few epochs (e.g., 3-5) to demonstrate the effect
        model, acc, sparsity = train_and_evaluate(l, epochs=5, device=device)
        results.append((l, acc, sparsity))
        if l == 1e-4:  # Choose a lambda that should show pruning for the plot
            best_gates = model.get_all_gate_values()

    print("\n--- Summary ---")
    print("Lambda\tAccuracy\tSparsity (%)")
    for l, acc, sparsity in results:
        print(f"{l}\t{acc:.2f}%\t\t{sparsity:.2f}%")

    # Save plot for the regularized model
    if best_gates is not None:
        plt.figure(figsize=(8, 6))
        plt.hist(
            best_gates,
            bins=50,
            range=(0, 1),
            alpha=0.75,
            color="blue",
            edgecolor="black",
        )
        plt.title("Distribution of Gate Values (Lambda = 1e-4)")
        plt.xlabel("Gate Value (after Sigmoid)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig("gate_distribution.png")
        print("\nSaved gate distribution plot to 'gate_distribution.png'")
