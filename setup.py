import os
import json
import numpy as np
from network import Network
from generation import Generation

def setup_memory():
    """Create memory folder and initial files"""
    # Create memory directory
    os.makedirs('memory', exist_ok=True)
    
    # Create initial metadata
    metadata = {
        'generation_num': 0,
        'best_fitness': 0,
    }
    
    # Save metadata
    with open('memory/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Create initial generation
    generation = Generation()
    
    # Save initial networks
    networks_data = []
    for genome in generation.genomes:
        network = {
            'W1': genome.W1.tolist(),
            'W2': genome.W2.tolist(),
            'fitness': 0,
            'steps_survived': 0
        }
        networks_data.append(network)
    
    # Save initial generation
    np.save('memory/generation_0.npy', networks_data)
    
    print("Created memory folder structure:")
    print("memory/")
    print("├── metadata.json")
    print("└── generation_0.npy")

if __name__ == "__main__":
    setup_memory() 