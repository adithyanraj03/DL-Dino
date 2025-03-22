# 🦖 DL-Dino: Deep Learning Chrome Dinosaur Player

![GitHub](https://img.shields.io/github/license/adithyanraj03/DL-Dino)
![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)

A deep reinforcement learning agent that learns to play the Chrome Dinosaur game. This project implements various Deep Q-Learning algorithms to compare performance and investigates the effect of techniques like batch normalization on training. The system uses computer vision, deep reinforcement learning, and genetic algorithms to create an AI capable of mastering the Chrome Dino game and surpassing average human performance.

![demo](https://github.com/user-attachments/assets/c7d3e99b-87f0-403c-a2f3-6566b043f115)


![image](https://github.com/user-attachments/assets/4a4d90c8-065f-4092-8f76-edcef205f3b8)

## ✨ Features

- 🧠 Multiple Deep Q-Learning algorithms implemented:
  - Standard DQN (Deep Q-Network)
  - Double DQN to reduce Q-value overestimation
  - Dueling DQN with separate value and advantage streams
  - DQN with Prioritized Experience Replay (PER)
  - Variants with Batch Normalization
- 🎮 Automated game interaction via Selenium WebDriver
- 👁️ Computer vision-based game state analysis with OpenCV
- 🚀 CUDA acceleration for image processing (when available)
- 📈 Real-time visualization of the neural network and decision-making
- 💾 Training state saving and loading for continuous improvement
- 🧬 Optional genetic algorithm approach for network evolution
- 📊 Performance comparison between algorithms and human players

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adithyanraj03/DL-Dino.git
   cd DL-Dino
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Chrome WebDriver:**
   - Make sure you have Chrome installed
   - Download the appropriate [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) version for your Chrome browser
   - Place it in the root directory of the project or in a location in your PATH

## 💻 Requirements

- Python 3.6+
- PyTorch
- OpenCV
- Selenium
- Pygame
- NumPy
- Matplotlib

## 🚀 Usage

### Initial Setup

Run the setup script to initialize the memory directory:
```bash
python setup.py
```

### Training the AI

Start the training process with:
```bash
python ai.py
```

The AI will go through these phases:
1. Calibration - You'll need to adjust the screen capture area to match your Chrome window
2. Exploration - The agent will try random actions to learn the game dynamics
3. Exploitation - As training progresses, the agent will use its learned policy more often
   


### Visualization

During training, two windows will appear:
- **Game View**: Shows the current game state with obstacle detection
- **Network Visualization**: Displays the neural network architecture, Q-values, and decision-making process
  
![image](https://github.com/user-attachments/assets/0245f62b-afd7-43c9-a368-f654b53251cd)

### Comparing Different Algorithms

The project allows you to compare the performance of different DQN variants:

1. Set the desired algorithm in `network.py`
2. Run multiple training sessions
3. Compare the results using the statistical metrics captured during training

## 📁 Project Structure

- `ai.py` - Main training loop and reward calculation
- `network.py` - Neural network architecture and training logic
- `scanner.py` - Game window interaction and computer vision
- `visualizer.py` - Real-time visualization of the neural network
- `generation.py` - Genetic algorithm implementation for network evolution
- `setup.py` - Initializes memory directory structure

## 🧩 How It Works

### 1. Game State Detection
The system captures the Chrome window and uses computer vision to:
- Detect the dinosaur position and state (jumping/ducking)
- Identify obstacles (cacti and birds) and their distances
- Calculate game speed and score

### 2. Reinforcement Learning
The project implements several reinforcement learning algorithms:

#### DQN (Deep Q-Network)
- Convolutional layers for image processing
- Experience replay to improve stability
- Epsilon-greedy exploration strategy
- Custom reward function based on successful obstacle avoidance

#### Double DQN
- Addresses the overestimation bias in standard DQN
- Decouples action selection and evaluation
- Uses two networks to reduce value overestimation

#### Dueling DQN
- Separates state value and action advantage estimation
- Allows better generalization across actions
- Shows the best overall performance in testing

#### DQN with Prioritized Experience Replay (PER)
- Weights experiences by their TD-error
- Focuses learning on the most informative transitions
- Note: Shows slower performance in real-time environments

#### Batch Normalization Variants
- Applied to all algorithm variants
- Significantly improves training stability
- Accelerates learning in most cases

### 3. Visualization
Real-time displays show:
- Neural network weights and activations
- Current game state analysis
- Learning metrics (loss, memory size)
- Action decisions and their corresponding Q-values

## 🔧 Advanced Configuration

### Hyperparameters

The project uses the following optimized hyperparameters:

| Hyperparameter  | Value     | Description |
| --------------- | --------- | ----------- |
| Memory Size     | 3 × 10^5  | Size of experience replay buffer |
| Batch Size      | 128       | Number of experiences per training batch |
| Gamma           | 0.99      | Discount factor for future rewards |
| Initial epsilon | 1 × 10^−1 | Starting exploration rate |
| Final epsilon   | 1 × 10^−4 | Minimum exploration rate |
| Explore steps   | 1 × 10^5  | Number of steps to decay epsilon |
| Learning Rate   | 2 × 10^−5 | Rate of network weight updates |

You can modify these parameters in the respective Python files:

- `network.py`: Learning rate, discount factor, epsilon decay rate, network architecture
- `ai.py`: Reward function parameters, save intervals, exploration settings
- `scanner.py`: Frame processing rate, window positions, image preprocessing

## Training Results
### Comparison of different DQN algorithms

Using tuned hyperparameters, we ran each algorithm for 200 epochs. Dueling DQN showed the best performance, while Prioritized Experience Replay performed poorly due to the computational overhead of weight updates.

![Training Comparison](https://github.com/user-attachments/assets/77c76689-63a9-441d-a7a0-cef05c622842 "Training Comparison")

### Comparison between DQN and DQN with Batch Normalization

Batch normalization significantly improved performance across most algorithms, with the exception of PER.

![BN Comparison](https://github.com/user-attachments/assets/b9dfc8ac-09c2-48c6-ada8-b667f3d53f99 "Batch Normalization Comparison")

### Statistical Results in Training

| Algorithm         | Mean    | Std     | Max   | 25%    | 50%   | 75%     | Time (h) |
| ----------------- | ------- | ------- | ----- | ------ | ----- | ------- | -------- |
| DQN               | 537.50  | 393.61  | 1915  | 195.75 | 481   | 820     | 25.87    |
| Double DQN        | 443.31  | 394.01  | 2366  | 97.75  | 337   | 662.25  | 21.36    |
| Dueling DQN       | 839.04  | 1521.40 | 25706 | 155    | 457   | 956.5   | 35.78    |
| DQN with PER      | 43.50   | 2.791   | 71    | 43     | 43    | 43      | 3.31     |
| DQN (BN)          | 777.54  | 917.26  | 8978  | 97.75  | 462.5 | 1139.25 | 32.59    |
| Double DQN (BN)   | 696.43  | 758.81  | 5521  | 79     | 430.5 | 1104.25 | 29.40    |
| Dueling DQN (BN)  | 1050.26 | 1477.00 | 14154 | 84     | 541.5 | 1520    | 40.12    |
| DQN with PER (BN) | 46.14   | 7.54    | 98    | 43     | 43    | 43      | 3.44     |

## 📊 Performance Results

### Key Findings

1. **Dueling DQN** shows the best performance overall, achieving the highest scores in both training and testing.
2. **Batch Normalization** significantly improves all algorithms except PER.
3. **Standard DQN with Batch Normalization** also performs exceptionally well.
4. The best models (**Dueling DQN** and **DQN with BN**) surpass average human performance.
5. **Prioritized Experience Replay** performs poorly in this real-time environment, likely due to computational overhead.

### Statistical Comparison

The best algorithm (Dueling DQN with Batch Normalization) achieves:
- Mean score of ~2000+ in testing
- Maximum scores over 14000
- Consistent performance above the 75th percentile of human players

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/adithyanraj03/DL-Dino/LICENSE.md) file for details.

## 💡 Acknowledgements

- [Chrome Dinosaur Game](chrome://dino) for providing an interesting RL challenge
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Selenium](https://www.selenium.dev/) for browser automation
- [Pygame](https://www.pygame.org/) for visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- DQN: [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
- Double DQN: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
- Dueling DQN: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
- Prioritized Experience Replay: [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
- Batch Normalization: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)

---

