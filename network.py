import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

class DQN(nn.Module):
    def __init__(self, input_shape=(4, 36, 40)):
        super(DQN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Calculate size after convolutions
        conv_out_size = self._get_conv_output(input_shape)
        
        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3 actions: nothing, jump, crouch
        )
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def _get_conv_output(self, shape):
        # Helper function to calculate conv output size
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        return int(np.prod(o.shape))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class Network:
    def __init__(self):
        # Check CUDA availability and print GPU info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            # Initialize gradient scaler for mixed precision training
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Initialize DQN model
        self.model = DQN()
        self.model.to(self.device)  # Ensure model is on correct device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        
        # Training parameters
        self.memory = []
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.01  # Lower minimum for more exploitation
        self.epsilon_decay = 0.00005  # Much slower decay
        self.batch_size = 64  # Larger batch size
        self.min_memory_size = 1000  # Wait for minimum experiences
        
        # Add tracking variables
        self.total_training_steps = 0
        self.running_loss = []
        self.max_loss_history = 1000
    
    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, 2)
        else:
            # Exploitation: best action from model
            with torch.no_grad():
                with autocast(enabled=torch.cuda.is_available()):
                    state = torch.FloatTensor(state).to(self.device)
                    state = state.permute(2, 0, 1).unsqueeze(0)  # Reshape for PyTorch (B,C,H,W)
                    state = state.float() / 255.0
                    q_values = self.model(state)
                    return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 50000:  # Increased memory size
            self.memory.pop(0)
    
    def train(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.min_memory_size:
            print(f"\rWaiting for more experiences: {len(self.memory)}/{self.min_memory_size}", end="")
            return 0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([x[0] for x in batch]).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor([x[3] for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        # Reshape states for PyTorch (B,C,H,W)
        states = states.permute(0, 3, 1, 2) / 255.0
        next_states = next_states.permute(0, 3, 1, 2) / 255.0
        
        # Use mixed precision training if GPU is available
        if torch.cuda.is_available():
            with autocast():
                # Get current Q values
                current_q_values = self.model(states)
                current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
                
                # Get next Q values
                with torch.no_grad():
                    next_q_values = self.model(next_states)
                    max_next_q = next_q_values.max(1)[0]
                    target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
                
                # Compute loss
                loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Regular training on CPU
            # Get current Q values
            current_q_values = self.model(states)
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
            
            # Get next Q values
            with torch.no_grad():
                next_q_values = self.model(next_states)
                max_next_q = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
            
            # Compute loss and optimize
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        
        # After loss calculation, before optimization:
        loss_val = loss.item()
        self.running_loss.append(loss_val)
        if len(self.running_loss) > self.max_loss_history:
            self.running_loss.pop(0)
        
        self.total_training_steps += 1
        if self.total_training_steps % 100 == 0:
            avg_loss = sum(self.running_loss) / len(self.running_loss)
            print(f"\nStep {self.total_training_steps}")
            print(f"Memory Size: {len(self.memory)}")
            print(f"Average Loss: {avg_loss:.5f}")
            print(f"Epsilon: {self.epsilon:.5f}")
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if self.scaler and checkpoint['scaler']:
            self.scaler.load_state_dict(checkpoint['scaler'])
