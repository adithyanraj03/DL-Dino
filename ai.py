from scanner import Scanner
from network import Network
import os
import json
import numpy as np
import time
from visualizer import NetworkVisualizer
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def save_training_state(scanner, network, save_dir='./memory'):
    """Save complete training state"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'total_games': scanner.total_games,
            'best_score': scanner.best_score,
            'epsilon': network.epsilon,
            'steps_survived': scanner.steps_survived
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save model state
        network.save_model(os.path.join(save_dir, 'model.pth'))
        
        # Save experience memory
        if network.memory:  # Only save if we have memories
            memory_data = {
                'states': [m[0].tolist() for m in network.memory],
                'actions': [int(m[1]) for m in network.memory],
                'rewards': [float(m[2]) for m in network.memory],
                'next_states': [m[3].tolist() for m in network.memory],
                'dones': [bool(m[4]) for m in network.memory]
            }
            np.save(os.path.join(save_dir, 'memory.npy'), memory_data)
            print(f"Saved {len(network.memory)} experiences to memory")
        
        print(f"Training state saved to {save_dir}")
        
    except Exception as e:
        print(f"Error saving training state: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full error trace

def load_training_state(scanner, network, save_dir='./memory'):
    """Load complete training state"""
    try:
        if not os.path.exists(save_dir):
            print(f"Directory not found: {save_dir}")
            return False
            
        # Load metadata
        metadata_path = os.path.join(save_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            return False
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            scanner.total_games = metadata.get('total_games', 0)
            scanner.best_score = metadata.get('best_score', 0)
            network.epsilon = metadata.get('epsilon', network.epsilon)
            scanner.steps_survived = metadata.get('steps_survived', 0)
            print(f"Loaded metadata: Games={scanner.total_games}, Best={scanner.best_score}, Epsilon={network.epsilon:.3f}")
        
        # Load model weights - using correct .pth extension
        model_path = os.path.join(save_dir, 'model.pth')
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
            
        try:
            network.load_model(model_path)
            print("Loaded model weights")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            return False
        
        # Load experience memory
        memory_path = os.path.join(save_dir, 'memory.npy')
        if os.path.exists(memory_path):
            try:
                memory_data = np.load(memory_path, allow_pickle=True).item()
                
                # Validate memory data structure
                required_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
                if not all(key in memory_data for key in required_keys):
                    print("Memory file is corrupted: missing required keys")
                    return False
                
                # Validate data lengths match
                data_lengths = [len(memory_data[key]) for key in required_keys]
                if not all(length == data_lengths[0] for length in data_lengths):
                    print("Memory file is corrupted: inconsistent data lengths")
                    return False
                
                try:
                    network.memory = [
                        (
                            np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)
                        )
                        for state, action, reward, next_state, done in zip(
                            memory_data['states'],
                            memory_data['actions'],
                            memory_data['rewards'],
                            memory_data['next_states'],
                            memory_data['dones']
                        )
                    ]
                    print(f"Loaded {len(network.memory)} experiences from memory")
                except (ValueError, TypeError) as e:
                    print(f"Error converting memory data: {str(e)}")
                    network.memory = []
                    return False
            except Exception as e:
                print(f"Error loading memory: {str(e)}")
                network.memory = []
                return False
        else:
            print(f"Memory file not found: {memory_path}")
            network.memory = []
        
        print(f"Successfully loaded training state from {save_dir}")
        return True
        
    except Exception as e:
        print(f"Error loading training state: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def calculate_reward(action, obstacle, done, dino_state):
    """Calculate reward based on action and closest obstacle"""
    if done:
        # Much more severe penalty for dying by hitting an obstacle
        if obstacle and obstacle['x'] < 30:  # If obstacle was very close when dying
            if obstacle['isBird']:
                return -300.0  # Big penalty for bird collision
            else:
                return -500.0  # Severe penalty for cactus collision (should be easily avoidable)
        return -100.0  # Regular death penalty (like falling into a pit)
    
    base_reward = 1.0  # Base survival reward
    
    # Add reward for beating best score and milestone rewards
    if hasattr(calculate_reward, 'last_score') and hasattr(calculate_reward, 'best_score'):
        current_score = dino_state.get('score', calculate_reward.last_score)
        
        # Check for new best score
        if current_score > calculate_reward.best_score:
            # Base reward for any improvement
            base_reward += 50.0
            
            # Check for milestone crossings (every 100 points)
            old_milestone = calculate_reward.best_score // 100
            new_milestone = current_score // 100
            
            if new_milestone > old_milestone:
                # Huge reward for crossing 100-point milestones
                milestone_reward = 500.0 * (new_milestone - old_milestone)
                print(f"\nMILESTONE ACHIEVED! Score {new_milestone * 100} reached! Reward: +{milestone_reward}")
                base_reward += milestone_reward
            
            calculate_reward.best_score = current_score
        calculate_reward.last_score = current_score
    else:
        calculate_reward.last_score = dino_state.get('score', 0)
        calculate_reward.best_score = dino_state.get('best_score', 0)
    
    # No obstacle case - strongly penalize unnecessary actions
    if not obstacle:
        if action == 1:  # Jump
            return base_reward - 10.0  # Strong penalty for jumping with no obstacle
        if action == 2:  # Duck
            return base_reward - 5.0  # Penalty for ducking with no obstacle
        return base_reward + 0.5  # Reward for running normally
    
    distance = obstacle['x']
    is_cluster = obstacle['type'] == 'CACTUS_CLUSTER'
    is_bird = obstacle['isBird']
    
    # Get game speed from dino state (if available)
    game_speed = dino_state.get('speed', 6.0)
    
    # Scale rewards based on game speed/difficulty
    difficulty_multiplier = min(2.0, max(1.0, game_speed / 6.0))
    
    # Define optimal jump distance range
    optimal_jump_start = 90  # Start jumping around this distance
    optimal_jump_end = 40    # Must jump by this distance
    
    # Track successful jumps over cacti
    if not hasattr(calculate_reward, 'last_obstacle_x'):
        calculate_reward.last_obstacle_x = None
        calculate_reward.successful_jump = False
    
    # Check if we've successfully passed an obstacle
    if calculate_reward.last_obstacle_x is not None:
        if distance > calculate_reward.last_obstacle_x:  # New obstacle
            if not done:  # If we're not dead, we successfully passed the last obstacle
                calculate_reward.successful_jump = True
    
    calculate_reward.last_obstacle_x = distance if obstacle else None
    
    # Strongly penalize very early jumps
    if action == 1 and distance > optimal_jump_start + 30:
        return base_reward - 15.0  # Heavy penalty for jumping way too early
    
    # Penalize early jumps less severely
    if action == 1 and distance > optimal_jump_start:
        return base_reward - 8.0  # Moderate penalty for jumping too early
    
    # Only give proactive rewards if we successfully jumped over the last obstacle
    if distance < optimal_jump_start and calculate_reward.successful_jump:
        # Reward decreases as distance gets smaller (more reactive)
        proactive_factor = distance / optimal_jump_start
        base_reward *= (1.0 + proactive_factor)
    
    # Progressive reward based on distance survived
    if distance < 30:  # Just passed an obstacle
        success_reward = 25.0 * difficulty_multiplier  # Increased base success reward
        
        # Extra reward for successful jumps over cacti
        if not is_bird and action == 1 and dino_state.get('jumping', False):
            success_reward *= 2.0  # Double reward for successful cactus jump
            calculate_reward.successful_jump = True  # Mark this as a successful jump
        
        base_reward += success_reward
        
        if is_cluster:
            base_reward += 20.0 * difficulty_multiplier  # Extra reward for passing clusters
    
    # Smarter action rewards with continuous feedback
    if is_bird:
        bird_height = obstacle['y']
        if distance < optimal_jump_start:
            if bird_height > 75:  # Low bird
                if action == 2:  # Ducking
                    # More reward for early duck, less for late duck
                    duck_timing_factor = min(1.0, distance / 50.0)
                    base_reward += 12.0 * duck_timing_factor  # Increased duck reward
                else:
                    # Penalty increases as bird gets closer
                    base_reward -= 8.0 * (1.0 - distance / optimal_jump_start)
            else:  # High bird
                if action == 1:  # Jumping
                    # More reward for early jump, less for late jump
                    jump_timing_factor = min(1.0, distance / 50.0)
                    base_reward += 12.0 * jump_timing_factor  # Increased jump reward
                else:
                    # Penalty increases as bird gets closer
                    base_reward -= 8.0 * (1.0 - distance / optimal_jump_start)
    else:  # Cactus or Cactus Cluster
        if distance < optimal_jump_start:
            if not dino_state['jumping']:  # Only reward first jump
                jump_reward = 15.0 * difficulty_multiplier  # Increased base jump reward
                
                if is_cluster:
                    # Higher reward for jumping over clusters
                    jump_reward *= 2.0  # Doubled cluster reward
                
                # Scale jump reward based on timing
                jump_timing_factor = min(1.0, distance / optimal_jump_end)
                
                if action == 1:
                    if distance < optimal_jump_end:  # Perfect timing
                        base_reward += jump_reward * 2.0  # Doubled perfect timing bonus
                    else:
                        base_reward += jump_reward * jump_timing_factor
                else:
                    # Increasing penalty for not jumping as obstacle gets closer
                    base_reward -= 20.0 * (1.0 - distance / optimal_jump_end)  # Doubled penalty
                
                # Add extra penalty for being too close without jumping
                if distance < optimal_jump_end and not dino_state['jumping']:
                    base_reward -= 30.0 * (1.0 - distance / optimal_jump_end)  # Doubled progressive penalty
    
    # Smooth transition rewards
    if hasattr(calculate_reward, 'last_action'):
        # Penalize rapid action changes
        if calculate_reward.last_action != action:
            # Don't penalize transitioning to/from doing nothing
            if not (action == 0 or calculate_reward.last_action == 0):
                base_reward -= 2.0
    
    # Store last action for next call
    calculate_reward.last_action = action
    
    # Additional penalty for unnecessary actions when obstacle is far
    if distance > optimal_jump_start + 50:  # Well before optimal jump distance
        if action == 1:  # Jump
            base_reward -= 12.0  # Strong penalty for very early jump
        elif action == 2:  # Duck
            base_reward -= 6.0   # Penalty for unnecessary duck
    
    return base_reward

# Initialize the last_action attribute
calculate_reward.last_action = 0  # Start with no action
calculate_reward.last_score = 0   # Initialize last score
calculate_reward.best_score = 0   # Initialize best score
calculate_reward.last_obstacle_x = None  # Track last obstacle position
calculate_reward.successful_jump = False  # Track if last jump was successful

class TrainingStats:
    def __init__(self):
        self.scores = []
        self.avg_scores = []
        self.rewards = []
        self.avg_rewards = []
        self.epsilons = []
        self.losses = []
        
    def update(self, score, reward, epsilon, loss):
        self.scores.append(score)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        
        # Calculate moving averages
        window = 100
        self.avg_scores.append(np.mean(self.scores[-window:]))
        self.avg_rewards.append(np.mean(self.rewards[-window:]))
    
    def plot(self, save_dir):
        plt.figure(figsize=(15, 10))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(self.scores, label='Score')
        plt.plot(self.avg_scores, label='Avg Score')
        plt.title('Game Scores')
        plt.legend()
        
        # Plot rewards
        plt.subplot(2, 2, 2)
        plt.plot(self.rewards, label='Reward')
        plt.plot(self.avg_rewards, label='Avg Reward')
        plt.title('Rewards')
        plt.legend()
        
        # Plot epsilon
        plt.subplot(2, 2, 3)
        plt.plot(self.epsilons)
        plt.title('Exploration Rate (ε)')
        
        # Plot loss
        plt.subplot(2, 2, 4)
        plt.plot(self.losses)
        plt.title('Loss')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_stats.png')
        plt.close()

def main():
    # Initialize game, network and visualizer
    scanner = Scanner()
    network = Network()
    visualizer = NetworkVisualizer()
    
    # Create new memory directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    memory_dir = f'./memory/run_{timestamp}'
    
    # Try to load previous training state before creating new directory
    try:
        if os.path.exists('./memory'):
            run_dirs = [d for d in os.listdir('./memory') if d.startswith('run_')]
            if run_dirs:
                # Sort by timestamp in reverse order (newest first)
                latest_run = sorted(run_dirs, key=lambda x: x.split('_')[1], reverse=True)[0]
                latest_dir = f'./memory/{latest_run}'
                print(f"Found previous training state in {latest_dir}")
                
                # Use the load_training_state function to load everything
                if load_training_state(scanner, network, latest_dir):
                    print(f"Successfully loaded previous training state from {latest_dir}")
                else:
                    print("Failed to load previous training state, starting fresh")
    except Exception as e:
        print(f"Error checking previous state: {str(e)}")
        print("Starting fresh training session")
    
    # Create new directory after loading attempt
    os.makedirs(memory_dir, exist_ok=True)
    print(f"Created new memory directory: {memory_dir}")
    
    # Initialize stats tracking
    stats = TrainingStats()
    
    # Training loop variables
    save_interval = 1000  # Save every 1000 steps
    last_save_time = time.time()
    save_interval_time = 300  # Save every 5 minutes
    steps_since_last_save = 0
    last_milestone = 0  # Track last milestone for progress messages
    
    try:
        print("\nStarting training loop. Training will continue until manually stopped.")
        print("Press Ctrl+C to stop training\n")
        
        while True:  # Run indefinitely
            # Handle both visualizer and game window events
            if not visualizer.handle_events():
                break
                
            try:
                scanner.game_window.update()
            except:
                break
            
            # Get current state with dino info
            state, is_game_over, obstacles, dino_state = scanner.get_game_state()
            
            if is_game_over:
                print(f"\nGame Over! Score: {scanner.current_score} (Best: {scanner.best_score})")
                # Print milestone progress
                current_milestone = scanner.best_score // 100
                if current_milestone > last_milestone:
                    print(f"New milestone reached! {current_milestone * 100} points!")
                    last_milestone = current_milestone
                scanner.restart()
                time.sleep(0.5)
                continue
            
            # Normalize state and get action
            state_normalized = torch.FloatTensor(state).to(network.device)
            state_normalized = state_normalized.permute(2, 0, 1).unsqueeze(0)  # BCHW format
            state_normalized = state_normalized.float() / 255.0

            # Get Q-values using the model
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    q_values = network.model(state_normalized)
                    q_values = q_values.cpu().numpy()[0]

            # Get action
            action = network.get_action(state)
            
            # Execute action
            if action == 1:  # Jump
                scanner.jump()
            elif action == 2:  # Crouch
                scanner.crouch()
            else:  # Do nothing
                scanner.release_crouch()
            
            # Get new state and calculate reward
            next_state, done, new_obstacles, new_dino_state = scanner.get_game_state()
            reward = calculate_reward(action, new_obstacles, done, dino_state)
            
            # Store experience
            network.remember(state, action, reward, next_state, done)
            
            # Train network
            loss = network.train()
            
            # Update visualization with proper model
            visualizer.draw_network(
                q_values=q_values,
                action=action,
                epsilon=network.epsilon,
                score=scanner.current_score,
                best_score=scanner.best_score,
                obstacle=obstacles,
                dino_state=dino_state,
                model=network
            )
            
            # Update stats
            stats.update(
                score=scanner.current_score,
                reward=reward,
                epsilon=network.epsilon,
                loss=loss if loss is not None else 0
            )
            
            # Save checkpoints based on steps and time
            steps_since_last_save += 1
            current_time = time.time()
            
            if (steps_since_last_save >= save_interval or 
                current_time - last_save_time >= save_interval_time):
                print(f"\nSaving checkpoint...")
                print(f"Current Best Score: {scanner.best_score}")
                print(f"Current Milestone: {scanner.best_score // 100 * 100}")
                print(f"Current Epsilon: {network.epsilon:.4f}")
                print(f"Memory Size: {len(network.memory)}")
                
                save_training_state(scanner, network, memory_dir)
                stats.plot(memory_dir)
                
                steps_since_last_save = 0
                last_save_time = current_time
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Best Score Achieved: {scanner.best_score}")
        print(f"Final Milestone: {scanner.best_score // 100 * 100}")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nSaving final state...")
        save_training_state(scanner, network, memory_dir)
        stats.plot(memory_dir)
        scanner.cleanup()
        visualizer.cleanup()

if __name__ == "__main__":
    main()
