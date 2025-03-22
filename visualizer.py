import pygame
import numpy as np
import torch
import os

class NetworkVisualizer:
    def __init__(self, width=1100, height=500):
        pygame.init()
        self.width = width
        self.height = height
        
        # Get the screen info
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        
        # Calculate position for bottom half of screen
        window_x = (screen_width - width) // 2  # Center horizontally
        window_y = (screen_height // 2) + ((screen_height // 2 - height) // 2)  # Center in bottom half
        
        # Set window position
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{window_x},{window_y}"
        
        # Create the window
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Dino AI Visualization")
        
        # Add model attribute
        self.model = None
        
        # Enhanced colors for better visibility
        self.bg_color = (20, 20, 20)  # Darker background
        self.node_color = (80, 180, 255)  # Slightly adjusted blue
        self.active_color = (255, 180, 60)  # Warmer orange
        self.line_color = (60, 60, 60)  # Darker lines
        self.text_color = (220, 220, 220)  # Softer white
        self.warning_color = (255, 80, 80)  # Brighter red
        self.header_color = (150, 150, 150)  # Gray for headers
        
        # Network layout
        self.layer_spacing = (width - 600) // 6  # More space for text panels
        self.node_radius = 8
        self.layer_sizes = [4, 32, 64, 256, 3]
        
        # Action labels
        self.action_labels = ["Do Nothing", "Jump", "Crouch"]
        
        # Adjusted padding
        self.network_start_x = 280  # More left padding
        self.right_panel_width = 280  # Increased right panel width
        self.left_panel_width = 250   # Increased left panel width
        
        # Add spacing constants for better layout
        self.top_padding = 100  # Space for layer labels
        self.network_center_y = self.height // 2 + 30  # Moved network down slightly
        
        # Add learning metrics
        self.loss_history = []
        self.max_loss_points = 100
        
    def draw_network(self, q_values=None, action=None, epsilon=None, score=None, best_score=None, obstacle=None, dino_state=None, model=None):
        self.model = model  # Store the model reference
        self.screen.fill(self.bg_color)
        
        # Draw network structure
        self._draw_network_structure(action)
        
        # Draw obstacle info if available
        if obstacle:
            self._draw_obstacle_info(obstacle)
        
        # Draw stats and dino info in sequence
        y_offset = self._draw_stats_panel(q_values, epsilon, score, best_score)
        
        # Draw dino info below stats
        if dino_state:
            self._draw_dino_info(dino_state, start_y=y_offset)
        
        # Draw learning metrics - ensure model is not None
        if model is not None and hasattr(model, 'memory'):
            memory_size = len(model.memory) if model.memory is not None else 0
            loss_history = model.running_loss if hasattr(model, 'running_loss') else []
            self._draw_learning_metrics(loss_history, memory_size)
        
        pygame.display.flip()
    
    def _draw_network_structure(self, current_action):
        def get_connection_color(weight, activation=None):
            # Base color based on weight
            if weight > 0:
                base_color = (0, min(255, int(weight * 255)), 0)  # Green
            else:
                base_color = (min(255, int(-weight * 255)), 0, 0)  # Red
            
            # Brighten color if neuron is activated
            if activation is not None and activation > 0.5:  # Threshold for activation
                # Make color brighter by mixing with white
                return tuple(min(255, c + 100) for c in base_color)
            return base_color

        # Get weights and activations from the model
        weights = []
        activations = []
        if self.model and hasattr(self.model, 'model'):  # Check if we have the network object and it has a model
            # Get weights from PyTorch model
            for name, param in self.model.model.named_parameters():
                if 'weight' in name:
                    weights.append(param.detach().cpu().numpy())
            
            # Get intermediate layer activations using a test input
            try:
                test_input = torch.randn(1, 4, 36, 40).to(next(self.model.model.parameters()).device)
                
                # Get activations for conv layers
                x = test_input
                conv_activations = []
                
                # Conv1
                x = self.model.model.conv1(x)
                conv_activations.append(x.mean((2, 3)).detach().cpu().numpy()[0])
                
                # Conv2
                x = self.model.model.conv2(x)
                conv_activations.append(x.mean((2, 3)).detach().cpu().numpy()[0])
                
                # Flatten
                x = x.view(x.size(0), -1)
                
                # FC layers
                for layer in self.model.model.fc:
                    if isinstance(layer, torch.nn.Linear):
                        x = layer(x)
                        conv_activations.append(x.detach().cpu().numpy()[0])
                
                activations = conv_activations
            except Exception as e:
                print(f"Error getting activations: {str(e)}")
                activations = [None] * len(weights)

        # Draw connections between layers
        for layer_idx in range(len(self.layer_sizes) - 1):
            start_nodes = min(self.layer_sizes[layer_idx], 10)
            end_nodes = min(self.layer_sizes[layer_idx + 1], 10)
            
            # Get activations for this layer
            curr_activation = activations[layer_idx] if layer_idx < len(activations) else None
            next_activation = activations[layer_idx + 1] if layer_idx + 1 < len(activations) else None
            
            for i in range(start_nodes):
                start_x = self.network_start_x + (layer_idx + 1) * self.layer_spacing
                start_y = self.network_center_y + (i - start_nodes // 2) * 30
                
                # Get activation value for the source neuron
                src_activation = curr_activation[i] if curr_activation is not None else None
                
                for j in range(end_nodes):
                    end_x = self.network_start_x + (layer_idx + 2) * self.layer_spacing
                    end_y = self.network_center_y + (j - end_nodes // 2) * 30
                    
                    # Get activation value for the target neuron
                    tgt_activation = next_activation[j] if next_activation is not None else None
                    
                    try:
                        weight = weights[layer_idx][i, j] if weights else 0
                        # Use both weight and activation for color
                        color = get_connection_color(weight, max(src_activation, tgt_activation))
                        thickness = max(1, min(3, int(abs(weight) * 3)))
                    except:
                        color = self.line_color
                        thickness = 1
                    
                    # Highlight connections to active output
                    if layer_idx == len(self.layer_sizes) - 2:  # Last layer
                        if j == current_action:
                            color = self.active_color
                            thickness = 2
                    
                    # Draw connection
                    pygame.draw.line(self.screen, color, (start_x, start_y),
                                   (end_x, end_y), thickness)

        # Draw nodes with activation-based colors
        font = pygame.font.Font(None, 24)
        layer_labels = ["Input", "Conv1", "Conv2", "Dense", "Output"]
        
        for i, size in enumerate(self.layer_sizes):
            x = self.network_start_x + (i + 1) * self.layer_spacing
            
            # Draw layer label with more space above nodes
            label = font.render(layer_labels[i], True, self.text_color)
            label_y = self.network_center_y - 180  # Increased space above nodes
            self.screen.blit(label, (x - 20, label_y))
            
            # Get activations for this layer
            layer_activation = activations[i] if i < len(activations) else None
            
            for j in range(min(size, 10)):
                # Calculate y position relative to network center
                y = self.network_center_y + (j - min(size, 10) // 2) * 30
                
                # Determine node color based on activation
                if i == len(self.layer_sizes) - 1:  # Output layer
                    color = self.active_color if j == current_action else self.node_color
                else:
                    # Color based on activation
                    activation = layer_activation[j] if layer_activation is not None else 0
                    if activation > 0.5:  # Activated
                        color = (min(255, self.node_color[0] + 100),
                                min(255, self.node_color[1] + 100),
                                min(255, self.node_color[2] + 100))
                    else:
                        color = self.node_color
                
                # Draw node with border
                pygame.draw.circle(self.screen, color, (x, y), self.node_radius)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), self.node_radius, 1)
                
                # Draw action labels for output layer with adjusted position
                if i == len(self.layer_sizes) - 1 and j < len(self.action_labels):
                    label = font.render(self.action_labels[j], True, color)
                    self.screen.blit(label, (x + 20, y - 10))
            
            # Show ellipsis if more nodes exist (adjusted position)
            if size > 10:
                y = self.network_center_y + (5 * 30)
                pygame.draw.circle(self.screen, self.node_color, (x, y), 3)
                pygame.draw.circle(self.screen, self.node_color, (x, y + 10), 3)
                pygame.draw.circle(self.screen, self.node_color, (x, y + 20), 3)
    
    def _draw_stats_panel(self, q_values, epsilon, score, best_score):
        font = pygame.font.Font(None, 32)
        header_font = pygame.font.Font(None, 36)
        y_offset = 20
        x_pos = self.width - self.right_panel_width
        
        # Draw Q-values with improved formatting
        if q_values is not None:
            self.screen.blit(header_font.render("Q-Values:", True, self.header_color), (x_pos, y_offset))
            y_offset += 35
            for i, q in enumerate(q_values):
                color = self.active_color if i == np.argmax(q_values) else self.text_color
                text = f"{self.action_labels[i]}: {q:.3f}"
                self.screen.blit(font.render(text, True, color), (x_pos + 10, y_offset))
                y_offset += 25
        
        # Draw epsilon with separator
        if epsilon is not None:
            y_offset += 15
            pygame.draw.line(self.screen, self.line_color, (x_pos, y_offset), 
                           (x_pos + 200, y_offset), 1)  # Wider separator
            y_offset += 15
            text = f"Exploration (ε): {epsilon:.3f}"
            self.screen.blit(font.render(text, True, self.text_color), (x_pos + 10, y_offset))
        
        # Draw scores with separator
        if score is not None:
            y_offset += 15
            pygame.draw.line(self.screen, self.line_color, (x_pos, y_offset), 
                           (x_pos + 200, y_offset), 1)  # Wider separator
            y_offset += 15
            self.screen.blit(font.render(f"Score: {int(score)}", True, self.text_color), 
                           (x_pos + 10, y_offset))
        
        if best_score is not None:
            y_offset += 25
            self.screen.blit(font.render(f"Best: {int(best_score)}", True, self.active_color), 
                           (x_pos + 10, y_offset))
            y_offset += 35  # Extra space before dino info

        return y_offset  # Return current y_offset for dino info positioning
    
    def _draw_obstacle_info(self, obstacles):
        font = pygame.font.Font(None, 32)
        header_font = pygame.font.Font(None, 36)
        y_offset = 20
        x_pos = 40
        
        # Draw header
        header = "Closest Obstacle:"
        self.screen.blit(header_font.render(header, True, self.header_color), (x_pos, y_offset))
        y_offset += 35
        
        if obstacles:  # Now obstacles is a single obstacle dict
            is_cluster = obstacles['type'] == 'CACTUS_CLUSTER'
            info_lines = [
                f"Type: {obstacles['type']}",
                f"Distance: {obstacles['x']:.1f}",
                f"Height: {obstacles['y']:.1f}",
                f"Size: {obstacles['width']:.1f}x{obstacles['height']:.1f}"
            ]
            
            for line in info_lines:
                color = self.warning_color if is_cluster else self.text_color
                text = font.render(line, True, color)
                self.screen.blit(text, (x_pos + 10, y_offset))
                y_offset += 25
            
            if obstacles['isBird']:
                text = font.render("Type: Bird", True, self.warning_color)
                self.screen.blit(text, (x_pos + 10, y_offset))
                y_offset += 25
            elif is_cluster:
                text = font.render("Warning: Multiple Cacti!", True, self.warning_color)
                self.screen.blit(text, (x_pos + 10, y_offset))
                y_offset += 25
        else:
            self.screen.blit(font.render("No obstacles", True, self.text_color), 
                           (x_pos + 10, y_offset))
    
    def _draw_dino_info(self, dino_state, start_y):
        if not dino_state:
            return
        
        font = pygame.font.Font(None, 32)
        header_font = pygame.font.Font(None, 36)
        x_pos = self.width - self.right_panel_width
        y_offset = start_y + 15  # Add some spacing from previous section
        
        # Draw separator
        pygame.draw.line(self.screen, self.line_color, (x_pos, y_offset), 
                        (x_pos + 200, y_offset), 1)
        y_offset += 15
        
        # Draw header
        self.screen.blit(header_font.render("Dino Info:", True, self.header_color), 
                        (x_pos, y_offset))
        y_offset += 35
        
        try:
            info_lines = [
                f"Position: ({dino_state.get('x', 0):.1f}, {dino_state.get('y', 0):.1f})",
                f"Speed: {dino_state.get('speed', 0):.1f}",
                f"Jumping: {dino_state.get('jumping', False)}",
                f"Ducking: {dino_state.get('ducking', False)}"
            ]
            
            for line in info_lines:
                text = font.render(line, True, self.text_color)
                self.screen.blit(text, (x_pos + 10, y_offset))
                y_offset += 25
        except Exception as e:
            print(f"Error drawing dino info: {str(e)}")
    
    def _draw_learning_metrics(self, loss_history, memory_size):
        """Draw learning metrics like loss history and memory size"""
        font = pygame.font.Font(None, 32)
        header_font = pygame.font.Font(None, 36)
        
        # Position in bottom left corner with padding
        x_pos = 20
        y_pos = self.height - 120
        
        # Draw background panel
        panel_width = 500
        panel_height = 110
        pygame.draw.rect(self.screen, (30, 30, 30), 
                        (x_pos, y_pos, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.line_color, 
                        (x_pos, y_pos, panel_width, panel_height), 1)
        
        # Draw header
        self.screen.blit(header_font.render("Learning Progress:", True, self.header_color), 
                        (x_pos + 10, y_pos + 5))
        
        # Draw memory info
        memory_color = self.warning_color if memory_size < 1000 else self.text_color
        memory_text = f"Memory: {memory_size:,} / 50,000 samples"
        self.screen.blit(font.render(memory_text, True, memory_color), 
                        (x_pos + 10, y_pos + 40))
        
        # Draw loss info if available
        if loss_history:
            avg_loss = sum(loss_history[-100:]) / len(loss_history[-100:])
            loss_text = f"Avg Loss (100): {avg_loss:.5f}"
            self.screen.blit(font.render(loss_text, True, self.text_color), 
                           (x_pos + 10, y_pos + 70))
            
            # Draw mini loss graph
            graph_width = 200
            graph_height = 40
            graph_x = x_pos + 280
            graph_y = y_pos + 45
            
            # Draw graph background
            pygame.draw.rect(self.screen, (40, 40, 40), 
                           (graph_x, graph_y, graph_width, graph_height))
            pygame.draw.rect(self.screen, self.line_color, 
                           (graph_x, graph_y, graph_width, graph_height), 1)
            
            # Draw loss line
            if len(loss_history) > 1:
                recent_losses = loss_history[-100:]
                if len(recent_losses) > 1:
                    max_loss = max(recent_losses)
                    min_loss = min(recent_losses)
                    loss_range = max_loss - min_loss if max_loss != min_loss else 1
                    
                    points = []
                    for i, loss in enumerate(recent_losses):
                        x = graph_x + (i * graph_width / len(recent_losses))
                        y = graph_y + graph_height - ((loss - min_loss) / loss_range * graph_height)
                        points.append((int(x), int(y)))
                    
                    if len(points) > 1:
                        pygame.draw.lines(self.screen, self.active_color, False, points, 2)
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def cleanup(self):
        pygame.quit() 