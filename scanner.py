from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import os
import cv2
import numpy as np
from PIL import ImageGrab, ImageDraw, Image
import time
import json
from collections import deque
import cv2.cuda as cuda  # For GPU acceleration if available
import tkinter as tk
from PIL import ImageTk
import mss
import threading
from queue import Queue

class Scanner:
    def __init__(self, game_url="chrome://dino"):
        # Store tk and ImageTk as class attributes
        self.tk = tk
        self.ImageTk = ImageTk
        
        # Create the game window at initialization
        self.game_window = self.tk.Tk()
        self.game_window.title("Game View")
        self.game_window.geometry("600x150+0+0")  # Initial size, will be updated
        self.label = self.tk.Label(self.game_window)
        self.label.pack()
        
        # Add frame processing attributes
        self.frame_queue = Queue(maxsize=2)
        self.running = True
        self.last_viz_update = time.time()
        self.window_update_counter = 0
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0
        self.last_window_update = time.time()
        self.viz_fps = 30  # Target FPS for visualization
        
        # Initialize screen capture (moved to thread)
        self.sct = None
        
        # Initialize Chrome with offline mode
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--window-position=0,0")
        # Add offline mode capabilities
        chrome_options.add_argument("--disable-network-connections")
        chrome_options.add_argument("--disable-web-security")
        self._driver = webdriver.Chrome(options=chrome_options)
        self._driver.set_window_size(800, 400)
        
        # Try multiple methods to access the dino game
        methods = [
            lambda: self._driver.get(game_url),  # Direct access
            lambda: self._driver.execute_script(
                "window.location.href = 'chrome://dino';"
            ),  # JavaScript navigation
            lambda: (  # Offline mode trigger
                self._driver.get("http://example.com"),
                self._driver.get(game_url)
            )
        ]
        
        success = False
        for method in methods:
            try:
                method()
                # Verify the game loaded by checking for the Runner object
                self._driver.execute_script("return Runner")
                success = True
                break
            except:
                continue
        
        if not success:
            raise Exception("Could not load the dino game through any available method")
        
        # Wait for game to load and start it
        time.sleep(1)
        
        # Press space to start the game
        body = self._driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.SPACE)
        
        # Load or create calibration
        self.game_bbox = self.load_or_calibrate()
        
        # Initialize CUDA if available
        try:
            self.use_gpu = cuda.getCudaEnabledDeviceCount() > 0
            if self.use_gpu:
                print("CUDA enabled for image processing")
        except:
            self.use_gpu = False
            print("CUDA not available, using CPU")
        
        # Try to modify game config for training (may not work in chrome://dino)
        try:
            self._driver.execute_script("Runner.config.ACCELERATION=0")
            self._driver.execute_script("Runner.config.SPEED=6")
        except:
            print("Could not modify game speed - using default speed")
        
        # Game state
        self.steps_survived = 0
        self.best_time = 0
        self.current_game_time = 0
        self.total_games = 0
        self.current_score = 0
        self.best_score = 0
        self.last_game_score = 0
        
        # Add frame buffer for smoother state representation
        self.frame_buffer = deque(maxlen=4)
        
        # Add performance tracking
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.avg_fps = 0
        
        # Start background processing thread AFTER calibration
        self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()
    
    def load_or_calibrate(self):
        """Load existing calibration or create new one"""
        import tkinter as tk
        from tkinter import messagebox
        
        CALIBRATION_DIR = './calibrations'
        CALIBRATION_FILE = f'{CALIBRATION_DIR}/game_area.json'
        
        # Create calibration directory if it doesn't exist
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
        
        # Check if calibration file exists
        if os.path.exists(CALIBRATION_FILE):
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            if messagebox.askyesno("Calibration", 
                "Found existing calibration. Would you like to use it?\n"
                "No to create new calibration."):
                with open(CALIBRATION_FILE, 'r') as f:
                    bbox = json.load(f)
                    return tuple(bbox)
        
        # Create new calibration
        bbox = self.calibrate_game_area()
        
        # Save calibration
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(list(bbox), f)
        
        return bbox
    
    def calibrate_game_area(self):
        """Create a window to calibrate game area"""
        import tkinter as tk
        from tkinter import ttk
        
        root = tk.Tk()
        root.title("Game Area Calibration")
        
        # Default values - initial guess
        default_x = self._driver.get_window_position()['x']
        default_y = self._driver.get_window_position()['y'] + 75  # Skip chrome header
        default_width = 600
        default_height = 150
        
        # Variables to store values
        x_var = tk.IntVar(value=default_x)
        y_var = tk.IntVar(value=default_y)
        width_var = tk.IntVar(value=default_width)
        height_var = tk.IntVar(value=default_height)
        
        def update_preview():
            try:
                # Get screenshot with current values
                bbox = (x_var.get(), y_var.get(), 
                       x_var.get() + width_var.get(), 
                       y_var.get() + height_var.get())
                preview = ImageGrab.grab(bbox=bbox)
                
                # Convert to OpenCV format and show
                frame = np.array(preview)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('Preview', frame_rgb)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Preview error: {str(e)}")
        
        def save_and_close():
            bbox = (x_var.get(), y_var.get(), 
                   x_var.get() + width_var.get(), 
                   y_var.get() + height_var.get())
            root.bbox = bbox
            root.quit()
        
        # Create sliders
        ttk.Label(root, text="X Position:").pack()
        ttk.Scale(root, from_=0, to=2000, variable=x_var, 
                  command=lambda _: update_preview()).pack()
        
        ttk.Label(root, text="Y Position:").pack()
        ttk.Scale(root, from_=0, to=2000, variable=y_var,
                  command=lambda _: update_preview()).pack()
        
        ttk.Label(root, text="Width:").pack()
        ttk.Scale(root, from_=100, to=1000, variable=width_var,
                  command=lambda _: update_preview()).pack()
        
        ttk.Label(root, text="Height:").pack()
        ttk.Scale(root, from_=50, to=500, variable=height_var,
                  command=lambda _: update_preview()).pack()
        
        # Save button
        ttk.Button(root, text="Save and Continue", command=save_and_close).pack(pady=10)
        
        # Show initial preview
        update_preview()
        
        # Start calibration window
        root.mainloop()
        
        # Clean up
        cv2.destroyAllWindows()
        
        # Get final bbox
        bbox = getattr(root, 'bbox', (default_x, default_y, 
                                     default_x + default_width, 
                                     default_y + default_height))
        root.destroy()
        return bbox
    
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")
    
    def restart(self):
        """Restart game and save last score"""
        self.last_game_score = self.current_score
        self.total_games += 1
        
        try:
            # Try the standard restart first
            self._driver.execute_script("Runner.instance_.restart()")
        except:
            # If that fails, try to restart by pressing space
            body = self._driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.SPACE)
        
        time.sleep(0.25)  # Wait for game to start
        self.current_score = 0
        self.steps_survived = 0
        self.current_game_time = 0
    
    def jump(self):
        # Updated to use new Selenium syntax
        body = self._driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ARROW_UP)
        
    def get_score(self):
        """Get actual score from game's display"""
        try:
            # Get score from the distance meter digits
            score_digits = self._driver.execute_script("""
                var digits = Runner.instance_.distanceMeter.digits;
                return digits.join('');
            """)
            
            # Convert to integer, handle empty string case
            self.current_score = int(score_digits) if score_digits else 0
            self.best_score = max(self.best_score, self.current_score)
            return self.current_score
        except Exception as e:
            print(f"Error getting score: {str(e)}")
            return self.current_score
    
    def get_high_score(self):
        """Get high score from game's display"""
        try:
            # Get high score digits
            high_score = self._driver.execute_script("""
                var highScore = Runner.instance_.distanceMeter.highScore;
                return highScore.join('');
            """)
            return int(high_score) if high_score else 0
        except Exception as e:
            print(f"Error getting high score: {str(e)}")
            return 0
    
    def preprocess_frame(self, frame):
        """Enhanced frame preprocessing"""
        try:
            if self.use_gpu:
                # Use GPU for image processing
                gpu_frame = cuda.GpuMat()
                gpu_frame.upload(frame)
                
                # GPU operations
                gpu_gray = cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                gpu_binary = cuda.threshold(gpu_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
                
                # Download results
                binary = gpu_binary.download()
            else:
                # CPU operations
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Additional preprocessing
            binary = cv2.medianBlur(binary, 3)  # Remove noise
            binary = cv2.dilate(binary, None, iterations=1)  # Enhance edges
            
            return binary
            
        except Exception as e:
            print(f"Preprocessing error: {str(e)}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def get_game_state(self):
        """Get current game state"""
        try:
            # Get latest processed frame
            if self.frame_queue.empty():
                return self.last_state if hasattr(self, 'last_state') else (np.zeros((36, 40, 4)), False, None, None)
            
            frame_data = self.frame_queue.get()
            frame = frame_data['original']
            binary = frame_data['processed']
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find contours on the preprocessed binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            obstacle_data = []
            dino_data = None
            DINO_X = 40  # Fixed dino x position
            
            # First pass - find the leftmost contour (dino)
            leftmost_contour = None
            leftmost_x = float('inf')
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Minimum area threshold
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                if x < leftmost_x and w * h > 100:
                    leftmost_x = x
                    leftmost_contour = contour
            
            # Process dino if found
            if leftmost_contour is not None:
                x, y, w, h = cv2.boundingRect(leftmost_contour)
                dino_data = {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'bottom': y + h
                }
                # Draw dino box in green
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_rgb, "DINO", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                DINO_X = x + w//2
            
            # Second pass - find obstacles
            potential_obstacles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip if this is the dino contour
                if leftmost_contour is not None and np.array_equal(contour, leftmost_contour):
                    continue
                
                # Filter out ground
                if y > binary.shape[0] - 20:
                    continue
                
                # Calculate relative distance from dino
                distance_from_dino = x - DINO_X
                
                # Only consider obstacles ahead of dino
                if distance_from_dino > 0:
                    is_bird = (y < binary.shape[0] // 2 and float(w)/h > 1.2) and distance_from_dino > 30
                    potential_obstacles.append({
                        'type': 'PTERODACTYL' if is_bird else 'CACTUS',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'isBird': is_bird,
                        'distance': distance_from_dino
                    })
            
            # Merge nearby cacti into clusters
            merged_obstacles = []
            used_indices = set()
            
            for i, obs1 in enumerate(potential_obstacles):
                if i in used_indices:
                    continue
                
                if obs1['isBird']:
                    merged_obstacles.append({
                        'type': 'PTERODACTYL',
                        'x': obs1['distance'],
                        'y': obs1['y'],
                        'width': obs1['width'],
                        'height': obs1['height'],
                        'isBird': True,
                        'canvasX': obs1['x'],
                        'canvasY': obs1['y']
                    })
                    used_indices.add(i)
                    continue
                
                cluster = {
                    'x_min': obs1['x'],
                    'x_max': obs1['x'] + obs1['width'],
                    'y_min': obs1['y'],
                    'y_max': obs1['y'] + obs1['height'],
                    'distance': obs1['distance']
                }
                used_indices.add(i)
                
                for j, obs2 in enumerate(potential_obstacles):
                    if j in used_indices or obs2['isBird']:
                        continue
                    
                    if abs(obs2['x'] - cluster['x_max']) < 20:
                        cluster['x_max'] = max(cluster['x_max'], obs2['x'] + obs2['width'])
                        cluster['y_min'] = min(cluster['y_min'], obs2['y'])
                        cluster['y_max'] = max(cluster['y_max'], obs2['y'] + obs2['height'])
                        cluster['distance'] = min(cluster['distance'], obs2['distance'])
                        used_indices.add(j)
                
                merged_obstacles.append({
                    'type': 'CACTUS_CLUSTER' if len(used_indices) > 1 else 'CACTUS',
                    'x': cluster['distance'],
                    'y': cluster['y_min'],
                    'width': cluster['x_max'] - cluster['x_min'],
                    'height': cluster['y_max'] - cluster['y_min'],
                    'isBird': False,
                    'canvasX': cluster['x_min'],
                    'canvasY': cluster['y_min']
                })
            
            # Find closest obstacle
            closest_obstacle = None
            if merged_obstacles:
                closest_obstacle = min(merged_obstacles, key=lambda x: x['x'])
                
                # Draw closest obstacle
                x = closest_obstacle['canvasX']
                y = closest_obstacle['canvasY']
                w = closest_obstacle['width']
                h = closest_obstacle['height']
                is_bird = closest_obstacle['isBird']
                is_cluster = closest_obstacle['type'] == 'CACTUS_CLUSTER'
                
                color = (255, 0, 0) if is_bird else (0, 0, 255)
                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), color, 2)
                label = f"{'BIRD' if is_bird else 'CACTUS CLUSTER' if is_cluster else 'CACTUS'} ({closest_obstacle['x']}px)"
                cv2.putText(frame_rgb, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw dino reference line
            cv2.line(frame_rgb, (DINO_X, 0), (DINO_X, frame_rgb.shape[0]), (0, 255, 0), 1)
            
            # Update visualization at target FPS
            current_time = time.time()
            if current_time - self.last_viz_update >= 1.0/self.viz_fps:
                # Convert to PIL Image for display
                game_view = Image.fromarray(frame_rgb)
                
                # Update window size less frequently
                if current_time - self.last_window_update >= 1.0:
                    if game_view.width != self.game_window.winfo_width() or \
                       game_view.height != self.game_window.winfo_height():
                        self.game_window.geometry(f"{game_view.width}x{game_view.height}+0+0")
                    self.last_window_update = current_time
                
                # Update the image
                photo = self.ImageTk.PhotoImage(game_view)
                self.label.configure(image=photo)
                self.label.image = photo
                self.game_window.update()
                
                self.last_viz_update = current_time
            
            # Get dino's state from game
            try:
                game_dino_state = self._driver.execute_script("""
                    var dino = Runner.instance_.tRex;
                    return {
                        jumping: dino.jumping || false,
                        ducking: dino.ducking || false,
                        yPos: dino.yPos || 0,
                        speed: Runner.instance_.currentSpeed || 6
                    }
                """)
                
                dino_state = {
                    **game_dino_state,
                    'detected_x': dino_data['x'] if dino_data else DINO_X,
                    'detected_y': dino_data['y'] if dino_data else 0,
                    'detected_height': dino_data['height'] if dino_data else 0,
                    'detected_bottom': dino_data['bottom'] if dino_data else 0
                }
            except Exception as e:
                print(f"Error getting dino state: {str(e)}")
                dino_state = {
                    'jumping': False,
                    'ducking': False,
                    'yPos': 0,
                    'speed': 6,
                    'detected_x': DINO_X,
                    'detected_y': 0,
                    'detected_height': 0,
                    'detected_bottom': 0
                }
            
            # Update frame buffer for state representation
            processed = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (40, 36))
            processed = cv2.Canny(processed, threshold1=100, threshold2=200)
            self.frame_buffer.append(processed)
            
            if len(self.frame_buffer) < 4:
                self.frame_buffer.extend([processed] * (4 - len(self.frame_buffer)))
            
            state = np.stack(list(self.frame_buffer), axis=2)
            
            # Check game state and update score
            is_game_over = self.get_crashed()
            
            if not is_game_over:
                self.steps_survived += 1
                self.current_game_time += 1
                self.current_score = self.get_score()
            
            # Store last state
            self.last_state = (state, is_game_over, closest_obstacle, dino_state)
            return self.last_state
            
        except Exception as e:
            print(f"Game state error: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros((36, 40, 4)), True, None, None
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
        try:
            if hasattr(self, 'sct') and self.sct:
                try:
                    self.sct.close()
                except:
                    pass  # Ignore mss cleanup errors
        except:
            pass
        try:
            self.game_window.destroy()
        except:
            pass
        try:
            self._driver.quit()
        except:
            pass
    
    def _process_frames(self):
        """Background thread for frame processing"""
        # Initialize MSS in the thread that will use it
        self.sct = mss.mss()
        
        while self.running:
            try:
                if self.frame_queue.full():
                    # Remove old frame if queue is full
                    self.frame_queue.get()
                
                # Capture screen using mss (faster than ImageGrab)
                monitor = {
                    "top": self.game_bbox[1],
                    "left": self.game_bbox[0],
                    "width": self.game_bbox[2] - self.game_bbox[0],
                    "height": self.game_bbox[3] - self.game_bbox[1]
                }
                
                try:
                    screenshot = np.array(self.sct.grab(monitor))
                except Exception as e:
                    print(f"Screen capture error: {str(e)}")
                    time.sleep(0.1)
                    continue
                
                # Convert BGRA to BGR
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                
                # Process frame
                if self.use_gpu:
                    # GPU processing
                    gpu_frame = cuda.GpuMat()
                    gpu_frame.upload(frame)
                    gpu_gray = cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                    gpu_binary = cuda.threshold(gpu_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
                    binary = gpu_binary.download()
                else:
                    # CPU processing
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                
                # Add processed frame to queue
                self.frame_queue.put({
                    'original': frame,
                    'processed': binary,
                    'timestamp': time.time()
                })
                
                # Control capture rate
                time.sleep(1/120)  # Cap at 120 FPS for capture
                
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                time.sleep(0.1)  # Wait before retrying
    
    def crouch(self):
        """Press down key to crouch"""
        body = self._driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ARROW_DOWN)
    
    def release_crouch(self):
        """Release down key to stop crouching"""
        body = self._driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ARROW_DOWN)  # Release key