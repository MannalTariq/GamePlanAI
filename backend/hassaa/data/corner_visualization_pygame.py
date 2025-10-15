#!/usr/bin/env python3
"""
Pygame-based Corner Kick Visualization Engine
Replaces the Matplotlib-based animation with a smoother, more realistic Pygame implementation
"""

import pygame
import math
import json
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Any

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
PITCH_COLOR = (35, 77, 32)  # Dark green
PITCH_LINE_COLOR = (255, 255, 255)  # White
ATTACKER_COLOR = (255, 68, 68)  # Red
DEFENDER_COLOR = (74, 163, 255)  # Blue
KEEPER_COLOR = (155, 89, 182)  # Purple
BALL_COLOR = (255, 255, 255)  # White
SHADOW_COLOR = (0, 0, 0, 128)  # Black with transparency
TRAIL_COLOR = (255, 255, 255, 100)  # White with transparency

class PlayerSprite:
    """Represents a player sprite with team-specific visuals"""
    
    def __init__(self, x: float, y: float, player_id: int, team: str):
        self.x = x
        self.y = y
        self.player_id = player_id
        self.team = team
        self.size = 20
        self.jump_height = 0
        self.will_jump = False
        self.jump_phase = 0.0
        self.direction = 0  # Angle in radians for facing direction
        self.speed = 0  # Current movement speed for visual effects
        
        # Set color based on team
        if team == 'attacker':
            self.color = ATTACKER_COLOR
        elif team == 'defender':
            self.color = DEFENDER_COLOR
        elif team == 'keeper':
            self.color = KEEPER_COLOR
        else:
            self.color = (200, 200, 200)  # Default gray
            
    def draw(self, screen, camera_x=0, camera_y=0):
        """Draw the player sprite with team-specific shape and details"""
        # Adjust position based on camera
        draw_x = self.x - camera_x
        draw_y = self.y - camera_y
        
        # Apply jump offset
        draw_y -= self.jump_height
        
        # Draw glow effect based on team and speed
        glow_intensity = 50 + int(self.speed * 20)  # More glow when moving faster
        glow_surf = pygame.Surface((self.size * 3, self.size * 3), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.color[:3], min(100, glow_intensity)), 
                          (self.size * 1.5, self.size * 1.5), self.size * 1.5)
        screen.blit(glow_surf, (draw_x - self.size * 1.5, draw_y - self.size * 1.5))
        
        # Draw player based on team with enhanced visuals
        if self.team == 'keeper':
            # Square for keeper with more details
            pygame.draw.rect(screen, self.color, 
                           (draw_x - self.size//2, draw_y - self.size//2, self.size, self.size))
            pygame.draw.rect(screen, (255, 255, 255), 
                           (draw_x - self.size//2, draw_y - self.size//2, self.size, self.size), 2)
            # Draw 'K' on keeper
            font = pygame.font.Font(None, 20)
            text = font.render("K", True, (255, 255, 255))
            text_rect = text.get_rect(center=(draw_x, draw_y))
            screen.blit(text, text_rect)
        elif self.team == 'defender':
            # Triangle for defender with more details
            points = [
                (draw_x, draw_y - self.size//2),
                (draw_x - self.size//2, draw_y + self.size//2),
                (draw_x + self.size//2, draw_y + self.size//2)
            ]
            pygame.draw.polygon(screen, self.color, points)
            pygame.draw.polygon(screen, (255, 255, 255), points, 2)
            # Draw 'D' on defender
            font = pygame.font.Font(None, 20)
            text = font.render("D", True, (255, 255, 255))
            text_rect = text.get_rect(center=(draw_x, draw_y))
            screen.blit(text, text_rect)
        else:
            # Circle for attacker with more details
            pygame.draw.circle(screen, self.color, (int(draw_x), int(draw_y)), self.size//2)
            pygame.draw.circle(screen, (255, 255, 255), (int(draw_x), int(draw_y)), self.size//2, 2)
            # Draw 'A' on attacker
            font = pygame.font.Font(None, 20)
            text = font.render("A", True, (255, 255, 255))
            text_rect = text.get_rect(center=(draw_x, draw_y))
            screen.blit(text, text_rect)
            
        # Draw player ID with better visibility
        font = pygame.font.Font(None, 20)
        text = font.render(str(self.player_id), True, (255, 255, 255))
        text_rect = text.get_rect(center=(draw_x, draw_y - self.size - 5))
        # Draw background for better visibility
        bg_rect = text_rect.inflate(6, 4)
        pygame.draw.rect(screen, (0, 0, 0, 200), bg_rect, border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255, 100), bg_rect, 1, border_radius=4)
        screen.blit(text, text_rect)
        
        # Draw direction indicator if moving significantly
        if abs(self.speed) > 0.1:
            end_x = draw_x + math.cos(self.direction) * self.size
            end_y = draw_y + math.sin(self.direction) * self.size
            pygame.draw.line(screen, (255, 255, 255), (draw_x, draw_y), (end_x, end_y), 2)
        
    def update_position(self, x: float, y: float, direction: float = 0):
        """Update player position and direction"""
        # Calculate speed based on position change
        if hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            dx = x - self.prev_x
            dy = y - self.prev_y
            self.speed = math.sqrt(dx*dx + dy*dy)
        else:
            self.speed = 0
            
        self.prev_x = x
        self.prev_y = y
        self.x = x
        self.y = y
        self.direction = direction
        
    def set_jump(self, will_jump: bool):
        """Set whether this player will jump"""
        self.will_jump = will_jump
        
    def update_jump(self, phase: float):
        """Update jump animation"""
        self.jump_phase = phase
        self.jump_height = 0.8 * math.sin(math.pi * phase) if phase > 0 else 0

class BallSprite:
    """Represents the ball with realistic movement and effects"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.z = 0  # Height above ground
        self.size = 12
        self.trail = []
        self.rotation = 0  # Rotation angle for visual effect
        self.sparkle_effect = []  # Sparkle effects when ball is kicked or received
        
    def draw(self, screen, camera_x=0, camera_y=0):
        """Draw the ball with shadow, trail, and rotation effects"""
        # Adjust position based on camera
        draw_x = self.x - camera_x
        draw_y = self.y - camera_y - self.z
        
        # Draw trail with fading effect and varying sizes
        for i, (trail_x, trail_y, trail_z, trail_time) in enumerate(self.trail):
            # Fade out older trail points
            age = len(self.trail) - i
            alpha = max(20, 255 - age * 8)
            
            # Vary size based on age (newer = larger)
            size = max(2, 6 - age * 0.2)
            
            trail_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (255, 255, 255, alpha), (size, size), size)
            screen.blit(trail_surf, (trail_x - camera_x - size, trail_y - camera_y - trail_z - size))
        
        # Draw sparkle effects
        current_time = pygame.time.get_ticks()
        for i, (sparkle_x, sparkle_y, sparkle_time, sparkle_size) in enumerate(self.sparkle_effect[:]):
            age = current_time - sparkle_time
            if age > 500:  # Remove after 500ms
                self.sparkle_effect.remove((sparkle_x, sparkle_y, sparkle_time, sparkle_size))
                continue
                
            # Fade out over time
            alpha = max(0, 255 - (age / 500) * 255)
            size = sparkle_size * (1 - age / 500)
            
            sparkle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(sparkle_surf, (255, 255, 200, alpha), (size, size), size)
            screen.blit(sparkle_surf, (sparkle_x - camera_x - size, sparkle_y - camera_y - size))
        
        # Draw shadow with scaling based on height
        shadow_size_x = max(4, self.size * (1 - self.z / 30))
        shadow_size_y = max(2, self.size * 0.3 * (1 - self.z / 30))
        shadow_surf = pygame.Surface((shadow_size_x * 2, shadow_size_y * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (*SHADOW_COLOR[:3], 120), 
                          (0, 0, shadow_size_x * 2, shadow_size_y * 2))
        screen.blit(shadow_surf, (draw_x - shadow_size_x, draw_y + self.size//2))
        
        # Draw ball with rotation effect
        ball_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        
        # Draw white ball with subtle gradient
        pygame.draw.circle(ball_surf, BALL_COLOR, (self.size, self.size), self.size)
        
        # Draw pentagons to simulate a soccer ball
        pentagon_color = (20, 20, 20)  # Dark pentagons
        # Draw several pentagons rotated at different angles
        for i in range(5):
            angle = self.rotation + i * (2 * math.pi / 5)
            pentagon_points = []
            for j in range(5):
                point_angle = angle + j * (2 * math.pi / 5)
                # Make pentagons smaller and closer to center
                point_x = self.size + 0.4 * self.size * math.cos(point_angle)
                point_y = self.size + 0.4 * self.size * math.sin(point_angle)
                pentagon_points.append((point_x, point_y))
            pygame.draw.polygon(ball_surf, pentagon_color, pentagon_points)
        
        # Draw outline
        pygame.draw.circle(ball_surf, (30, 30, 30), (self.size, self.size), self.size, 1)
        
        screen.blit(ball_surf, (draw_x - self.size, draw_y - self.size))
        
    def update_position(self, x: float, y: float, z: float = 0):
        """Update ball position, add to trail, and update rotation"""
        # Update rotation based on movement
        if hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            # Calculate movement direction and update rotation
            dx = x - self.prev_x
            dy = y - self.prev_y
            if dx != 0 or dy != 0:
                self.rotation += math.sqrt(dx*dx + dy*dy) * 0.2
        
        self.prev_x = x
        self.prev_y = y
        self.x = x
        self.y = y
        self.z = z
        
        # Add to trail (limit to 40 points)
        current_time = pygame.time.get_ticks()
        self.trail.append((x, y, z, current_time))
        if len(self.trail) > 40:
            self.trail.pop(0)
            
    def add_sparkle(self, x: float, y: float, size: float = 5):
        """Add a sparkle effect at the given position"""
        current_time = pygame.time.get_ticks()
        self.sparkle_effect.append((x, y, current_time, size))
            
    def clear_trail(self):
        """Clear the ball trail"""
        self.trail = []
        self.rotation = 0
        self.sparkle_effect = []

class PitchRenderer:
    """Renders the attacking third of the pitch"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Attacking third coordinates (x: 70-105, y: 0-68 in meters)
        self.pitch_x_min = 70
        self.pitch_x_max = 105
        self.pitch_y_min = 0
        self.pitch_y_max = 68
        
    def meters_to_pixels(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
        """Convert pitch coordinates (meters) to screen coordinates (pixels)"""
        x_ratio = (x_meters - self.pitch_x_min) / (self.pitch_x_max - self.pitch_x_min)
        y_ratio = (y_meters - self.pitch_y_min) / (self.pitch_y_max - self.pitch_y_min)
        
        x_pixel = int(x_ratio * self.screen_width)
        y_pixel = int((1 - y_ratio) * self.screen_height)  # Flip Y axis
        
        return x_pixel, y_pixel
        
    def draw_pitch(self, screen):
        """Draw the attacking third of the pitch with detailed markings"""
        # Draw pitch background
        screen.fill(PITCH_COLOR)
        
        # Draw pitch outline
        pygame.draw.rect(screen, PITCH_LINE_COLOR, (0, 0, self.screen_width, self.screen_height), 2)
        
        # Draw center line (partial)
        center_x = int((16.5 / (self.pitch_x_max - self.pitch_x_min)) * self.screen_width)
        pygame.draw.line(screen, PITCH_LINE_COLOR, (center_x, 0), (center_x, self.screen_height), 1)
        
        # Draw penalty area
        penalty_width = int((16.5 / (self.pitch_x_max - self.pitch_x_min)) * self.screen_width)
        penalty_height = int((40.0 / (self.pitch_y_max - self.pitch_y_min)) * self.screen_height)
        penalty_x = self.screen_width - penalty_width
        penalty_y = (self.screen_height - penalty_height) // 2
        
        pygame.draw.rect(screen, PITCH_LINE_COLOR, 
                        (penalty_x, penalty_y, penalty_width, penalty_height), 2)
        
        # Draw goal area
        goal_width = int((5.5 / (self.pitch_x_max - self.pitch_x_min)) * self.screen_width)
        goal_height = int((12.0 / (self.pitch_y_max - self.pitch_y_min)) * self.screen_height)
        goal_x = self.screen_width - goal_width
        goal_y = (self.screen_height - goal_height) // 2
        
        pygame.draw.rect(screen, PITCH_LINE_COLOR, 
                        (goal_x, goal_y, goal_width, goal_height), 2)
        
        # Draw penalty spot
        penalty_spot_x = self.screen_width - int((11.0 / (self.pitch_x_max - self.pitch_x_min)) * self.screen_width)
        penalty_spot_y = self.screen_height // 2
        pygame.draw.circle(screen, PITCH_LINE_COLOR, (penalty_spot_x, penalty_spot_y), 5)
        
        # Draw penalty arc
        arc_rect = pygame.Rect(penalty_spot_x - 60, penalty_spot_y - 60, 120, 120)
        pygame.draw.arc(screen, PITCH_LINE_COLOR, arc_rect, -0.4, 0.4, 2)
        
        # Draw goal
        goal_post_height = int((7.32 / (self.pitch_y_max - self.pitch_y_min)) * self.screen_height)
        goal_post_x = self.screen_width
        goal_post_y = (self.screen_height - goal_post_height) // 2
        
        pygame.draw.rect(screen, PITCH_LINE_COLOR, 
                        (goal_post_x, goal_post_y, 15, goal_post_height), 2)
        
        # Draw corner flag area
        corner_x = self.screen_width
        corner_y = self.screen_height
        pygame.draw.arc(screen, PITCH_LINE_COLOR, 
                       (corner_x - 40, corner_y - 40, 80, 80), 
                       math.pi, 1.5 * math.pi, 2)
        
        # Draw distance markers (every 5 meters)
        font = pygame.font.Font(None, 20)
        for i in range(1, 8):
            x_meter = self.pitch_x_min + i * 5
            if x_meter <= self.pitch_x_max:
                x_pixel = int(((x_meter - self.pitch_x_min) / (self.pitch_x_max - self.pitch_x_min)) * self.screen_width)
                pygame.draw.line(screen, (100, 100, 100), (x_pixel, 0), (x_pixel, self.screen_height), 1)
                # Add label
                label = font.render(f"{x_meter}m", True, (200, 200, 200))
                screen.blit(label, (x_pixel - 15, 5))
                
        for i in range(1, 14):
            y_meter = self.pitch_y_min + i * 5
            if y_meter <= self.pitch_y_max:
                y_pixel = int((1 - (y_meter - self.pitch_y_min) / (self.pitch_y_max - self.pitch_y_min)) * self.screen_height)
                pygame.draw.line(screen, (100, 100, 100), (0, y_pixel), (self.screen_width, y_pixel), 1)
                # Add label
                label = font.render(f"{y_meter}m", True, (200, 200, 200))
                screen.blit(label, (5, y_pixel - 10))
        
class CornerKickVisualization:
    """Main Pygame visualization engine for corner kicks"""
    
    def __init__(self, strategy_data: Dict[str, Any]):
        self.strategy_data = strategy_data
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Corner Kick Visualization - Pygame Engine")
        self.clock = pygame.time.Clock()
        
        # Initialize pitch renderer
        self.pitch_renderer = PitchRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Initialize players
        self.players = []
        self.initialize_players()
        
        # Initialize ball
        self.ball = BallSprite(0, 0)
        self.initialize_ball()
        
        # Animation parameters
        self.current_frame = 0
        self.total_frames = 360  # 6 seconds at 60 FPS
        self.cross_frames = 240  # 4 seconds for ball flight
        self.reception_frames = 60  # 1 second for reception
        self.shot_frames = 60  # 1 second for shot
        
        # Camera settings
        self.camera_mode = "fixed"  # "fixed", "follow", or "smart"
        self.camera_x = 0
        self.camera_y = 0
        self.target_camera_x = 0
        self.target_camera_y = 0
        self.camera_zoom = 1.0  # Zoom level (1.0 = normal)
        
        # Track events
        self.ball_kicked = False
        self.ball_received = False
        self.shot_attempted = False
        
        # Primary receiver for targeting
        self.primary_receiver_id = strategy_data.get("primary", {}).get("id")
        
    def initialize_players(self):
        """Initialize player sprites from strategy data"""
        for player_data in self.strategy_data.get("players", []):
            position = player_data.get("position", {"x": 0, "y": 0})
            player = PlayerSprite(
                position["x"], 
                position["y"], 
                player_data["id"], 
                player_data["team"]
            )
            
            # Set jump flag for attackers near the target
            if player_data["team"] == "attacker":
                target = player_data.get("target", [95, 34])
                distance_to_target = math.hypot(
                    position["x"] - target[0], 
                    position["y"] - target[1]
                )
                player.set_jump(distance_to_target < 15)
                
            self.players.append(player)
            
    def initialize_ball(self):
        """Initialize ball position from corner flag"""
        corner_flag = self.strategy_data.get("corner_flag", {"x": 105, "y": 0})
        # Convert to actual pitch coordinates
        self.ball.update_position(corner_flag["x"], corner_flag["y"])
        
    def meters_to_pixels(self, x_meters: float, y_meters: float) -> Tuple[int, int]:
        """Convert meters to pixels using pitch renderer"""
        return self.pitch_renderer.meters_to_pixels(x_meters, y_meters)
        
    def ease_in_out_cubic(self, t: float) -> float:
        """Easing function for smooth animations"""
        if t < 0:
            return 0
        if t > 1:
            return 1
        return 3 * t**2 - 2 * t**3
        
    def quadratic_bezier(self, p0: Tuple[float, float], p1: Tuple[float, float], 
                        p2: Tuple[float, float], t: float) -> Tuple[float, float]:
        """Calculate point on quadratic bezier curve"""
        if t < 0:
            t = 0
        if t > 1:
            t = 1
        x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
        return (x, y)
        
    def update_players_cross_phase(self, frame: int):
        """Update player positions during the cross phase with realistic movement"""
        t_raw = min(max(frame, 0), self.cross_frames) / max(1, self.cross_frames)
        t = self.ease_in_out_cubic(t_raw)
        
        primary_target = self.strategy_data.get("primary", {}).get("target", [95, 34])
        primary_x, primary_y = primary_target
        
        for player in self.players:
            # Get original position
            original_pos = None
            for p_data in self.strategy_data.get("players", []):
                if p_data["id"] == player.player_id:
                    original_pos = p_data.get("position", {"x": player.x, "y": player.y})
                    break
            
            if not original_pos:
                continue
                
            # Calculate target position based on player role
            target_x, target_y = player.x, player.y
            
            if player.team == "attacker":
                # Attackers move toward the target area
                if player.player_id == self.primary_receiver_id:
                    # Primary receiver moves toward the primary target
                    target_x = primary_x
                    target_y = primary_y
                else:
                    # Other attackers move toward nearby positions
                    target_x = original_pos["x"] + (primary_x - original_pos["x"]) * 0.5
                    target_y = original_pos["y"] + (primary_y - original_pos["y"]) * 0.5
                    
                # Add some randomness for natural movement
                target_x += math.sin(t * 5 + player.player_id * 0.1) * 1.0
                target_y += math.cos(t * 3 + player.player_id * 0.1) * 1.0
                
            elif player.team == "defender":
                # Defenders move to mark attackers or cover zones
                # Move toward the ball trajectory projection
                ball_end_x, ball_end_y = primary_x, primary_y  # Approximate ball landing spot
                target_x = original_pos["x"] + (ball_end_x - original_pos["x"]) * 0.6
                target_y = original_pos["y"] + (ball_end_y - original_pos["y"]) * 0.6
                
                # Add some defensive positioning
                target_x += math.sin(t * 2 + player.player_id * 0.2) * 0.5
                target_y += math.cos(t * 2 + player.player_id * 0.2) * 0.5
                
            elif player.team == "keeper":
                # Keeper moves to defend the goal
                goal_center_x, goal_center_y = 105, 34
                target_x = original_pos["x"] + (goal_center_x - original_pos["x"]) * 0.3
                target_y = original_pos["y"] + (goal_center_y - original_pos["y"]) * 0.3
                
                # Keepers move more conservatively
                target_x += math.sin(t * 1.5 + player.player_id * 0.05) * 0.3
                target_y += math.cos(t * 1.5 + player.player_id * 0.05) * 0.3
                
            # Calculate new position with easing
            new_x = original_pos["x"] + (target_x - original_pos["x"]) * t
            new_y = original_pos["y"] + (target_y - original_pos["y"]) * t
            
            # Calculate direction of movement
            direction = math.atan2(target_y - original_pos["y"], target_x - original_pos["x"])
            
            player.update_position(new_x, new_y, direction)
            
    def update_players_jump_phase(self, frame: int):
        """Update player jump animations"""
        # Approximate reception frame
        recv_start = self.cross_frames - 20  # Start jump before ball arrives
        recv_end = self.cross_frames + self.reception_frames
        
        # Jump progression
        if frame < recv_start:
            jp = 0.0
        elif frame <= recv_end:
            # Map frame to 0..1
            jp = (frame - recv_start) / (recv_end - recv_start)
            jp = min(max(jp, 0.0), 1.0)
            jp = self.ease_in_out_cubic(jp)
        else:
            jp = 0.0
            
        # Apply jump to players who will jump
        for player in self.players:
            if player.will_jump:
                player.update_jump(jp)
                
    def update_ball_cross_phase(self, frame: int):
        """Update ball position during cross phase with curved trajectory"""
        t_raw = frame / max(1, self.cross_frames - 1)
        t = self.ease_in_out_cubic(t_raw)
        
        # Get ball start and end positions
        corner_flag = self.strategy_data.get("corner_flag", {"x": 105, "y": 0})
        ball_start = (corner_flag["x"], corner_flag["y"])
        
        # Target near goal (primary receiver position)
        primary_target = self.strategy_data.get("primary", {}).get("target", [104, 34])
        ball_end = (primary_target[0], primary_target[1])
        
        # Control point for curve (higher in the middle for realistic arc)
        # The ball should start low, go high in the middle, then come down
        ctrl_x = ball_start[0] - 12  # Pull the curve back
        ctrl_y = ball_start[1] + 20  # Peak height
        ball_ctrl = (ctrl_x, ctrl_y)
        
        # Calculate position on bezier curve
        bx, by = self.quadratic_bezier(ball_start, ball_ctrl, ball_end, t)
        
        # Calculate height for z-axis effect (parabolic arc)
        # Max height in the middle of the flight
        z = math.sin(math.pi * t) * 6  # Max height of 6 meters
        
        self.ball.update_position(bx, by, z)
        
    def update_ball_reception_phase(self, frame: int):
        """Update ball position during reception phase with realistic behavior"""
        # Calculate how far we are into the reception phase
        reception_frame = frame - self.cross_frames
        t_raw = reception_frame / max(1, self.reception_frames - 1)
        t = self.ease_in_out_cubic(t_raw)
        
        # Get the reception position
        primary_target = self.strategy_data.get("primary", {}).get("target", [104, 34])
        reception_pos = (primary_target[0], primary_target[1])
        
        # When the ball is received, it should have a small bounce or movement
        # Simulate the ball being controlled by the player
        if t < 0.5:
            # Initial contact - small upward movement
            z = 0.5 * math.sin(math.pi * t * 2)  # Small bounce
        else:
            # Settling - coming to rest
            z = 0.5 * (1 - t)  # Decrease height from 0.5 to 0
            
        # Small horizontal movement as the player controls the ball
        dx = 0.3 * math.sin(t * 3)  # Small side-to-side movement
        dy = 0.1 * math.cos(t * 2)  # Small forward/back movement
        
        self.ball.update_position(reception_pos[0] + dx, reception_pos[1] + dy, z)
        
    def update_ball_shot_phase(self, shot_frame: int):
        """Update ball position during shot phase with realistic trajectory"""
        t_raw = shot_frame / max(1, self.shot_frames - 1)
        t = self.ease_in_out_cubic(t_raw)
        
        # Get receiver position (where ball was received)
        primary_target = self.strategy_data.get("primary", {}).get("target", [104, 34])
        shot_start = (primary_target[0], primary_target[1])
        
        # Goal target (center of goal with some randomness for realism)
        goal_center = (105.5, 34 + math.sin(t * 10) * 0.5)  # Slight movement for realism
        
        # Calculate shot trajectory
        shot_x = shot_start[0] + (goal_center[0] - shot_start[0]) * t
        shot_y = shot_start[1] + (goal_center[1] - shot_start[1]) * t
        
        # Add a more pronounced arc to the shot
        # The ball should rise quickly then drop sharply
        if t < 0.7:
            # Rising phase
            z = math.sin(math.pi * t / 0.7) * 4  # Max height of 4 meters
        else:
            # Dropping phase
            drop_t = (t - 0.7) / 0.3  # Normalize to 0-1 for the drop
            z = math.sin(math.pi * (1 - drop_t)) * 4 * (1 - drop_t)
        
        self.ball.update_position(shot_x, shot_y, z)
        
        # Update goalkeeper reaction during shot
        self.update_goalkeeper_reaction(shot_frame)
        
    def update_goalkeeper_reaction(self, shot_frame: int):
        """Update goalkeeper reaction to the shot"""
        if shot_frame < 0:
            return
            
        t_raw = shot_frame / max(1, self.shot_frames - 1)
        t = self.ease_in_out_cubic(t_raw)
        
        # Find the goalkeeper
        goalkeeper = None
        for player in self.players:
            if player.team == "keeper":
                goalkeeper = player
                break
                
        if not goalkeeper:
            return
            
        # Get keeper's original position
        original_pos = None
        for p_data in self.strategy_data.get("players", []):
            if p_data["id"] == goalkeeper.player_id:
                original_pos = p_data.get("position", {"x": goalkeeper.x, "y": goalkeeper.y})
                break
                
        if not original_pos:
            return
            
        # Goalkeeper reacts to the shot:
        # 1. Initial stance (0.0 - 0.3): Stay in position, prepare
        # 2. Dive/catch (0.3 - 0.7): Move toward ball trajectory
        # 3. Follow through (0.7 - 1.0): Continue movement, then settle
        
        if t < 0.3:
            # Preparation phase - small adjustments
            dx = math.sin(t * 20) * 0.2
            dy = math.cos(t * 15) * 0.1
            goalkeeper.update_position(original_pos["x"] + dx, original_pos["y"] + dy)
            
        elif t < 0.7:
            # Dive/catch phase - move toward ball
            # Normalize time for this phase
            phase_t = (t - 0.3) / 0.4
            
            # Get ball position at this time
            ball_pos_t = phase_t
            primary_target = self.strategy_data.get("primary", {}).get("target", [104, 34])
            shot_start = (primary_target[0], primary_target[1])
            goal_center = (105.5, 34)
            ball_x = shot_start[0] + (goal_center[0] - shot_start[0]) * ball_pos_t
            ball_y = shot_start[1] + (goal_center[1] - shot_start[1]) * ball_pos_t
            
            # Move toward ball with anticipation
            target_x = original_pos["x"] + (ball_x - original_pos["x"]) * phase_t * 0.8
            target_y = original_pos["y"] + (ball_y - original_pos["y"]) * phase_t * 0.8
            
            # Add dive effect - goalkeepers lean forward
            dive_distance = 1.5 * math.sin(math.pi * phase_t)
            
            # Calculate direction to ball
            direction = math.atan2(ball_y - target_y, ball_x - target_x)
            target_x += math.cos(direction) * dive_distance
            target_y += math.sin(direction) * dive_distance
            
            goalkeeper.update_position(target_x, target_y, direction)
            
            # Add dive animation - stretch effect
            goalkeeper.size = 20 + 5 * math.sin(math.pi * phase_t)  # Stretch during dive
            
        else:
            # Follow through and settle phase
            phase_t = (t - 0.7) / 0.3
            
            # Return to original position with some overshoot
            target_x = original_pos["x"] + (original_pos["x"] - goalkeeper.x) * phase_t * 0.3
            target_y = original_pos["y"] + (original_pos["y"] - goalkeeper.y) * phase_t * 0.3
            
            # Reduce size back to normal
            goalkeeper.size = 25 - 5 * phase_t  # Shrink back to normal
            
            # Face the goal
            direction = math.atan2(34 - target_y, 105 - target_x)
            goalkeeper.update_position(target_x, target_y, direction)
            
    def update_camera(self):
        """Update camera position based on mode with smooth following"""
        if self.camera_mode == "follow":
            # Follow the ball with smooth interpolation
            self.target_camera_x = max(0, self.ball.x - SCREEN_WIDTH // 3)
            self.target_camera_y = max(0, self.ball.y - SCREEN_HEIGHT // 2)
            
            # Smoothly interpolate to target position
            self.camera_x += (self.target_camera_x - self.camera_x) * 0.05
            self.camera_y += (self.target_camera_y - self.camera_y) * 0.05
            
        elif self.camera_mode == "smart":
            # Smart camera that follows action and adjusts zoom
            # Focus on the most important action (ball or key players)
            if self.current_frame < self.cross_frames:
                # During cross, follow the ball
                self.target_camera_x = max(0, self.ball.x - SCREEN_WIDTH // 3)
                self.target_camera_y = max(0, self.ball.y - SCREEN_HEIGHT // 2)
                # Zoom out slightly to show more context
                self.camera_zoom = 0.9
            elif self.current_frame < self.cross_frames + self.reception_frames:
                # During reception, focus on the receiver
                receiver = self.get_primary_receiver()
                if receiver:
                    self.target_camera_x = max(0, receiver.x - SCREEN_WIDTH // 2)
                    self.target_camera_y = max(0, receiver.y - SCREEN_HEIGHT // 2)
                    # Zoom in for detail
                    self.camera_zoom = 1.1
                else:
                    # Fallback to ball
                    self.target_camera_x = max(0, self.ball.x - SCREEN_WIDTH // 2)
                    self.target_camera_y = max(0, self.ball.y - SCREEN_HEIGHT // 2)
                    self.camera_zoom = 1.0
            else:
                # During shot, follow the ball and show goal
                self.target_camera_x = max(0, self.ball.x - SCREEN_WIDTH // 2.5)
                self.target_camera_y = max(0, self.ball.y - SCREEN_HEIGHT // 2)
                # Normal zoom
                self.camera_zoom = 1.0
                
            # Smoothly interpolate to target position
            self.camera_x += (self.target_camera_x - self.camera_x) * 0.08
            self.camera_y += (self.target_camera_y - self.camera_y) * 0.08
            
        # For fixed mode, camera stays at 0,0 (showing attacking third)
        
    def get_primary_receiver(self):
        """Get the primary receiver player"""
        for player in self.players:
            if player.player_id == self.primary_receiver_id:
                return player
        return None
        
    def update(self):
        """Update animation state for current frame"""
        # Update based on current phase
        if self.current_frame < self.cross_frames:
            # Cross phase
            if not self.ball_kicked:
                print("⚽ Ball kicked from corner")
                self.ball_kicked = True
                # Add sparkle effect when ball is kicked
                self.ball.add_sparkle(self.ball.x, self.ball.y, 8)
                
            self.update_players_cross_phase(self.current_frame)
            self.update_ball_cross_phase(self.current_frame)
            
        elif self.current_frame < self.cross_frames + self.reception_frames:
            # Reception phase
            if not self.ball_received:
                print("🎯 Ball arriving at receiver")
                self.ball_received = True
                # Add sparkle effect when ball is received
                self.ball.add_sparkle(self.ball.x, self.ball.y, 6)
                
            self.update_ball_reception_phase(self.current_frame - self.cross_frames)
            
        else:
            # Shot phase
            shot_frame = self.current_frame - self.cross_frames - self.reception_frames
            if shot_frame == 0 and not self.shot_attempted:
                print("🔫 Shot attempted")
                self.shot_attempted = True
                # Add sparkle effect when shot is attempted
                self.ball.add_sparkle(self.ball.x, self.ball.y, 10)
                
            self.update_ball_shot_phase(shot_frame)
            
        # Update player jumps throughout the animation
        self.update_players_jump_phase(self.current_frame)
        
        # Update camera
        self.update_camera()
        
        # Advance frame
        self.current_frame += 1
        
        # Loop animation if needed
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
            self.ball.clear_trail()
            self.ball_kicked = False
            self.ball_received = False
            self.shot_attempted = False
            
    def draw(self):
        """Draw the current frame with camera transformations"""
        # Draw pitch
        self.pitch_renderer.draw_pitch(self.screen)
        
        # Apply camera transformations (simplified for now)
        # In a more advanced implementation, we would apply scaling for zoom
        
        # Draw players
        for player in self.players:
            player.draw(self.screen, self.camera_x, self.camera_y)
            
        # Draw ball
        self.ball.draw(self.screen, self.camera_x, self.camera_y)
        
        # Draw HUD
        font = pygame.font.Font(None, 36)
        title = font.render("Corner Kick Visualization", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))
        
        # Draw strategy info
        strategy_text = self.strategy_data.get("best_strategy", "No strategy")
        strategy_surface = font.render(strategy_text, True, (255, 255, 255))
        self.screen.blit(strategy_surface, (10, 50))
        
        # Draw frame counter
        frame_text = font.render(f"Frame: {self.current_frame}/{self.total_frames}", True, (255, 255, 255))
        self.screen.blit(frame_text, (10, 90))
        
        # Draw camera mode and zoom
        camera_text = font.render(f"Camera: {self.camera_mode} (Zoom: {self.camera_zoom:.1f}x)", True, (255, 255, 255))
        self.screen.blit(camera_text, (10, 130))
        
        # Draw controls help
        controls_font = pygame.font.Font(None, 24)
        controls = controls_font.render("Controls: F - Cycle Camera, R - Reset, ESC - Quit", True, (200, 200, 200))
        self.screen.blit(controls, (10, SCREEN_HEIGHT - 30))
        
    def run(self):
        """Run the visualization loop"""
        print("Starting Pygame corner kick visualization...")
        print("Controls: F - Cycle Camera Mode, R - Reset Animation, ESC - Quit")
        
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_f:
                        # Cycle through camera modes
                        if self.camera_mode == "fixed":
                            self.camera_mode = "follow"
                        elif self.camera_mode == "follow":
                            self.camera_mode = "smart"
                        else:
                            self.camera_mode = "fixed"
                        print(f"Camera mode: {self.camera_mode}")
                    elif event.key == pygame.K_r:
                        # Reset animation
                        self.current_frame = 0
                        self.ball.clear_trail()
                        self.ball_kicked = False
                        self.ball_received = False
                        self.shot_attempted = False
                        print("Animation reset")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Zoom in
                        self.camera_zoom = min(2.0, self.camera_zoom + 0.1)
                        print(f"Zoom: {self.camera_zoom:.1f}x")
                    elif event.key == pygame.K_MINUS:
                        # Zoom out
                        self.camera_zoom = max(0.5, self.camera_zoom - 0.1)
                        print(f"Zoom: {self.camera_zoom:.1f}x")
                        
            # Update animation
            self.update()
            
            # Draw everything
            self.draw()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        print("Pygame visualization finished.")
        sys.exit()

def main():
    """Main function to test the Pygame visualization"""
    # Sample strategy data for testing
    sample_strategy = {
        "corner_id": 1,
        "best_strategy": "Far Post - Target #244553",
        "confidence": 0.81,
        "corner_flag": {"x": 105, "y": 0},
        "primary": {"id": 244553, "position": [94, 34]},
        "players": [
            {"id": 244553, "team": "attacker", "position": {"x": 90, "y": 36}},
            {"id": 232293, "team": "attacker", "position": {"x": 88, "y": 30}},
            {"id": 221992, "team": "attacker", "position": {"x": 86, "y": 42}},
            {"id": 900001, "team": "defender", "position": {"x": 96, "y": 32}},
            {"id": 900002, "team": "defender", "position": {"x": 97, "y": 38}},
            {"id": 900003, "team": "defender", "position": {"x": 93, "y": 44}},
            {"id": 100000, "team": "keeper", "position": {"x": 105, "y": 34}},
            {"id": 888888, "team": "attacker", "position": {"x": 103, "y": 6}}
        ]
    }
    
    # Create and run visualization
    visualization = CornerKickVisualization(sample_strategy)
    visualization.run()

if __name__ == "__main__":
    main()