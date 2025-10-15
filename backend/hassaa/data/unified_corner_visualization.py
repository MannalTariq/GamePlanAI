#!/usr/bin/env python3

"""

Unified Corner Kick Visualization — Fixed Version

- Handles placement without crashing

- Uses SCREEN_WIDTH/SCREEN_HEIGHT properly

- Adds error logging and crash protection

"""

import pygame
import math
import json
import numpy as np
import os
import sys
import traceback
from typing import Tuple
import torch
from torch_geometric.data import Data
import threading
import time

# === GLOBAL SETTINGS ===
pygame.init()
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# === COLORS ===
PITCH_COLOR = (35, 77, 32)
PITCH_LINE_COLOR = (255, 255, 255)
ATTACKER_COLOR = (255, 68, 68)
DEFENDER_COLOR = (74, 163, 255)
KEEPER_COLOR = (155, 89, 182)
BALL_COLOR = (255, 255, 255)
SHADOW_COLOR = (0, 0, 0, 128)  # Black with transparency

# ---------------- EASING AND BEZIER FUNCTIONS ----------------
def ease_in_out(t: float) -> float:
    """Smooth ease-in-out motion for more natural ball movement"""
    return 3 * t**2 - 2 * t**3

def bezier_point(p0, p1, p2, t):
    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return (x, y)

# ---------------- BUTTON CLASS ----------------
class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
        self.font = pygame.font.Font(None, 24)

    def draw(self, screen):
        color = (100, 149, 237) if self.hovered else (70, 130, 180)
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2, border_radius=5)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def check_click(self, pos, event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(pos)

# ---------------- BALL CLASS ----------------
class Ball:
    """Represents the ball with realistic curved movement"""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 - 80  # raised midpoint for curved cross
        self.control = (mid_x, mid_y)
        self.pos = start
        self.size = 12
        self.trail_points = []  # For drawing trail

    def update(self, t):
        """Update ball position with curved trajectory using Bezier curve"""
        # easing for natural speed
        t_eased = 3 * t**2 - 2 * t**3
        x = (1 - t_eased)**2 * self.start[0] + 2 * (1 - t_eased) * t_eased * self.control[0] + t_eased**2 * self.end[0]
        y = (1 - t_eased)**2 * self.start[1] + 2 * (1 - t_eased) * t_eased * self.control[1] + t_eased**2 * self.end[1]
        self.pos = (x, y)
        
        # Add current position to trail
        self.trail_points.append(self.pos)
        if len(self.trail_points) > 20:
            self.trail_points.pop(0)

    def draw(self, screen):
        """Draw the ball with trail effect"""
        # Draw trail
        for i, p in enumerate(self.trail_points):
            alpha = int(255 * (i / len(self.trail_points)))  # Fade out trail
            radius = max(2, int(6 - i * 0.2))  # Size reduction from 6 to 2 pixels
            # Create a surface for the trail point with alpha
            trail_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (255, 255, 255, alpha), (radius, radius), radius)
            screen.blit(trail_surf, (int(p[0]) - radius, int(p[1]) - radius))
        
        # Draw shadow
        shadow_surf = pygame.Surface((self.size * 2, self.size), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (*SHADOW_COLOR[:3], 120), (0, 0, self.size * 2, self.size))
        screen.blit(shadow_surf, (int(self.pos[0]) - self.size, int(self.pos[1])))
        
        # Draw ball
        pygame.draw.circle(screen, BALL_COLOR, (int(self.pos[0]), int(self.pos[1])), self.size)
        pygame.draw.circle(screen, (30, 30, 30), (int(self.pos[0]), int(self.pos[1])), self.size, 1)

# ---------------- PLAYER CLASS ----------------
class Player:
    def __init__(self, x, y, player_id, team):
        self.x = x
        self.y = y
        self.player_id = player_id
        self.team = team
        self.size = 20
        # Add attributes for simulation
        self.original_x = x
        self.original_y = y
        self.target_x = x
        self.target_y = y
        self.jump_height = 0
        self.speed = 0
        self.will_jump = False

    def draw(self, screen):
        # Ensure proper bounds checking
        try:
            draw_x = int((self.x - 70) / (105 - 70) * SCREEN_WIDTH)
            draw_y = int((1 - (self.y / 68)) * SCREEN_HEIGHT)
            
            # Bounds check to prevent drawing outside screen
            if draw_x < -self.size or draw_x > SCREEN_WIDTH + self.size or \
               draw_y < -self.size or draw_y > SCREEN_HEIGHT + self.size:
                return

            if self.team == "attacker":
                color = ATTACKER_COLOR
                pygame.draw.circle(screen, color, (draw_x, draw_y), self.size // 2)
            elif self.team == "defender":
                color = DEFENDER_COLOR
                points = [(draw_x, draw_y - self.size // 2),
                          (draw_x - self.size // 2, draw_y + self.size // 2),
                          (draw_x + self.size // 2, draw_y + self.size // 2)]
                pygame.draw.polygon(screen, color, points)
            else:
                color = KEEPER_COLOR
                pygame.draw.rect(screen, color,
                                 (draw_x - self.size // 2, draw_y - self.size // 2, self.size, self.size))

            font = pygame.font.Font(None, 20)
            label = font.render(str(self.player_id), True, (255, 255, 255))
            screen.blit(label, (draw_x - 10, draw_y - self.size - 10))
        except Exception as e:
            print(f"[DRAW ERROR] Failed to draw player {self.player_id}: {e}")

    def set_target(self, target_x, target_y):
        """Set target position for movement"""
        self.target_x = target_x
        self.target_y = target_y

    def set_jump(self, will_jump):
        """Set whether this player will jump"""
        self.will_jump = will_jump
        
    def update_position(self, t):
        """Update player position based on interpolation parameter t (0 to 1)"""
        self.x = self.original_x + (self.target_x - self.original_x) * t
        self.y = self.original_y + (self.target_y - self.original_y) * t

# ---------------- MAIN CLASS ----------------
class UnifiedCornerVisualization:
    def __init__(self):
        try:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Unified Corner Kick Visualization (Fixed)")
            self.clock = pygame.time.Clock()
            self.players = []
            self.placement_mode = "attacker"
            self.attackers_placed = 0
            self.defenders_placed = 0
            self.keepers_placed = 0
            self.max_attackers = 10
            self.max_defenders = 10
            self.max_keepers = 1  # Limit to 1 keeper
            self.player_id_counter = 1
            # Add simulation attributes
            self.mode = "placement"  # "placement" or "simulation"
            self.strategy_data = None
            self.current_frame = 0
            self.total_frames = 180  # 3 seconds at 60 fps
            self.ball_kicked = False
            self.ball_received = False
            self.shot_attempted = False
            # Ball movement attributes
            self.ball_progress = 0.0        # from 0 to 1
            self.ball_speed = 0.008         # tune this for realism
            self.ball_active = False        # becomes True when "Done" pressed
            # Initialize ball at corner position (105m, 0m)
            ball_start_x = int((105 - 70) / (105 - 70) * SCREEN_WIDTH)
            ball_start_y = int((1 - (0 / 68)) * SCREEN_HEIGHT)
            ball_end_x = int((105 - 70) / (105 - 70) * SCREEN_WIDTH)  # Goal x (same x)
            ball_end_y = int((1 - (34 / 68)) * SCREEN_HEIGHT)  # Goal y (middle)
            self.ball = Ball((ball_start_x, ball_start_y), (ball_end_x, ball_end_y))
            self.setup_ui()
            print("[INIT] Visualization initialized successfully")
        except Exception as e:
            print(f"[INIT ERROR] Failed to initialize visualization: {e}")
            traceback.print_exc()
            raise

    def setup_ui(self):
        try:
            self.btn_done = Button(20, SCREEN_HEIGHT - 60, 120, 40, "Done")
            self.btn_reset = Button(160, SCREEN_HEIGHT - 60, 120, 40, "Reset")
            print("[UI] UI elements created successfully")
        except Exception as e:
            print(f"[UI ERROR] Failed to create UI elements: {e}")
            traceback.print_exc()

    def pixels_to_meters(self, x_pixels, y_pixels):
        try:
            # Ensure proper bounds
            x_pixels = max(0, min(SCREEN_WIDTH, x_pixels))
            y_pixels = max(0, min(SCREEN_HEIGHT, y_pixels))
            
            x_ratio = x_pixels / SCREEN_WIDTH
            y_ratio = 1 - (y_pixels / SCREEN_HEIGHT)
            x_meters = 70 + x_ratio * (105 - 70)
            y_meters = 0 + y_ratio * 68
            return x_meters, y_meters
        except Exception as e:
            print(f"[CONVERSION ERROR] Failed to convert pixels to meters: {e}")
            # Return safe defaults
            return 70.0, 0.0

    def handle_placement_click(self, pos):
        try:
            x_meters, y_meters = self.pixels_to_meters(pos[0], pos[1])
            total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
            
            if total_players >= 22:
                print("[WARN] Max players reached")
                return

            # Team selection logic
            if self.placement_mode == "attacker":
                team = "attacker"
                self.attackers_placed += 1
                if self.attackers_placed >= self.max_attackers:
                    self.placement_mode = "defender"
            elif self.placement_mode == "defender":
                team = "defender"
                self.defenders_placed += 1
                if self.defenders_placed >= self.max_defenders:
                    self.placement_mode = "keeper"
            else:
                team = "keeper"
                # Check if keeper already placed
                if self.keepers_placed >= self.max_keepers:
                    print("⚠️ Only one keeper allowed.")
                    return
                self.keepers_placed += 1

            player = Player(x_meters, y_meters, self.player_id_counter, team)
            self.player_id_counter += 1
            self.players.append(player)
            print(f"[PLACED] Player {player.player_id} ({team}) at ({x_meters:.2f}, {y_meters:.2f})")
        except Exception as e:
            print(f"[PLACEMENT ERROR] Failed to place player: {e}")
            traceback.print_exc()

    def draw_pitch(self):
        try:
            self.screen.fill(PITCH_COLOR)
            pygame.draw.rect(self.screen, PITCH_LINE_COLOR, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 3)
            
            # Penalty box
            box_w = (16.5 / (105 - 70)) * SCREEN_WIDTH
            box_h = (40 / 68) * SCREEN_HEIGHT
            box_x = SCREEN_WIDTH - box_w
            box_y = (SCREEN_HEIGHT - box_h) / 2
            pygame.draw.rect(self.screen, PITCH_LINE_COLOR, (box_x, box_y, box_w, box_h), 2)
            
            # Goal
            goal_h = (7.32 / 68) * SCREEN_HEIGHT
            goal_x = SCREEN_WIDTH
            goal_y = (SCREEN_HEIGHT - goal_h) / 2
            pygame.draw.rect(self.screen, PITCH_LINE_COLOR, (goal_x - 5, goal_y, 5, goal_h), 2)
        except Exception as e:
            print(f"[PITCH ERROR] Failed to draw pitch: {e}")

    def reset(self):
        try:
            self.players.clear()
            self.attackers_placed = 0
            self.defenders_placed = 0
            self.keepers_placed = 0
            self.placement_mode = "attacker"
            self.player_id_counter = 1
            # Reset simulation attributes
            self.mode = "placement"
            self.strategy_data = None
            self.current_frame = 0
            self.ball_kicked = False
            self.ball_received = False
            self.shot_attempted = False
            # Reset ball movement attributes
            self.ball_progress = 0.0
            self.ball_speed = 0.008
            self.ball_active = False
            # Reset ball position
            ball_start_x = int((105 - 70) / (105 - 70) * SCREEN_WIDTH)
            ball_start_y = int((1 - (0 / 68)) * SCREEN_HEIGHT)
            ball_end_x = int((105 - 70) / (105 - 70) * SCREEN_WIDTH)  # Goal x (same x)
            ball_end_y = int((1 - (34 / 68)) * SCREEN_HEIGHT)  # Goal y (middle)
            self.ball = Ball((ball_start_x, ball_start_y), (ball_end_x, ball_end_y))
            print("[RESET] All players removed.")
        except Exception as e:
            print(f"[RESET ERROR] Failed to reset: {e}")
            traceback.print_exc()

    def generate_strategy(self):
        """Generate a simple strategy for simulation"""
        try:
            if len(self.players) == 0:
                return None
                
            # Find the first attacker as primary receiver
            primary_player = None
            for player in self.players:
                if player.team == "attacker":
                    primary_player = player
                    break
                    
            # If no attacker found, use first player
            if primary_player is None and len(self.players) > 0:
                primary_player = self.players[0]
                
            if primary_player is None:
                return None
                
            # Create strategy data
            strategy_data = {
                "corner_id": 1,
                "best_strategy": f"Target Player #{primary_player.player_id}",
                "confidence": 0.75,
                "primary": {
                    "id": primary_player.player_id,
                    "position": [94, 34],
                    "target": [94, 34]
                },
                "players": []
            }
            
            # Add all players to strategy
            for player in self.players:
                intent = "unknown"
                if player.team == "attacker":
                    if player.player_id == primary_player.player_id:
                        intent = "primary_target"
                    else:
                        intent = "support_runner"
                elif player.team == "defender":
                    intent = "marking"
                elif player.team == "keeper":
                    intent = "goalkeeping"
                    
                strategy_data["players"].append({
                    "id": player.player_id,
                    "team": player.team,
                    "intent": intent,
                    "position": {
                        "x": player.x,
                        "y": player.y
                    },
                    "start": [player.x, player.y],
                    "target": [player.x + 2, player.y]  # Simple target
                })
                
            return strategy_data
        except Exception as e:
            print(f"[STRATEGY ERROR] Failed to generate strategy: {e}")
            return None

    def start_simulation(self):
        """Start the simulation mode — non-blocking with threaded strategy generation."""
        print("🚀 Starting threaded strategy generation...")
        self.strategy_data = None

        def generate():
            try:
                t0 = time.time()
                data = self.generate_strategy()
                self.strategy_data = data
                print(f"✅ Strategy generated in {time.time()-t0:.2f} sec")
            except Exception as e:
                print(f"❌ Strategy generation failed: {e}")
                self.strategy_data = None

        # Run strategy generation in background
        t = threading.Thread(target=generate, daemon=True)
        t.start()

        # Wait a max of 5 seconds for strategy
        max_wait = 5
        waited = 0
        while self.strategy_data is None and waited < max_wait:
            pygame.time.wait(100)
            waited += 0.1

        if self.strategy_data is None:
            print("⚠️ Strategy generation timed out or failed. Using fallback.")
            # Minimal fallback strategy
            if len(self.players) > 0:
                self.strategy_data = {
                    "corner_id": 1,
                    "best_strategy": f"Fallback Target #{self.players[0].player_id}",
                    "confidence": 0.5,
                    "primary": {"id": self.players[0].player_id},
                    "players": []
                }
            else:
                print("❌ No players to run simulation.")
                return

        # Switch to simulation mode after strategy is ready
        self.mode = "simulation"
        self.current_frame = 0
        self.ball_kicked = False
        self.ball_received = False
        self.shot_attempted = False
        # Reset ball movement attributes
        self.ball_progress = 0.0
        self.ball_active = True
        # After identifying the receiver target, create new ball
        ball_start_x = int((105 - 70) / (105 - 70) * SCREEN_WIDTH)
        ball_start_y = int((1 - (0 / 68)) * SCREEN_HEIGHT)
        # Get receiver position (for now using default)
        ball_end_x = int((105 - 70) / (105 - 70) * SCREEN_WIDTH)  # Goal x (same x)
        ball_end_y = int((1 - (34 / 68)) * SCREEN_HEIGHT)  # Goal y (middle)
        self.ball = Ball((ball_start_x, ball_start_y), (ball_end_x, ball_end_y))

        # Reset player positions and targets
        for player in self.players:
            player.x = player.original_x
            player.y = player.original_y
            player.jump_height = 0
            player.speed = 0

        primary_id = self.strategy_data["primary"]["id"]
        for player in self.players:
            if player.player_id == primary_id:
                player.set_target(94, 34)
                player.set_jump(True)
            elif player.team == "attacker":
                player.set_target(player.x + 3, player.y + 1)
                player.set_jump(False)
            elif player.team == "defender":
                player.set_target(player.x + 2, player.y)
                player.set_jump(False)
            elif player.team == "keeper":
                player.set_target(105, 34)
                player.set_jump(False)

        print("🎬 Simulation starting now!")
        
    def update_simulation(self):
        """Update simulation frames and positions"""
        # Update ball movement
        if self.ball_active:
            self.ball_progress += self.ball_speed
            if self.ball_progress > 1.0:
                self.ball_progress = 1.0
                self.ball_active = False  # stop after reaching target
            self.ball.update(self.ball_progress)

        # Initialize total frames if not set
        if not hasattr(self, "total_frames"):
            self.total_frames = 180  # 3 seconds at 60 fps
        if not hasattr(self, "current_frame"):
            self.current_frame = 0

        if self.current_frame < self.total_frames:
            self.current_frame += 1

            # Update each player's movement toward their target
            t = self.current_frame / self.total_frames
            for p in self.players:
                p.update_position(t)

        else:
            print("✅ Simulation finished.")
            self.mode = "placement"
            self.current_frame = 0

    def draw_simulation(self):
        """Draw the simulation with trail effect"""
        self.draw_pitch()
        for p in self.players:
            p.draw(self.screen)
        self.ball.draw(self.screen)

        # Optional visual debug
        # print(f"🎨 Drawing simulation frame {self.current_frame}/{self.total_frames}")

    def run(self):
        try:
            running = True
            print("[RUN] Starting visualization loop...")
            while running:
                try:
                    mouse_pos = pygame.mouse.get_pos()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.MOUSEBUTTONDOWN:
                            try:
                                if self.btn_done.check_click(mouse_pos, event):
                                    print("[INFO] Done clicked")
                                    if self.mode == "placement":
                                        # Check if we have enough players
                                        total_players = self.attackers_placed + self.defenders_placed + self.keepers_placed
                                        if total_players >= 3:  # Minimum for a meaningful simulation
                                            self.start_simulation()
                                        else:
                                            print("Need at least 3 players to start simulation")
                                elif self.btn_reset.check_click(mouse_pos, event):
                                    self.reset()
                                else:
                                    if self.mode == "placement":
                                        self.handle_placement_click(mouse_pos)
                            except Exception as e:
                                print(f"[EVENT ERROR] {e}")
                                traceback.print_exc()

                    self.btn_done.check_hover(mouse_pos)
                    self.btn_reset.check_hover(mouse_pos)

                    # Main mode handling
                    if self.mode == "placement":
                        self.draw_pitch()
                        for player in self.players:
                            player.draw(self.screen)
                        self.btn_done.draw(self.screen)
                        self.btn_reset.draw(self.screen)
                    elif self.mode == "simulation":
                        self.update_simulation()
                        self.draw_simulation()

                    pygame.display.flip()
                    self.clock.tick(FPS)
                except Exception as e:
                    print(f"[LOOP ERROR] Error in main loop: {e}")
                    traceback.print_exc()
                    # Continue running unless it's a critical error
                    continue

            pygame.quit()
            sys.exit()
        except Exception as e:
            print(f"[FATAL ERROR] Visualization crashed: {e}")
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

# ---------------- MAIN ----------------
def main():
    try:
        viz = UnifiedCornerVisualization()
        viz.run()
    except Exception as e:
        print(f"🔥 FATAL ERROR: {e}")
        traceback.print_exc()
        pygame.quit()

if __name__ == "__main__":
    main()