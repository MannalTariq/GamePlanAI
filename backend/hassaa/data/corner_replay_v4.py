#!/usr/bin/env python3
"""
corner_replay_v4.py
Polished, cinematic corner-kick replay with:
- realistic Bezier ball arc with z-height
- goalkeeper dive and defender jump/clearance
- player icon upgrades (triangles, squares, circles) with perspective scaling
- header/shot animation and goal-net flash
- HUD, slow-motion, save mp4/gif fallback, pause/resume (space)
- maps normalized coords (0-100) to local attacking-third window
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.path as mpath
import matplotlib.patheffects as PathEffects
import argparse
import json
import os
import math
import random
import sys
import traceback
from functools import partial

# ---------------- CONFIG ----------------
FPS = 60  # high-fidelity smoothness
DEFAULT_VIEW = "corner_close"  # options: corner_close, tactical, broadcast
DEFAULT_SPEED = "normal"  # slow, normal, fast

# Local camera windows (attacking right side: uses x near 85..105)
VIEW_WINDOWS = {
    "corner_close": {"x_min": 85.0, "x_max": 105.0, "y_min": 15.0, "y_max": 53.0},
    "tactical": {"x_min": 78.0, "x_max": 105.0, "y_min": 0.0, "y_max": 68.0},
    "broadcast": {"x_min": 80.0, "x_max": 105.0, "y_min": 10.0, "y_max": 58.0}
}

# Aesthetic colors
COLORS = {
    "pitch": "#234d20",
    "pitch_strip": "#2d6b2e",
    "attacker": "#ff7f50",
    "primary": "#ff3b3b",
    "alternate": "#ffb86b",
    "defender": "#4aa3ff",
    "keeper": "#9b59b6",
    "ball": "#ffffff",
    "shadow": "#000000"
}

# ---------------- Utils ----------------
def ease_in_out_cubic(t):
    return 3 * t**2 - 2 * t**3

def lerp(a, b, t):
    return a + (b - a) * t

def quadratic_bezier(p0, p1, p2, t):
    x = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    y = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return (x, y)

def map_norm_to_local(x_norm, y_norm, x_min, x_max, y_min, y_max):
    if 0.0 <= x_norm <= 100.0 and 0.0 <= y_norm <= 100.0:
        x = x_min + (x_norm / 100.0) * (x_max - x_min)
        y = y_min + (y_norm / 100.0) * (y_max - y_min)
        return (x, y)
    else:
        return (x_norm, y_norm)

def create_triangle_path(size=1.0):
    Path = mpath.Path
    verts = [(0, 0.8), (-0.6, -0.4), (0.6, -0.4), (0, 0.8)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    verts = [(x * size, y * size) for x, y in verts]
    return Path(verts, codes)

def create_square_path(size=1.0):
    Path = mpath.Path
    verts = [(-0.6, -0.6), (-0.6, 0.6), (0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)]
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    verts = [(x * size, y * size) for x, y in verts]
    return Path(verts, codes)

# ---------------- Main class ----------------
class CornerReplayV4:
    def __init__(self, strategy_data=None, corner_id=1, view=DEFAULT_VIEW, speed=DEFAULT_SPEED, fps=FPS):
        self.corner_id = corner_id
        self.view = view if view in VIEW_WINDOWS else DEFAULT_VIEW
        self.speed = speed
        self.fps = fps

        # duration params (s) for phases
        speed_map = {
            "slow": (4.0, 0.8, 1.2),
            "normal": (3.0, 0.6, 0.9),
            "fast": (2.2, 0.4, 0.6)
        }
        cross_s, recv_s, shot_s = speed_map.get(self.speed, speed_map["normal"])
        self.cross_frames = int(cross_s * self.fps)
        self.recv_frames = int(recv_s * self.fps)
        self.shot_frames = int(shot_s * self.fps)
        self.total_frames = self.cross_frames + self.recv_frames + self.shot_frames

        # load strategy
        self.strategy_data = strategy_data if strategy_data else self._sample_strategy()
        self.window = VIEW_WINDOWS[self.view]

        # prepare geometry and actors
        self._prepare_strategy()

        # set up figure
        self._setup_figure()

        # initialize visual artists
        self._init_visuals()

        # interactive controls
        self.paused = False

    def _sample_strategy(self):
        # normalized coords 0..100 for convenience
        return {
            "corner_id": 1,
            "best_strategy": "Far Post - Target #244553",
            "confidence": 0.81,
            "corner_flag": {"x": 100, "y": 0},   # normalized
            "primary": {"id": 244553, "start": [82, 36], "target": [94, 34]},
            "alternates":[
                {"id": 232293, "start":[80,30], "target":[92,30]},
                {"id": 221992, "start":[78,42], "target":[90,36]}
            ],
            "players":[
                {"id":244553, "team":"attacker","start":[82,36],"intent":"far_post","target":[94,34]},
                {"id":232293, "team":"attacker","start":[80,30],"intent":"near_post","target":[92,30]},
                {"id":221992, "team":"attacker","start":[78,42],"intent":"penalty_spot","target":[92,34]},
                {"id":900001,"team":"defender","start":[92,32],"intent":"mark","target":[93,33]},
                {"id":900002,"team":"defender","start":[93,38],"intent":"shift","target":[94,36]},
                {"id":900003,"team":"defender","start":[89,44],"intent":"edge_box","target":[90,42]},
                {"id":100000,"team":"keeper","start":[105,34],"intent":"keeper","target":[103,34]},
                {"id":888888,"team":"attacker","start":[100,6],"intent":"corner_taker","target":[99,6]}
            ],
            "cluster_zone":[93,34]
        }

    def _prepare_strategy(self):
        sd = self.strategy_data
        self.corner_raw = (sd['corner_flag']['x'], sd['corner_flag']['y'])
        self.corner = map_norm_to_local(self.corner_raw[0], self.corner_raw[1],
                                        self.window['x_min'], self.window['x_max'],
                                        self.window['y_min'], self.window['y_max'])
        # cluster & primary
        self.cluster_raw = tuple(sd.get('cluster_zone', sd['primary']['target']))
        self.cluster = map_norm_to_local(self.cluster_raw[0], self.cluster_raw[1],
                                         self.window['x_min'], self.window['x_max'],
                                         self.window['y_min'], self.window['y_max'])

        p = sd.get('primary', {})
        self.primary_id = p.get('id')
        self.primary_start = map_norm_to_local(p.get('start',[82,36])[0], p.get('start',[82,36])[1],
                                               self.window['x_min'], self.window['x_max'],
                                               self.window['y_min'], self.window['y_max'])
        self.primary_target = map_norm_to_local(p.get('target', self.cluster_raw)[0], p.get('target', self.cluster_raw)[1],
                                                self.window['x_min'], self.window['x_max'],
                                                self.window['y_min'], self.window['y_max'])

        # players list
        self.players = []
        for pl in sd.get('players', []):
            s = pl.get('start', [80,34]); t = pl.get('target', s)
            s_m = map_norm_to_local(s[0], s[1], self.window['x_min'], self.window['x_max'], self.window['y_min'], self.window['y_max'])
            t_m = map_norm_to_local(t[0], t[1], self.window['x_min'], self.window['x_max'], self.window['y_min'], self.window['y_max'])
            # nudge attacker target closer to goal if needed
            if pl.get('team') == 'attacker' and t_m[0] < s_m[0]:
                t_m = (min(self.window['x_max'] - 0.5, s_m[0] + 3.5), t_m[1])
            self.players.append({
                'id': pl['id'],
                'team': pl.get('team','attacker'),
                'intent': pl.get('intent',''),
                'start': s_m,
                'target': t_m,
                'current': s_m
            })

        # ball geometry: start, end, control
        self.ball_start = self.corner
        # choose ball_end near primary_target but ensure it sits inside penalty area
        penalty_x = self.window['x_max'] - 16.5
        bx = min(self.primary_target[0], self.window['x_max'] - 0.8)
        if bx < penalty_x:
            bx = penalty_x + 3.0  # nudge into box
        by = self.primary_target[1]
        self.ball_end = (bx, by)
        # control point mid + lift
        mid = ((self.ball_start[0] + self.ball_end[0]) / 2.0, (self.ball_start[1] + self.ball_end[1]) / 2.0)
        lift = max(6.0, (self.ball_end[0] - self.ball_start[0]) * 0.13)
        ctrl_x = min(self.window['x_max'] - 1.0, mid[0] + (self.ball_end[0] - mid[0]) * 0.15 + 2.2)
        ctrl_y = mid[1] + lift
        self.ball_ctrl = (ctrl_x, ctrl_y)

        # player bezier control points
        for pl in self.players:
            s = pl['start']; t = pl['target']
            cp_mid = ((s[0] + t[0]) / 2.0, (s[1] + t[1]) / 2.0)
            # small lateral randomization and bias toward cluster
            cp = (cp_mid[0] + random.uniform(-1.0, 1.0) + 0.3*(self.cluster[0]-cp_mid[0]),
                  cp_mid[1] + random.uniform(-0.8, 0.8))
            pl['control'] = cp
            # add jump flag for those who will try to head (if near cluster)
            dist_to_cluster = math.hypot(t[0] - self.cluster[0], t[1] - self.cluster[1])
            pl['will_jump'] = (pl['team'] == 'attacker' and dist_to_cluster < 6.5)

    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.fig.patch.set_facecolor('#07130b')
        self.ax.set_facecolor(COLORS['pitch'])
        self.ax.set_xlim(self.window['x_min'], self.window['x_max'])
        self.ax.set_ylim(self.window['y_min'], self.window['y_max'])
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self._draw_local_pitch()

    def _draw_local_pitch(self):
        x0, x1 = self.window['x_min'], self.window['x_max']
        y0, y1 = self.window['y_min'], self.window['y_max']
        # pitch patch
        self.ax.add_patch(patches.Rectangle((x0,y0), x1-x0, y1-y0, facecolor=COLORS['pitch_strip'], zorder=0))
        # penalty area
        pa_x = x1 - 16.5
        pa_height = 40.0
        pa_y = (y0 + y1)/2.0 - pa_height/2.0
        self.ax.add_patch(patches.Rectangle((pa_x, pa_y), 16.5, pa_height, edgecolor='white', facecolor='none', linewidth=1.5, zorder=2))
        # 6-yard
        six_x = x1 - 5.5
        six_y = (y0+y1)/2.0 - 6.0
        self.ax.add_patch(patches.Rectangle((six_x, six_y), 5.5, 12.0, edgecolor='white', facecolor='none', linewidth=1.2, zorder=2))
        # penalty spot and arc
        ps_x = x1 - 11.0; ps_y = (y0+y1)/2.0
        self.ax.add_patch(patches.Circle((ps_x, ps_y), 0.35, color='white', zorder=3))
        arc = patches.Arc((ps_x,ps_y), 2*9.15, 2*9.15, angle=0, theta1=50, theta2=130, edgecolor='white', linewidth=1.2, zorder=2)
        self.ax.add_patch(arc)
        # goal posts marker (thin)
        goal_bottom = ps_y - 7.32/2.0
        self.ax.add_patch(patches.Rectangle((x1, goal_bottom), 0.8, 7.32, edgecolor='white', facecolor='none', linewidth=1.5, zorder=4))
        # corner arcs (small)
        self.ax.add_patch(patches.Arc((x1, y0), 2.0, 2.0, theta1=90, theta2=180, edgecolor='white', zorder=3))
        self.ax.add_patch(patches.Arc((x1, y1), 2.0, 2.0, theta1=180, theta2=270, edgecolor='white', zorder=3))
        # cluster zone with gradient effect
        cluster_circle = patches.Circle(self.cluster, radius=3.0, facecolor='cyan', alpha=0.15, zorder=1)
        self.ax.add_patch(cluster_circle)

    def _init_visuals(self):
        # HUD & status
        self.ax.set_title(f"Corner Kick — {self.strategy_data.get('best_strategy','')}", color='white', fontsize=14, pad=14)
        self.hud_conf = self.ax.text(0.98, 0.95, f"Conf: {self.strategy_data.get('confidence',0.0):.2f}", transform=self.ax.transAxes, ha='right', color='yellow', bbox=dict(facecolor='black', alpha=0.6), fontsize=10)
        self.status_text = self.ax.text(0.02, 0.02, "", transform=self.ax.transAxes, ha='left', color='white', bbox=dict(facecolor='black', alpha=0.6), fontsize=10)

        # create artist shapes for each player with enhanced styling
        self.artists = {}
        tri_path = create_triangle_path(size=1.2)  # Slightly larger
        square_path = create_square_path(size=1.2)  # Slightly larger
        for pl in self.players:
            x, y = pl['start']
            if pl['team'] == 'defender':
                marker = patches.PathPatch(tri_path, facecolor=COLORS['defender'], edgecolor='white', lw=1.2, zorder=6)
                # Add subtle glow effect
                glow = patches.Circle((x, y), radius=2.0, facecolor=COLORS['defender'], alpha=0.3, zorder=5)
                self.ax.add_patch(glow)
            elif pl['team'] == 'keeper':
                marker = patches.PathPatch(square_path, facecolor=COLORS['keeper'], edgecolor='white', lw=1.2, zorder=7)
                # Add subtle glow effect
                glow = patches.Circle((x, y), radius=2.0, facecolor=COLORS['keeper'], alpha=0.3, zorder=5)
                self.ax.add_patch(glow)
            else:
                # attackers: circle created by scatter with glow effect
                glow = patches.Circle((x, y), radius=2.0, facecolor=COLORS['attacker'], alpha=0.3, zorder=5)
                self.ax.add_patch(glow)
                marker = self.ax.scatter([x], [y], s=280, color=COLORS['attacker'], edgecolors='white', linewidths=1.2, zorder=6)

            if isinstance(marker, patches.PathPatch):
                self.ax.add_patch(marker)
                marker.set_transform(self.ax.transData)
                marker.set_clip_on(True)
            else:
                # placeholder scatter for attackers
                sc = self.ax.scatter([x], [y], s=220, color=COLORS['attacker'], edgecolors='white', linewidths=0.8, zorder=6)
                marker = sc

            txt = self.ax.text(x, y + 1.0, f"#{pl['id']}", color='white', fontsize=9, ha='center', weight='bold', zorder=8,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.8))

            # path artist (for triangles/squares) we'll move via set_transform translate
            self.artists[pl['id']] = {
                'artist': marker,
                'text': txt,
                'start': pl['start'],
                'target': pl['target'],
                'control': pl['control'],
                'team': pl['team'],
                'intent': pl['intent'],
                'will_jump': pl.get('will_jump', False),
                'current': pl['start'],
                'jump_phase': 0.0
            }

        # ensure corner taker visible (corner flag)
        self.corner_flag_artist = self.ax.scatter([self.ball_start[0]], [self.ball_start[1]], s=80, color='black', edgecolors='white', zorder=5)
        self.corner_flag_text = self.ax.text(self.ball_start[0], self.ball_start[1]-1.1, "Corner", color='white', ha='center', fontsize=8)

        # ball + shadow with enhanced styling
        self.ball_artist = self.ax.scatter([self.ball_start[0]], [self.ball_start[1]], s=150, color=COLORS['ball'], 
                                          edgecolors='black', linewidths=1.5, zorder=15)
        self.ball_shadow = self.ax.scatter([self.ball_start[0]], [self.ball_start[1]-0.3], s=45, 
                                          color=COLORS['shadow'], alpha=0.5, zorder=4)

        # trail container
        self.trail = []

        # goal flash
        goal_bottom = (self.window['y_min'] + self.window['y_max']) / 2.0 - 7.32/2.0
        self.goal_flash = patches.Rectangle((self.window['x_max'], goal_bottom), 0.8, 7.32, color='yellow', alpha=0.0, zorder=20)
        self.ax.add_patch(self.goal_flash)

        # assign artist transforms for polygons: store initial offsets
        # For PathPatch, we'll use set_xy by offsetting the path coords (we'll compute translations manually)
        for pid, info in self.artists.items():
            art = info['artist']
            if isinstance(art, patches.PathPatch):
                # store original path verts for reusing
                info['path'] = art.get_path()
                info['orig_verts'] = info['path'].vertices.copy()
            # scatter has set_offsets

        # event logging times
        self.kick_logged = False
        self.receive_logged = False
        self.shot_logged = False

        # connect keypress for pause/resume
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def _update_players(self, frame):
        # players move along quadratic bezier from start->control->target until reception
        total_move_frames = self.cross_frames + self.recv_frames
        t_raw = min(max(frame, 0), total_move_frames) / max(1, total_move_frames)
        t = ease_in_out_cubic(t_raw)
        for pid, info in self.artists.items():
            s = info['start']; c = info['control']; tg = info['target']
            nx, ny = quadratic_bezier(s, c, tg, t)
            # if this player will jump, apply vertical offset later
            info['current'] = (nx, ny)
            # update polygon or scatter
            art = info['artist']
            if isinstance(art, patches.PathPatch):
                # translate stored vertices
                verts = info['orig_verts'] + np.array([nx, ny])
                new_path = mpath.Path(verts, info['path'].codes)
                art.set_path(new_path)
            else:
                # scatter
                art.set_offsets([nx, ny])
            info['text'].set_position([nx, ny + 0.8])

    def _players_jump_phase(self, frame):
        # if ball about to land, attackers that will_jump perform a small vertical 'jump' animation
        # approximate reception frame:
        recv_start = self.cross_frames
        recv_end = self.cross_frames + self.recv_frames
        # jump progression (0..1) triggered around recv_start
        for pid, info in self.artists.items():
            if not info['will_jump']:
                continue
            # compute relative timing
            if frame < recv_start - 4:
                jp = 0.0
            elif frame <= recv_end:
                # map frame in [recv_start-4, recv_end] to 0..1
                jp = (frame - (recv_start - 4)) / (recv_end - (recv_start - 4))
                jp = min(max(jp, 0.0), 1.0)
                jp = ease_in_out_cubic(jp)
            else:
                jp = 0.0
            info['jump_phase'] = jp

            # apply vertical offset and slight scale for heading effect
            nx, ny = info['current']
            # Enhanced jump animation
            vertical = 0.8 * math.sin(math.pi * jp)  # Increased height
            # enlarge a bit at peak
            scale = 1.0 + 0.35 * math.sin(math.pi * jp)  # Increased scale
            art = info['artist']
            if isinstance(art, patches.PathPatch):
                verts = info['orig_verts']*scale + np.array([nx, ny + vertical])
                new_path = mpath.Path(verts, info['path'].codes)
                art.set_path(new_path)
            else:
                art.set_offsets([nx, ny + vertical])
                art.set_sizes([220 * scale])  # scatter sizes
            info['text'].set_position([nx, ny + 0.8 + vertical])

    def _keepers_and_defenders_react(self, frame):
        # defenders move slightly faster toward the ball line when cross arrives
        if frame < self.cross_frames * 0.6:
            return
        # defenders shift with extra lerp toward ball_end
        for pid, info in self.artists.items():
            if info['team'] == 'defender' or info['team'] == 'keeper':
                sx, sy = info['start']
                tx, ty = info['target']
                # bias target toward ball_end slightly
                bias = 0.25
                bx, by = self.ball_end
                new_target = (lerp(tx, bx, bias), lerp(ty, by, bias))
                # progression after crossframes
                t_raw = min(max(frame - self.cross_frames, 0), self.recv_frames + self.shot_frames) / max(1, self.recv_frames + self.shot_frames)
                t = ease_in_out_cubic(t_raw)
                nx = lerp(sx, new_target[0], t)
                ny = lerp(sy, new_target[1], t)
                info['current'] = (nx, ny)
                art = info['artist']
                if isinstance(art, patches.PathPatch):
                    verts = info['orig_verts'] + np.array([nx, ny])
                    art.set_path(mpath.Path(verts, info['path'].codes))
                else:
                    art.set_offsets([nx, ny])
                info['text'].set_position([nx, ny + 0.8])

    def _update_ball_cross(self, frame):
        # compute position along bezier with ease
        t_raw = frame / max(1, self.cross_frames - 1)
        t = ease_in_out_cubic(t_raw)
        bx, by = quadratic_bezier(self.ball_start, self.ball_ctrl, self.ball_end, t)
        # z-height for visual scaling with enhanced effect
        z = math.sin(math.pi * t)  # 0..1
        # ball vertical offset so ball appears above pitch
        ball_y_disp = by + z * 1.2  # Increased height
        self.ball_artist.set_offsets([bx, ball_y_disp])
        # scale ball size slightly with z
        base_size = 120
        size = base_size * (1 + 0.9 * z)
        self.ball_artist.set_sizes([size])
        # shadow shrinks as ball gets higher
        shadow_size = 36 * (1 - 0.7 * z)
        self.ball_shadow.set_offsets([bx, by - 0.15])
        self.ball_shadow.set_sizes([shadow_size])
        self.ball_shadow.set_alpha(0.45 * (1 - 0.6 * z))
        # add trail point (fade)
        trail_point = self.ax.scatter([bx], [by], s=15, color='white', alpha=0.3, zorder=9,
                                     edgecolors='gold', linewidths=0.5)
        self.trail.append(trail_point)
        if len(self.trail) > 28:
            old = self.trail.pop(0)
            try:
                old.remove()
            except Exception:
                pass

    def _update_ball_reception(self):
        bx, by = self.ball_end
        # ball lingers with small bounce
        self.ball_artist.set_offsets([bx, by + 0.9])
        self.ball_artist.set_sizes([140])
        self.ball_shadow.set_offsets([bx, by - 0.12])
        self.ball_shadow.set_sizes([22])
        # fade trail
        for i, t in enumerate(self.trail):
            t.set_alpha(max(0.02, 0.25 - i*0.007))

    def _update_ball_shot(self, shot_frame):
        # shot from receiver to goal center
        t_raw = shot_frame / max(1, self.shot_frames - 1)
        t = ease_in_out_cubic(t_raw)
        # get current receiver pos (may have jump)
        rec_info = self.artists.get(self.primary_id, None)
        if rec_info:
            rx, ry = rec_info['current']
        else:
            rx, ry = self.ball_end
        goal_center = (self.window['x_max'] + 0.5, (self.window['y_min'] + self.window['y_max']) / 2.0)
        sx = lerp(rx, goal_center[0], t)
        sy = lerp(ry, goal_center[1], t)
        # shot is fast, little arc with enhanced effect
        z = 0.12 * math.sin(math.pi * t)  # Increased arc
        self.ball_artist.set_offsets([sx, sy + z*0.8])  # Enhanced offset
        self.ball_shadow.set_offsets([sx, sy - 0.1])
        self.ball_shadow.set_sizes([16])
        # defenders attempt clearance: if close when shot near, animate a lunge
        if t > 0.3:
            for pid, info in self.artists.items():
                if info['team'] == 'defender':
                    # if defender is near shot line, make a quick lunge visual
                    dx = abs(info['current'][0] - sx)
                    dy = abs(info['current'][1] - sy)
                    if dx < 3.2 and dy < 3.2:
                        # quick offset outwards to simulate jump/clear
                        lp = math.sin((t - 0.3) / (1 - 0.3) * math.pi)
                        nx = info['current'][0] + 0.6 * lp
                        ny = info['current'][1] - 0.6 * lp
                        art = info['artist']
                        if isinstance(art, patches.PathPatch):
                            verts = info['orig_verts'] + np.array([nx, ny])
                            art.set_path(mpath.Path(verts, info['path'].codes))
                        else:
                            art.set_offsets([nx, ny])

        # goal flash with enhanced effect
        if t > 0.9:
            self.goal_flash.set_alpha(min(0.95, (t - 0.9) * 12))  # Increased alpha
        else:
            self.goal_flash.set_alpha(0.0)

    def animate(self, frame):
        if self.paused:
            return []

        # NB: phases
        cross_end = self.cross_frames
        recv_end = cross_end + self.recv_frames

        # log kickoff
        if frame == 0 and not self.kick_logged:
            print("⚽ Ball kicked from corner")
            self.kick_logged = True
            self.status_text.set_text("Ball kicked")

        # players update
        self._update_players(frame)
        # players jump/wave
        self._players_jump_phase(frame)
        # keepers/defenders react
        self._keepers_and_defenders_react(frame)

        # ball update per phase
        if frame < cross_end:
            self._update_ball_cross(frame)
        elif frame < recv_end:
            # reception
            if not self.receive_logged:
                print(f"🎯 Ball arriving at reception at frame {frame}")
                self.receive_logged = True
            self._update_ball_reception()
            if frame == cross_end:
                self.status_text.set_text(f"Ball received by Player #{self.primary_id}")
        else:
            # shot
            shot_frame = frame - recv_end
            if not self.shot_logged and shot_frame == 0:
                print("🔫 Shot attempted")
                self.shot_logged = True
            self._update_ball_shot(shot_frame)
            self.status_text.set_text("Shot attempted")

        # collect artists
        artists = [self.ball_artist, self.ball_shadow, self.goal_flash, self.status_text, self.hud_conf]
        # add trail
        artists += self.trail
        for pid, info in self.artists.items():
            art = info['artist']
            txt = info['text']
            artists.append(art)
            artists.append(txt)
        # corner flag
        artists.append(self.corner_flag_artist); artists.append(self.corner_flag_text)
        return artists

    def run(self, save_video=True):
        print(f"Starting cinematic corner replay (v4) for corner_id={self.corner_id}")
        print(f"Ball start: {self.ball_start}, landing: {self.ball_end}, ctrl: {self.ball_ctrl}")

        anim = FuncAnimation(self.fig, self.animate, frames=self.total_frames, interval=1000.0/self.fps, blit=True, repeat=False)

        if save_video:
            filename = f"corner_animation_{self.corner_id}_v4.mp4"
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            try:
                print("Saving MP4 (requires ffmpeg). This may take a little time...")
                anim.save(path, fps=self.fps, dpi=150, writer='ffmpeg', bitrate=5000,
                          savefig_kwargs={'facecolor': self.fig.get_facecolor()})
                print(f"✅ Saved {filename}")
            except Exception as e:
                print(f"⚠️ MP4 save failed: {e}; attempting GIF fallback")
                gifpath = path.replace('.mp4', '.gif')
                try:
                    anim.save(gifpath, fps=max(10, self.fps//6), writer='pillow', dpi=100)
                    print(f"✅ Saved GIF fallback {os.path.basename(gifpath)}")
                except Exception as e2:
                    print(f"❌ Could not save animation: {e2}")

        plt.tight_layout()
        plt.show()
        print("Replay completed")

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Cinematic Corner Replay v4")
    parser.add_argument("--corner_id", type=int, default=1)
    parser.add_argument("--view", choices=['corner_close','broadcast','tactical'], default=DEFAULT_VIEW)
    parser.add_argument("--speed", choices=['slow','normal','fast'], default=DEFAULT_SPEED)
    parser.add_argument("--no-save", action='store_true', help="Do not save video/gif")
    args = parser.parse_args()

    strategy_json = None
    json_path = f"strategy_corner_{args.corner_id}.json"
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                strategy_json = json.load(f)
                print(f"Loaded strategy JSON: {json_path}")
        except Exception as e:
            print(f"Failed to load {json_path}: {e}")

    replay = CornerReplayV4(strategy_data=strategy_json, corner_id=args.corner_id, view=args.view, speed=args.speed, fps=FPS)
    replay.run(save_video=(not args.no_save))

if __name__ == "__main__":
    main()