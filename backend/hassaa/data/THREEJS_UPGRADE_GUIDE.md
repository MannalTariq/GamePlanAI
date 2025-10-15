# Three.js Upgrade Guide

This document outlines how to upgrade the Pygame-based corner kick visualization engine to a Three.js implementation for web-based 3D visualization.

## Architecture Overview

The current Pygame implementation follows a modular design that makes it easy to upgrade to Three.js:

1. **Data Model**: Strategy data is stored in a standardized JSON format
2. **Animation Engine**: Handles timing, easing, and state management
3. **Rendering Engine**: Separated rendering logic from animation logic
4. **Entity Classes**: PlayerSprite and BallSprite encapsulate entity behavior

## Key Design Decisions for Three.js Upgrade

### 1. Data Model Compatibility

The current JSON strategy format is already web-friendly:

```json
{
  "corner_id": 1,
  "best_strategy": "Far Post - Target #244553",
  "confidence": 0.81,
  "corner_flag": {"x": 105, "y": 0},
  "primary": {"id": 244553, "position": [94, 34]},
  "players": [
    {"id": 244553, "team": "attacker", "position": {"x": 90, "y": 36}}
  ]
}
```

This format can be directly used in Three.js without modification.

### 2. Separation of Concerns

The Pygame implementation separates:
- **Animation Logic**: Easing functions, timing, state management
- **Rendering Logic**: Drawing sprites, handling camera
- **Entity Logic**: Player and ball behavior

This separation makes it easy to replace the Pygame renderer with a Three.js renderer.

### 3. Coordinate System

The current implementation uses a metric-based coordinate system (0-105m x 0-68m) which maps directly to Three.js world coordinates.

## Three.js Implementation Plan

### 1. Core Classes Translation

#### PlayerSprite → Three.js PlayerMesh
```javascript
class PlayerMesh extends THREE.Mesh {
  constructor(playerData) {
    // Create geometry based on team
    const geometry = getPlayerGeometry(playerData.team);
    const material = getPlayerMaterial(playerData.team);
    super(geometry, material);
    
    this.playerId = playerData.id;
    this.team = playerData.team;
    this.jumpHeight = 0;
  }
  
  updatePosition(x, y, z = 0) {
    this.position.set(x, z, y); // Note: Y and Z swapped for 3D
  }
}
```

#### BallSprite → Three.js BallMesh
```javascript
class BallMesh extends THREE.Mesh {
  constructor() {
    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
    const material = new THREE.MeshStandardMaterial({ 
      color: 0xFFFFFF,
      roughness: 0.2,
      metalness: 0.1
    });
    super(geometry, material);
  }
  
  updatePosition(x, y, z) {
    this.position.set(x, z, y);
  }
}
```

### 2. Animation System

Replace Pygame's animation loop with Three.js animation:

```javascript
class CornerKickAnimation {
  constructor(strategyData) {
    this.strategyData = strategyData;
    this.clock = new THREE.Clock();
    this.currentFrame = 0;
    
    // Initialize Three.js scene
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    
    this.initializeEntities();
  }
  
  animate() {
    requestAnimationFrame(() => this.animate());
    
    const delta = this.clock.getDelta();
    this.update(delta);
    this.renderer.render(this.scene, this.camera);
  }
}
```

### 3. Pitch Rendering

Create a 3D pitch using Three.js:

```javascript
class Pitch3D {
  constructor() {
    this.pitch = new THREE.Group();
    this.createPitchLines();
    this.createGoals();
    this.createGrass();
  }
  
  createPitchLines() {
    // Create lines for pitch markings using THREE.Line
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xFFFFFF });
    
    // Outer rectangle
    const outerPoints = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(105, 0, 0),
      new THREE.Vector3(105, 0, 68),
      new THREE.Vector3(0, 0, 68),
      new THREE.Vector3(0, 0, 0)
    ];
    
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(outerPoints);
    const line = new THREE.Line(lineGeometry, lineMaterial);
    this.pitch.add(line);
  }
}
```

### 4. Camera System

Enhance the camera system for 3D:

```javascript
class CameraController {
  constructor(camera) {
    this.camera = camera;
    this.mode = 'fixed'; // 'fixed', 'follow', 'smart'
    this.targetPosition = new THREE.Vector3();
  }
  
  update(ballPosition, receiverPosition) {
    switch(this.mode) {
      case 'follow':
        this.targetPosition.set(
          ballPosition.x - 20,
          15,
          ballPosition.z
        );
        break;
      case 'smart':
        // Implement smart camera logic
        break;
    }
    
    // Smooth interpolation
    this.camera.position.lerp(this.targetPosition, 0.05);
    this.camera.lookAt(ballPosition);
  }
}
```

## Migration Steps

### 1. Create Three.js Project Structure
```
corner-kick-3d/
├── index.html
├── src/
│   ├── main.js
│   ├── entities/
│   │   ├── PlayerMesh.js
│   │   ├── BallMesh.js
│   ├── animation/
│   │   ├── CornerKickAnimation.js
│   │   ├── EasingFunctions.js
│   ├── rendering/
│   │   ├── Pitch3D.js
│   │   ├── CameraController.js
│   └── utils/
│       └── DataConverter.js
└── assets/
    └── textures/
```

### 2. Data Conversion Layer
Create a utility to convert Pygame strategy data to Three.js format:

```javascript
class DataConverter {
  static strategyTo3D(strategyData) {
    // Convert 2D coordinates to 3D
    // Add any additional 3D-specific properties
    return {
      ...strategyData,
      players: strategyData.players.map(player => ({
        ...player,
        position: {
          x: player.position.x,
          y: 0, // Ground level
          z: player.position.y
        }
      }))
    };
  }
}
```

### 3. Entity Implementation
Implement 3D versions of PlayerSprite and BallSprite with:
- Proper 3D geometries
- Materials with realistic textures
- Animation capabilities
- Team-specific appearances

### 4. Animation System
Port the animation logic from Pygame to Three.js:
- Easing functions
- State management
- Timing controls

### 5. Rendering System
Implement the rendering pipeline:
- Scene setup
- Lighting
- Camera controls
- Post-processing effects

## Benefits of Three.js Upgrade

1. **Web Deployment**: Run in browsers without Python dependencies
2. **3D Visualization**: Add depth and realism to the visualization
3. **Interactive Controls**: Enhanced user interaction possibilities
4. **Cross-Platform**: Works on any device with a modern browser
5. **Performance**: Hardware-accelerated rendering
6. **Sharing**: Easy to share visualizations online

## Considerations

1. **Learning Curve**: Team will need to learn Three.js if not already familiar
2. **Performance**: 3D rendering is more resource-intensive
3. **Compatibility**: Ensure fallback for older browsers
4. **Assets**: Need 3D models and textures for realistic visualization

## Conclusion

The current Pygame implementation is designed with modularity and separation of concerns that makes upgrading to Three.js straightforward. The data model is web-compatible, and the architecture allows for easy replacement of the rendering engine while preserving animation and game logic.