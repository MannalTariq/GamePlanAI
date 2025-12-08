import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { FaChevronDown } from 'react-icons/fa';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import interact from 'interactjs';
import {
  OVERLAP_RADIUS_PX,
  MAX_RED,
  MAX_BLUE,
  hasOverlapPx,
  validatePlacements as validateAllPlacements,
  pctToPx,
} from '../../utils/placements';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const Simbody = () => {
  const [selectedSetPiece, setSelectedSetPiece] = useState('');
  // Show field once the user chooses a set-piece (Corner/Free kick)
  const isSetupActive = Boolean(selectedSetPiece);
  const [isSetPieceOpen, setIsSetPieceOpen] = useState(false);
  const chartRef = useRef(null);
  const fieldRef = useRef(null);

  const [placedPlayers, setPlacedPlayers] = useState([]);
  const [placeMode, setPlaceMode] = useState('red'); // 'red' | 'blue' | 'gk' | 'ball'
  const [placementMode, setPlacementMode] = useState('manual'); // 'manual' | 'automatic'
  const [autoPlaceType, setAutoPlaceType] = useState('all'); // 'all' | 'attackers' | 'defenders'
  const [isPlacementModeOpen, setIsPlacementModeOpen] = useState(false);
  const [isAutoPlaceTypeOpen, setIsAutoPlaceTypeOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const undoStack = useRef([]);
  const [predictionBoard, setPredictionBoard] = useState(null); // stubbed backend output
  
  // Animation state
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationStep, setAnimationStep] = useState(0);
  const [ballPosition, setBallPosition] = useState(null);
  const [targetPosition, setTargetPosition] = useState(null);
  const [shotBallPosition, setShotBallPosition] = useState(null); // Ball position during shot phase
  const [cornerPosition, setCornerPosition] = useState({ x: 95, y: 5 }); // Default corner position
  const [goalPosition, setGoalPosition] = useState({ x: 95, y: 50 }); // Default goal position
  const [cornerSide, setCornerSide] = useState('right'); // Track current corner side
  const [cornerDescription, setCornerDescription] = useState('Bottom-Right Corner');
  const [freekickPosition, setFreekickPosition] = useState({ x: 50, y: 30 }); // Default freekick position
  const [freekickDecision, setFreekickDecision] = useState('cross'); // Track freekick decision
  const [ballPositionSet, setBallPositionSet] = useState(false); // Track if ball position is set for freekick
  const [isProcessing, setIsProcessing] = useState(false); // Track if simulation is processing
  const animationRef = useRef(null);
  const [playerStates, setPlayerStates] = useState({});
  const [tacticalHighlights, setTacticalHighlights] = useState({});

  const setPieceOptions = ['Corner Kick', 'Free Kick'];

  // removed legacy demo player list and insight text

  const [successRateData, setSuccessRateData] = useState({
    labels: ['Shot Confidence', 'Remaining'],
    datasets: [
      {
        data: [0, 100],
        backgroundColor: ['#688834', '#C41616'],
        borderWidth: 0,
      },
    ],
  });

  const updateChartWithShotConfidence = (shotConfidence) => {
    setSuccessRateData((prev) => ({
      ...prev,
      datasets: [
        {
          ...prev.datasets[0],
          data: [Math.round(shotConfidence), Math.max(0, 100 - Math.round(shotConfidence))],
        },
      ],
    }));
  };

  // Animation utility functions based on Python logic
  const calculateBezierPoint = (start, control, end, t) => {
    const x = Math.pow(1 - t, 2) * start.x + 2 * (1 - t) * t * control.x + Math.pow(t, 2) * end.x;
    const y = Math.pow(1 - t, 2) * start.y + 2 * (1 - t) * t * control.y + Math.pow(t, 2) * end.y;
    return { x, y };
  };

  // Realistic shot trajectory with physics-based motion
  const calculateRealisticShotTrajectory = (start, end, t) => {
    // Calculate distance and angle
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const distance = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx);
    
    // Realistic shot parameters
    // Shot starts fast, maintains speed, slight deceleration near end
    const speedCurve = t < 0.3 
      ? t * 3.33 // Fast acceleration in first 30%
      : t < 0.8
      ? 1.0 + (t - 0.3) * 0.2 // Maintain high speed (80-100% of max)
      : 1.0 - (t - 0.8) * 0.5; // Slight deceleration in last 20%
    
    // Use ease-out for realistic deceleration combined with speed curve
    const easedT = 1 - Math.pow(1 - t, 1.8); // Ease-out for natural motion
    const finalProgress = easedT * speedCurve;
    
    // Calculate horizontal position with realistic speed curve
    const x = start.x + dx * finalProgress;
    
    // Calculate vertical position with realistic parabolic arc
    // Arc is more pronounced for longer shots
    const arcIntensity = Math.min(distance / 30, 1.5); // Stronger arc for longer shots
    const arcHeight = arcIntensity * 8 * Math.sin(t * Math.PI); // Parabolic arc (rises then falls)
    
    // Add slight side spin effect for realism (ball curves slightly)
    const spinEffect = Math.sin(t * Math.PI * 2) * 0.5; // Subtle side movement
    
    const y = start.y + dy * finalProgress - arcHeight + spinEffect;
    
    return { x, y };
  };

  const calculateDistance = (point1, point2) => {
    return Math.sqrt(Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2));
  };

  const getPlayerLabel = (playerId) => {
    const player = placedPlayers.find(p => p.id === playerId);
    return player ? player.label : `Player ${playerId}`;
  };

  const findPrimaryReceiver = useCallback((players, prediction) => {
    // Find the primary receiver based on prediction or closest red player to goal
    const redPlayers = players.filter(p => p.role === 'red');
    if (redPlayers.length === 0) return null;
    
    // Use prediction if available, otherwise find closest to goal
    if (prediction && prediction.primaryPlayer) {
      const playerId = prediction.primaryPlayer.match(/\d+/)?.[0];
      return redPlayers.find(p => p.label.includes(playerId)) || redPlayers[0];
    }
    
    // Find closest red player to goal
    return redPlayers.reduce((closest, current) => {
      const currentDist = calculateDistance(current, goalPosition);
      const closestDist = calculateDistance(closest, goalPosition);
      return currentDist < closestDist ? current : closest;
    });
  }, [goalPosition]);

  const initializePlayerStates = useCallback((players, primaryReceiver) => {
    const states = {};
    players.forEach(player => {
      const startPos = { x: player.xPct, y: player.yPct };
      let targetPos = startPos;
      
      if (player.id === primaryReceiver?.id) {
        // Primary receiver moves toward goal
        const dx = (goalPosition.x - startPos.x) * 0.3;
        const dy = (goalPosition.y - startPos.y) * 0.2;
        targetPos = {
          x: Math.max(0, Math.min(100, startPos.x + dx)),
          y: Math.max(0, Math.min(100, startPos.y + dy))
        };
      } else if (player.role === 'red') {
        // Supporting attackers create space
        const dx = (startPos.x - primaryReceiver?.xPct || 0) * 0.1;
        const dy = (startPos.y - primaryReceiver?.yPct || 0) * 0.1;
        targetPos = {
          x: Math.max(0, Math.min(100, startPos.x + dx)),
          y: Math.max(0, Math.min(100, startPos.y + dy))
        };
      } else if (player.role === 'blue') {
        // Defenders move to mark nearest attackers
        const nearestAttacker = players
          .filter(p => p.role === 'red')
          .reduce((closest, current) => {
            const currentDist = calculateDistance(player, current);
            const closestDist = calculateDistance(player, closest);
            return currentDist < closestDist ? current : closest;
          }, players.find(p => p.role === 'red'));
        
        if (nearestAttacker) {
          const dx = (nearestAttacker.xPct - startPos.x) * 0.7;
          const dy = (nearestAttacker.yPct - startPos.y) * 0.7;
          targetPos = {
            x: Math.max(0, Math.min(100, startPos.x + dx)),
            y: Math.max(0, Math.min(100, startPos.y + dy))
          };
        }
      }
      
      states[player.id] = {
        startPos,
        targetPos,
        currentPos: startPos,
        movementSpeed: player.role === 'red' ? 0.8 : player.role === 'blue' ? 0.6 : 0.4,
        role: player.id === primaryReceiver?.id ? 'Primary Receiver' : 
              player.role === 'red' ? 'Support Runner' : 
              player.role === 'blue' ? 'Marker' : 'Goalkeeper'
      };
    });
    return states;
  }, [goalPosition]);

  const exportToPDF = () => {
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const tempContainer = document.createElement('div');
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    tempContainer.style.top = '-9999px';
    tempContainer.style.width = '800px';
    tempContainer.style.height = '600px';
    tempContainer.style.backgroundColor = '#ffffff';
    document.body.appendChild(tempContainer);

    const reportSection = document.querySelector('.report-card');
    if (reportSection) {
      const reportClone = reportSection.cloneNode(true);
      const printButton = reportClone.querySelector('.print-report-button');
      if (printButton) printButton.remove();
      
      // Ensure the clone has proper styling
      reportClone.style.position = 'relative';
      reportClone.style.transform = 'none';
      tempContainer.appendChild(reportClone);

      // Ensure chart is rendered before capture
      const chartCanvas = reportClone.querySelector('.success-rate-chart canvas');
      if (chartCanvas && chartRef.current) {
        const chartInstance = chartRef.current.chartInstance;
        if (chartInstance) {
          chartInstance.render();
        }
      }

      // Wait for chart to render and then capture
      setTimeout(() => {
        // Re-render the chart to ensure it's visible
        if (chartRef.current && chartRef.current.chartInstance) {
          chartRef.current.chartInstance.render();
        }
        
        html2canvas(tempContainer, { 
          useCORS: true, 
          scale: 2,
          allowTaint: true,
          backgroundColor: '#ffffff',
          logging: false
        }).then((canvas) => {
          const imgData = canvas.toDataURL('image/png');
          const imgWidth = 210;
          const pageHeight = 297;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;
          let heightLeft = imgHeight;
          let position = 0;

          doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
          heightLeft -= pageHeight;

          while (heightLeft > 0) {
            position = heightLeft - imgHeight;
            doc.addPage();
            doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
            heightLeft -= pageHeight;
          }

          doc.save('report.pdf');
          document.body.removeChild(tempContainer);
        });
      }, 500);
    }
  };

  // setup activates as soon as a set-piece is selected

  const pushUndo = (state) => {
    undoStack.current.push(JSON.stringify(state));
    if (undoStack.current.length > 50) undoStack.current.shift();
  };

  const handleFieldClick = (e) => {
    if (!fieldRef.current) return;
    
    // Allow manual placement in manual mode OR when in automatic mode with partial placement
    const isFullAutomatic = placementMode === 'automatic' && autoPlaceType === 'all';
    if (isFullAutomatic) {
      setMessage('Switch to Manual mode or use partial auto-placement to place players manually');
      setTimeout(() => setMessage(''), 2000);
      return;
    }

    const rect = fieldRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const xPct = (clickX / rect.width) * 100;
    const yPct = (clickY / rect.height) * 100;

    // Handle ball position setting for freekick
    if (selectedSetPiece === 'Free Kick' && placeMode === 'ball') {
      handleBallPositionSet(xPct, yPct);
      return;
    }

    const newRole = placeMode;
    const redCount = placedPlayers.filter(p => p.role === 'red').length;
    const blueCount = placedPlayers.filter(p => p.role === 'blue').length;
    const hasGK = placedPlayers.some(p => p.role === 'gk');

    // Check if trying to place a role that was auto-placed (prevent overwriting)
    if (placementMode === 'automatic' && autoPlaceType === 'attackers' && newRole === 'red') {
      setMessage('Attackers are auto-placed. Switch to "Place Defenders Only" or Manual mode to modify.');
      setTimeout(() => setMessage(''), 3000);
      return;
    }
    if (placementMode === 'automatic' && autoPlaceType === 'defenders' && (newRole === 'blue' || newRole === 'gk')) {
      setMessage('Defenders are auto-placed. Switch to "Place Attackers Only" or Manual mode to modify.');
      setTimeout(() => setMessage(''), 3000);
      return;
    }

    if (newRole === 'red' && redCount >= MAX_RED) { setMessage('Maximum players reached for Red.'); return; }
    if (newRole === 'blue' && blueCount >= MAX_BLUE) { setMessage('Maximum players reached for Blue.'); return; }
    if (newRole === 'gk' && hasGK) { setMessage('Goalkeeper already placed.'); return; }

    // prevent overlap in pixels
    const willOverlap = hasOverlapPx(
      placedPlayers,
      { xPct, yPct },
      rect.width,
      rect.height,
      OVERLAP_RADIUS_PX
    );
    if (willOverlap) { setMessage('Too close to an existing marker.'); return; }

    const id = Date.now();
    const label = newRole === 'red' ? `Attacker${redCount + 1}` : newRole === 'blue' ? `Defender${blueCount + 1}` : 'Goalkeeper';
    const newPlayer = { id, xPct, yPct, role: newRole, label };
    pushUndo(placedPlayers);
    setPlacedPlayers((prev) => [...prev, newPlayer]);
    setSelectedId(id);
    setMessage('');
  };

  // Helper function to add random variation to positions
  const addRandomVariation = (basePos, variation = 2) => {
    return {
      x: basePos.x + (Math.random() - 0.5) * variation * 2,
      y: basePos.y + (Math.random() - 0.5) * variation * 2
    };
  };

  // Generate realistic automatic positions for corner kick with randomization
  const generateCornerKickPositions = (placeType = 'all') => {
    const players = [];
    const isRightCorner = cornerSide === 'right';
    const goalX = isRightCorner ? 95 : 5;
    const goalY = 50;
    
    // Base positions with randomization for variety
    // Attackers (Red) - Near goal in penalty area (MAX_RED = 9)
    if (placeType === 'all' || placeType === 'attackers') {
      const baseAttackerPositions = [
        { x: goalX - 8, y: goalY - 10 },   // Near post
        { x: goalX - 8, y: goalY - 3 },    // Center near (spaced)
        { x: goalX - 8, y: goalY + 3 },    // Center near (spaced)
        { x: goalX - 8, y: goalY + 10 },   // Far post near
        { x: goalX - 13, y: goalY - 13 },  // Edge of box (more spacing)
        { x: goalX - 13, y: goalY - 5 },   // Edge of box
        { x: goalX - 13, y: goalY + 5 },   // Edge of box
        { x: goalX - 13, y: goalY + 13 },  // Edge of box
        { x: goalX - 20, y: goalY },       // Deeper center
      ];
      
      baseAttackerPositions.forEach((basePos, idx) => {
        const pos = addRandomVariation(basePos, 1.5); // Add 1.5% random variation
        players.push({
          id: Date.now() + idx * 1000 + Math.random() * 100,
          xPct: Math.max(0, Math.min(100, pos.x)),
          yPct: Math.max(0, Math.min(100, pos.y)),
          role: 'red',
          label: `Attacker${idx + 1}`
        });
      });
    }
    
    // Defenders (Blue) - Marking attackers with proper spacing
    if (placeType === 'all' || placeType === 'defenders') {
      const baseDefenderPositions = [
        { x: goalX - 5, y: goalY - 12 },   // Marking near post (spaced)
        { x: goalX - 5, y: goalY - 4 },    // Marking center (spaced)
        { x: goalX - 5, y: goalY + 4 },    // Marking center (spaced)
        { x: goalX - 5, y: goalY + 12 },   // Marking far post (spaced)
        { x: goalX - 11, y: goalY - 16 },  // Edge marking (more spacing)
        { x: goalX - 11, y: goalY - 7 },   // Edge marking
        { x: goalX - 11, y: goalY + 7 },   // Edge marking
        { x: goalX - 11, y: goalY + 16 },  // Edge marking
        { x: goalX - 16, y: goalY - 9 },   // Deeper marking (more spacing)
        { x: goalX - 16, y: goalY + 9 },   // Deeper marking
      ];
      
      baseDefenderPositions.forEach((basePos, idx) => {
        const pos = addRandomVariation(basePos, 1.5); // Add 1.5% random variation
        players.push({
          id: Date.now() + (idx + 10) * 1000 + Math.random() * 100,
          xPct: Math.max(0, Math.min(100, pos.x)),
          yPct: Math.max(0, Math.min(100, pos.y)),
          role: 'blue',
          label: `Defender${idx + 1}`
        });
      });
    }
    
    // Goalkeeper - In goal area (always placed)
    if (placeType === 'all' || placeType === 'defenders') {
      players.push({
        id: Date.now() + 20000,
        xPct: goalX,
        yPct: goalY,
        role: 'gk',
        label: 'Goalkeeper'
      });
    }
    
    return players;
  };

  // Generate realistic automatic positions for free kick with randomization
  const generateFreekickPositions = (placeType = 'all') => {
    const players = [];
    const fkX = freekickPosition.x;
    const fkY = freekickPosition.y;
    const goalX = 95;
    const goalY = 50;
    
    // Base positions with randomization for variety
    // Attackers (Red) - Around freekick position and near goal (MAX_RED = 9)
    if (placeType === 'all' || placeType === 'attackers') {
      const baseAttackerPositions = [
        { x: fkX + 4, y: fkY },           // Support near ball (spaced)
        { x: fkX + 7, y: fkY - 5 },       // Support left (more spacing)
        { x: fkX + 7, y: fkY + 5 },       // Support right (more spacing)
        { x: goalX - 8, y: goalY - 10 },  // Near post (spaced)
        { x: goalX - 8, y: goalY - 3 },   // Center (spaced)
        { x: goalX - 8, y: goalY + 3 },   // Center (spaced)
        { x: goalX - 8, y: goalY + 10 },   // Far post (spaced)
        { x: goalX - 13, y: goalY - 5 },   // Edge of box (more spacing)
        { x: goalX - 13, y: goalY + 5 },   // Edge of box
      ];
      
      baseAttackerPositions.forEach((basePos, idx) => {
        const pos = addRandomVariation(basePos, 1.5); // Add 1.5% random variation
        players.push({
          id: Date.now() + idx * 1000 + Math.random() * 100,
          xPct: Math.max(0, Math.min(100, pos.x)),
          yPct: Math.max(0, Math.min(100, pos.y)),
          role: 'red',
          label: `Attacker${idx + 1}`
        });
      });
    }
    
    // Defenders (Blue) - Wall and marking with proper spacing
    if (placeType === 'all' || placeType === 'defenders') {
      const baseDefenderPositions = [
        { x: fkX + 9, y: fkY - 4 },       // Wall left (spaced)
        { x: fkX + 9, y: fkY },          // Wall center (spaced)
        { x: fkX + 9, y: fkY + 4 },      // Wall right (spaced)
        { x: goalX - 5, y: goalY - 12 },  // Marking near post (spaced)
        { x: goalX - 5, y: goalY - 4 },   // Marking center (spaced)
        { x: goalX - 5, y: goalY + 4 },   // Marking center (spaced)
        { x: goalX - 5, y: goalY + 12 },  // Marking far post (spaced)
        { x: goalX - 11, y: goalY - 7 },  // Edge marking (more spacing)
        { x: goalX - 11, y: goalY + 7 },  // Edge marking
        { x: goalX - 16, y: goalY },      // Deeper marking (more spacing)
      ];
      
      baseDefenderPositions.forEach((basePos, idx) => {
        const pos = addRandomVariation(basePos, 1.5); // Add 1.5% random variation
        players.push({
          id: Date.now() + (idx + 10) * 1000 + Math.random() * 100,
          xPct: Math.max(0, Math.min(100, pos.x)),
          yPct: Math.max(0, Math.min(100, pos.y)),
          role: 'blue',
          label: `Defender${idx + 1}`
        });
      });
      
      // Goalkeeper - In goal area (always placed with defenders)
      players.push({
        id: Date.now() + 20000,
        xPct: goalX,
        yPct: goalY,
        role: 'gk',
        label: 'Goalkeeper'
      });
    }
    
    return players;
  };

  // Auto-place players based on selected mode
  const handleAutoPlace = () => {
    if (!fieldRef.current) return;
    
    pushUndo(placedPlayers);
    
    let newPlayers = [];
    if (selectedSetPiece === 'Corner Kick') {
      newPlayers = generateCornerKickPositions(autoPlaceType);
    } else if (selectedSetPiece === 'Free Kick') {
      if (!ballPositionSet && autoPlaceType !== 'defenders') {
        setMessage('Please set ball position first for free kick');
        setTimeout(() => setMessage(''), 3000);
        return;
      }
      newPlayers = generateFreekickPositions(autoPlaceType);
    } else {
      setMessage('Please select a set piece first');
      setTimeout(() => setMessage(''), 2000);
      return;
    }
    
    // Merge with existing players if placing only attackers or only defenders
    let finalPlayers = [];
    if (autoPlaceType === 'all') {
      // Replace all players
      finalPlayers = newPlayers;
    } else if (autoPlaceType === 'attackers') {
      // Keep existing defenders and goalkeeper, replace attackers
      const existingDefenders = placedPlayers.filter(p => p.role === 'blue' || p.role === 'gk');
      finalPlayers = [...newPlayers, ...existingDefenders];
    } else if (autoPlaceType === 'defenders') {
      // Keep existing attackers, replace defenders and goalkeeper
      const existingAttackers = placedPlayers.filter(p => p.role === 'red');
      finalPlayers = [...existingAttackers, ...newPlayers];
    }
    
    // Validate placements before setting
    const rect = fieldRef.current.getBoundingClientRect();
    const validation = validateAllPlacements(finalPlayers, rect, { setPiece: selectedSetPiece });
    
    if (!validation.valid) {
      setMessage(`Auto-placement error: ${validation.message}. Trying with adjusted positions...`);
      // Try to fix overlaps by adjusting positions slightly
      const adjustedPlayers = adjustPlayerPositions(finalPlayers, rect);
      const revalidation = validateAllPlacements(adjustedPlayers, rect, { setPiece: selectedSetPiece });
      
      if (revalidation.valid) {
        setPlacedPlayers(adjustedPlayers);
        const typeLabel = autoPlaceType === 'all' ? 'all players' : autoPlaceType === 'attackers' ? 'attackers' : 'defenders';
        setMessage(`Auto-placed ${typeLabel} (positions adjusted)`);
        setTimeout(() => setMessage(''), 3000);
      } else {
        setMessage(`Auto-placement failed: ${revalidation.message}`);
        setTimeout(() => setMessage(''), 5000);
      }
    } else {
      setPlacedPlayers(finalPlayers);
      const typeLabel = autoPlaceType === 'all' ? 'all players' : autoPlaceType === 'attackers' ? 'attackers' : 'defenders';
      setMessage(`Auto-placed ${typeLabel} in realistic positions`);
      setTimeout(() => setMessage(''), 3000);
    }
  };

  // Adjust player positions to prevent overlaps
  const adjustPlayerPositions = (players, rect) => {
    const adjusted = [...players];
    const minDistancePx = OVERLAP_RADIUS_PX + 5; // Add buffer
    
    for (let i = 0; i < adjusted.length; i++) {
      for (let j = i + 1; j < adjusted.length; j++) {
        const a = pctToPx(adjusted[i].xPct, adjusted[i].yPct, rect.width, rect.height);
        const b = pctToPx(adjusted[j].xPct, adjusted[j].yPct, rect.width, rect.height);
        const dist = Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
        
        if (dist < minDistancePx) {
          // Move players apart
          const angle = Math.atan2(b.y - a.y, b.x - a.x);
          const moveDistance = (minDistancePx - dist) / 2;
          const moveX = Math.cos(angle) * moveDistance;
          const moveY = Math.sin(angle) * moveDistance;
          
          // Adjust positions (convert back to percentage)
          adjusted[i].xPct = Math.max(0, Math.min(100, adjusted[i].xPct - (moveX / rect.width) * 100));
          adjusted[i].yPct = Math.max(0, Math.min(100, adjusted[i].yPct - (moveY / rect.height) * 100));
          adjusted[j].xPct = Math.max(0, Math.min(100, adjusted[j].xPct + (moveX / rect.width) * 100));
          adjusted[j].yPct = Math.max(0, Math.min(100, adjusted[j].yPct + (moveY / rect.height) * 100));
        }
      }
    }
    
    return adjusted;
  };

  // Auto-place when corner side changes in automatic mode
  useEffect(() => {
    if (placementMode === 'automatic' && selectedSetPiece === 'Corner Kick' && placedPlayers.length > 0) {
      // Re-generate positions when corner side changes
      handleAutoPlace();
    }
  }, [cornerSide]);

  const resetPlacement = () => {
    setPlacedPlayers([]);
    updateChartWithShotConfidence(0);
    setMessage('');
    setSelectedId(null);
    
    // Reset report section
    setPredictionBoard(null);
    
    // Reset animation state
    setIsAnimating(false);
    setAnimationStep(0);
    setBallPosition(null);
    setTargetPosition(null);
    setShotBallPosition(null);
    setPlayerStates({});
    setTacticalHighlights({});
    
    // Reset corner position to default
    setCornerPosition({ x: 95, y: 5 });
    setGoalPosition({ x: 95, y: 50 });
    setCornerSide('right');
    setCornerDescription('Bottom-Right Corner');
    
    // Reset freekick position to default
    setFreekickPosition({ x: 50, y: 30 });
    setFreekickDecision('cross');
    setBallPositionSet(false);
    setIsProcessing(false);
    
    // Cancel any ongoing animation
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  const undo = () => {
    if (undoStack.current.length === 0) return;
    const prev = JSON.parse(undoStack.current.pop());
    setPlacedPlayers(prev);
  };

  // removed legacy probability calculation in favor of predictionBoard.successRate

  // Initialize drag and drop functionality for markers
  useEffect(() => {
    if (!isSetupActive) return;
    const playerElements = document.querySelectorAll('.marker[data-id]');
    playerElements.forEach((playerEl) => {
      interact(playerEl).draggable({
        listeners: {
          move(event) {
            const target = event.target;
            const x = (parseFloat(target.getAttribute('data-x')) || 0) + event.dx;
            const y = (parseFloat(target.getAttribute('data-y')) || 0) + event.dy;
            target.style.transform = `translate(${x}px, ${y}px)`;
            target.setAttribute('data-x', x);
            target.setAttribute('data-y', y);
          },
          end(event) {
            const target = event.target;
            const id = target.getAttribute('data-id');
            if (!fieldRef.current || !id) return;
            const rect = fieldRef.current.getBoundingClientRect();
            const leftPct = parseFloat(target.getAttribute('data-left-pct')) || 0;
            const topPct = parseFloat(target.getAttribute('data-top-pct')) || 0;
            const currentDX = parseFloat(target.getAttribute('data-x')) || 0;
            const currentDY = parseFloat(target.getAttribute('data-y')) || 0;

            const leftPx = (leftPct / 100) * rect.width + currentDX;
            const topPx = (topPct / 100) * rect.height + currentDY;
            const newXPct = (leftPx / rect.width) * 100;
            const newYPct = (topPx / rect.height) * 100;

            const candidate = { xPct: newXPct, yPct: newYPct };
            const list = placedPlayers.filter(p => String(p.id) !== String(id));
            if (hasOverlapPx(list, candidate, rect.width, rect.height, OVERLAP_RADIUS_PX)) {
              // revert visual
              target.style.transform = 'translate(0px, 0px)';
              target.setAttribute('data-x', '0');
              target.setAttribute('data-y', '0');
              setMessage('Too close to an existing marker.');
              return;
            }

            pushUndo(placedPlayers);
            setPlacedPlayers((prev) =>
              prev.map((p) => (String(p.id) === String(id) ? { ...p, xPct: newXPct, yPct: newYPct } : p))
            );

            target.style.transform = 'translate(0px, 0px)';
            target.setAttribute('data-x', '0');
            target.setAttribute('data-y', '0');
            target.setAttribute('data-left-pct', String(newXPct));
            target.setAttribute('data-top-pct', String(newYPct));
          },
        },
        modifiers: [
          interact.modifiers.restrictRect({ restriction: 'parent' }),
        ],
      });
    });
  }, [isSetupActive, placedPlayers]);

  // keyboard deletion handler
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Backspace' && selectedId) {
        pushUndo(placedPlayers);
        setPlacedPlayers((prev) => prev.filter(p => p.id !== selectedId));
        setSelectedId(null);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [selectedId, placedPlayers]);

  // API endpoints
  const getPlacements = () => placedPlayers;
  const setPlacement = (id, xPct, yPct) => {
    pushUndo(placedPlayers);
    setPlacedPlayers((prev) => prev.map(p => p.id === id ? { ...p, xPct, yPct } : p));
  };
  const removePlacement = (id) => {
    pushUndo(placedPlayers);
    setPlacedPlayers((prev) => prev.filter(p => p.id !== id));
  };
  const validatePlacements = () => validateAllPlacements(placedPlayers, fieldRef.current?.getBoundingClientRect());

  useEffect(() => {
    window.simPlacementAPI = { getPlacements, setPlacement, removePlacement, validatePlacements };
  });

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Convert internal yPct 0-100 visual scale to logical 0-70 scale for backend
  const toLogicalCoords = (p) => ({
    id: p.id,
    role: p.role,
    label: p.label,
    x: Math.max(0, Math.min(100, p.xPct)),
    y: Math.max(0, Math.min(70, (p.yPct / 100) * 70)),
  });

  // Animation function based on Python logic
  const startCornerKickAnimation = useCallback((prediction) => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setAnimationStep(0);
    
    // Find primary receiver
    const primaryReceiver = findPrimaryReceiver(placedPlayers, prediction);
    if (!primaryReceiver) {
      setMessage('No valid receiver found');
      setIsAnimating(false);
      return;
    }
    
    // Set ball and target positions
    setBallPosition(cornerPosition);
    setTargetPosition({ x: primaryReceiver.xPct, y: primaryReceiver.yPct });
    
    // Initialize player states
    const states = initializePlayerStates(placedPlayers, primaryReceiver);
    setPlayerStates(states);
    
    // Set tactical highlights
    const primaryPlayerMatch = prediction.prediction.primaryPlayer.match(/\((\d+)%\)/);
    const primaryPlayerPercentage = primaryPlayerMatch ? parseInt(primaryPlayerMatch[1]) : 0;
    setTacticalHighlights({
      primaryReceiver: primaryReceiver.id,
      shotConfidence: primaryPlayerPercentage, // Using shotConfidence field to store primary player percentage
      tacticalDecision: prediction.prediction.tacticalDecision
    });
    
    // Start animation loop
    // Slowed down: increased steps for slower animation
    let step = 0;
    const maxSteps = 120; // Increased from 60 to 120 for slower animation (~4 seconds at 30fps)
    const ballArrivalStep = 60; // Ball arrives at step 60 (halfway through animation)
    let frameCounter = 0;
    const frameSkip = 1; // Update every frame
    
    const animate = () => {
      frameCounter++;
      if (frameCounter % frameSkip !== 0) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }
      
      if (step >= maxSteps) {
        setIsAnimating(false);
        setAnimationStep(0);
        setBallPosition(null);
        setTargetPosition(null);
        setShotBallPosition(null);
        return;
      }
      
      setAnimationStep(step);
      
      // Calculate unified progress for both ball and players (0 to 1)
      // This ensures perfect synchronization
      const ballProgress = Math.min(1, step / ballArrivalStep);
      const overallProgress = Math.min(1, step / maxSteps);
      
      // Ball flight phase (ball arrives at ballArrivalStep)
      if (step <= ballArrivalStep) {
        const controlPoint = {
          x: (cornerPosition.x + primaryReceiver.xPct) / 2,
          y: (cornerPosition.y + primaryReceiver.yPct) / 2 - 15 // High arc
        };
        const ballPos = calculateBezierPoint(cornerPosition, controlPoint, { x: primaryReceiver.xPct, y: primaryReceiver.yPct }, ballProgress);
        setBallPosition(ballPos);
        setShotBallPosition(null); // Hide shot ball during initial flight
      } else {
        // Ball stays at target position after arrival (until shot starts)
        if (step <= ballArrivalStep + 5) {
          setBallPosition({ x: primaryReceiver.xPct, y: primaryReceiver.yPct });
          setShotBallPosition(null);
        } else {
          // Start shot phase: Ball moves from primary receiver to goal
          const shotStartStep = ballArrivalStep + 5;
          const shotEndStep = maxSteps;
          const shotDuration = shotEndStep - shotStartStep;
          const shotProgress = Math.min(1, (step - shotStartStep) / shotDuration);
          
          // Get primary receiver's final position (where ball arrived)
          const primaryReceiverFinalPos = { x: primaryReceiver.xPct, y: primaryReceiver.yPct };
          
          // Get goalkeeper position (goal center)
          const goalkeeper = placedPlayers.find(p => p.role === 'gk');
          const goalTarget = goalkeeper 
            ? { x: goalPosition.x, y: goalPosition.y } // Goal center
            : goalPosition; // Fallback to goal position
          
          // Use realistic physics-based trajectory for shot
          // Apply ease-in-out for smooth acceleration and deceleration
          const easedProgress = shotProgress < 0.5
            ? 2 * shotProgress * shotProgress
            : 1 - Math.pow(-2 * shotProgress + 2, 2) / 2;
          
          // Calculate realistic shot trajectory with physics
          const shotBallPos = calculateRealisticShotTrajectory(
            primaryReceiverFinalPos,
            goalTarget,
            easedProgress
          );
          setShotBallPosition(shotBallPos);
          
          // Hide original ball during shot
          setBallPosition(null);
        }
      }
      
      // Player movement phase - PERFECTLY SYNCHRONIZED with ball movement
      // All players use the same time-based progress for smooth, synchronized visualization
      if (step >= 0) {
        setPlayerStates(prevStates => {
          const newStates = { ...prevStates };
          
          Object.keys(newStates).forEach(playerId => {
            const state = newStates[playerId];
            if (state && state.startPos && state.targetPos) {
              const isPrimaryReceiver = playerId === primaryReceiver.id;
              
              // Use synchronized progress: primary receiver matches ball, others use overall progress
              // This ensures primary receiver arrives exactly when ball arrives
              let movementProgress;
              if (isPrimaryReceiver) {
                // Primary receiver: perfectly synchronized with ball arrival
                movementProgress = ballProgress;
              } else {
                // Other players: move smoothly throughout animation using same time base
                movementProgress = overallProgress;
              }
              
              // Apply smooth easing for natural movement (no speed multiplier to maintain sync)
              // Use ease-in-out for smooth acceleration and deceleration
              const easedProgress = movementProgress < 0.5
                ? 2 * movementProgress * movementProgress
                : 1 - Math.pow(-2 * movementProgress + 2, 2) / 2;
              
              const currentX = state.startPos.x + (state.targetPos.x - state.startPos.x) * easedProgress;
              const currentY = state.startPos.y + (state.targetPos.y - state.startPos.y) * easedProgress;
              newStates[playerId] = { ...state, currentPos: { x: currentX, y: currentY } };
            }
          });
          
          return newStates;
        });
      }
      
      step++;
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
  }, [isAnimating, placedPlayers, cornerPosition, goalPosition, playerStates, findPrimaryReceiver, initializePlayerStates]);

  // Enhanced animation function that uses API data
  const startCornerKickAnimationWithAPI = useCallback((apiResult) => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setAnimationStep(0);
    
    // Extract data from API response
    const ballTrajectory = apiResult.simulation.ballTrajectory;
    const playerMovements = apiResult.simulation.playerMovements;
    const primaryReceiver = apiResult.simulation.primaryReceiver;
    
    // Set ball and target positions from API
    setBallPosition(ballTrajectory.start);
    setTargetPosition(ballTrajectory.end);
    
    // Initialize player states from API movements
    const states = {};
    playerMovements.forEach(movement => {
      states[movement.playerId] = {
        startPos: movement.startPos,
        targetPos: movement.targetPos,
        currentPos: movement.startPos,
        movementSpeed: movement.movementSpeed || 0.8,
        role: movement.role === 'attacker' ? 
              (movement.playerId === primaryReceiver.player_id ? 'Primary Receiver' : 'Support Runner') :
              movement.role === 'defender' ? 'Marker' : 'Goalkeeper'
      };
    });
    setPlayerStates(states);
    
    // Set tactical highlights from API
    const apiPrimaryPlayerMatch = apiResult.prediction.primaryPlayer.match(/\((\d+)%\)/);
    const apiPrimaryPlayerPercentage = apiPrimaryPlayerMatch ? parseInt(apiPrimaryPlayerMatch[1]) : 0;
    setTacticalHighlights({
      primaryReceiver: primaryReceiver.player_id,
      shotConfidence: apiPrimaryPlayerPercentage, // Using shotConfidence field to store primary player percentage
      tacticalDecision: apiResult.prediction.tacticalDecision
    });
    
    // Start enhanced animation loop with API trajectory
    // Slowed down: increased steps and added frame throttling
    let step = 0;
    const maxSteps = 120; // Increased from 60 to 120 for slower animation (~4 seconds at 30fps)
    const ballArrivalStep = 60; // Ball arrives at step 60 (halfway through animation)
    const trajectoryPoints = ballTrajectory.points;
    let frameCounter = 0;
    const frameSkip = 1; // Update every frame (can increase to 2 for even slower)
    
    const animate = () => {
      frameCounter++;
      if (frameCounter % frameSkip !== 0) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }
      
      if (step >= maxSteps) {
        setIsAnimating(false);
        setAnimationStep(0);
        setBallPosition(null);
        setTargetPosition(null);
        setShotBallPosition(null);
        return;
      }
      
      setAnimationStep(step);
      
      // Calculate unified progress for both ball and players (0 to 1)
      // This ensures perfect synchronization
      const ballProgress = Math.min(1, step / ballArrivalStep);
      const overallProgress = Math.min(1, step / maxSteps);
      
      // Ball flight phase using API trajectory (ball arrives at ballArrivalStep)
      if (step <= ballArrivalStep) {
        const pointIndex = Math.floor(ballProgress * (trajectoryPoints.length - 1));
        const ballPos = trajectoryPoints[pointIndex];
        setBallPosition(ballPos);
        setShotBallPosition(null); // Hide shot ball during initial flight
      } else {
        // Ball stays at target position after arrival (until shot starts)
        if (step <= ballArrivalStep + 5) {
          setBallPosition(ballTrajectory.end);
          setShotBallPosition(null);
        } else {
          // Start shot phase: Ball moves from primary receiver to goal
          const shotStartStep = ballArrivalStep + 5;
          const shotEndStep = maxSteps;
          const shotDuration = shotEndStep - shotStartStep;
          const shotProgress = Math.min(1, (step - shotStartStep) / shotDuration);
          
          // Get primary receiver's final position (where ball arrived)
          const primaryReceiverFinalPos = ballTrajectory.end;
          
          // Get goalkeeper position (goal center)
          const goalkeeper = placedPlayers.find(p => p.role === 'gk');
          const goalTarget = goalkeeper 
            ? { x: goalPosition.x, y: goalPosition.y } // Goal center
            : goalPosition; // Fallback to goal position
          
          // Use realistic physics-based trajectory for shot
          // Apply ease-in-out for smooth acceleration and deceleration
          const easedProgress = shotProgress < 0.5
            ? 2 * shotProgress * shotProgress
            : 1 - Math.pow(-2 * shotProgress + 2, 2) / 2;
          
          // Calculate realistic shot trajectory with physics
          const shotBallPos = calculateRealisticShotTrajectory(
            primaryReceiverFinalPos,
            goalTarget,
            easedProgress
          );
          setShotBallPosition(shotBallPos);
          
          // Hide original ball during shot
          setBallPosition(null);
        }
      }
      
      // Player movement phase - SYNCHRONIZED with ball using same time-based progress
      // All players use the same time reference for smooth, synchronized movement
      if (step >= 0) {
        setPlayerStates(prevStates => {
          const newStates = { ...prevStates };
          
          Object.keys(newStates).forEach(playerId => {
            const state = newStates[playerId];
            if (state && state.startPos && state.targetPos) {
              const isPrimaryReceiver = playerId === primaryReceiver.player_id;
              
              // Use synchronized progress: primary receiver matches ball, others use overall progress
              // This ensures primary receiver arrives exactly when ball arrives
              let movementProgress;
              if (isPrimaryReceiver) {
                // Primary receiver: perfectly synchronized with ball arrival
                movementProgress = ballProgress;
              } else {
                // Other players: move smoothly throughout animation using same time base
                // Use a slightly slower start but same overall timing for smoothness
                movementProgress = overallProgress;
              }
              
              // Apply smooth easing for natural movement (no speed multiplier to maintain sync)
              // Use ease-in-out for smooth acceleration and deceleration
              const easedProgress = movementProgress < 0.5
                ? 2 * movementProgress * movementProgress
                : 1 - Math.pow(-2 * movementProgress + 2, 2) / 2;
              
              const currentX = state.startPos.x + (state.targetPos.x - state.startPos.x) * easedProgress;
              const currentY = state.startPos.y + (state.targetPos.y - state.startPos.y) * easedProgress;
              newStates[playerId] = { ...state, currentPos: { x: currentX, y: currentY } };
            }
          });
          
          return newStates;
        });
      }
      
      step++;
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
  }, [isAnimating, playerStates, placedPlayers, goalPosition]);

  // Freekick animation function that uses API data
  const startFreekickAnimationWithAPI = useCallback((apiResult) => {
    if (isAnimating) return;
    
    setIsAnimating(true);
    setAnimationStep(0);
    
    // Extract data from API response
    const ballTrajectory = apiResult.simulation.ballTrajectory;
    const playerMovements = apiResult.simulation.playerMovements;
    const primaryReceiver = apiResult.simulation.primaryReceiver;
    const freekickPos = apiResult.simulation.freekickPosition;
    
    // Convert freekick position from meters to percentage if needed
    const freekickStartPos = freekickPos 
      ? { x: (freekickPos.x / 105) * 100, y: (freekickPos.y / 68) * 100 }
      : ballTrajectory.start;
    
    // Set ball and target positions from API
    // Ball should go to where primary receiver will receive it (ballTrajectory.end)
    setBallPosition(freekickStartPos);
    setTargetPosition(ballTrajectory.end);
    
    // Initialize player states from API movements
    // IMPORTANT: Include ALL placed players, not just those in movements
    const states = {};
    
    // First, add all placed players with their current positions
    placedPlayers.forEach(player => {
      states[player.id] = {
        startPos: { x: player.xPct, y: player.yPct },
        targetPos: { x: player.xPct, y: player.yPct }, // Default: stay in place
        currentPos: { x: player.xPct, y: player.yPct },
        movementSpeed: 0.8,
        role: player.role === 'red' ? 'Support Runner' : 
              player.role === 'blue' ? 'Marker' : 'Goalkeeper'
      };
    });
    
    // Then, update with movement data from API
    playerMovements.forEach(movement => {
      if (states[movement.playerId]) {
        const isPrimaryReceiver = movement.playerId === primaryReceiver.player_id;
        
        if (isPrimaryReceiver) {
          // Primary receiver: use their movement target position (where they will be after off-ball movement)
          // The ball trajectory is already calculated to go to this target position
          states[movement.playerId] = {
            ...states[movement.playerId],
            startPos: movement.startPos, // Their starting position
            targetPos: movement.targetPos, // Their target position (where ball will arrive)
            movementSpeed: movement.movementSpeed || 1.0, // Normal speed for off-ball movement
            role: 'Primary Receiver'
          };
        } else {
          // Other players use their movement target positions
          states[movement.playerId] = {
            ...states[movement.playerId],
            startPos: movement.startPos,
            targetPos: movement.targetPos,
            movementSpeed: movement.movementSpeed || 0.8,
            role: movement.role === 'attacker' ? 'Support Runner' :
                  movement.role === 'defender' ? 'Marker' : 'Goalkeeper'
          };
        }
      }
    });
    
    setPlayerStates(states);
    
    // Set tactical highlights from API
    const apiPrimaryPlayerMatch = apiResult.prediction.primaryPlayer.match(/\((\d+)%\)/);
    const apiPrimaryPlayerPercentage = apiPrimaryPlayerMatch ? parseInt(apiPrimaryPlayerMatch[1]) : 0;
    const tacticalDecision = apiResult.prediction.tacticalDecision;
    const isDirectShot = tacticalDecision === "Direct free kick shot" || tacticalDecision === "Direct shot" || tacticalDecision === "Direct freekick shot";
    
    setTacticalHighlights({
      primaryReceiver: primaryReceiver.player_id,
      shotConfidence: apiPrimaryPlayerPercentage,
      tacticalDecision: tacticalDecision
    });
    
    // Start enhanced animation loop with API trajectory
    // Synchronized: ball and players move at same pace
    let step = 0;
    const maxSteps = 120; // Total animation steps (~4 seconds at 30fps)
    const ballArrivalStep = 60; // Ball arrives at step 60 (halfway through animation)
    const trajectoryPoints = ballTrajectory.points;
    let frameCounter = 0;
    const frameSkip = 1; // Update every frame
    
    const animate = () => {
      frameCounter++;
      if (frameCounter % frameSkip !== 0) {
        animationRef.current = requestAnimationFrame(animate);
        return;
      }
      
      if (step >= maxSteps) {
        setIsAnimating(false);
        setAnimationStep(0);
        setBallPosition(null);
        setTargetPosition(null);
        setShotBallPosition(null);
        return;
      }
      
      setAnimationStep(step);
      
      // Calculate unified progress for both ball and players (0 to 1)
      // This ensures perfect synchronization across all elements
      const ballProgress = Math.min(1, step / ballArrivalStep);
      const overallProgress = Math.min(1, step / maxSteps);
      
      // Ball flight phase using API trajectory (ball arrives at ballArrivalStep)
      if (step <= ballArrivalStep) {
        const pointIndex = Math.floor(ballProgress * (trajectoryPoints.length - 1));
        const ballPos = trajectoryPoints[pointIndex];
        setBallPosition(ballPos);
        setShotBallPosition(null); // Hide shot ball during initial flight
      } else {
        // Ball stays at target position after arrival (until shot starts)
        if (step <= ballArrivalStep + 5) {
          setBallPosition(ballTrajectory.end);
          setShotBallPosition(null);
        } else {
          // Start shot phase: Ball moves from primary receiver to goal
          const shotStartStep = ballArrivalStep + 5;
          const shotEndStep = maxSteps;
          const shotDuration = shotEndStep - shotStartStep;
          const shotProgress = Math.min(1, (step - shotStartStep) / shotDuration);
          
          // Get primary receiver's final position (where ball arrived)
          const primaryReceiverFinalPos = ballTrajectory.end;
          
          // Get goalkeeper position (goal center)
          const goalkeeper = placedPlayers.find(p => p.role === 'gk');
          const goalTarget = goalkeeper 
            ? { x: goalPosition.x, y: goalPosition.y } // Goal center
            : goalPosition; // Fallback to goal position
          
          // Use realistic physics-based trajectory for shot
          // Apply ease-in-out for smooth acceleration and deceleration
          const easedProgress = shotProgress < 0.5
            ? 2 * shotProgress * shotProgress
            : 1 - Math.pow(-2 * shotProgress + 2, 2) / 2;
          
          // Calculate realistic shot trajectory with physics
          const shotBallPos = calculateRealisticShotTrajectory(
            primaryReceiverFinalPos,
            goalTarget,
            easedProgress
          );
          setShotBallPosition(shotBallPos);
          
          // Hide original ball during shot
          setBallPosition(null);
        }
      }
      
      // Player movement phase - PERFECTLY SYNCHRONIZED with ball movement
      // All players use the same time-based progress for smooth, synchronized visualization
      if (step >= 0) {
        setPlayerStates(prevStates => {
          const newStates = { ...prevStates };
          
          Object.keys(newStates).forEach(playerId => {
            const state = newStates[playerId];
            if (state && state.startPos && state.targetPos) {
              const isPrimaryReceiver = playerId === primaryReceiver.player_id;
              
              // Use synchronized progress based on scenario
              let movementProgress;
              if (isDirectShot) {
                // Direct shot: all players move synchronized with ball flight
                // Ball goes to goal, players move in sync
                movementProgress = ballProgress;
              } else {
                // Cross/pass: primary receiver perfectly synchronized with ball arrival
                if (isPrimaryReceiver) {
                  movementProgress = ballProgress; // Perfect sync with ball
                } else {
                  // Other players: smooth movement throughout, using same time base
                  movementProgress = overallProgress;
                }
              }
              
              // Apply smooth easing for natural movement (no speed multiplier to maintain perfect sync)
              // Ease-in-out provides smooth acceleration and deceleration
              const easedProgress = movementProgress < 0.5
                ? 2 * movementProgress * movementProgress
                : 1 - Math.pow(-2 * movementProgress + 2, 2) / 2;
              
              const currentX = state.startPos.x + (state.targetPos.x - state.startPos.x) * easedProgress;
              const currentY = state.startPos.y + (state.targetPos.y - state.startPos.y) * easedProgress;
              newStates[playerId] = { ...state, currentPos: { x: currentX, y: currentY } };
            }
          });
          
          return newStates;
        });
      }
      
      step++;
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animationRef.current = requestAnimationFrame(animate);
  }, [isAnimating, playerStates, placedPlayers, goalPosition]);

  // Corner button API functions
  const handleCornerLeft = async () => {
    try {
      console.log('📍 Left corner button clicked');
      
      const response = await fetch(`${process.env.REACT_APP_CORNER_API || 'http://localhost:5000/api'}/corner/left`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          currentCornerPosition: cornerPosition,
          currentGoalPosition: goalPosition
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('📥 Left corner response:', result);

      if (result.success) {
        // Clear any ongoing animation when corner changes
        setIsAnimating(false);
        setBallPosition(null);
        setTargetPosition(null);
        setAnimationStep(0);
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
          animationRef.current = null;
        }
        
        // Update frontend state with new positions
        setCornerPosition(result.cornerPosition);
        setGoalPosition(result.goalPosition);
        setCornerSide(result.cornerSide);
        setCornerDescription(result.cornerDescription);
        setMessage(result.message);
        
        // Clear message after 3 seconds
        setTimeout(() => setMessage(''), 3000);
      } else {
        throw new Error(result.error || 'Unknown API error');
      }

    } catch (error) {
      console.error('❌ Left corner API call failed:', error);
      setMessage(`Error: ${error.message}`);
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const handleBallPositionSet = async (xPct, yPct) => {
    try {
      console.log('📍 Ball position set:', { x: xPct, y: yPct });
      
      const response = await fetch(`${process.env.REACT_APP_FK_API || 'http://localhost:5001/api/freekick'}/position`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          freekickPosition: { x: xPct, y: yPct }
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('📥 Ball position response:', result);

      if (result.success) {
        // Update frontend state with new ball position
        setFreekickPosition({ x: xPct, y: yPct });
        setFreekickDecision(result.freekickDecision);
        setBallPositionSet(true);
        setPlaceMode('red'); // Switch back to player placement mode
        setMessage(result.message);
        
        // Clear message after 3 seconds
        setTimeout(() => setMessage(''), 3000);
      } else {
        throw new Error(result.error || 'Unknown API error');
      }

    } catch (error) {
      console.error('❌ Ball position API call failed:', error);
      setMessage(`Error: ${error.message}`);
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const handleFreekickPositionSet = async (xPct, yPct) => {
    try {
      console.log('📍 Freekick position set:', { x: xPct, y: yPct });
      
      const response = await fetch(`${process.env.REACT_APP_FK_API || 'http://localhost:5001/api/freekick'}/position`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          freekickPosition: { x: xPct, y: yPct }
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('📥 Freekick position response:', result);

      if (result.success) {
        // Update frontend state with new freekick position
        setFreekickPosition({ x: xPct, y: yPct });
        setFreekickDecision(result.freekickDecision);
        setMessage(result.message);
        
        // Clear message after 3 seconds
        setTimeout(() => setMessage(''), 3000);
      } else {
        throw new Error(result.error || 'Unknown API error');
      }

    } catch (error) {
      console.error('❌ Freekick position API call failed:', error);
      setMessage(`Error: ${error.message}`);
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const handleCornerRight = async () => {
    try {
      console.log('📍 Right corner button clicked');
      
      const response = await fetch(`${process.env.REACT_APP_CORNER_API || 'http://localhost:5000/api'}/corner/right`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          currentCornerPosition: cornerPosition,
          currentGoalPosition: goalPosition
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('📥 Right corner response:', result);

      if (result.success) {
        // Clear any ongoing animation when corner changes
        setIsAnimating(false);
        setBallPosition(null);
        setTargetPosition(null);
        setAnimationStep(0);
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
          animationRef.current = null;
        }
        
        // Update frontend state with new positions
        setCornerPosition(result.cornerPosition);
        setGoalPosition(result.goalPosition);
        setCornerSide(result.cornerSide);
        setCornerDescription(result.cornerDescription);
        setMessage(result.message);
        
        // Clear message after 3 seconds
        setTimeout(() => setMessage(''), 3000);
      } else {
        throw new Error(result.error || 'Unknown API error');
      }

    } catch (error) {
      console.error('❌ Right corner API call failed:', error);
      setMessage(`Error: ${error.message}`);
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const handleGenerateAndSimulate = async () => {
    if (!fieldRef.current) return;
    
    // Check ball position for freekick
    if (selectedSetPiece === 'Free Kick' && !ballPositionSet) {
      setMessage('Please set ball position first');
      return;
    }
    
    const reds = placedPlayers.filter(p => p.role === 'red').length;
    const blues = placedPlayers.filter(p => p.role === 'blue').length;
    const gks = placedPlayers.filter(p => p.role === 'gk').length;
    if (reds < MAX_RED || blues < MAX_BLUE || gks !== 1) {
      setMessage('please place all players');
      return;
    }

    // Set processing state
    setIsProcessing(true);
    setMessage('Processing simulation... Please wait');

    // Validate placements - but make it less strict for now
    if (!fieldRef.current) {
      setIsProcessing(false);
      setMessage('Field not ready');
      return;
    }
    
    const validation = validateAllPlacements(placedPlayers, fieldRef.current.getBoundingClientRect(), { setPiece: selectedSetPiece });
    if (!validation.valid) {
      setIsProcessing(false);
      setMessage(validation.message || 'Invalid placements');
      setTimeout(() => setMessage(''), 5000);
      return;
    }

    // Show loading message
    setMessage('Generating simulation...');
    setIsAnimating(false); // Reset animation state

    try {
      // Prepare data for API call
      const apiData = {
        players: placedPlayers,
        cornerPosition: cornerPosition,
        goalPosition: goalPosition,
        freekickPosition: freekickPosition,
        setPiece: selectedSetPiece
      };

      console.log('📤 Sending simulation request to API:', apiData);

      // Call backend API - route to correct API based on set piece
      const apiUrl = selectedSetPiece === 'Free Kick' 
        ? `${process.env.REACT_APP_FK_API || 'http://localhost:5001/api/freekick'}/simulate`
        : `${process.env.REACT_APP_CORNER_API || 'http://localhost:5000/api'}/simulate`;
      
      // Add timeout to prevent hanging
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(apiData),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('📥 Received simulation response:', result);

      if (result.success) {
        // Clear processing state
        setIsProcessing(false);
        
        // Update prediction board with API response
        setPredictionBoard(result);
        // Extract primary player percentage from the primaryPlayer string (e.g., "Attacker1 (75%)")
        const primaryPlayerMatch = result.prediction.primaryPlayer.match(/\((\d+)%\)/);
        const primaryPlayerPercentage = primaryPlayerMatch ? parseInt(primaryPlayerMatch[1]) : 0;
        updateChartWithShotConfidence(result.prediction.shotConfidence);
        setMessage('Simulation completed successfully!');
        
        // Start animation with API data - use appropriate function based on set piece
        if (selectedSetPiece === 'Free Kick') {
          startFreekickAnimationWithAPI(result);
        } else {
          startCornerKickAnimationWithAPI(result);
        }
        
        // Clear success message after 3 seconds
        setTimeout(() => setMessage(''), 3000);
      } else {
        throw new Error(result.error || 'Unknown API error');
      }

    } catch (error) {
      console.error('❌ API call failed:', error);
      setIsProcessing(false);
      
      if (error.name === 'AbortError') {
        setMessage('Request timeout - API server may be slow or not responding. Please try again.');
      } else if (error.message.includes('Failed to fetch')) {
        setMessage(`Cannot connect to API server. Make sure the backend is running on ${selectedSetPiece === 'Free Kick' ? 'port 5001' : 'port 5000'}`);
      } else {
        setMessage(`API Error: ${error.message}`);
      }
      
      setTimeout(() => setMessage(''), 8000);
    }
  };

  return (
    <div className="Simbody">
      {/* Selection Section */}
      <div className="obody-section selection-section">
        <div className="controls-container">
          <div className="dropdowns-row">
            <div className="custom-dropdown">
              <div
                className={`custom-dropdown-header ${isSetPieceOpen ? 'open' : ''}`}
                onClick={() => setIsSetPieceOpen(!isSetPieceOpen)}
              >
                {selectedSetPiece || 'Select Set-piece'}
                <FaChevronDown className="arrow" />
              </div>
              {isSetPieceOpen && (
                <ul className="custom-dropdown-options">
                  {setPieceOptions.map((option, index) => (
                    <li
                      key={index}
                      className="custom-option"
                      onClick={() => {
                        setSelectedSetPiece(option);
                        setIsSetPieceOpen(false);
                      }}
                    >
                      {option}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            {selectedSetPiece && (
              <>
                <div className="custom-dropdown">
                  <div
                    className={`custom-dropdown-header ${isPlacementModeOpen ? 'open' : ''}`}
                    onClick={() => setIsPlacementModeOpen(!isPlacementModeOpen)}
                  >
                    {placementMode === 'manual' ? 'Manual Placement' : 'Automatic Placement'}
                    <FaChevronDown className="arrow" />
                  </div>
                  {isPlacementModeOpen && (
                    <ul className="custom-dropdown-options">
                      <li
                        className="custom-option"
                        onClick={() => {
                          setPlacementMode('manual');
                          setIsPlacementModeOpen(false);
                          setMessage('Manual placement mode - Click on field to place players');
                          setTimeout(() => setMessage(''), 3000);
                        }}
                      >
                        Manual Placement
                      </li>
                      <li
                        className="custom-option"
                        onClick={() => {
                          setPlacementMode('automatic');
                          setIsPlacementModeOpen(false);
                          handleAutoPlace();
                        }}
                      >
                        Automatic Placement
                      </li>
                    </ul>
                  )}
                </div>
                {placementMode === 'automatic' && (
                  <div className="custom-dropdown">
                    <div
                      className={`custom-dropdown-header ${isAutoPlaceTypeOpen ? 'open' : ''}`}
                      onClick={() => setIsAutoPlaceTypeOpen(!isAutoPlaceTypeOpen)}
                    >
                      {autoPlaceType === 'all' ? 'Place All Players' : 
                       autoPlaceType === 'attackers' ? 'Place Attackers Only' : 
                       'Place Defenders Only'}
                      <FaChevronDown className="arrow" />
                    </div>
                    {isAutoPlaceTypeOpen && (
                      <ul className="custom-dropdown-options">
                        <li
                          className="custom-option"
                          onClick={() => {
                            setAutoPlaceType('all');
                            setIsAutoPlaceTypeOpen(false);
                            handleAutoPlace();
                          }}
                        >
                          Place All Players
                        </li>
                        <li
                          className="custom-option"
                          onClick={() => {
                            setAutoPlaceType('attackers');
                            setIsAutoPlaceTypeOpen(false);
                            handleAutoPlace();
                          }}
                        >
                          Place Attackers Only
                        </li>
                        <li
                          className="custom-option"
                          onClick={() => {
                            setAutoPlaceType('defenders');
                            setIsAutoPlaceTypeOpen(false);
                            handleAutoPlace();
                          }}
                        >
                          Place Defenders Only
                        </li>
                      </ul>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
          <div className="button-row">
            <div className="status-text">
              {selectedSetPiece ? 'Setup Ready' : 'Choose Set-piece Above'}
            </div>
          </div>
        </div>
      </div>

      {/* Simulation Section */}
      {isSetupActive && (
        <div className="obody-section simulation-section">
          <div className="field-simulation">
            <div className={`field ${isAnimating ? 'animating' : ''}`} ref={fieldRef} onClick={handleFieldClick}>
              {/* Corner flag - only show for corner kicks */}
              {selectedSetPiece === 'Corner Kick' && (
                <div 
                  className="corner-flag" 
                  style={{ 
                    left: `${cornerPosition.x}%`, 
                    top: `${cornerPosition.y}%`,
                    position: 'absolute',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 10
                  }}
                >
                  <div className="flag-marker">⚽</div>
                </div>
              )}


              {/* Ball - show at appropriate position when not animating, or at animation position when animating */}
              {(ballPosition || (!isAnimating && (selectedSetPiece === 'Corner Kick' || (selectedSetPiece === 'Free Kick' && ballPositionSet)))) && (
                <div 
                  className="ball" 
                  style={{ 
                    left: `${(ballPosition || (selectedSetPiece === 'Free Kick' ? freekickPosition : cornerPosition)).x}%`, 
                    top: `${(ballPosition || (selectedSetPiece === 'Free Kick' ? freekickPosition : cornerPosition)).y}%`,
                    position: 'absolute',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 20
                  }}
                >
                  <div className="ball-marker">⚽</div>
                </div>
              )}

              {/* Shot ball - shows ball trajectory from player to goal during shot phase */}
              {shotBallPosition && (
                <div 
                  className="ball shot-ball" 
                  style={{ 
                    left: `${shotBallPosition.x}%`, 
                    top: `${shotBallPosition.y}%`,
                    position: 'absolute',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 21,
                    transition: 'none' // Instant position updates for realistic motion
                  }}
                >
                  <div className="ball-marker">⚽</div>
                </div>
              )}

              {/* Ball trail - only show during animation */}
              {ballPosition && isAnimating && animationStep > 3 && (
                <div 
                  className="ball-trail" 
                  style={{ 
                    left: `${ballPosition.x}%`, 
                    top: `${ballPosition.y}%`,
                    position: 'absolute',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 15
                  }}
                >
                  <div className="trail-dot"></div>
                </div>
              )}

              {/* Players with animation support */}
              {placedPlayers.map((p) => {
                const playerState = playerStates[p.id];
                const currentPos = playerState?.currentPos || { x: p.xPct, y: p.yPct };
                const isPrimaryReceiver = tacticalHighlights.primaryReceiver === p.id;
                
                return (
                <div
                  key={p.id}
                    className={`marker ${p.role} ${selectedId === p.id ? 'selected' : ''} ${isPrimaryReceiver ? 'primary-receiver' : ''}`}
                  data-id={p.id}
                    data-left-pct={currentPos.x}
                    data-top-pct={currentPos.y}
                    style={{ 
                      left: `${currentPos.x}%`, 
                      top: `${currentPos.y}%`,
                      transition: isAnimating ? 'none' : 'all 0.1s ease'
                    }}
                  onClick={(ev) => { ev.stopPropagation(); setSelectedId(p.id); }}
                  onContextMenu={(ev) => { ev.preventDefault(); pushUndo(placedPlayers); setPlacedPlayers(prev => prev.filter(x => x.id !== p.id)); }}
                >
                  <div className="circle" />
                  <div className="label">{p.label}</div>
                    
                    {/* Primary receiver highlight */}
                    {isPrimaryReceiver && (
                      <div className="receiver-highlight">
                        <div className="highlight-ring"></div>
                        <div className="receiver-label">TARGET</div>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Tactical overlay */}
              {isAnimating && tacticalHighlights.primaryReceiver && (
                <div className="tactical-overlay">
                  <div className="overlay-content">
                    <div className="gnn-prediction">🧠 GNN PREDICTION</div>
                    <div className="primary-info">
                      Primary: {getPlayerLabel(tacticalHighlights.primaryReceiver)} (72%)
                    </div>
                    <div className="shot-confidence">
                      {/* Primary Player Percentage: {tacticalHighlights.shotConfidence}% */}
                    </div>
                    <div className="tactical-decision">
                      Decision: {tacticalHighlights.tacticalDecision}
                    </div>
                  </div>
                </div>
              )}


              {/* Shot outcome animation */}
              {isAnimating && animationStep >= 45 && targetPosition && (
                <div className="shot-outcome">
                  <div 
                    className="shot-arrow"
                    style={{
                      left: `${targetPosition.x}%`,
                      top: `${targetPosition.y}%`,
                      position: 'absolute',
                      transform: 'translate(-50%, -50%)',
                      zIndex: 25
                    }}
                  >
                    <div className="arrow-line"></div>
                    <div className="shot-label">SHOT!</div>
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="placement-toolbar">
            <div className="counts">
              <span>Red remaining: {Math.max(0, MAX_RED - placedPlayers.filter(p=>p.role==='red').length)} /{MAX_RED}</span>
              <span>Blue remaining: {Math.max(0, MAX_BLUE - placedPlayers.filter(p=>p.role==='blue').length)} /{MAX_BLUE}</span>
              <span>GK placed: {placedPlayers.some(p=>p.role==='gk') ? 'Yes' : 'No'}</span>
            </div>
            <div className="modes">
              <button 
                className={`optimize-button ${placeMode==='red'?'active':''}`} 
                onClick={()=>setPlaceMode('red')}
                disabled={placementMode === 'automatic' && autoPlaceType === 'attackers'}
                title={placementMode === 'automatic' && autoPlaceType === 'attackers' ? 'Attackers are auto-placed' : ''}
              >
                Place Attacker
              </button>
              <button 
                className={`optimize-button ${placeMode==='blue'?'active':''}`} 
                onClick={()=>setPlaceMode('blue')}
                disabled={placementMode === 'automatic' && autoPlaceType === 'defenders'}
                title={placementMode === 'automatic' && autoPlaceType === 'defenders' ? 'Defenders are auto-placed' : ''}
              >
                Place Defender
              </button>
              <button 
                className={`optimize-button ${placeMode==='gk'?'active':''}`} 
                onClick={()=>setPlaceMode('gk')}
                disabled={placementMode === 'automatic' && autoPlaceType === 'defenders'}
                title={placementMode === 'automatic' && autoPlaceType === 'defenders' ? 'Goalkeeper is auto-placed' : ''}
              >
                Place Goal Keeper
              </button>
              {selectedSetPiece === 'Free Kick' && (
                <button 
                  className={`optimize-button ${placeMode==='ball'?'active':''} ${ballPositionSet?'completed':''}`} 
                  onClick={()=>setPlaceMode('ball')}
                >
                  {ballPositionSet ? '✓ Ball Position Set' : 'Add Ball Position'}
                </button>
              )}
            </div>
            <div className="actions">
              {placementMode === 'automatic' && (
                <button 
                  className="optimize-button" 
                  onClick={handleAutoPlace}
                  title={`Re-generate ${autoPlaceType === 'all' ? 'all players' : autoPlaceType === 'attackers' ? 'attackers' : 'defenders'}`}
                >
                  {autoPlaceType === 'all' ? 'Auto-Place All' : 
                   autoPlaceType === 'attackers' ? 'Auto-Place Attackers' : 
                   'Auto-Place Defenders'}
                </button>
              )}
              <button className="optimize-button" onClick={resetPlacement}>Reset</button>
              {(placementMode === 'manual' || (placementMode === 'automatic' && autoPlaceType !== 'all')) && (
                <button className="optimize-button" onClick={undo}>Undo</button>
              )}
              <button 
                className={`optimize-button ${isProcessing ? 'processing' : ''}`} 
                onClick={handleGenerateAndSimulate}
                disabled={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Generate & Simulate'}
              </button>
            </div>
            {placementMode === 'automatic' && autoPlaceType !== 'all' && (
              <div className="placement-info" style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: '#666', textAlign: 'center' }}>
                {autoPlaceType === 'attackers' 
                  ? '✓ Attackers auto-placed. You can now manually place defenders and goalkeeper.'
                  : '✓ Defenders auto-placed. You can now manually place attackers.'}
              </div>
            )}
            {selectedSetPiece === 'Corner Kick' && (
              <div className="corner-controls">
                <div className="corner-info">
                  <span>Corner: {cornerDescription}</span>
                </div>
                <div className="corner-buttons">
                  <button 
                    className={`corner-button ${cornerSide === 'left' ? 'active' : ''}`} 
                    onClick={handleCornerLeft}
                  >
                    ← Left Corner
                  </button>
                  <button 
                    className={`corner-button ${cornerSide === 'right' ? 'active' : ''}`} 
                    onClick={handleCornerRight}
                  >
                    Right Corner →
                  </button>
                </div>
              </div>
            )}
            {selectedSetPiece === 'Free Kick' && (
              <div className="freekick-controls">
                <div className="freekick-info">
                  <span>Freekick: {freekickDecision.replace('_', ' ').toUpperCase()}</span>
                </div>
                <div className="freekick-instruction">
                  {!ballPositionSet ? (
                    <span>Click "Add Ball Position" button, then click on field to set ball position</span>
                  ) : (
                    <span>Ball position set! Now place players on the field</span>
                  )}
                </div>
                {isProcessing && (
                  <div className="processing-indicator">
                    <div className="spinner"></div>
                    <span>Processing freekick simulation...</span>
                  </div>
                )}
              </div>
            )}
          </div>
          {message && <div className="placement-message">{message}</div>}
        </div>
      )}

      {/* Report Section */}
      {predictionBoard && (
        <div className="report-section-container">
          <div className="report-card">
            <div className="report-header">
              <h2>Report</h2>
            </div>
            
            <div className="report-content">
              <div className="prediction-section">
                <div className="prediction-title">Prediction :</div>
                <div className="prediction-details">
                  <div className="primary-player">Primary {predictionBoard.prediction.primaryPlayer.replace(/\s*\(\d+%\)/g, '')}</div>
                  {predictionBoard.prediction.kicker && (
                    <div className="kicker-info">Kicker: {predictionBoard.prediction.kicker.replace(/\s*\(\d+%\)/g, '')}</div>
                  )}
                  {/* <div className="primary-player-percentage">Primary Player Percentage: {(() => {
                    const primaryPlayerMatch = predictionBoard.prediction.primaryPlayer.match(/\((\d+)%\)/);
                    return primaryPlayerMatch ? parseInt(primaryPlayerMatch[1]) : 0;
                  })()}%</div> */}
                </div>
                
                <div className="backend-metrics-section">
                  <div className="backend-metrics-title">Corner Kick Analysis:</div>
                  <div className="backend-metrics-details">
                    <div className="shot-confidence-metric">
                      Shot Confidence: {Math.round(predictionBoard.prediction.shotConfidence)}
                    </div>
                    <div className="tactical-decision-metric">
                      Tactical Decision: {predictionBoard.prediction.tacticalDecision}
                    </div>
                    <div className="total-alternatives-metric">
                      Total Alternatives: {predictionBoard.prediction.alternatives.length}
                    </div>
                  </div>
                </div>
                
                <div className="alternatives-section">
                  <div className="alternatives-title">Alternatives:</div>
                  <div className="alternatives-list">
                    {predictionBoard.prediction.alternatives.map((alt, index) => (
                      <div key={index} className="alternative-item">
                        {index + 1}. {alt.player.replace(/\s*\(\d+%\)/g, '')}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              
              <div className="success-rate-section">
                <div className="success-rate-title">Shot Confidence</div>
                <div className="success-rate-chart">
                  <Doughnut
                    ref={chartRef}
                    data={successRateData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { 
                        legend: { display: false },
                        tooltip: {
                          callbacks: {
                            label: function(context) {
                              const label = context.label || '';
                              const value = context.parsed;
                              return `${label}: ${value}`;
                            }
                          }
                        }
                      },
                      cutout: '70%',
                    }}
                  />
                </div>
                <div className="success-rate-value">
                  {Math.round(predictionBoard.prediction.shotConfidence)}
                </div>
              </div>
            </div>
            
            <button className="print-report-button" onClick={exportToPDF}>
              Print Report
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Simbody;