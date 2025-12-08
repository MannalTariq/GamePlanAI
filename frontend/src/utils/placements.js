// Placement utilities for simulation

export const MAX_RED = 9;
export const MAX_BLUE = 10;
export const OVERLAP_RADIUS_PX = 20;

export function pctToPx(xPct, yPct, width, height) {
  return { x: (xPct / 100) * width, y: (yPct / 100) * height };
}

export function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

export function hasOverlapPx(existing, candidatePct, width, height, minDistPx) {
  const candidate = pctToPx(candidatePct.xPct, candidatePct.yPct, width, height);
  for (const p of existing) {
    const pt = pctToPx(p.xPct, p.yPct, width, height);
    if (distance(candidate, pt) < minDistPx) return true;
  }
  return false;
}

export function validatePlacements(placed, rect, options = {}) {
  if (!rect) return { valid: false, message: 'Field not ready' };
  const reds = placed.filter(p=>p.role==='red');
  const blues = placed.filter(p=>p.role==='blue');
  const gks = placed.filter(p=>p.role==='gk');
  if (reds.length > MAX_RED) return { valid: false, message: 'Too many red players' };
  if (blues.length > MAX_BLUE) return { valid: false, message: 'Too many blue players' };
  if (gks.length !== 1) return { valid: false, message: 'Goalkeeper required (exactly 1)' };

  // GK validation - made more lenient
  const gk = gks[0];
  const gkPx = pctToPx(gk.xPct, gk.yPct, rect.width, rect.height);
  
  // For freekicks, be more lenient with goalkeeper position
  const isFreekick = options.setPiece === 'Free Kick';
  
  if (isFreekick) {
    // For freekicks, just check goalkeeper is on defending side (right half)
    const isInRightHalf = gkPx.x >= rect.width * 0.4; // More lenient for freekicks
    if (!isInRightHalf) {
      return { valid: false, message: 'Goalkeeper should be on the defending side' };
    }
  } else {
    // For corner kicks, check penalty box area (more lenient)
    const boxLeft = rect.width * 0.70; // More lenient - right penalty box start
    const boxRight = rect.width * 1.0;  // Allow up to edge
    const boxTop = rect.height * 0.20;  // More lenient top
    const boxBottom = rect.height * 0.80; // More lenient bottom
    
    // Check if goalkeeper is at least in the right half of the field (defending side)
    const isInRightHalf = gkPx.x >= rect.width * 0.5;
    const isInBox = (gkPx.x >= boxLeft && gkPx.x <= boxRight && gkPx.y >= boxTop && gkPx.y <= boxBottom);
    
    if (!isInRightHalf) {
      return { valid: false, message: 'Goalkeeper must be on the defending side (right half of field)' };
    }
    
    // Warn but don't block if slightly outside box
    if (!isInBox && gkPx.x < boxLeft) {
      return { valid: false, message: 'Goalkeeper should be closer to the goal area' };
    }
  }

  // overlap check among all
  for (let i=0;i<placed.length;i++) {
    for (let j=i+1;j<placed.length;j++) {
      const a = pctToPx(placed[i].xPct, placed[i].yPct, rect.width, rect.height);
      const b = pctToPx(placed[j].xPct, placed[j].yPct, rect.width, rect.height);
      if (distance(a,b) < OVERLAP_RADIUS_PX) {
        return { valid: false, message: 'Players overlap too closely' };
      }
    }
  }

  return { valid: true };
}

export function toSimulationInput(placed) {
  return placed.map(p => ({ id: p.id, role: p.role, label: p.label, xPct: p.xPct, yPct: p.yPct }));
}


