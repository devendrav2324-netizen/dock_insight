"""
Port Compatibility Engine

Evaluates physical constraints, handling suitability, turnaround time,
and congestion for vessels at Indian East Coast ports.

NOTE: All limits, handling rates, and congestion factors are marked as 
DEMO/SYNTHETIC and should be replaced with authoritative port data in production.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PortInfo:
    """
    Physical and operational specifications of a port.
    Values should be treated as SYNTHETIC/DEMO data.
    """
    port_id: str
    port_name: str
    max_draft_m: float
    max_loa_m: float
    max_beam_m: float
    cargo_handling_rate_tpd: float # Tonnes per day
    berthing_capacity: int
    current_congestion_factor: float = 1.0 # 1.0 = normal, >1.0 = congested

@dataclass
class VesselInfo:
    """
    Physical dimensions and cargo requirements of a vessel.
    """
    class_name: str
    draft_m: float
    loa_m: float
    beam_m: float
    cargo_to_handle_t: float


def check_vessel_port_compatibility(vessel: VesselInfo, port: PortInfo) -> Dict[str, Any]:
    """
    Evaluates if a vessel can operate at a specific port.
    
    Args:
        vessel: VesselInfo instance
        port: PortInfo instance
        
    Returns:
        dict containing compatibility status, score, failed constraints, warnings, and an explanation.
    """
    failed_constraints = []
    warnings = []
    
    # Base score
    score = 100.0
    
    # 1. Draft compatibility
    # No constraint is sometimes marked as negative or extreme
    if port.max_draft_m > 0 and vessel.draft_m > port.max_draft_m:
        failed_constraints.append(f"Draft ({vessel.draft_m}m) exceeds permitted limit ({port.max_draft_m}m)")
        score = 0.0
        
    # 2. LOA compatibility
    if port.max_loa_m > 0 and vessel.loa_m > port.max_loa_m:
        failed_constraints.append(f"LOA ({vessel.loa_m}m) exceeds permitted limit ({port.max_loa_m}m)")
        score = 0.0
        
    # 3. Beam compatibility
    if port.max_beam_m > 0 and vessel.beam_m > port.max_beam_m:
        failed_constraints.append(f"Beam ({vessel.beam_m}m) exceeds permitted limit ({port.max_beam_m}m)")
        score = 0.0
        
    # We do not strictly check cargo capacity against max DWT here because the prompt asks
    # to evaluate handling suitability and estimated turnaround. We'll use cargo_to_handle_t for that.
    
    if len(failed_constraints) > 0:
        return {
            "compatible": False,
            "score": 0,
            "failed_constraints": failed_constraints,
            "warnings": warnings,
            "explanation": "Vessel physically cannot operate at this port due to size constraints."
        }
        
    # Soft Constraints / Warnings
    
    # 4 & 5 & 6. Cargo capacity & Handling suitability & Estimated turnaround
    if port.cargo_handling_rate_tpd <= 0:
        warnings.append("Port has no documented cargo handling rate.")
        score -= 10
    else:
        estimated_turnaround_days = vessel.cargo_to_handle_t / port.cargo_handling_rate_tpd
        if estimated_turnaround_days > 7.0:
            warnings.append(f"Slow turnaround warning: estimated {estimated_turnaround_days:.1f} days due to port handling rate.")
            score -= (estimated_turnaround_days - 7.0) * 2 # penalize for slow turnaround
        elif estimated_turnaround_days < 1.0:
            score += 5 # Bonus for extremely fast turnaround
            
    # 7. Congestion
    if port.current_congestion_factor > 1.2:
        warnings.append("High port congestion expected.")
        score -= 15
    elif port.current_congestion_factor > 1.0:
        warnings.append("Moderate port congestion expected.")
        score -= 5
        
    # Ensure score is within 0-100 bounds
    score = max(0, min(100, int(score)))
    
    if len(warnings) > 0:
        explanation = f"Vessel is compatible, but has {len(warnings)} operational warning(s)."
    else:
        explanation = "Vessel is fully compatible with optimal operational conditions."
        
    return {
        "compatible": True,
        "score": int(score),
        "failed_constraints": failed_constraints,
        "warnings": warnings,
        "explanation": explanation
    }
