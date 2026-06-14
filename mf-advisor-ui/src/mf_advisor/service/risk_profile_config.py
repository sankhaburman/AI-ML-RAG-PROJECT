"""
Risk profile-aware scoring configuration and utilities.
Modulates scoring weights and thresholds based on investor risk profile.
"""
from typing import Dict, Tuple
from enum import Enum

class RiskProfile(Enum):
    """Risk profile categories."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# Risk Profile Specific Scoring Configurations
RISK_PROFILE_CONFIGS: Dict[str, Dict] = {
    "CONSERVATIVE": {
        "name": "Conservative",
        "description": "Capital preservation focus, lower volatility tolerance",
        "scoring_weights": {
            "predicted_return": 0.30,      # Lower weight - focus on stability
            "sharpe_ratio": 0.50,          # Highest weight - risk-adjusted returns matter most
            "volatility": -0.20            # Standard penalty
        },
        "thresholds": {
            "increase": 6.0,               # Stricter threshold
            "hold": 3.0,
            "reduce": float('-inf')
        },
        "max_single_recommendation": 0.30, # Max 30% in INCREASE
        "min_hold_percentage": 0.40,       # At least 40% in HOLD or better
        "volatility_tolerance": "LOW"
    },
    
    "MODERATE": {
        "name": "Moderate/Balanced",
        "description": "Balanced growth and preservation",
        "scoring_weights": {
            "predicted_return": 0.50,      # Balanced approach
            "sharpe_ratio": 0.30,
            "volatility": -0.20
        },
        "thresholds": {
            "increase": 5.0,
            "hold": 2.0,
            "reduce": float('-inf')
        },
        "max_single_recommendation": 0.40, # Max 40% in INCREASE
        "min_hold_percentage": 0.30,       # At least 30% in HOLD or better
        "volatility_tolerance": "MEDIUM"
    },
    
    "AGGRESSIVE": {
        "name": "Aggressive",
        "description": "Growth focus, higher volatility tolerance",
        "scoring_weights": {
            "predicted_return": 0.60,      # Highest weight - growth priority
            "sharpe_ratio": 0.25,
            "volatility": -0.10            # Minimal penalty - accept more risk
        },
        "thresholds": {
            "increase": 4.0,               # Looser threshold
            "hold": 1.0,
            "reduce": float('-inf')
        },
        "max_single_recommendation": 0.50, # Max 50% in INCREASE (more concentrated)
        "min_hold_percentage": 0.10,       # At least 10% in HOLD or better
        "volatility_tolerance": "HIGH"
    }
}


def get_risk_config(risk_profile: str) -> Dict:
    """
    Get configuration for a given risk profile.
    
    Args:
        risk_profile: Risk profile (CONSERVATIVE, MODERATE, AGGRESSIVE)
        
    Returns:
        Configuration dictionary for the risk profile
        
    Raises:
        ValueError: If risk profile not recognized
    """
    profile_upper = risk_profile.upper()
    
    if profile_upper not in RISK_PROFILE_CONFIGS:
        raise ValueError(
            f"Unknown risk profile: {risk_profile}. "
            f"Must be one of: {list(RISK_PROFILE_CONFIGS.keys())}"
        )
    
    return RISK_PROFILE_CONFIGS[profile_upper]


def get_scoring_weights(risk_profile: str) -> Tuple[float, float, float]:
    """
    Get scoring weights for the risk profile.
    
    Args:
        risk_profile: Risk profile
        
    Returns:
        Tuple of (predicted_return_weight, sharpe_ratio_weight, volatility_weight)
    """
    config = get_risk_config(risk_profile)
    weights = config["scoring_weights"]
    return (
        weights["predicted_return"],
        weights["sharpe_ratio"],
        weights["volatility"]
    )


def get_thresholds(risk_profile: str) -> Dict[str, float]:
    """
    Get recommendation thresholds for the risk profile.
    
    Args:
        risk_profile: Risk profile
        
    Returns:
        Dictionary with 'increase' and 'hold' thresholds
    """
    config = get_risk_config(risk_profile)
    return config["thresholds"]


def normalize_risk_profile(risk_profile: str) -> str:
    """
    Normalize risk profile to standard format.
    
    Args:
        risk_profile: Risk profile (any case)
        
    Returns:
        Normalized risk profile (uppercase)
    """
    if not risk_profile:
        return "MODERATE"
    
    normalized = risk_profile.upper()
    
    # Handle common aliases
    if normalized in ["LOW", "CONSERVATIVE", "CONSERVATIVE_LOW"]:
        return "CONSERVATIVE"
    elif normalized in ["MEDIUM", "MODERATE", "BALANCED", "MODERATE_BALANCED"]:
        return "MODERATE"
    elif normalized in ["HIGH", "AGGRESSIVE", "AGGRESSIVE_HIGH"]:
        return "AGGRESSIVE"
    else:
        return "MODERATE"  # Default to moderate
