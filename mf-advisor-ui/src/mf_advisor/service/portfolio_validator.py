"""
Portfolio-level validation and constraint checking.
Ensures recommendations don't create extreme or risky portfolio scenarios.
"""
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class PortfolioConstraintValidator:
    """Validates portfolio-level constraints and flags extreme scenarios."""
    
    def __init__(self, risk_config: Dict[str, Any]):
        """
        Initialize validator with risk profile constraints.
        
        Args:
            risk_config: Risk profile configuration dictionary
        """
        self.config = risk_config
        self.max_single_recommendation = risk_config.get("max_single_recommendation", 0.40)
        self.min_hold_percentage = risk_config.get("min_hold_percentage", 0.30)
    
    def validate(self, recommendations: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate portfolio recommendations against constraints.
        
        Args:
            recommendations: List of recommendation actions (INCREASE, HOLD, REDUCE)
            
        Returns:
            Tuple of (validation_result_dict, list_of_warnings)
        """
        warnings = []
        
        if not recommendations:
            return {
                "is_valid": False,
                "recommendation_count": 0,
                "distribution": {},
                "violations": ["No recommendations provided"]
            }, warnings
        
        total = len(recommendations)
        distribution = Counter(recommendations)
        
        result = {
            "is_valid": True,
            "recommendation_count": total,
            "distribution": dict(distribution),
            "violations": []
        }
        
        # Check: All funds recommended for REDUCE (liquidation risk)
        if distribution.get("REDUCE", 0) == total:
            warning = (
                f"CRITICAL: All {total} funds recommended for REDUCE. "
                f"This suggests liquidating entire portfolio into cash. "
                f"Consider if this is intentional."
            )
            result["violations"].append(warning)
            warnings.append(warning)
            result["is_valid"] = False
            logger.warning(warning)
        
        # Check: All funds recommended for INCREASE (over-concentration)
        if distribution.get("INCREASE", 0) == total:
            warning = (
                f"WARNING: All {total} funds recommended for INCREASE. "
                f"Portfolio may become over-concentrated. "
                f"Recommend manual review of top performers."
            )
            result["violations"].append(warning)
            warnings.append(warning)
            logger.warning(warning)
        
        # Check: No HOLD funds (no stability)
        if distribution.get("HOLD", 0) == 0 and total > 0:
            warning = (
                f"WARNING: No funds in HOLD status. "
                f"Portfolio lacks stability positions. "
                f"Consider reallocating some INCREASE funds to HOLD."
            )
            warnings.append(warning)
            logger.warning(warning)
        
        # Check: INCREASE concentration exceeds limit
        increase_pct = distribution.get("INCREASE", 0) / total if total > 0 else 0
        if increase_pct > self.max_single_recommendation:
            warning = (
                f"WARNING: {increase_pct:.1%} of funds in INCREASE status. "
                f"Exceeds risk profile limit of {self.max_single_recommendation:.1%}. "
                f"Consider downgrading some INCREASE to HOLD."
            )
            warnings.append(warning)
            logger.warning(warning)
        
        # Check: HOLD coverage below minimum
        hold_or_better = (distribution.get("INCREASE", 0) + distribution.get("HOLD", 0)) / total if total > 0 else 0
        if hold_or_better < self.min_hold_percentage:
            warning = (
                f"WARNING: Only {hold_or_better:.1%} funds in HOLD or better status. "
                f"Below risk profile minimum of {self.min_hold_percentage:.1%}. "
                f"Consider more conservative positioning."
            )
            warnings.append(warning)
            logger.warning(warning)
        
        # Check: Highly imbalanced distribution
        max_recommendation = max(distribution.values()) if distribution else 0
        if max_recommendation / total > 0.7:
            recommendation_type = [k for k, v in distribution.items() if v == max_recommendation][0]
            warning = (
                f"WARNING: Portfolio heavily skewed toward {recommendation_type} ({max_recommendation/total:.1%}). "
                f"Consider more balanced approach for diversification."
            )
            warnings.append(warning)
            logger.warning(warning)
        
        return result, warnings
    
    def adjust_recommendations_if_needed(
        self,
        recommendations: List[str],
        strict_mode: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Adjust recommendations to comply with constraints (if strict_mode=True).
        
        Args:
            recommendations: Original recommendations
            strict_mode: If True, enforce constraints; if False, just warn
            
        Returns:
            Tuple of (adjusted_recommendations, adjustment_notes)
        """
        if not strict_mode:
            return recommendations, []
        
        notes = []
        adjusted = recommendations.copy()
        total = len(adjusted)
        
        if total == 0:
            return adjusted, notes
        
        # If all REDUCE, mark some as HOLD
        if all(r == "REDUCE" for r in adjusted):
            adjust_count = max(1, int(total * 0.2))  # Keep at least 20% as HOLD
            for i in range(adjust_count):
                adjusted[i] = "HOLD"
            notes.append(f"Converted {adjust_count} REDUCE to HOLD (liquidation risk mitigation)")
            logger.info(notes[-1])
        
        # If all INCREASE, mark some as HOLD
        elif all(r == "INCREASE" for r in adjusted):
            adjust_count = max(1, int(total * 0.3))  # Mark 30% as HOLD
            for i in range(adjust_count):
                adjusted[i] = "HOLD"
            notes.append(f"Converted {adjust_count} INCREASE to HOLD (over-concentration mitigation)")
            logger.info(notes[-1])
        
        return adjusted, notes


def calculate_recommendation_distribution_stats(recommendations: List[str]) -> Dict[str, Any]:
    """
    Calculate statistics about recommendation distribution.
    
    Args:
        recommendations: List of recommendations
        
    Returns:
        Dictionary with distribution statistics
    """
    if not recommendations:
        return {}
    
    distribution = Counter(recommendations)
    total = len(recommendations)
    
    stats = {
        "total_funds": total,
        "increase_count": distribution.get("INCREASE", 0),
        "hold_count": distribution.get("HOLD", 0),
        "reduce_count": distribution.get("REDUCE", 0),
        "increase_pct": distribution.get("INCREASE", 0) / total,
        "hold_pct": distribution.get("HOLD", 0) / total,
        "reduce_pct": distribution.get("REDUCE", 0) / total,
        "diversification_ratio": len(distribution) / 3.0,  # How many recommendation types used
    }
    
    return stats
