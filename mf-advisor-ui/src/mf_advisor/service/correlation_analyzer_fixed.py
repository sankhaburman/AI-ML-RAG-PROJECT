"""
Portfolio correlation and diversification analysis.
Detects over-concentrated or highly correlated fund positions.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyzes correlation between funds and diversification risk."""
    
    def __init__(self, correlation_threshold: float = 0.75):
        """
        Initialize analyzer.
        
        Args:
            correlation_threshold: Flag correlations above this threshold (default 0.75)
        """
        self.threshold = correlation_threshold
    
    def calculate_pairwise_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pairwise correlations between funds.
        
        Args:
            df: DataFrame with fund metrics (rows=funds, columns=metrics)
            
        Returns:
            Correlation matrix (funds x funds)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if df.empty or len(numeric_cols) == 0:
            logger.warning("No numeric columns for correlation calculation")
            return pd.DataFrame()
        
        profile_cols = [
            col for col in numeric_cols
            if any(metric in col.lower() for metric in
                   ["return", "sharpe", "volatility", "cagr", "moving"])
        ]
        
        if not profile_cols:
            profile_cols = numeric_cols.tolist()[:6]
        
        correlation_matrix = df[profile_cols].corr()
        logger.debug(f"Calculated correlations using {len(profile_cols)} metrics")
        return correlation_matrix
    
    def identify_highly_correlated_pairs(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[Tuple[Any, Any, float]]:
        """
        Identify fund pairs with high correlation.
        
        Args:
            correlation_matrix: Correlation matrix from calculate_pairwise_correlations
            
        Returns:
            List of (fund1, fund2, correlation_coefficient) tuples
        """
        pairs = []
        
        if correlation_matrix.empty:
            return pairs
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > self.threshold:
                    fund1 = correlation_matrix.index[i]
                    fund2 = correlation_matrix.columns[j]
                    pairs.append((fund1, fund2, float(corr_value)))
        
        if pairs:
            logger.warning(
                f"Found {len(pairs)} fund pairs with correlation > {self.threshold:.2f}"
            )
        
        return pairs
    
    def analyze_portfolio_concentration(
        self,
        recommendations: List[str],
        scheme_codes: List[int],
        correlation_pairs: List[Tuple[Any, Any, float]]
    ) -> Dict[str, Any]:
        """
        Analyze portfolio concentration risk based on correlations.
        
        Args:
            recommendations: List of INCREASE/HOLD/REDUCE actions
            scheme_codes: List of scheme codes (order matches recommendations)
            correlation_pairs: List of highly correlated pairs
            
        Returns:
            Concentration analysis dictionary
        """
        analysis = {
            "highly_correlated_increases": [],
            "concentration_risk": "LOW",
            "diversification_score": 1.0,
            "recommendations": []
        }
        
        if not recommendations or len(recommendations) != len(scheme_codes):
            return analysis
        
        increases = {
            scheme_codes[i]: recommendations[i]
            for i in range(len(recommendations))
            if recommendations[i] == "INCREASE"
        }
        
        if not increases:
            return analysis
        
        for fund1, fund2, corr in correlation_pairs:
            if fund1 in increases and fund2 in increases:
                analysis["highly_correlated_increases"].append({
                    "fund1": fund1,
                    "fund2": fund2,
                    "correlation": corr,
                    "risk": "HIGH"
                })
        
        num_increases = len(increases)
        total_funds = len(recommendations)
        increase_ratio = num_increases / total_funds
        correlated_pairs = len(analysis["highly_correlated_increases"])
        
        if correlated_pairs > 2 or increase_ratio > 0.6:
            analysis["concentration_risk"] = "HIGH"
            analysis["diversification_score"] = 0.4
        elif correlated_pairs > 0 or increase_ratio > 0.4:
            analysis["concentration_risk"] = "MEDIUM"
            analysis["diversification_score"] = 0.6
        else:
            analysis["concentration_risk"] = "LOW"
            analysis["diversification_score"] = 0.9
        
        if analysis["concentration_risk"] == "HIGH":
            analysis["recommendations"].append(
                f"High concentration risk detected ({num_increases}/{total_funds} in INCREASE). "
                f"Consider downgrading some INCREASE positions to HOLD."
            )
            
            if correlated_pairs > 0:
                analysis["recommendations"].append(
                    f"{correlated_pairs} pairs of INCREASE funds are highly correlated. "
                    f"Reduce allocation to one fund in each pair."
                )
        
        elif analysis["concentration_risk"] == "MEDIUM":
            analysis["recommendations"].append(
                f"Moderate concentration ({num_increases}/{total_funds} in INCREASE). "
                f"Monitor for diversification."
            )
        
        return analysis
    
    def suggest_diversifiers(
        self,
        recommendations: List[str],
        scheme_codes: List[int],
        correlation_matrix: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Suggest which HOLD/REDUCE funds could improve diversification.
        
        Args:
            recommendations: List of INCREASE/HOLD/REDUCE actions
            scheme_codes: List of scheme codes
            correlation_matrix: Correlation matrix
            
        Returns:
            List of diversification suggestions
        """
        suggestions = []
        
        if not recommendations or correlation_matrix.empty:
            return suggestions
        
        increase_indices = [
            i for i in range(len(recommendations))
            if recommendations[i] == "INCREASE"
        ]
        
        if not increase_indices:
            return suggestions
        
        other_indices = [
            i for i in range(len(recommendations))
            if recommendations[i] in ["HOLD", "REDUCE"]
        ]
        
        for other_idx in other_indices:
            avg_correlation = 0
            count = 0
            
            for inc_idx in increase_indices:
                if inc_idx < len(correlation_matrix) and other_idx < len(correlation_matrix.columns):
                    try:
                        corr = correlation_matrix.iloc[inc_idx, other_idx]
                        if pd.notna(corr):
                            avg_correlation += abs(corr)
                            count += 1
                    except (IndexError, KeyError):
                        continue
            
            if count > 0:
                avg_correlation /= count
                
                if avg_correlation < 0.4:
                    suggestion = {
                        "scheme_code": scheme_codes[other_idx],
                        "current_action": recommendations[other_idx],
                        "avg_correlation_with_increases": float(avg_correlation),
                        "diversification_benefit": "HIGH",
                        "reason": (
                            f"Low correlation ({avg_correlation:.2f}) with INCREASE funds. "
                            f"Could improve portfolio diversification if promoted to HOLD/INCREASE."
                        )
                    }
                    suggestions.append(suggestion)
        
        return suggestions[:3]


def create_correlation_summary(analysis: Dict[str, Any]) -> str:
    """
    Create human-readable summary of correlation analysis.
    
    Args:
        analysis: Analysis dictionary from analyze_portfolio_concentration
        
    Returns:
        Formatted summary string
    """
    summary_lines = []
    
    summary_lines.append(f"Concentration Risk: {analysis.get('concentration_risk', 'UNKNOWN')}")
    summary_lines.append(f"Diversification Score: {analysis.get('diversification_score', 0):.1%}")
    
    corr_pairs = analysis.get("highly_correlated_increases", [])
    if corr_pairs:
        summary_lines.append(f"Highly Correlated INCREASE Pairs: {len(corr_pairs)}")
        for pair in corr_pairs[:2]:
            summary_lines.append(
                f"  - {pair['fund1']} ↔ {pair['fund2']}: {pair['correlation']:.2f}"
            )
    
    for rec in analysis.get("recommendations", []):
        summary_lines.append(f"✓ {rec}")
    
    return "\n".join(summary_lines)
