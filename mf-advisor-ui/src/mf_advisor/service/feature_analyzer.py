"""
Feature analysis, data quality checks, and temporal weighting utilities.
Provides insights into which features drive recommendations and data quality.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class FeatureQualityAnalyzer:
    """Analyzes feature data quality and flags issues."""
    
    FEATURE_CATEGORIES = {
        "returns": ["daily_return_pct", "weekly_return_pct", "monthly_return_pct"],
        "rolling_returns": ["rolling_return_30d_pct", "rolling_return_90d_pct"],
        "moving_averages": ["moving_avg_7d", "moving_avg_30d", "moving_avg_90d", "moving_avg_200d"],
        "risk_metrics": ["cagr_percent", "sharpe_ratio", "annualized_volatility"]
    }
    
    def __init__(self, missing_threshold: float = 0.3):
        """
        Initialize analyzer.
        
        Args:
            missing_threshold: Warn if >X% of a feature is missing (default 30%)
        """
        self.missing_threshold = missing_threshold
    
    def analyze_data_quality(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Analyze data quality of features in DataFrame.
        
        Args:
            df: Pandas DataFrame with feature data
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with quality metrics and warnings
        """
        quality_report = {
            "total_records": len(df),
            "missing_values": {},
            "zero_values": {},
            "warnings": [],
            "data_quality_score": 1.0  # 0-1 scale
        }
        
        if df.empty:
            quality_report["warnings"].append("DataFrame is empty")
            quality_report["data_quality_score"] = 0.0
            return quality_report
        
        total_issues = 0
        
        for col in feature_columns:
            if col not in df.columns:
                quality_report["warnings"].append(f"Missing column: {col}")
                total_issues += 1
                continue
            
            # Check for missing values
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)
            
            if missing_count > 0:
                quality_report["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_pct)
                }
                
                if missing_pct > self.missing_threshold:
                    warning = (
                        f"Column '{col}' has {missing_pct:.1%} missing values. "
                        f"Exceeds threshold of {self.missing_threshold:.1%}"
                    )
                    quality_report["warnings"].append(warning)
                    total_issues += 1
                    logger.warning(warning)
            
            # Check for zero values (might indicate missing data coded as 0)
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_pct = zero_count / len(df)
                quality_report["zero_values"][col] = {
                    "count": int(zero_count),
                    "percentage": float(zero_pct)
                }
                
                # Warn if >50% of a feature is zero (suspicious)
                if zero_pct > 0.5 and col not in ["daily_return_pct"]:
                    warning = (
                        f"Column '{col}' has {zero_pct:.1%} zero values. "
                        f"May indicate missing data coding."
                    )
                    quality_report["warnings"].append(warning)
                    total_issues += 1
                    logger.warning(warning)
        
        # Calculate composite quality score (reduce by 0.1 for each issue, floor at 0)
        quality_report["data_quality_score"] = max(0.0, 1.0 - (total_issues * 0.1))
        
        return quality_report
    
    def add_confidence_score(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        quality_report: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Add confidence score to each record based on data completeness.
        
        Args:
            df: DataFrame to enhance
            feature_columns: Feature columns
            quality_report: Quality report from analyze_data_quality
            
        Returns:
            DataFrame with added 'data_confidence_score' column
        """
        df = df.copy()
        
        # Calculate confidence per row: % of non-missing features
        def calculate_confidence(row):
            valid_features = sum(1 for col in feature_columns if pd.notna(row.get(col)))
            return valid_features / len(feature_columns) if feature_columns else 0.0
        
        df["data_confidence_score"] = df.apply(calculate_confidence, axis=1)
        
        return df


class TemporalWeightingAnalyzer:
    """Analyzes recency of data and applies temporal weighting."""
    
    # Temporal feature weights (favor recent data)
    TEMPORAL_WEIGHTS = {
        "daily_return_pct": 0.40,        # Most recent - highest weight
        "weekly_return_pct": 0.25,
        "monthly_return_pct": 0.15,
        "rolling_return_30d_pct": 0.10,
        "rolling_return_90d_pct": 0.05,  # Historical - lower weight
        "cagr_percent": 0.03,
        "moving_avg_7d": 0.35,
        "moving_avg_30d": 0.25,
        "moving_avg_90d": 0.20,
        "moving_avg_200d": 0.10,          # Very historical
        "sharpe_ratio": 0.15,             # Risk metric - moderate recency weight
        "annualized_volatility": 0.10     # Risk metric - less time-sensitive
    }
    
    @staticmethod
    def get_temporal_weight(feature_name: str) -> float:
        """
        Get temporal weight for a feature (0-1 scale).
        Higher weight = more recent/relevant.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Temporal weight
        """
        return TemporalWeightingAnalyzer.TEMPORAL_WEIGHTS.get(feature_name, 0.1)
    
    @staticmethod
    def create_weighted_score(
        row: pd.Series,
        feature_columns: List[str],
        base_weights: Dict[str, float]
    ) -> float:
        """
        Create a weighted score considering feature recency.
        
        Args:
            row: DataFrame row
            feature_columns: Features to consider
            base_weights: Original scoring weights (e.g., from risk profile)
            
        Returns:
            Recency-adjusted score
        """
        weighted_score = 0.0
        
        for col in feature_columns:
            if col not in row or pd.isna(row[col]):
                continue
            
            feature_value = row[col]
            temporal_weight = TemporalWeightingAnalyzer.get_temporal_weight(col)
            
            # Combine base weight (from risk profile) with temporal weight
            combined_weight = base_weights.get(col, 1.0) * temporal_weight
            weighted_score += feature_value * combined_weight
        
        return weighted_score


def detect_temporal_anomalies(
    df: pd.DataFrame,
    z_score_threshold: float = 2.0
) -> Dict[str, List[int]]:
    """
    Detect anomalies in recent vs historical returns.
    
    Args:
        df: DataFrame with return columns
        z_score_threshold: Z-score threshold for anomaly detection
        
    Returns:
        Dictionary mapping fund indices to detected anomalies
    """
    anomalies = {}
    
    # Check if recent returns deviate significantly from historical
    if "daily_return_pct" not in df.columns or "moving_avg_7d" not in df.columns:
        return anomalies
    
    for idx, row in df.iterrows():
        daily = row.get("daily_return_pct", 0)
        ma7 = row.get("moving_avg_7d", 0)
        
        if pd.notna(daily) and pd.notna(ma7) and ma7 != 0:
            deviation = abs(daily - ma7) / abs(ma7)
            
            # Flag if recent return deviates >2x from moving average
            if deviation > 2.0:
                if idx not in anomalies:
                    anomalies[idx] = []
                anomalies[idx].append(
                    f"Daily return ({daily:.2f}%) deviates significantly "
                    f"from 7-day MA ({ma7:.2f}%)"
                )
    
    if anomalies:
        logger.warning(f"Detected {len(anomalies)} funds with temporal anomalies")
    
    return anomalies


def flag_data_quality_concerns(
    df: pd.DataFrame,
    feature_columns: List[str],
    quality_report: Dict[str, Any]
) -> List[Tuple[int, str]]:
    """
    Flag records with data quality concerns.
    
    Args:
        df: DataFrame
        feature_columns: Feature columns
        quality_report: Quality analysis report
        
    Returns:
        List of (record_index, concern_description) tuples
    """
    concerns = []
    
    for idx, row in df.iterrows():
        record_concerns = []
        
        # Check for too many missing features
        missing_count = sum(1 for col in feature_columns if pd.isna(row.get(col)))
        if missing_count > len(feature_columns) * 0.3:
            record_concerns.append(
                f"{missing_count}/{len(feature_columns)} features missing"
            )
        
        # Check for suspicious patterns (all zeros, all same values)
        values = [row.get(col) for col in feature_columns if pd.notna(row.get(col))]
        if values:
            if all(v == 0 for v in values):
                record_concerns.append("All feature values are zero (data quality issue)")
            elif len(set(values)) == 1:
                record_concerns.append("All feature values are identical (data quality issue)")
        
        if record_concerns:
            for concern in record_concerns:
                concerns.append((idx, concern))
    
    return concerns
