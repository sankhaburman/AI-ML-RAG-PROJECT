"""
Input validation schemas and validators for portfolio rebalancing service.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FundInput:
    """Represents a fund in the portfolio request."""
    name: str
    
    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Fund name must be a non-empty string")
        self.name = self.name.strip()


@dataclass
class PortfolioRequest:
    """Validates and represents a portfolio rebalancing request."""
    funds: List[FundInput]
    risk_profile: Optional[str] = None
    
    VALID_RISK_PROFILES = {"LOW", "MEDIUM", "HIGH"}
    
    def __post_init__(self):
        """Validate portfolio request structure and values."""
        if not isinstance(self.funds, list):
            raise ValueError("Funds must be a list")
        
        if len(self.funds) == 0:
            raise ValueError("At least one fund is required")
        
        # Convert dict items to FundInput objects
        validated_funds = []
        for fund in self.funds:
            if isinstance(fund, dict):
                validated_funds.append(FundInput(name=fund.get("name")))
            elif isinstance(fund, FundInput):
                validated_funds.append(fund)
            else:
                raise ValueError(f"Invalid fund format: {fund}")
        
        self.funds = validated_funds
        
        # Validate risk profile
        if self.risk_profile is None:
            self.risk_profile = "MEDIUM"  # default
        else:
            risk_profile_upper = str(self.risk_profile).upper()
            if risk_profile_upper not in self.VALID_RISK_PROFILES:
                raise ValueError(
                    f"Invalid risk profile. Must be one of: {self.VALID_RISK_PROFILES}"
                )
            self.risk_profile = risk_profile_upper
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PortfolioRequest":
        """
        Create a PortfolioRequest from a dictionary.
        
        Args:
            data: Dictionary containing 'funds' and optional 'risk_profile'
            
        Returns:
            Validated PortfolioRequest instance
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValueError("Portfolio request must be a dictionary")
        
        funds = data.get("funds", [])
        risk_profile = data.get("risk_profile", None)
        
        try:
            return cls(funds=funds, risk_profile=risk_profile)
        except ValueError as e:
            logger.error(f"Portfolio validation failed: {e}")
            raise


def validate_scheme_codes(scheme_codes: List[int]) -> List[int]:
    """
    Validate scheme codes.
    
    Args:
        scheme_codes: List of scheme codes
        
    Returns:
        Validated scheme codes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(scheme_codes, list):
        raise ValueError("Scheme codes must be a list")
    
    if len(scheme_codes) == 0:
        raise ValueError("At least one scheme code is required")
    
    validated = []
    for code in scheme_codes:
        try:
            validated.append(int(code))
        except (ValueError, TypeError):
            raise ValueError(f"Invalid scheme code: {code}")
    
    return validated
