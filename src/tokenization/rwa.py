"""
ERC-3643 Compliant RWA Tokenization Framework.

This module implements a security token for real-world assets (RWA)
following the ERC-3643 (T-REX) standard with full compliance checks,
investor verification, transfer restrictions, and dividend distribution.
"""

from enum import Enum
from typing import Any
from datetime import datetime, timedelta


class ComplianceStatus(Enum):
    """Compliance status for investors."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    REJECTED = "rejected"


class InvestorStatus(Enum):
    """Investor verification status."""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    FROZEN = "frozen"


# Sanctioned jurisdictions (ISO country codes)
SANCTIONED_JURISDICTIONS = {"IR", "KP", "SY", "CU"}

# Maximum holding as percentage of total supply (10%)
MAX_HOLDING_PERCENT = 0.10

# KYC validity period in days
KYC_VALIDITY_DAYS = 365


class RWATokenizer:
    """
    ERC-3643 compliant RWA Tokenizer.

    Tokenizes real-world assets with full compliance checks,
    investor verification, transfer restrictions, and dividend distribution.
    """

    def __init__(
        self,
        w3: Any,
        owner_address: str,
        token_name: str,
        token_symbol: str,
        initial_supply: int,
        asset_type: str,
        asset_value: int,
        jurisdiction: str,
    ):
        """
        Initialize the RWA Tokenizer.

        Args:
            w3: Web3 instance
            owner_address: Token owner address
            token_name: Name of the token
            token_symbol: Symbol of the token
            initial_supply: Initial token supply
            asset_type: Type of real-world asset (e.g., "real_estate")
            asset_value: Value of the underlying asset in cents
            jurisdiction: Jurisdiction code (ISO country code)
        """
        self.w3 = w3
        self.owner = owner_address
        self.token_name = token_name
        self.token_symbol = token_symbol
        self.initial_supply = initial_supply
        self.asset_type = asset_type
        self.asset_value = asset_value
        self.jurisdiction = jurisdiction

        # Investor registry: address -> investor info
        self.investors: dict[str, dict[str, Any]] = {}

        # Token balances: address -> balance
        self.balances: dict[str, int] = {}

        # Dividend distribution history
        self.distributions: list[dict[str, Any]] = []

        # ERC-3643 compliance module
        self.compliance_module = {
            "name": "ERC3643Compliance",
            "version": "1.0.0",
            "rules": [
                "investor_verification",
                "jurisdiction_check",
                "max_holding_limit",
                "freeze_support",
            ],
        }

        # ERC-3643 identity registry
        self.identity_registry = {
            "name": "ERC3643IdentityRegistry",
            "version": "1.0.0",
            "identities": {},
        }

        # Deploy on-chain
        self._deploy()

    def _deploy(self) -> None:
        """Deploy the token contract on-chain."""
        # In production, this would deploy the actual ERC-3643 contract
        # For now, we simulate the deployment
        if self.w3 and hasattr(self.w3.eth, 'send_raw_transaction'):
            self.w3.eth.send_raw_transaction(b"deploy_tx")

    # ─── Compliance Checks ────────────────────────────────────────────────

    def check_compliance(self, investor_address: str) -> ComplianceStatus:
        """
        Check if an investor is compliant with regulations.

        Args:
            investor_address: Address of the investor

        Returns:
            ComplianceStatus.COMPLIANT if compliant, NON_COMPLIANT otherwise
        """
        if investor_address not in self.investors:
            return ComplianceStatus.NON_COMPLIANT

        investor = self.investors[investor_address]

        # Check verification status
        if investor["status"] != InvestorStatus.VERIFIED:
            return ComplianceStatus.NON_COMPLIANT

        # Check KYC expiry
        if investor["kyc_expiry"] < datetime.now().timestamp():
            return ComplianceStatus.NON_COMPLIANT

        # Check jurisdiction sanctions
        if investor["jurisdiction"] in SANCTIONED_JURISDICTIONS:
            return ComplianceStatus.NON_COMPLIANT

        # Check if frozen
        if investor.get("frozen", False):
            return ComplianceStatus.NON_COMPLIANT

        return ComplianceStatus.COMPLIANT

    # ─── Investor Verification ─────────────────────────────────────────────

    def register_investor(
        self,
        investor_address: str,
        jurisdiction: str,
        accredited: bool,
    ) -> None:
        """
        Register a new investor with pending status.

        Args:
            investor_address: Address of the investor
            jurisdiction: Investor's jurisdiction (ISO country code)
            accredited: Whether the investor is accredited
        """
        self.investors[investor_address] = {
            "status": InvestorStatus.PENDING,
            "kyc_expiry": 0,
            "jurisdiction": jurisdiction,
            "accredited": accredited,
            "frozen": False,
            "registered_at": datetime.now().timestamp(),
        }

        # Update identity registry
        self.identity_registry["identities"][investor_address] = {
            "status": InvestorStatus.PENDING,
            "jurisdiction": jurisdiction,
        }

    def verify_investor(self, investor_address: str) -> None:
        """
        Verify a registered investor (KYC/AML passed).

        Args:
            investor_address: Address of the investor

        Raises:
            ValueError: If investor is not registered
        """
        if investor_address not in self.investors:
            raise ValueError("Investor not registered")

        self.investors[investor_address]["status"] = InvestorStatus.VERIFIED
        self.investors[investor_address]["kyc_expiry"] = (
            datetime.now() + timedelta(days=KYC_VALIDITY_DAYS)
        ).timestamp()

        # Update identity registry
        self.identity_registry["identities"][investor_address]["status"] = (
            InvestorStatus.VERIFIED
        )

    def reject_investor(self, investor_address: str) -> None:
        """
        Reject an investor's registration.

        Args:
            investor_address: Address of the investor

        Raises:
            ValueError: If investor is not registered
        """
        if investor_address not in self.investors:
            raise ValueError("Investor not registered")

        self.investors[investor_address]["status"] = InvestorStatus.REJECTED

        # Update identity registry
        self.identity_registry["identities"][investor_address]["status"] = (
            InvestorStatus.REJECTED
        )

    def freeze_investor(self, investor_address: str) -> None:
        """
        Freeze an investor's account (ERC-3643 compliance feature).

        Args:
            investor_address: Address of the investor

        Raises:
            ValueError: If investor is not registered
        """
        if investor_address not in self.investors:
            raise ValueError("Investor not registered")

        self.investors[investor_address]["frozen"] = True
        self.investors[investor_address]["status"] = InvestorStatus.FROZEN

    def unfreeze_investor(self, investor_address: str) -> None:
        """
        Unfreeze an investor's account.

        Args:
            investor_address: Address of the investor

        Raises:
            ValueError: If investor is not registered
        """
        if investor_address not in self.investors:
            raise ValueError("Investor not registered")

        self.investors[investor_address]["frozen"] = False
        self.investors[investor_address]["status"] = InvestorStatus.VERIFIED

    # ─── Transfer Restrictions ─────────────────────────────────────────────

    def can_transfer(
        self,
        from_address: str,
        to_address: str,
        amount: int,
    ) -> bool:
        """
        Check if a transfer is allowed under compliance rules.

        Args:
            from_address: Sender address
            to_address: Recipient address
            amount: Amount to transfer

        Returns:
            True if transfer is allowed, False otherwise
        """
        # Both parties must be compliant
        if self.check_compliance(from_address) != ComplianceStatus.COMPLIANT:
            return False

        if self.check_compliance(to_address) != ComplianceStatus.COMPLIANT:
            return False

        # Check max holding limit for recipient
        recipient_balance = self.balances.get(to_address, 0)
        max_holding = int(self.initial_supply * MAX_HOLDING_PERCENT)
        if recipient_balance + amount > max_holding:
            return False

        # Check sender has sufficient balance
        sender_balance = self.balances.get(from_address, 0)
        if sender_balance < amount:
            return False

        return True

    def transfer(
        self,
        from_address: str,
        to_address: str,
        amount: int,
    ) -> bool:
        """
        Execute a transfer if compliant.

        Args:
            from_address: Sender address
            to_address: Recipient address
            amount: Amount to transfer

        Returns:
            True if transfer succeeded, False otherwise
        """
        if not self.can_transfer(from_address, to_address, amount):
            return False

        self.balances[from_address] -= amount
        self.balances[to_address] = self.balances.get(to_address, 0) + amount
        return True

    def force_transfer(
        self,
        from_address: str,
        to_address: str,
        amount: int,
    ) -> bool:
        """
        Force a transfer (owner only, for compliance purposes).

        Args:
            from_address: Sender address
            to_address: Recipient address
            amount: Amount to transfer

        Returns:
            True if transfer succeeded, False otherwise
        """
        # Only owner can force transfer
        # In production, this would check msg.sender == owner
        sender_balance = self.balances.get(from_address, 0)
        if sender_balance < amount:
            return False

        self.balances[from_address] -= amount
        self.balances[to_address] = self.balances.get(to_address, 0) + amount
        return True

    # ─── Dividend Distribution ─────────────────────────────────────────────

    def distribute_dividends(self, total_amount: int) -> dict[str, Any]:
        """
        Distribute dividends to all verified investors proportionally.

        Args:
            total_amount: Total dividend amount in cents

        Returns:
            Distribution record with details
        """
        # Get all verified investors with balances
        verified_investors = [
            addr for addr, info in self.investors.items()
            if info["status"] == InvestorStatus.VERIFIED and not info.get("frozen", False)
        ]

        # Calculate total balance of verified investors
        total_verified_balance = sum(
            self.balances.get(addr, 0) for addr in verified_investors
        )

        if total_verified_balance == 0:
            return {
                "total_amount": total_amount,
                "recipients": 0,
                "distribution_time": datetime.now().timestamp(),
            }

        # Distribute proportionally
        distribution_record = {
            "total_amount": total_amount,
            "recipients": len(verified_investors),
            "distribution_time": datetime.now().timestamp(),
            "details": [],
        }

        for addr in verified_investors:
            balance = self.balances.get(addr, 0)
            share = int(total_amount * (balance / total_verified_balance))
            self.balances[addr] = balance + share
            distribution_record["details"].append({
                "address": addr,
                "amount": share,
            })

        self.distributions.append(distribution_record)
        return distribution_record

    # ─── Asset Valuation ───────────────────────────────────────────────────

    def get_asset_value(self) -> int:
        """Get the current asset value in cents."""
        return self.asset_value

    def get_token_price(self) -> float:
        """
        Get the current token price in cents.

        Returns:
            Token price (asset_value / initial_supply)
        """
        if self.initial_supply == 0:
            return 0.0
        return self.asset_value / self.initial_supply

    def update_asset_value(self, new_value: int) -> None:
        """
        Update the asset value.

        Args:
            new_value: New asset value in cents
        """
        self.asset_value = new_value

    # ─── Batch Operations ──────────────────────────────────────────────────

    def batch_register_investors(
        self,
        addresses: list[str],
        jurisdiction: str,
        accredited: bool,
    ) -> None:
        """
        Register multiple investors at once.

        Args:
            addresses: List of investor addresses
            jurisdiction: Jurisdiction for all investors
            accredited: Accredited status for all investors
        """
        for addr in addresses:
            self.register_investor(addr, jurisdiction, accredited)

    def batch_verify_investors(self, addresses: list[str]) -> None:
        """
        Verify multiple investors at once.

        Args:
            addresses: List of investor addresses to verify
        """
        for addr in addresses:
            if addr in self.investors:
                self.verify_investor(addr)

    # ─── ERC-3643 Interface ────────────────────────────────────────────────

    def get_compliance_module(self) -> dict[str, Any]:
        """Get the compliance module info."""
        return self.compliance_module

    def get_identity_registry(self) -> dict[str, Any]:
        """Get the identity registry."""
        return self.identity_registry

    def get_investor_info(self, investor_address: str) -> dict[str, Any] | None:
        """
        Get investor information.

        Args:
            investor_address: Address of the investor

        Returns:
            Investor info dict or None if not registered
        """
        return self.investors.get(investor_address)

    def get_balance(self, address: str) -> int:
        """
        Get token balance for an address.

        Args:
            address: Address to check

        Returns:
            Token balance
        """
        return self.balances.get(address, 0)

    def get_total_supply(self) -> int:
        """Get the total token supply."""
        return self.initial_supply

    def get_distributions(self) -> list[dict[str, Any]]:
        """Get all dividend distribution records."""
        return self.distributions
