"""
TDD tests for RWA Tokenization Framework.
ERC-3643 compliant security token for real-world assets.
"""
import pytest
from unittest.mock import MagicMock
from eth_account import Account

from src.tokenization.rwa import RWATokenizer, ComplianceStatus, InvestorStatus


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def w3():
    """Web3 instance with mocked provider."""
    mock_w3 = MagicMock()
    mock_w3.eth.chain_id = 1
    mock_w3.eth.get_transaction_count.return_value = 0
    mock_w3.eth.gas_price = 20_000_000_000
    mock_w3.eth.estimate_gas.return_value = 21000
    mock_w3.eth.wait_for_transaction_receipt.return_value = {
        "status": 1,
        "transactionHash": b"\x00" * 32,
        "blockNumber": 1,
    }
    return mock_w3


@pytest.fixture
def owner_account():
    return Account.create()


@pytest.fixture
def investor_account():
    return Account.create()


@pytest.fixture
def tokenizer(w3, owner_account):
    return RWATokenizer(
        w3=w3,
        owner_address=owner_account.address,
        token_name="Real Estate Token",
        token_symbol="RET",
        initial_supply=1_000_000,
        asset_type="real_estate",
        asset_value=500_000_000,  # $500k in cents
        jurisdiction="US",
    )


# ─── Test 1: Token Deployment ────────────────────────────────────────────────

class TestTokenDeployment:
    def test_token_deployment_sets_name_and_symbol(self, tokenizer):
        assert tokenizer.token_name == "Real Estate Token"
        assert tokenizer.token_symbol == "RET"

    def test_token_deployment_sets_initial_supply(self, tokenizer):
        assert tokenizer.initial_supply == 1_000_000

    def test_token_deployment_sets_asset_type(self, tokenizer):
        assert tokenizer.asset_type == "real_estate"

    def test_token_deployment_sets_asset_value(self, tokenizer):
        assert tokenizer.asset_value == 500_000_000

    def test_token_deployment_sets_jurisdiction(self, tokenizer):
        assert tokenizer.jurisdiction == "US"

    def test_token_deployment_sets_owner(self, tokenizer, owner_account):
        assert tokenizer.owner == owner_account.address

    def test_token_deployment_registers_on_chain(self, tokenizer, w3):
        """Token deployment should call contract registration."""
        w3.eth.send_raw_transaction.assert_called()


# ─── Test 2: Compliance Checks ───────────────────────────────────────────────

class TestCompliance:
    def test_compliance_check_passes_for_verified_investor(self, tokenizer, investor_account):
        tokenizer.investors[investor_account.address] = {
            "status": InvestorStatus.VERIFIED,
            "kyc_expiry": 9999999999,
            "jurisdiction": "US",
            "accredited": True,
        }
        result = tokenizer.check_compliance(investor_account.address)
        assert result == ComplianceStatus.COMPLIANT

    def test_compliance_check_fails_for_unverified_investor(self, tokenizer, investor_account):
        tokenizer.investors[investor_account.address] = {
            "status": InvestorStatus.PENDING,
            "kyc_expiry": 9999999999,
            "jurisdiction": "US",
            "accredited": False,
        }
        result = tokenizer.check_compliance(investor_account.address)
        assert result == ComplianceStatus.NON_COMPLIANT

    def test_compliance_check_fails_for_expired_kyc(self, tokenizer, investor_account):
        tokenizer.investors[investor_account.address] = {
            "status": InvestorStatus.VERIFIED,
            "kyc_expiry": 0,  # expired
            "jurisdiction": "US",
            "accredited": True,
        }
        result = tokenizer.check_compliance(investor_account.address)
        assert result == ComplianceStatus.NON_COMPLIANT

    def test_compliance_check_fails_for_unknown_investor(self, tokenizer, investor_account):
        result = tokenizer.check_compliance(investor_account.address)
        assert result == ComplianceStatus.NON_COMPLIANT

    def test_compliance_check_fails_for_sanctioned_jurisdiction(self, tokenizer, investor_account):
        tokenizer.investors[investor_account.address] = {
            "status": InvestorStatus.VERIFIED,
            "kyc_expiry": 9999999999,
            "jurisdiction": "IR",  # sanctioned
            "accredited": True,
        }
        result = tokenizer.check_compliance(investor_account.address)
        assert result == ComplianceStatus.NON_COMPLIANT


# ─── Test 3: Investor Verification ──────────────────────────────────────────

class TestInvestorVerification:
    def test_register_investor_sets_pending_status(self, tokenizer, investor_account):
        tokenizer.register_investor(
            investor_address=investor_account.address,
            jurisdiction="US",
            accredited=True,
        )
        assert investor_account.address in tokenizer.investors
        assert tokenizer.investors[investor_account.address]["status"] == InvestorStatus.PENDING

    def test_verify_investor_sets_verified_status(self, tokenizer, investor_account):
        tokenizer.register_investor(
            investor_address=investor_account.address,
            jurisdiction="US",
            accredited=True,
        )
        tokenizer.verify_investor(investor_account.address)
        assert tokenizer.investors[investor_account.address]["status"] == InvestorStatus.VERIFIED

    def test_verify_investor_sets_kyc_expiry(self, tokenizer, investor_account):
        tokenizer.register_investor(
            investor_address=investor_account.address,
            jurisdiction="US",
            accredited=True,
        )
        tokenizer.verify_investor(investor_account.address)
        assert tokenizer.investors[investor_account.address]["kyc_expiry"] > 0

    def test_reject_investor_sets_rejected_status(self, tokenizer, investor_account):
        tokenizer.register_investor(
            investor_address=investor_account.address,
            jurisdiction="US",
            accredited=True,
        )
        tokenizer.reject_investor(investor_account.address)
        assert tokenizer.investors[investor_account.address]["status"] == InvestorStatus.REJECTED

    def test_cannot_verify_unregistered_investor(self, tokenizer, investor_account):
        with pytest.raises(ValueError, match="Investor not registered"):
            tokenizer.verify_investor(investor_account.address)


# ─── Test 4: Transfer Restrictions ───────────────────────────────────────────

class TestTransferRestrictions:
    def test_transfer_allowed_between_verified_investors(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)

        # Set up balances
        tokenizer.balances[owner_account.address] = 500_000
        tokenizer.balances[investor_account.address] = 50_000

        result = tokenizer.can_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=1000,
        )
        assert result is True

    def test_transfer_blocked_from_unverified_sender(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        # owner not verified
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)

        result = tokenizer.can_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=1000,
        )
        assert result is False

    def test_transfer_blocked_to_unverified_recipient(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        # investor not verified

        result = tokenizer.can_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=1000,
        )
        assert result is False

    def test_transfer_blocked_exceeds_max_holding(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)

        # Set up balances
        tokenizer.balances[owner_account.address] = 500_000
        tokenizer.balances[investor_account.address] = 50_000

        # Try to transfer more than 10% of supply (max holding limit)
        result = tokenizer.can_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=200_000,  # 20% of 1M supply
        )
        assert result is False

    def test_transfer_blocked_for_sanctioned_jurisdiction(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "IR", True)  # sanctioned
        tokenizer.verify_investor(investor_account.address)

        result = tokenizer.can_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=1000,
        )
        assert result is False


# ─── Test 5: Dividend Distribution ──────────────────────────────────────────

class TestDividendDistribution:
    def test_distribute_dividends_updates_balances(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)

        # Set up balances
        tokenizer.balances[owner_account.address] = 600_000
        tokenizer.balances[investor_account.address] = 400_000

        # Distribute $10,000 in dividends (10000 cents)
        tokenizer.distribute_dividends(total_amount=10_000)

        # Owner should get 60% = 6000
        assert tokenizer.balances[owner_account.address] == 606_000
        # Investor should get 40% = 4000
        assert tokenizer.balances[investor_account.address] == 404_000

    def test_distribute_dividends_only_to_verified_investors(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        # investor NOT verified

        tokenizer.balances[owner_account.address] = 600_000
        tokenizer.balances[investor_account.address] = 400_000

        tokenizer.distribute_dividends(total_amount=10_000)

        # Only verified owner gets dividends
        assert tokenizer.balances[owner_account.address] == 610_000
        # Unverified investor gets nothing
        assert tokenizer.balances[investor_account.address] == 400_000

    def test_distribute_dividends_records_distribution(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.balances[owner_account.address] = 1_000_000

        tokenizer.distribute_dividends(total_amount=5_000)

        assert len(tokenizer.distributions) == 1
        assert tokenizer.distributions[0]["total_amount"] == 5_000
        assert tokenizer.distributions[0]["recipients"] == 1


# ─── Test 6: ERC-3643 Compliance ────────────────────────────────────────────

class TestERC3643Compliance:
    def test_token_implements_erc3643_interface(self, tokenizer):
        """ERC-3643 tokens must implement the T-REX interface."""
        assert hasattr(tokenizer, 'check_compliance')
        assert hasattr(tokenizer, 'can_transfer')
        assert hasattr(tokenizer, 'register_investor')
        assert hasattr(tokenizer, 'verify_investor')

    def test_token_has_compliance_module(self, tokenizer):
        """ERC-3643 requires a compliance module."""
        assert tokenizer.compliance_module is not None

    def test_token_has_identity_registry(self, tokenizer):
        """ERC-3643 requires an identity registry for investor tracking."""
        assert tokenizer.identity_registry is not None

    def test_token_supports_freeze(self, tokenizer, investor_account):
        """ERC-3643 tokens support freezing investor balances."""
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)
        tokenizer.balances[investor_account.address] = 1000

        tokenizer.freeze_investor(investor_account.address)
        assert tokenizer.investors[investor_account.address]["frozen"] is True

    def test_frozen_investor_cannot_transfer(self, tokenizer, owner_account, investor_account):
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)

        tokenizer.freeze_investor(investor_account.address)

        result = tokenizer.can_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=1000,
        )
        assert result is False

    def test_token_supports_force_transfer(self, tokenizer, owner_account, investor_account):
        """ERC-3643 allows force transfer by owner for compliance."""
        tokenizer.register_investor(owner_account.address, "US", True)
        tokenizer.verify_investor(owner_account.address)
        tokenizer.register_investor(investor_account.address, "US", True)
        tokenizer.verify_investor(investor_account.address)

        tokenizer.balances[owner_account.address] = 5000
        tokenizer.balances[investor_account.address] = 3000

        # Force transfer from owner to investor
        tokenizer.force_transfer(
            from_address=owner_account.address,
            to_address=investor_account.address,
            amount=1000,
        )

        assert tokenizer.balances[owner_account.address] == 4000
        assert tokenizer.balances[investor_account.address] == 4000


# ─── Test 7: Asset Valuation ────────────────────────────────────────────────

class TestAssetValuation:
    def test_get_asset_value_returns_correct_value(self, tokenizer):
        assert tokenizer.get_asset_value() == 500_000_000

    def test_get_token_price_calculates_correctly(self, tokenizer):
        # Token price = asset_value / initial_supply
        expected_price = 500_000_000 / 1_000_000  # 500 cents = $5.00
        assert tokenizer.get_token_price() == expected_price

    def test_update_asset_value(self, tokenizer):
        tokenizer.update_asset_value(600_000_000)
        assert tokenizer.get_asset_value() == 600_000_000


# ─── Test 8: Batch Operations ───────────────────────────────────────────────

class TestBatchOperations:
    def test_batch_register_investors(self, tokenizer, investor_account):
        accounts = [Account.create() for _ in range(5)]
        addresses = [a.address for a in accounts]

        tokenizer.batch_register_investors(addresses, "US", True)

        for addr in addresses:
            assert addr in tokenizer.investors
            assert tokenizer.investors[addr]["status"] == InvestorStatus.PENDING

    def test_batch_verify_investors(self, tokenizer):
        accounts = [Account.create() for _ in range(3)]
        addresses = [a.address for a in accounts]

        tokenizer.batch_register_investors(addresses, "US", True)
        tokenizer.batch_verify_investors(addresses)

        for addr in addresses:
            assert tokenizer.investors[addr]["status"] == InvestorStatus.VERIFIED
