"""
TDD tests for Blockchain Analytics Engine.

Covers:
- Blockchain data analysis (blocks, transactions, gas, addresses)
- DeFi protocol analytics (TVL, liquidity, AMM, lending)
- Tokenomics (supply, distribution, inflation, concentration, velocity)
- Consensus mechanism analysis (PoW/PoS, validators, finality, Nakamoto coefficient)
"""
import pytest
from unittest.mock import MagicMock, patch

from src.blockchain.analytics import (
    BlockchainAnalyticsEngine,
    BlockMetrics,
    TransactionMetrics,
    DeFiProtocolMetrics,
    TokenomicsMetrics,
    ConsensusMetrics,
    ConsensusType,
    ProtocolType,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_w3():
    """Web3 instance with mocked provider."""
    mock = MagicMock()
    mock.eth.chain_id = 1
    mock.eth.block_number = 1_000
    mock.eth.get_block.return_value = {
        "number": 999,
        "timestamp": 1_700_000_000,
        "transactions": [b"\x00" * 32] * 150,
        "gasUsed": 15_000_000,
        "gasLimit": 30_000_000,
        "difficulty": 0,
        "totalDifficulty": 100_000_000,
        "nonce": b"\x00" * 8,
        "mixHash": b"\x00" * 32,
    }
    mock.eth.get_transaction_count.return_value = 42
    mock.eth.gas_price = 20_000_000_000
    mock.eth.get_balance.return_value = 1_000_000_000_000_000_000  # 1 ETH
    mock.eth.get_code.return_value = b"\x60\x80\x60\x40"
    mock.to_wei.side_effect = lambda x, unit="ether": int(x * 1e18) if unit == "ether" else int(x * 1e9)
    mock.from_wei.side_effect = lambda x, unit="ether": x / 1e18 if unit == "ether" else x / 1e9
    mock.is_address.side_effect = lambda x: isinstance(x, str) and len(x) == 42 and x.startswith("0x")
    mock.is_checksum_address.side_effect = lambda x: isinstance(x, str) and len(x) == 42
    mock.keccak.side_effect = lambda x: b"\x00" * 32
    return mock


@pytest.fixture
def engine(mock_w3):
    return BlockchainAnalyticsEngine(w3=mock_w3)


@pytest.fixture
def sample_blocks():
    """Sample block data for testing."""
    return [
        {
            "number": i,
            "timestamp": 1_700_000_000 + i * 12,
            "transactions": [b"\x00" * 32] * (100 + i * 10),
            "gasUsed": 10_000_000 + i * 1_000_000,
            "gasLimit": 30_000_000,
            "difficulty": 1_000_000 if i < 5 else 0,
            "totalDifficulty": 100_000_000 + i * 1_000_000,
            "nonce": b"\x00" * 8,
            "mixHash": b"\x00" * 32,
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_txs():
    """Sample transaction data for testing."""
    return [
        {
            "hash": b"\x01" * 32,
            "from": "0x" + "11" * 20,
            "to": "0x" + "22" * 20,
            "value": 1_000_000_000_000_000_000,
            "gas": 21000,
            "gasPrice": 20_000_000_000,
            "nonce": 0,
            "blockNumber": 100,
        },
        {
            "hash": b"\x02" * 32,
            "from": "0x" + "33" * 20,
            "to": "0x" + "44" * 20,
            "value": 500_000_000_000_000_000,
            "gas": 65000,
            "gasPrice": 30_000_000_000,
            "nonce": 1,
            "blockNumber": 100,
        },
        {
            "hash": b"\x03" * 32,
            "from": "0x" + "55" * 20,
            "to": None,  # Contract creation
            "value": 0,
            "gas": 200_000,
            "gasPrice": 25_000_000_000,
            "nonce": 2,
            "blockNumber": 101,
        },
    ]


@pytest.fixture
def sample_token_holders():
    """Sample token holder distribution."""
    return [
        {"address": "0x" + "aa" * 20, "balance": 500_000_000, "percent": 50.0},
        {"address": "0x" + "bb" * 20, "balance": 200_000_000, "percent": 20.0},
        {"address": "0x" + "cc" * 20, "balance": 100_000_000, "percent": 10.0},
        {"address": "0x" + "dd" * 20, "balance": 50_000_000, "percent": 5.0},
        {"address": "0x" + "ee" * 20, "balance": 30_000_000, "percent": 3.0},
        {"address": "0x" + "ff" * 20, "balance": 20_000_000, "percent": 2.0},
        {"address": "0x" + "11" * 20, "balance": 10_000_000, "percent": 1.0},
        {"address": "0x" + "22" * 20, "balance": 5_000_000, "percent": 0.5},
        {"address": "0x" + "33" * 20, "balance": 3_000_000, "percent": 0.3},
        {"address": "0x" + "44" * 20, "balance": 2_000_000, "percent": 0.2},
    ]


@pytest.fixture
def sample_validators():
    """Sample validator set for consensus analysis."""
    return [
        {"address": "0x" + "a1" * 20, "stake": 1_000_000, "commission": 0.05, "uptime": 0.999},
        {"address": "0x" + "a2" * 20, "stake": 800_000, "commission": 0.06, "uptime": 0.998},
        {"address": "0x" + "a3" * 20, "stake": 600_000, "commission": 0.04, "uptime": 0.997},
        {"address": "0x" + "a4" * 20, "stake": 400_000, "commission": 0.07, "uptime": 0.996},
        {"address": "0x" + "a5" * 20, "stake": 300_000, "commission": 0.05, "uptime": 0.995},
        {"address": "0x" + "a6" * 20, "stake": 200_000, "commission": 0.08, "uptime": 0.994},
        {"address": "0x" + "a7" * 20, "stake": 150_000, "commission": 0.03, "uptime": 0.993},
        {"address": "0x" + "a8" * 20, "stake": 100_000, "commission": 0.05, "uptime": 0.992},
        {"address": "0x" + "a9" * 20, "stake": 80_000, "commission": 0.06, "uptime": 0.991},
        {"address": "0x" + "b1" * 20, "stake": 50_000, "commission": 0.04, "uptime": 0.990},
    ]


# ─── Test 1: Engine Initialization ──────────────────────────────────────────


class TestEngineInitialization:
    def test_engine_initializes_with_w3(self, engine, mock_w3):
        assert engine.w3 is mock_w3
        assert engine.chain_id == 1

    def test_engine_initializes_without_w3(self):
        engine = BlockchainAnalyticsEngine()
        assert engine.w3 is None
        assert engine.chain_id is None

    def test_engine_initializes_with_custom_rpc(self):
        with patch("src.blockchain.analytics.Web3") as mock_web3_class:
            mock_instance = MagicMock()
            mock_instance.eth.chain_id = 137
            mock_web3_class.HTTPProvider.return_value = MagicMock()
            mock_web3_class.return_value = mock_instance
            engine = BlockchainAnalyticsEngine(rpc_url="https://polygon-rpc.com")
            assert engine.chain_id == 137


# ─── Test 2: Block Metrics ──────────────────────────────────────────────────


class TestBlockMetrics:
    def test_analyze_block_time(self, engine, sample_blocks):
        metrics = engine.analyze_block_time(sample_blocks)
        assert isinstance(metrics, BlockMetrics)
        assert metrics.avg_block_time == pytest.approx(12.0, rel=0.01)
        assert metrics.min_block_time == 12
        assert metrics.max_block_time == 12

    def test_analyze_block_time_empty(self, engine):
        with pytest.raises(ValueError, match="No blocks provided"):
            engine.analyze_block_time([])

    def test_analyze_gas_utilization(self, engine, sample_blocks):
        metrics = engine.analyze_gas_utilization(sample_blocks)
        assert isinstance(metrics, BlockMetrics)
        assert 0 < metrics.avg_gas_utilization < 1
        assert metrics.avg_gas_utilization == pytest.approx(0.483, rel=0.01)

    def test_analyze_block_size_trend(self, engine, sample_blocks):
        metrics = engine.analyze_block_size_trend(sample_blocks)
        assert isinstance(metrics, BlockMetrics)
        assert metrics.avg_transactions_per_block > 0
        assert metrics.tx_growth_rate > 0


# ─── Test 3: Transaction Metrics ────────────────────────────────────────────


class TestTransactionMetrics:
    def test_analyze_transaction_throughput(self, engine, sample_txs):
        metrics = engine.analyze_transaction_throughput(sample_txs)
        assert isinstance(metrics, TransactionMetrics)
        assert metrics.total_transactions == 3
        assert metrics.total_value_transferred > 0
        assert metrics.avg_gas_used > 0

    def test_analyze_transaction_throughput_empty(self, engine):
        with pytest.raises(ValueError, match="No transactions provided"):
            engine.analyze_transaction_throughput([])

    def test_detect_contract_creations(self, engine, sample_txs):
        metrics = engine.analyze_transaction_throughput(sample_txs)
        assert metrics.contract_creations == 1

    def test_analyze_address_activity(self, engine, sample_txs):
        metrics = engine.analyze_address_activity(sample_txs)
        assert isinstance(metrics, dict)
        assert "0x" + "11" * 20 in metrics
        assert metrics["0x" + "11" * 20]["sent"] == 1
        assert metrics["0x" + "11" * 20]["received"] == 0


# ─── Test 4: DeFi Protocol Analytics ────────────────────────────────────────


class TestDeFiProtocolAnalytics:
    def test_analyze_amm_pool(self, engine):
        pool_data = {
            "token0": "0x" + "aa" * 20,
            "token1": "0x" + "bb" * 20,
            "reserve0": 1_000_000,
            "reserve1": 500_000,
            "fee_tier": 3000,
            "volume_24h": 10_000_000,
            "tvl": 2_000_000,
        }
        metrics = engine.analyze_amm_pool(pool_data)
        assert isinstance(metrics, DeFiProtocolMetrics)
        assert metrics.protocol_type == ProtocolType.AMM
        assert metrics.tvl == 2_000_000
        assert metrics.volume_24h == 10_000_000
        assert metrics.utilization_rate > 0

    def test_analyze_lending_protocol(self, engine):
        lending_data = {
            "total_deposits": 100_000_000,
            "total_borrows": 60_000_000,
            "liquidation_threshold": 0.8,
            "reserve_factor": 0.1,
            "assets": [
                {"symbol": "ETH", "deposits": 50_000_000, "borrows": 30_000_000},
                {"symbol": "USDC", "deposits": 50_000_000, "borrows": 30_000_000},
            ],
        }
        metrics = engine.analyze_lending_protocol(lending_data)
        assert isinstance(metrics, DeFiProtocolMetrics)
        assert metrics.protocol_type == ProtocolType.LENDING
        assert metrics.tvl == 100_000_000
        assert metrics.utilization_rate == pytest.approx(0.6, rel=0.01)

    def test_analyze_yield_farm(self, engine):
        farm_data = {
            "pool_token": "0x" + "cc" * 20,
            "staked_amount": 5_000_000,
            "reward_rate": 1000,
            "reward_token_price": 1.5,
            "staking_token_price": 10.0,
        }
        metrics = engine.analyze_yield_farm(farm_data)
        assert isinstance(metrics, DeFiProtocolMetrics)
        assert metrics.protocol_type == ProtocolType.YIELD_FARM
        assert metrics.apy > 0

    def test_calculate_impermanent_loss(self, engine):
        il = engine.calculate_impermanent_loss(
            initial_price_ratio=1.0,
            final_price_ratio=2.0,
        )
        assert isinstance(il, float)
        assert il == pytest.approx(0.0572, rel=0.01)  # ~5.72% IL for 2x price change

    def test_calculate_impermanent_loss_no_change(self, engine):
        il = engine.calculate_impermanent_loss(
            initial_price_ratio=1.0,
            final_price_ratio=1.0,
        )
        assert il == pytest.approx(0.0, abs=1e-10)


# ─── Test 5: Tokenomics ─────────────────────────────────────────────────────


class TestTokenomics:
    def test_analyze_token_distribution(self, engine, sample_token_holders):
        metrics = engine.analyze_token_distribution(sample_token_holders)
        assert isinstance(metrics, TokenomicsMetrics)
        assert metrics.total_supply == 920_000_000
        assert metrics.hhi > 0  # Herfindahl-Hirschman Index
        assert metrics.gini_coefficient > 0
        assert metrics.top_10_concentration > 0

    def test_analyze_token_distribution_empty(self, engine):
        with pytest.raises(ValueError, match="No holders provided"):
            engine.analyze_token_distribution([])

    def test_calculate_token_velocity(self, engine):
        velocity = engine.calculate_token_velocity(
            transaction_volume=100_000_000,
            circulating_supply=1_000_000_000,
            period_days=30,
        )
        assert isinstance(velocity, float)
        assert velocity == pytest.approx(0.1, rel=0.01)

    def test_calculate_inflation_rate(self, engine):
        inflation = engine.calculate_inflation_rate(
            initial_supply=1_000_000,
            current_supply=1_050_000,
            period_years=1,
        )
        assert isinstance(inflation, float)
        assert inflation == pytest.approx(0.05, rel=0.01)

    def test_analyze_vesting_schedule(self, engine):
        vesting_data = {
            "total_allocated": 10_000_000,
            "cliff_months": 12,
            "vesting_months": 36,
            "start_date": "2023-01-01",
            "end_date": "2026-01-01",
            "claimed": 2_000_000,
        }
        metrics = engine.analyze_vesting_schedule(vesting_data)
        assert isinstance(metrics, dict)
        assert metrics["vesting_progress"] == pytest.approx(0.2, rel=0.01)
        assert metrics["unvested"] == 8_000_000


# ─── Test 6: Consensus Analysis ─────────────────────────────────────────────


class TestConsensusAnalysis:
    def test_analyze_pos_validator_distribution(self, engine, sample_validators):
        metrics = engine.analyze_validator_distribution(
            sample_validators,
            consensus_type=ConsensusType.POS,
        )
        assert isinstance(metrics, ConsensusMetrics)
        assert metrics.consensus_type == ConsensusType.POS
        assert metrics.total_validators == 10
        assert metrics.total_stake == 3_680_000
        assert metrics.nakamoto_coefficient >= 1

    def test_analyze_pos_validator_distribution_empty(self, engine):
        with pytest.raises(ValueError, match="No validators provided"):
            engine.analyze_validator_distribution([], consensus_type=ConsensusType.POS)

    def test_calculate_nakamoto_coefficient(self, engine, sample_validators):
        nc = engine.calculate_nakamoto_coefficient(sample_validators)
        assert isinstance(nc, int)
        assert nc >= 1
        assert nc <= len(sample_validators)

    def test_analyze_pow_mining_distribution(self, engine):
        miners = [
            {"address": "0x" + "m1" * 20, "hashrate": 400_000_000},
            {"address": "0x" + "m2" * 20, "hashrate": 300_000_000},
            {"address": "0x" + "m3" * 20, "hashrate": 200_000_000},
            {"address": "0x" + "m4" * 20, "hashrate": 100_000_000},
        ]
        metrics = engine.analyze_mining_distribution(
            miners,
            consensus_type=ConsensusType.POW,
        )
        assert isinstance(metrics, ConsensusMetrics)
        assert metrics.consensus_type == ConsensusType.POW
        assert metrics.nakamoto_coefficient == 2  # Top 2 control >50%

    def test_estimate_finality_time(self, engine):
        finality = engine.estimate_finality_time(
            consensus_type=ConsensusType.POS,
            block_time=12,
            validator_count=100,
        )
        assert isinstance(finality, float)
        assert finality > 0

    def test_estimate_finality_time_pow(self, engine):
        finality = engine.estimate_finality_time(
            consensus_type=ConsensusType.POW,
            block_time=600,
            confirmations=6,
        )
        assert isinstance(finality, float)
        assert finality == pytest.approx(3600, rel=0.01)

    def test_calculate_staking_apy(self, engine):
        apy = engine.calculate_staking_apy(
            total_stake=10_000_000,
            annual_rewards=500_000,
            commission=0.05,
        )
        assert isinstance(apy, float)
        assert apy == pytest.approx(0.0475, rel=0.01)  # 5% * (1 - 0.05)


# ─── Test 7: On-Chain Analytics ─────────────────────────────────────────────


class TestOnChainAnalytics:
    def test_analyze_wallet_activity(self, engine, mock_w3):
        mock_w3.eth.get_transaction_count.return_value = 150
        mock_w3.eth.get_balance.return_value = 5_000_000_000_000_000_000
        mock_w3.eth.get_code.return_value = b"\x60\x80\x60\x40"

        result = engine.analyze_wallet_activity("0x" + "ab" * 20)
        assert isinstance(result, dict)
        assert result["transaction_count"] == 150
        assert result["balance"] > 0
        assert result["is_contract"] is True

    def test_analyze_wallet_activity_eoa(self, engine, mock_w3):
        mock_w3.eth.get_code.return_value = b""
        result = engine.analyze_wallet_activity("0x" + "cd" * 20)
        assert result["is_contract"] is False

    def test_detect_whale_wallets(self, engine, sample_token_holders):
        whales = engine.detect_whale_wallets(sample_token_holders, threshold_percent=5.0)
        assert isinstance(whales, list)
        assert len(whales) == 4  # Top 4 holders have >=5%
        assert whales[0]["address"] == "0x" + "aa" * 20


# ─── Test 8: Gas Analytics ──────────────────────────────────────────────────


class TestGasAnalytics:
    def test_analyze_gas_trends(self, engine, sample_blocks):
        result = engine.analyze_gas_trends(sample_blocks)
        assert isinstance(result, dict)
        assert "avg_gas_used" in result
        assert "gas_utilization" in result
        assert "gas_efficiency" in result

    def test_estimate_transaction_cost(self, engine):
        cost = engine.estimate_transaction_cost(
            gas_used=21000,
            gas_price_gwei=20,
            eth_price=2000,
        )
        assert isinstance(cost, float)
        assert cost == pytest.approx(0.84, rel=0.01)  # 21000 * 20e-9 * 2000


# ─── Test 9: Network Health ─────────────────────────────────────────────────


class TestNetworkHealth:
    def test_calculate_network_health_score(self, engine, sample_blocks, sample_txs):
        score = engine.calculate_network_health_score(sample_blocks, sample_txs)
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_detect_network_congestion(self, engine):
        congested_blocks = [
            {
                "number": i,
                "timestamp": 1_700_000_000 + i * 12,
                "transactions": [b"\x00" * 32] * 300,
                "gasUsed": 29_000_000,
                "gasLimit": 30_000_000,
                "difficulty": 0,
                "totalDifficulty": 100_000_000,
                "nonce": b"\x00" * 8,
                "mixHash": b"\x00" * 32,
            }
            for i in range(5)
        ]
        result = engine.detect_network_congestion(congested_blocks)
        assert isinstance(result, dict)
        assert result["is_congested"] is True
        assert result["avg_utilization"] > 0.9


# ─── Test 10: MEV Detection ─────────────────────────────────────────────────


class TestMEVDetection:
    def test_detect_sandwich_attack(self, engine):
        txs = [
            {"hash": b"\x01" * 32, "from": "0x" + "aa" * 20, "to": "0x" + "bb" * 20, "value": 100, "gasPrice": 20},
            {"hash": b"\x02" * 32, "from": "0x" + "cc" * 20, "to": "0x" + "bb" * 20, "value": 100, "gasPrice": 50},
            {"hash": b"\x03" * 32, "from": "0x" + "cc" * 20, "to": "0x" + "dd" * 20, "value": 100, "gasPrice": 15},
        ]
        result = engine.detect_sandwich_attack(txs)
        assert isinstance(result, dict)
        assert result["is_sandwich"] is True
        assert result["attacker"] == "0x" + "cc" * 20

    def test_detect_sandwich_attack_clean(self, engine):
        txs = [
            {"hash": b"\x01" * 32, "from": "0x" + "aa" * 20, "to": "0x" + "bb" * 20, "value": 100, "gasPrice": 20},
            {"hash": b"\x02" * 32, "from": "0x" + "cc" * 20, "to": "0x" + "dd" * 20, "value": 100, "gasPrice": 20},
        ]
        result = engine.detect_sandwich_attack(txs)
        assert result["is_sandwich"] is False


# ─── Test 11: Cross-Chain Analytics ─────────────────────────────────────────


class TestCrossChainAnalytics:
    def test_compare_chain_metrics(self, engine):
        chain_a = {"tvl": 1_000_000_000, "tx_count": 500_000, "avg_gas_price": 20}
        chain_b = {"tvl": 500_000_000, "tx_count": 300_000, "avg_gas_price": 10}
        result = engine.compare_chain_metrics(chain_a, chain_b)
        assert isinstance(result, dict)
        assert result["tvl_ratio"] == pytest.approx(2.0, rel=0.01)
        assert result["tx_ratio"] == pytest.approx(1.67, rel=0.01)

    def test_estimate_bridge_risk(self, engine):
        risk = engine.estimate_bridge_risk(
            bridge_tvl=100_000_000,
            daily_volume=10_000_000,
            validator_count=7,
            threshold=3,
        )
        assert isinstance(risk, dict)
        assert risk["risk_level"] in ["low", "medium", "high"]
        assert risk["validator_risk"] is not None


# ─── Test 12: Smart Contract Risk ───────────────────────────────────────────


class TestSmartContractRisk:
    def test_assess_contract_risk(self, engine):
        contract_data = {
            "is_verified": True,
            "audit_count": 2,
            "age_days": 365,
            "tvl": 10_000_000,
            "has_ownership_renounced": True,
            "has_timelock": True,
        }
        risk = engine.assess_contract_risk(contract_data)
        assert isinstance(risk, dict)
        assert risk["risk_score"] < 30  # Low risk
        assert risk["risk_level"] == "low"

    def test_assess_contract_risk_high_risk(self, engine):
        contract_data = {
            "is_verified": False,
            "audit_count": 0,
            "age_days": 7,
            "tvl": 100_000,
            "has_ownership_renounced": False,
            "has_timelock": False,
        }
        risk = engine.assess_contract_risk(contract_data)
        assert risk["risk_score"] > 70
        assert risk["risk_level"] == "high"


# ─── Test 13: Oracle Analytics ──────────────────────────────────────────────


class TestOracleAnalytics:
    def test_analyze_oracle_deviation(self, engine):
        prices = [100.0, 101.0, 99.5, 100.5, 100.2]
        result = engine.analyze_oracle_deviation(prices, reference_price=100.0)
        assert isinstance(result, dict)
        assert result["max_deviation"] > 0
        assert result["avg_deviation"] > 0

    def test_detect_oracle_manipulation(self, engine):
        prices = [100.0, 100.1, 100.2, 150.0, 100.3]  # Spike at index 3
        result = engine.detect_oracle_manipulation(prices, threshold=0.1)
        assert isinstance(result, dict)
        assert result["is_manipulation"] is True
        assert result["manipulation_index"] == 3


# ─── Test 14: Governance Analytics ──────────────────────────────────────────


class TestGovernanceAnalytics:
    def test_analyze_governance_participation(self, engine):
        proposals = [
            {"votes_for": 1000, "votes_against": 200, "total_supply": 2000},
            {"votes_for": 800, "votes_against": 400, "total_supply": 2000},
            {"votes_for": 1500, "votes_against": 100, "total_supply": 2000},
        ]
        result = engine.analyze_governance_participation(proposals)
        assert isinstance(result, dict)
        assert result["avg_participation"] > 0
        assert result["avg_approval_rate"] > 0

    def test_calculate_voting_power_concentration(self, engine):
        voters = [
            {"address": "0x" + "a1" * 20, "votes": 500},
            {"address": "0x" + "a2" * 20, "votes": 300},
            {"address": "0x" + "a3" * 20, "votes": 200},
        ]
        result = engine.calculate_voting_power_concentration(voters)
        assert isinstance(result, float)
        assert 0 <= result <= 1
