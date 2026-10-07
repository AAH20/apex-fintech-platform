"""
Blockchain Analytics Engine for DeFi, Tokenomics, and Consensus Analysis.

This module provides a comprehensive analytics engine for blockchain data,
DeFi protocols, tokenomics, and consensus mechanisms.

References:
    - Buterin, V. et al. (2020). Combining GHOST and Casper.
    - Gudgeon, L. et al. (2020). SoK: Layer-Two Blockchain Protocols.
    - CFA Institute (2023). Blockchain and Digital Assets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from web3 import Web3


class ConsensusType(str, Enum):
    """Consensus mechanism types."""

    POW = "pow"
    POS = "pos"
    DPOS = "dpos"
    POA = "poa"


class ProtocolType(str, Enum):
    """DeFi protocol types."""

    AMM = "amm"
    LENDING = "lending"
    YIELD_FARM = "yield_farm"
    STAKING = "staking"
    DERIVATIVES = "derivatives"
    BRIDGE = "bridge"


@dataclass
class BlockMetrics:
    """Block-level metrics."""

    avg_block_time: float = 0.0
    min_block_time: float = 0.0
    max_block_time: float = 0.0
    avg_gas_utilization: float = 0.0
    avg_transactions_per_block: float = 0.0
    tx_growth_rate: float = 0.0


@dataclass
class TransactionMetrics:
    """Transaction-level metrics."""

    total_transactions: int = 0
    total_value_transferred: float = 0.0
    avg_gas_used: float = 0.0
    avg_gas_price: float = 0.0
    contract_creations: int = 0
    unique_senders: int = 0
    unique_receivers: int = 0


@dataclass
class DeFiProtocolMetrics:
    """DeFi protocol metrics."""

    protocol_type: ProtocolType = ProtocolType.AMM
    tvl: float = 0.0
    volume_24h: float = 0.0
    utilization_rate: float = 0.0
    apy: float = 0.0
    impermanent_loss: float = 0.0
    fee_revenue: float = 0.0
    unique_users: int = 0


@dataclass
class TokenomicsMetrics:
    """Tokenomics metrics."""

    total_supply: float = 0.0
    circulating_supply: float = 0.0
    hhi: float = 0.0
    gini_coefficient: float = 0.0
    top_10_concentration: float = 0.0
    velocity: float = 0.0
    inflation_rate: float = 0.0
    holder_count: int = 0


@dataclass
class ConsensusMetrics:
    """Consensus mechanism metrics."""

    consensus_type: ConsensusType = ConsensusType.POS
    total_validators: int = 0
    total_stake: float = 0.0
    nakamoto_coefficient: int = 0
    avg_commission: float = 0.0
    avg_uptime: float = 0.0
    finality_time: float = 0.0
    staking_apy: float = 0.0


class BlockchainAnalyticsEngine:
    """Main blockchain analytics engine.

    Analyzes blockchain data, DeFi protocols, tokenomics, and consensus
    mechanisms using web3.py for on-chain data access.
    """

    def __init__(self, w3: Web3 | None = None, rpc_url: str | None = None):
        """Initialize the analytics engine.

        Args:
            w3: Web3 instance
            rpc_url: RPC URL for creating a Web3 instance
        """
        if w3 is not None:
            self.w3 = w3
        elif rpc_url is not None:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        else:
            self.w3 = None

        self.chain_id = self.w3.eth.chain_id if self.w3 else None

    # ─── Block Analytics ─────────────────────────────────────────────────

    def analyze_block_time(self, blocks: list[dict]) -> BlockMetrics:
        """Analyze block time statistics.

        Args:
            blocks: List of block data dictionaries

        Returns:
            BlockMetrics with block time statistics
        """
        if not blocks:
            raise ValueError("No blocks provided")

        timestamps = [b["timestamp"] for b in blocks]
        block_times = [
            timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]

        if not block_times:
            block_times = [0]

        return BlockMetrics(
            avg_block_time=sum(block_times) / len(block_times),
            min_block_time=min(block_times),
            max_block_time=max(block_times),
        )

    def analyze_gas_utilization(self, blocks: list[dict]) -> BlockMetrics:
        """Analyze gas utilization across blocks.

        Args:
            blocks: List of block data dictionaries

        Returns:
            BlockMetrics with gas utilization statistics
        """
        if not blocks:
            raise ValueError("No blocks provided")

        utilizations = [b["gasUsed"] / b["gasLimit"] for b in blocks]

        return BlockMetrics(
            avg_gas_utilization=sum(utilizations) / len(utilizations),
        )

    def analyze_block_size_trend(self, blocks: list[dict]) -> BlockMetrics:
        """Analyze block size trends.

        Args:
            blocks: List of block data dictionaries

        Returns:
            BlockMetrics with block size statistics
        """
        if not blocks:
            raise ValueError("No blocks provided")

        tx_counts = [len(b["transactions"]) for b in blocks]
        avg_tx = sum(tx_counts) / len(tx_counts)

        if len(tx_counts) > 1 and tx_counts[0] > 0:
            growth_rate = (tx_counts[-1] - tx_counts[0]) / tx_counts[0]
        else:
            growth_rate = 0.0

        return BlockMetrics(
            avg_transactions_per_block=avg_tx,
            tx_growth_rate=growth_rate,
        )

    # ─── Transaction Analytics ───────────────────────────────────────────

    def analyze_transaction_throughput(
        self, transactions: list[dict]
    ) -> TransactionMetrics:
        """Analyze transaction throughput metrics.

        Args:
            transactions: List of transaction data dictionaries

        Returns:
            TransactionMetrics with throughput statistics
        """
        if not transactions:
            raise ValueError("No transactions provided")

        total_value = sum(tx.get("value", 0) for tx in transactions)
        gas_used = [tx.get("gas", 0) for tx in transactions]
        gas_prices = [tx.get("gasPrice", 0) for tx in transactions]
        contract_creations = sum(1 for tx in transactions if tx.get("to") is None)
        unique_senders = len(set(tx.get("from") for tx in transactions))
        unique_receivers = len(set(tx.get("to") for tx in transactions if tx.get("to")))

        return TransactionMetrics(
            total_transactions=len(transactions),
            total_value_transferred=total_value,
            avg_gas_used=sum(gas_used) / len(gas_used) if gas_used else 0,
            avg_gas_price=sum(gas_prices) / len(gas_prices) if gas_prices else 0,
            contract_creations=contract_creations,
            unique_senders=unique_senders,
            unique_receivers=unique_receivers,
        )

    def analyze_address_activity(self, transactions: list[dict]) -> dict:
        """Analyze address activity from transactions.

        Args:
            transactions: List of transaction data dictionaries

        Returns:
            Dict mapping address to activity metrics
        """
        activity: dict[str, dict[str, int]] = {}

        for tx in transactions:
            sender = tx.get("from")
            receiver = tx.get("to")

            if sender:
                if sender not in activity:
                    activity[sender] = {"sent": 0, "received": 0}
                activity[sender]["sent"] += 1

            if receiver:
                if receiver not in activity:
                    activity[receiver] = {"sent": 0, "received": 0}
                activity[receiver]["received"] += 1

        return activity

    # ─── DeFi Protocol Analytics ─────────────────────────────────────────

    def analyze_amm_pool(self, pool_data: dict) -> DeFiProtocolMetrics:
        """Analyze AMM pool metrics.

        Args:
            pool_data: Pool data with reserves, volume, TVL

        Returns:
            DeFiProtocolMetrics for the AMM pool
        """
        tvl = pool_data.get("tvl", 0)
        volume_24h = pool_data.get("volume_24h", 0)
        utilization = volume_24h / tvl if tvl > 0 else 0

        return DeFiProtocolMetrics(
            protocol_type=ProtocolType.AMM,
            tvl=tvl,
            volume_24h=volume_24h,
            utilization_rate=utilization,
        )

    def analyze_lending_protocol(self, lending_data: dict) -> DeFiProtocolMetrics:
        """Analyze lending protocol metrics.

        Args:
            lending_data: Lending protocol data

        Returns:
            DeFiProtocolMetrics for the lending protocol
        """
        total_deposits = lending_data.get("total_deposits", 0)
        total_borrows = lending_data.get("total_borrows", 0)
        utilization = total_borrows / total_deposits if total_deposits > 0 else 0

        return DeFiProtocolMetrics(
            protocol_type=ProtocolType.LENDING,
            tvl=total_deposits,
            utilization_rate=utilization,
        )

    def analyze_yield_farm(self, farm_data: dict) -> DeFiProtocolMetrics:
        """Analyze yield farm metrics.

        Args:
            farm_data: Yield farm data

        Returns:
            DeFiProtocolMetrics for the yield farm
        """
        staked = farm_data.get("staked_amount", 0)
        reward_rate = farm_data.get("reward_rate", 0)
        reward_price = farm_data.get("reward_token_price", 0)
        staking_price = farm_data.get("staking_token_price", 0)

        annual_rewards = reward_rate * reward_price * 365
        tvl = staked * staking_price
        apy = (annual_rewards / tvl * 100) if tvl > 0 else 0

        return DeFiProtocolMetrics(
            protocol_type=ProtocolType.YIELD_FARM,
            tvl=tvl,
            apy=apy,
        )

    def calculate_impermanent_loss(
        self, initial_price_ratio: float, final_price_ratio: float
    ) -> float:
        """Calculate impermanent loss for AMM liquidity providers.

        IL = 2 * sqrt(P1/P0) / (1 + P1/P0) - 1

        Args:
            initial_price_ratio: Initial price ratio (P0)
            final_price_ratio: Final price ratio (P1)

        Returns:
            Impermanent loss as a positive float
        """
        if initial_price_ratio <= 0 or final_price_ratio <= 0:
            return 0.0

        price_ratio = final_price_ratio / initial_price_ratio
        il = 2 * math.sqrt(price_ratio) / (1 + price_ratio) - 1
        return abs(il)

    # ─── Tokenomics ──────────────────────────────────────────────────────

    def analyze_token_distribution(
        self, holders: list[dict]
    ) -> TokenomicsMetrics:
        """Analyze token distribution metrics.

        Args:
            holders: List of token holder data

        Returns:
            TokenomicsMetrics with distribution statistics
        """
        if not holders:
            raise ValueError("No holders provided")

        balances = [h["balance"] for h in holders]
        total_supply = sum(balances)

        # Herfindahl-Hirschman Index
        hhi = sum((b / total_supply) ** 2 for b in balances) if total_supply > 0 else 0

        # Gini coefficient
        sorted_balances = sorted(balances)
        n = len(sorted_balances)
        if n > 0 and total_supply > 0:
            cumsum = sum((i + 1) * b for i, b in enumerate(sorted_balances))
            gini = (2 * cumsum) / (n * total_supply) - (n + 1) / n
        else:
            gini = 0.0

        # Top 10 concentration
        top_10 = sorted(balances, reverse=True)[:10]
        top_10_concentration = (
            sum(top_10) / total_supply if total_supply > 0 else 0
        )

        return TokenomicsMetrics(
            total_supply=total_supply,
            hhi=hhi,
            gini_coefficient=gini,
            top_10_concentration=top_10_concentration,
            holder_count=len(holders),
        )

    def calculate_token_velocity(
        self,
        transaction_volume: float,
        circulating_supply: float,
        period_days: int,
    ) -> float:
        """Calculate token velocity.

        Args:
            transaction_volume: Total transaction volume
            circulating_supply: Circulating supply
            period_days: Period in days

        Returns:
            Token velocity
        """
        if circulating_supply <= 0 or period_days <= 0:
            return 0.0
        return transaction_volume / circulating_supply

    def calculate_inflation_rate(
        self,
        initial_supply: float,
        current_supply: float,
        period_years: float,
    ) -> float:
        """Calculate token inflation rate.

        Args:
            initial_supply: Initial token supply
            current_supply: Current token supply
            period_years: Period in years

        Returns:
            Annual inflation rate
        """
        if initial_supply <= 0 or period_years <= 0:
            return 0.0
        return (current_supply - initial_supply) / initial_supply / period_years

    def analyze_vesting_schedule(self, vesting_data: dict) -> dict:
        """Analyze token vesting schedule.

        Args:
            vesting_data: Vesting schedule data

        Returns:
            Dict with vesting metrics
        """
        total = vesting_data.get("total_allocated", 0)
        claimed = vesting_data.get("claimed", 0)

        return {
            "vesting_progress": claimed / total if total > 0 else 0,
            "unvested": total - claimed,
            "vested": claimed,
            "total_allocated": total,
        }

    # ─── Consensus Analysis ──────────────────────────────────────────────

    def analyze_validator_distribution(
        self,
        validators: list[dict],
        consensus_type: ConsensusType,
    ) -> ConsensusMetrics:
        """Analyze validator distribution for PoS.

        Args:
            validators: List of validator data
            consensus_type: Consensus type

        Returns:
            ConsensusMetrics with validator statistics
        """
        if not validators:
            raise ValueError("No validators provided")

        total_stake = sum(v["stake"] for v in validators)
        commissions = [v.get("commission", 0) for v in validators]
        uptimes = [v.get("uptime", 0) for v in validators]

        return ConsensusMetrics(
            consensus_type=consensus_type,
            total_validators=len(validators),
            total_stake=total_stake,
            nakamoto_coefficient=self.calculate_nakamoto_coefficient(validators),
            avg_commission=sum(commissions) / len(commissions) if commissions else 0,
            avg_uptime=sum(uptimes) / len(uptimes) if uptimes else 0,
        )

    def calculate_nakamoto_coefficient(self, validators: list[dict]) -> int:
        """Calculate Nakamoto coefficient.

        The minimum number of entities that control >50% of the stake/hashrate.

        Args:
            validators: List of validator/miner data

        Returns:
            Nakamoto coefficient
        """
        if not validators:
            return 0

        sorted_validators = sorted(
            validators, key=lambda v: v.get("stake", v.get("hashrate", 0)), reverse=True
        )
        total = sum(v.get("stake", v.get("hashrate", 0)) for v in sorted_validators)

        if total <= 0:
            return 0

        cumulative = 0
        for i, v in enumerate(sorted_validators):
            cumulative += v.get("stake", v.get("hashrate", 0))
            if cumulative > total / 2:
                return i + 1

        return len(sorted_validators)

    def analyze_mining_distribution(
        self,
        miners: list[dict],
        consensus_type: ConsensusType,
    ) -> ConsensusMetrics:
        """Analyze mining distribution for PoW.

        Args:
            miners: List of miner data
            consensus_type: Consensus type

        Returns:
            ConsensusMetrics with mining statistics
        """
        if not miners:
            raise ValueError("No miners provided")

        total_hashrate = sum(m["hashrate"] for m in miners)

        return ConsensusMetrics(
            consensus_type=consensus_type,
            total_validators=len(miners),
            total_stake=total_hashrate,
            nakamoto_coefficient=self.calculate_nakamoto_coefficient(miners),
        )

    def estimate_finality_time(
        self,
        consensus_type: ConsensusType,
        block_time: float,
        validator_count: int = 0,
        confirmations: int = 0,
    ) -> float:
        """Estimate finality time.

        Args:
            consensus_type: Consensus type
            block_time: Average block time in seconds
            validator_count: Number of validators (for PoS)
            confirmations: Number of confirmations (for PoW)

        Returns:
            Estimated finality time in seconds
        """
        if consensus_type == ConsensusType.POW:
            return float(block_time * confirmations)
        elif consensus_type == ConsensusType.POS:
            # Simplified: 2 epochs for finality
            epoch_time = block_time * 32
            return float(epoch_time * 2)
        else:
            return float(block_time * 12)

    def calculate_staking_apy(
        self,
        total_stake: float,
        annual_rewards: float,
        commission: float,
    ) -> float:
        """Calculate staking APY.

        Args:
            total_stake: Total staked amount
            annual_rewards: Annual rewards
            commission: Validator commission rate

        Returns:
            Net staking APY
        """
        if total_stake <= 0:
            return 0.0
        gross_apy = annual_rewards / total_stake
        return gross_apy * (1 - commission)

    # ─── On-Chain Analytics ──────────────────────────────────────────────

    def analyze_wallet_activity(self, address: str) -> dict:
        """Analyze wallet activity.

        Args:
            address: Wallet address

        Returns:
            Dict with wallet activity metrics
        """
        if not self.w3:
            return {"transaction_count": 0, "balance": 0, "is_contract": False}

        tx_count = self.w3.eth.get_transaction_count(address)
        balance = self.w3.eth.get_balance(address)
        code = self.w3.eth.get_code(address)

        return {
            "transaction_count": tx_count,
            "balance": balance,
            "is_contract": len(code) > 0,
        }

    def detect_whale_wallets(
        self, holders: list[dict], threshold_percent: float
    ) -> list[dict]:
        """Detect whale wallets above a threshold.

        Args:
            holders: List of token holder data
            threshold_percent: Minimum percentage to be considered a whale

        Returns:
            List of whale wallets
        """
        return [h for h in holders if h.get("percent", 0) >= threshold_percent]

    # ─── Gas Analytics ───────────────────────────────────────────────────

    def analyze_gas_trends(self, blocks: list[dict]) -> dict:
        """Analyze gas usage trends.

        Args:
            blocks: List of block data

        Returns:
            Dict with gas trend metrics
        """
        if not blocks:
            return {"avg_gas_used": 0, "gas_utilization": 0, "gas_efficiency": 0}

        gas_used = [b["gasUsed"] for b in blocks]
        gas_limits = [b["gasLimit"] for b in blocks]
        utilizations = [g / lim for g, lim in zip(gas_used, gas_limits)]

        return {
            "avg_gas_used": sum(gas_used) / len(gas_used),
            "gas_utilization": sum(utilizations) / len(utilizations),
            "gas_efficiency": sum(gas_used) / sum(gas_limits) if sum(gas_limits) > 0 else 0,
        }

    def estimate_transaction_cost(
        self,
        gas_used: int,
        gas_price_gwei: float,
        eth_price: float,
    ) -> float:
        """Estimate transaction cost in USD.

        Args:
            gas_used: Gas units used
            gas_price_gwei: Gas price in Gwei
            eth_price: ETH price in USD

        Returns:
            Transaction cost in USD
        """
        return gas_used * gas_price_gwei * 1e-9 * eth_price

    # ─── Network Health ──────────────────────────────────────────────────

    def calculate_network_health_score(
        self, blocks: list[dict], transactions: list[dict]
    ) -> float:
        """Calculate overall network health score.

        Args:
            blocks: List of block data
            transactions: List of transaction data

        Returns:
            Health score between 0 and 100
        """
        if not blocks:
            return 0.0

        # Gas utilization (lower is better for health)
        gas_utils = [b["gasUsed"] / b["gasLimit"] for b in blocks]
        avg_gas_util = sum(gas_utils) / len(gas_utils)
        gas_score = max(0, 100 - avg_gas_util * 100)

        # Transaction throughput
        tx_score = min(100, len(transactions) * 2)

        # Block time consistency
        timestamps = [b["timestamp"] for b in blocks]
        block_times = [
            timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
        ]
        if block_times:
            avg_bt = sum(block_times) / len(block_times)
            variance = sum((bt - avg_bt) ** 2 for bt in block_times) / len(block_times)
            consistency_score = max(0, 100 - variance)
        else:
            consistency_score = 100

        return (gas_score + tx_score + consistency_score) / 3

    def detect_network_congestion(self, blocks: list[dict]) -> dict:
        """Detect network congestion.

        Args:
            blocks: List of block data

        Returns:
            Dict with congestion metrics
        """
        if not blocks:
            return {"is_congested": False, "avg_utilization": 0}

        utilizations = [b["gasUsed"] / b["gasLimit"] for b in blocks]
        avg_util = sum(utilizations) / len(utilizations)

        return {
            "is_congested": avg_util > 0.9,
            "avg_utilization": avg_util,
        }

    # ─── MEV Detection ───────────────────────────────────────────────────

    def detect_sandwich_attack(self, transactions: list[dict]) -> dict:
        """Detect sandwich attack patterns.

        Args:
            transactions: List of transaction data

        Returns:
            Dict with detection results
        """
        if len(transactions) < 3:
            return {"is_sandwich": False, "attacker": None}

        # Look for pattern: victim tx, attacker front-run, attacker back-run
        for i in range(len(transactions) - 2):
            victim = transactions[i]
            front_run = transactions[i + 1]
            back_run = transactions[i + 2]

            # Same attacker address for front-run and back-run
            if front_run.get("from") == back_run.get("from"):
                # Front-run has higher gas price
                if front_run.get("gasPrice", 0) > victim.get("gasPrice", 0):
                    # Back-run has lower gas price
                    if back_run.get("gasPrice", 0) < front_run.get("gasPrice", 0):
                        return {
                            "is_sandwich": True,
                            "attacker": front_run.get("from"),
                            "victim": victim.get("from"),
                        }

        return {"is_sandwich": False, "attacker": None}

    # ─── Cross-Chain Analytics ───────────────────────────────────────────

    def compare_chain_metrics(
        self, chain_a: dict, chain_b: dict
    ) -> dict:
        """Compare metrics between two chains.

        Args:
            chain_a: Metrics for chain A
            chain_b: Metrics for chain B

        Returns:
            Dict with comparison ratios
        """
        tvl_a = chain_a.get("tvl", 0)
        tvl_b = chain_b.get("tvl", 0)
        tx_a = chain_a.get("tx_count", 0)
        tx_b = chain_b.get("tx_count", 0)

        return {
            "tvl_ratio": tvl_a / tvl_b if tvl_b > 0 else 0,
            "tx_ratio": tx_a / tx_b if tx_b > 0 else 0,
        }

    def estimate_bridge_risk(
        self,
        bridge_tvl: float,
        daily_volume: float,
        validator_count: int,
        threshold: int,
    ) -> dict:
        """Estimate bridge risk.

        Args:
            bridge_tvl: Bridge TVL
            daily_volume: Daily volume
            validator_count: Number of validators
            threshold: Minimum validator threshold

        Returns:
            Dict with risk assessment
        """
        # Validator risk
        validator_risk = validator_count < threshold

        # Volume risk
        volume_risk = daily_volume / bridge_tvl > 0.5 if bridge_tvl > 0 else True

        # Overall risk level
        risk_score = sum([validator_risk, volume_risk])
        if risk_score == 0:
            risk_level = "low"
        elif risk_score == 1:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "risk_level": risk_level,
            "validator_risk": validator_risk,
            "volume_risk": volume_risk,
        }

    # ─── Smart Contract Risk ─────────────────────────────────────────────

    def assess_contract_risk(self, contract_data: dict) -> dict:
        """Assess smart contract risk.

        Args:
            contract_data: Contract data

        Returns:
            Dict with risk assessment
        """
        score = 0

        # Verification
        if not contract_data.get("is_verified", False):
            score += 20

        # Audits
        audit_count = contract_data.get("audit_count", 0)
        if audit_count == 0:
            score += 20
        elif audit_count == 1:
            score += 10

        # Age
        age_days = contract_data.get("age_days", 0)
        if age_days < 30:
            score += 20
        elif age_days < 90:
            score += 10

        # TVL
        tvl = contract_data.get("tvl", 0)
        if tvl < 1_000_000:
            score += 15
        elif tvl < 10_000_000:
            score += 5

        # Ownership
        if not contract_data.get("has_ownership_renounced", False):
            score += 15

        # Timelock
        if not contract_data.get("has_timelock", False):
            score += 10

        # Risk level
        if score < 30:
            risk_level = "low"
        elif score < 70:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "risk_score": score,
            "risk_level": risk_level,
        }

    # ─── Oracle Analytics ────────────────────────────────────────────────

    def analyze_oracle_deviation(
        self, prices: list[float], reference_price: float
    ) -> dict:
        """Analyze oracle price deviation.

        Args:
            prices: List of oracle prices
            reference_price: Reference price

        Returns:
            Dict with deviation metrics
        """
        if not prices or reference_price <= 0:
            return {"max_deviation": 0, "avg_deviation": 0}

        deviations = [abs(p - reference_price) / reference_price for p in prices]

        return {
            "max_deviation": max(deviations),
            "avg_deviation": sum(deviations) / len(deviations),
        }

    def detect_oracle_manipulation(
        self, prices: list[float], threshold: float
    ) -> dict:
        """Detect oracle manipulation.

        Args:
            prices: List of oracle prices
            threshold: Deviation threshold

        Returns:
            Dict with detection results
        """
        if len(prices) < 2:
            return {"is_manipulation": False, "manipulation_index": None}

        for i in range(1, len(prices)):
            prev_price = prices[i - 1]
            curr_price = prices[i]

            if prev_price > 0:
                deviation = abs(curr_price - prev_price) / prev_price
                if deviation > threshold:
                    return {
                        "is_manipulation": True,
                        "manipulation_index": i,
                    }

        return {"is_manipulation": False, "manipulation_index": None}

    # ─── Governance Analytics ────────────────────────────────────────────

    def analyze_governance_participation(
        self, proposals: list[dict]
    ) -> dict:
        """Analyze governance participation.

        Args:
            proposals: List of proposal data

        Returns:
            Dict with participation metrics
        """
        if not proposals:
            return {"avg_participation": 0, "avg_approval_rate": 0}

        participations = []
        approval_rates = []

        for p in proposals:
            total_votes = p.get("votes_for", 0) + p.get("votes_against", 0)
            total_supply = p.get("total_supply", 0)

            if total_supply > 0:
                participations.append(total_votes / total_supply)

            if total_votes > 0:
                approval_rates.append(p.get("votes_for", 0) / total_votes)

        return {
            "avg_participation": (
                sum(participations) / len(participations) if participations else 0
            ),
            "avg_approval_rate": (
                sum(approval_rates) / len(approval_rates) if approval_rates else 0
            ),
        }

    def calculate_voting_power_concentration(
        self, voters: list[dict]
    ) -> float:
        """Calculate voting power concentration (HHI).

        Args:
            voters: List of voter data

        Returns:
            HHI score between 0 and 1
        """
        if not voters:
            return 0.0

        total_votes = sum(v.get("votes", 0) for v in voters)
        if total_votes <= 0:
            return 0.0

        return sum((v.get("votes", 0) / total_votes) ** 2 for v in voters)
