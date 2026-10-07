"""API Gateway — exposes all engines via REST API with auth and rate limiting."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.altdata.engine import AlternativeDataEngine
from src.marketmaking.engine import MarketMakingEngine
from src.regtech.compliance import (
    ComplianceEngine,
    ComplianceFramework,
    ComplianceStatus,
)
from src.tokenization.rwa import RWATokenizer


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        # Clean old entries
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > window_start
        ]
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        self._requests[client_id].append(now)
        return True

    def reset(self, client_id: str) -> None:
        self._requests.pop(client_id, None)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QuoteRequest(BaseModel):
    mid_price: float = Field(gt=0)
    inventory: int = 0
    time_remaining: float = Field(ge=0)
    gamma: float = Field(default=0.1, gt=0)
    sigma: float = Field(default=0.5, gt=0)
    k: float = Field(default=1.5, gt=0)
    A: float = Field(default=0.1, gt=0)
    dt: float = Field(default=1.0, gt=0)
    max_inventory: int = Field(default=10, gt=0)
    adverse_selection: float = 0.0


class BatchQuoteRequest(BaseModel):
    mid_prices: list[float]
    inventories: list[int]
    times_remaining: list[float]
    gamma: float = Field(default=0.1, gt=0)
    sigma: float = Field(default=0.5, gt=0)
    k: float = Field(default=1.5, gt=0)
    A: float = Field(default=0.1, gt=0)
    dt: float = Field(default=1.0, gt=0)
    max_inventory: int = Field(default=10, gt=0)
    adverse_selection: float = 0.0


class IngestRequest(BaseModel):
    ticker: str
    data: dict[str, Any]


class InvestorRegisterRequest(BaseModel):
    address: str
    jurisdiction: str
    accredited: bool = False


class TransferRequest(BaseModel):
    from_address: str
    to_address: str
    amount: int = Field(gt=0)


class MintRequest(BaseModel):
    address: str
    amount: int = Field(gt=0)


class ControlRegisterRequest(BaseModel):
    framework: str
    control_id: str
    name: str
    description: str


class AssessRequest(BaseModel):
    status: str
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# API Gateway
# ---------------------------------------------------------------------------


class APIGateway:
    """API Gateway that exposes all engines via REST API."""

    def __init__(
        self,
        api_key: str = "default-key",
        rate_limit: int = 100,
        rate_window: int = 60,
    ):
        self.api_key = api_key
        self.rate_limiter = RateLimiter(
            max_requests=rate_limit, window_seconds=rate_window
        )
        # Engine instances
        self._marketmaking_engine: Optional[MarketMakingEngine] = None
        self.altdata_engine = AlternativeDataEngine()
        self.tokenization_engine = RWATokenizer(
            w3=None,
            owner_address="0x0000000000000000000000000000000000000000",
            token_name="ApexRWA",
            token_symbol="ARWA",
            initial_supply=1_000_000,
            asset_type="real_estate",
            asset_value=100_000_000,
            jurisdiction="US",
        )
        self.regtech_engine = ComplianceEngine()

    @property
    def marketmaking_engine(self) -> MarketMakingEngine:
        if self._marketmaking_engine is None:
            self._marketmaking_engine = MarketMakingEngine(
                gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10
            )
        return self._marketmaking_engine

    def _check_auth(self, request: Request) -> None:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    def _check_rate_limit(self, request: Request) -> None:
        client_id = request.client.host if request.client else "unknown"
        if not self.rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

    def get_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="Apex Fintech Platform API",
            description="API Gateway for all fintech engines",
            version="0.1.0",
        )

        @app.middleware("http")
        async def auth_and_rate_limit(request: Request, call_next):
            # Skip auth for health endpoint
            if request.url.path == "/health":
                return await call_next(request)
            # Check auth
            api_key = request.headers.get("X-API-Key")
            if not api_key or api_key != self.api_key:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
            # Check rate limit
            client_id = request.client.host if request.client else "unknown"
            if not self.rate_limiter.is_allowed(client_id):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )
            return await call_next(request)

        # ─── Health ────────────────────────────────────────────────────

        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            return JSONResponse(
                status_code=400,
                content={"detail": str(exc)},
            )

        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "version": "0.1.0",
                "engines": {
                    "marketmaking": {"status": "available", "endpoints": ["/marketmaking/quotes", "/marketmaking/quotes/batch"]},
                    "altdata": {"status": "available", "endpoints": ["/altdata/satellite/{ticker}", "/altdata/sentiment/{ticker}", "/altdata/transactions/{ticker}", "/altdata/composite/{ticker}"]},
                    "tokenization": {"status": "available", "endpoints": ["/tokenization/investors", "/tokenization/transfer"]},
                    "regtech": {"status": "available", "endpoints": ["/regtech/controls", "/regtech/report"]},
                },
            }

        # ─── Engine Discovery ──────────────────────────────────────────

        @app.get("/engines")
        async def list_engines():
            return {
                "marketmaking": {
                    "status": "available",
                    "endpoints": ["/marketmaking/quotes", "/marketmaking/quotes/batch"],
                },
                "altdata": {
                    "status": "available",
                    "endpoints": [
                        "/altdata/satellite/{ticker}",
                        "/altdata/sentiment/{ticker}",
                        "/altdata/transactions/{ticker}",
                        "/altdata/composite/{ticker}",
                    ],
                },
                "tokenization": {
                    "status": "available",
                    "endpoints": ["/tokenization/investors", "/tokenization/transfer"],
                },
                "regtech": {
                    "status": "available",
                    "endpoints": ["/regtech/controls", "/regtech/report"],
                },
            }

        # ─── Market Making ─────────────────────────────────────────────

        @app.post("/marketmaking/quotes")
        async def compute_quotes(req: QuoteRequest):
            try:
                engine = MarketMakingEngine(
                    gamma=req.gamma,
                    sigma=req.sigma,
                    k=req.k,
                    A=req.A,
                    dt=req.dt,
                    max_inventory=req.max_inventory,
                    adverse_selection=req.adverse_selection,
                )
                quote = engine.compute_quotes(
                    mid_price=req.mid_price,
                    inventory=req.inventory,
                    time_remaining=req.time_remaining,
                )
                return {
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "mid": quote.mid,
                    "spread": quote.spread,
                    "reservation_price": quote.reservation_price,
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.post("/marketmaking/quotes/batch")
        async def compute_quotes_batch(req: BatchQuoteRequest):
            if not (len(req.mid_prices) == len(req.inventories) == len(req.times_remaining)):
                raise HTTPException(
                    status_code=400, detail="All input arrays must have the same length"
                )
            try:
                engine = MarketMakingEngine(
                    gamma=req.gamma,
                    sigma=req.sigma,
                    k=req.k,
                    A=req.A,
                    dt=req.dt,
                    max_inventory=req.max_inventory,
                    adverse_selection=req.adverse_selection,
                )
                quotes = engine.compute_quotes_batch(
                    mid_prices=req.mid_prices,
                    inventories=req.inventories,
                    times_remaining=req.times_remaining,
                )
                return [
                    {
                        "bid": q.bid,
                        "ask": q.ask,
                        "mid": q.mid,
                        "spread": q.spread,
                        "reservation_price": q.reservation_price,
                    }
                    for q in quotes
                ]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # ─── AltData ───────────────────────────────────────────────────

        @app.get("/altdata/satellite/{ticker}")
        async def satellite_analysis(ticker: str):
            return self.altdata_engine.analyze_satellite_trends(ticker)

        @app.get("/altdata/sentiment/{ticker}")
        async def sentiment_analysis(ticker: str):
            return self.altdata_engine.analyze_sentiment_trends(ticker)

        @app.get("/altdata/transactions/{ticker}")
        async def transaction_analysis(ticker: str):
            return self.altdata_engine.analyze_transaction_trends(ticker)

        @app.get("/altdata/composite/{ticker}")
        async def composite_signal(ticker: str):
            return self.altdata_engine.generate_composite_signal(ticker)

        @app.post("/altdata/ingest/satellite")
        async def ingest_satellite(req: IngestRequest):
            data = {**req.data, "ticker": req.ticker}
            self.altdata_engine.ingest_satellite_data([data])
            return {"status": "ingested", "ticker": req.ticker}

        @app.post("/altdata/ingest/sentiment")
        async def ingest_sentiment(req: IngestRequest):
            data = {**req.data, "ticker": req.ticker}
            self.altdata_engine.ingest_sentiment_data([data])
            return {"status": "ingested", "ticker": req.ticker}

        @app.post("/altdata/ingest/transactions")
        async def ingest_transactions(req: IngestRequest):
            data = {**req.data, "ticker": req.ticker}
            self.altdata_engine.ingest_transaction_data([data])
            return {"status": "ingested", "ticker": req.ticker}

        # ─── Tokenization ──────────────────────────────────────────────

        @app.post("/tokenization/investors")
        async def register_investor(req: InvestorRegisterRequest):
            self.tokenization_engine.register_investor(
                investor_address=req.address,
                jurisdiction=req.jurisdiction,
                accredited=req.accredited,
            )
            return {
                "address": req.address,
                "status": "pending",
                "jurisdiction": req.jurisdiction,
            }

        @app.post("/tokenization/investors/{address}/verify")
        async def verify_investor(address: str):
            try:
                self.tokenization_engine.verify_investor(address)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return {"address": address, "status": "verified"}

        @app.get("/tokenization/investors/{address}/compliance")
        async def check_compliance(address: str):
            status = self.tokenization_engine.check_compliance(address)
            return {"address": address, "status": status.value}

        @app.get("/tokenization/investors/{address}/balance")
        async def get_balance(address: str):
            balance = self.tokenization_engine.get_balance(address)
            return {"address": address, "balance": balance}

        @app.post("/tokenization/transfer")
        async def transfer_tokens(req: TransferRequest):
            success = self.tokenization_engine.transfer(
                from_address=req.from_address,
                to_address=req.to_address,
                amount=req.amount,
            )
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail="Transfer failed — check compliance and balances",
                )
            return {"success": True, "from": req.from_address, "to": req.to_address, "amount": req.amount}

        @app.post("/tokenization/mint")
        async def mint_tokens(req: MintRequest):
            self.tokenization_engine.balances[req.address] = (
                self.tokenization_engine.balances.get(req.address, 0) + req.amount
            )
            return {"address": req.address, "minted": req.amount}

        # ─── RegTech ───────────────────────────────────────────────────

        @app.post("/regtech/controls")
        async def register_control(req: ControlRegisterRequest):
            try:
                framework = ComplianceFramework(req.framework)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid framework. Must be one of: {[f.value for f in ComplianceFramework]}",
                )
            try:
                control = self.regtech_engine.register_control(
                    framework=framework,
                    control_id=req.control_id,
                    name=req.name,
                    description=req.description,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {
                "control_id": control.control_id,
                "framework": control.framework.value,
                "name": control.name,
                "status": control.status.value,
            }

        @app.get("/regtech/controls")
        async def list_controls():
            return [
                {
                    "control_id": c.control_id,
                    "framework": c.framework.value,
                    "name": c.name,
                    "status": c.status.value,
                }
                for c in self.regtech_engine.controls.values()
            ]

        @app.post("/regtech/controls/{control_id}/assess")
        async def assess_control(control_id: str, req: AssessRequest):
            try:
                status = ComplianceStatus(req.status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of: {[s.value for s in ComplianceStatus]}",
                )
            try:
                self.regtech_engine.assess_control(control_id, status, req.notes)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))
            return {"control_id": control_id, "status": status.value}

        @app.get("/regtech/report")
        async def generate_report():
            report = self.regtech_engine.generate_report()
            return {
                "total_controls": report.total_controls,
                "compliant_count": report.compliant_count,
                "non_compliant_count": report.non_compliant_count,
                "pending_count": report.pending_count,
                "not_assessed_count": report.not_assessed_count,
                "framework_breakdown": {
                    fw.value: counts
                    for fw, counts in report.framework_breakdown.items()
                },
            }

        return app
