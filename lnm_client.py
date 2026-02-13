import asyncio
import logging
from typing import Any

from lnmarkets_sdk.v3.http.client import APIAuthContext, APIClientConfig, LNMClient

logging.basicConfig(level=logging.INFO)


class lnm_client:
    """LN Markets SDK v3 client wrapper used by live strategies."""

    def __init__(self, options: dict[str, Any] | None):
        self.options = options or {}
        self._loop = asyncio.new_event_loop()
        self._entered = False
        self.lnm = self._build_client()
        self.get_running_trades()  # auth check
        logging.info("Connection to LN Markets SDK v3 ok!")

    def __del__(self):
        try:
            if self._entered and hasattr(self.lnm, "__aexit__"):
                self._run(self.lnm.__aexit__(None, None, None))
            if hasattr(self, "_loop") and not self._loop.is_closed():
                self._loop.close()
        except Exception:
            pass

    def _build_client(self):
        config = APIClientConfig(
            authentication=APIAuthContext(
                key=self.options.get("key"),
                secret=self.options.get("secret"),
                passphrase=self.options.get("passphrase"),
            ),
            network=str(self.options.get("network", "mainnet")).lower(),
        )
        client = LNMClient(config)
        if hasattr(client, "__aenter__"):
            self._run(client.__aenter__())
            self._entered = True
        return client

    def _run(self, maybe_awaitable):
        if asyncio.iscoroutine(maybe_awaitable):
            asyncio.set_event_loop(self._loop)
            try:
                return self._loop.run_until_complete(maybe_awaitable)
            finally:
                asyncio.set_event_loop(None)
        return maybe_awaitable

    def _to_object(self, payload):
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "dict"):
            return payload.dict()
        return payload

    def _get_path_callable(self, path: str):
        target = self.lnm
        for attr in path.split("."):
            if not hasattr(target, attr):
                return None
            target = getattr(target, attr)
        return target if callable(target) else None

    def _call_path(self, paths: list[str], *args, **kwargs):
        for path in paths:
            method = self._get_path_callable(path)
            if method is None:
                continue
            return self._to_object(self._run(method(*args, **kwargs)))
        raise RuntimeError(f"Missing SDK method. Tried: {paths}")

    def _call_with_variants(self, paths: list[str], variants: list[dict[str, Any]]):
        last_error = None
        for kwargs in variants:
            try:
                return self._call_path(paths, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"No compatible SDK signature for {paths}: {last_error}") from last_error
        raise RuntimeError(f"No compatible SDK method. Tried: {paths}")

    @staticmethod
    def _normalize_trade(trade: dict[str, Any]) -> dict[str, Any]:
        if "id" not in trade:
            for key in ("trade_id", "position_id", "uid"):
                if key in trade:
                    trade["id"] = trade[key]
                    break
        return trade

    def get_ticker(self) -> dict[str, Any]:
        ticker = self._call_path(["futures.get_ticker", "futures.isolated.get_ticker"])
        return ticker if isinstance(ticker, dict) else {"raw": ticker}

    def get_last_price(self) -> float:
        ticker = self.get_ticker()
        for key in ("lastPrice", "last_price", "last", "price", "mark_price"):
            if key in ticker:
                return float(ticker[key])
        raise KeyError(f"Ticker response has no last price field: {ticker}")

    def get_last(self) -> dict[str, Any]:
        return self.get_ticker()

    def _new_trade(self, side: str, quantity, leverage, takeprofit, stoploss):
        side_value = "b" if side == "long" else "s"
        payload = self._call_with_variants(
            ["futures.isolated.new_trade", "futures.new_trade"],
            [
                {
                    "side": side_value,
                    "type": "m",
                    "quantity": quantity,
                    "leverage": leverage,
                    "takeprofit": takeprofit,
                    "stoploss": stoploss,
                },
                {
                    "side": side,
                    "type": "market",
                    "quantity": quantity,
                    "leverage": leverage,
                    "takeprofit": takeprofit,
                    "stoploss": stoploss,
                },
                {
                    "side": side,
                    "quantity": quantity,
                    "leverage": leverage,
                    "takeprofit": takeprofit,
                    "stoploss": stoploss,
                },
            ],
        )
        return payload if isinstance(payload, dict) else {"raw": payload}

    def market_long(self, quantity, leverage, takeprofit, stoploss) -> dict[str, Any]:
        return self._normalize_trade(
            self._new_trade(
                side="long",
                quantity=quantity,
                leverage=leverage,
                takeprofit=takeprofit,
                stoploss=stoploss,
            )
        )

    def market_short(self, quantity, leverage, takeprofit, stoploss) -> dict[str, Any]:
        return self._normalize_trade(
            self._new_trade(
                side="short",
                quantity=quantity,
                leverage=leverage,
                takeprofit=takeprofit,
                stoploss=stoploss,
            )
        )

    def close_position(self, operation_id):
        return self._call_with_variants(
            ["futures.isolated.close_trade", "futures.close_trade"],
            [{"id": operation_id}, {"trade_id": operation_id}],
        )

    def get_running_trades(self) -> list[dict[str, Any]]:
        payload = self._call_path(
            ["futures.isolated.get_running_trades", "futures.get_running_trades"]
        )
        if not isinstance(payload, list):
            return []
        return [self._normalize_trade(p) if isinstance(p, dict) else {"raw": p} for p in payload]

    def get_closed_trades(self) -> list[dict[str, Any]]:
        payload = self._call_path(
            ["futures.isolated.get_closed_trades", "futures.get_closed_trades"]
        )
        if not isinstance(payload, list):
            return []
        return [self._normalize_trade(p) if isinstance(p, dict) else {"raw": p} for p in payload]

    def get_trades(self, type_trade):
        trade_type = str(type_trade).lower()
        if trade_type == "running":
            return self.get_running_trades()
        if trade_type == "closed":
            return self.get_closed_trades()
        raise ValueError(f"Unsupported trade type: {type_trade}")
