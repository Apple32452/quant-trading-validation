"""Reproducible educational models for blackjack, option pricing, and backtests.

The card-game win probabilities are synthetic, not actual casino odds.
The backtesting demonstration uses synthetic returns, not real market data.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np


@dataclass(frozen=True)
class BlackjackSimulation:
    true_counts: np.ndarray
    adaptive_cumulative: np.ndarray
    flat_cumulative: np.ndarray

    @property
    def adaptive_final(self) -> np.ndarray:
        return self.adaptive_cumulative[:, -1]

    @property
    def flat_final(self) -> np.ndarray:
        return self.flat_cumulative[:, -1]


def simulate_blackjack(
    n_sessions: int = 1_000,
    n_decks: int = 430,
    *,
    seed: int = 6,
    batch_decks: int = 32,
    bet_threshold: float = 1.0,
    low_bet: int = 50,
    high_bet: int = 500,
) -> BlackjackSimulation:
    """Toy Hi-Lo simulation at 5-card checkpoints, with paired flat-bet baseline.

    Each shuffled 52-card deck has 20 low (+1), 12 neutral (0), and
    20 high (-1) cards. At checkpoints of 5, 10, ..., 35 revealed cards:

      TC = running_count / (remaining_cards / 52)
      P(win) = clip(0.5 + (TC - 1) / 400, 0, 1)

    Outcomes are even-money wins/losses, omitting pushes, natural blackjacks,
    doubles, splits, dealer rules and all other casino-specific mechanics.
    """
    if min(n_sessions, n_decks, batch_decks, low_bet, high_bet) <= 0:
        raise ValueError("sessions, decks, batch size and bets must be positive")

    rng = np.random.default_rng(seed)
    deck = np.repeat(np.array([1, 0, -1], dtype=np.int8), [20, 12, 20])
    checkpoints = np.arange(5, 36, 5)
    n_checks = checkpoints.size
    remaining = 52 - checkpoints
    true_counts = np.empty((n_sessions, n_decks, n_checks), dtype=np.float64)
    adaptive_profits = np.empty((n_sessions, n_decks, n_checks), dtype=np.int32)
    flat_profits = np.empty((n_sessions, n_decks, n_checks), dtype=np.int32)

    # Keep peak memory bounded even for the original 1,000 x 430 scenario.
    for start in range(0, n_decks, batch_decks):
        stop = min(start + batch_decks, n_decks)
        batch = np.broadcast_to(deck, (n_sessions, stop - start, 52))
        shuffled = rng.permuted(batch, axis=-1)
        running = shuffled.cumsum(axis=-1, dtype=np.int16)
        tc = running[..., checkpoints - 1] * (52.0 / remaining)
        p = np.clip(0.5 + (tc - 1.0) / 400.0, 0.0, 1.0)
        outcome = np.where(rng.random(p.shape) < p, 1, -1)
        bet = np.where(tc > bet_threshold, high_bet, low_bet)

        true_counts[:, start:stop, :] = tc
        adaptive_profits[:, start:stop, :] = outcome * bet
        flat_profits[:, start:stop, :] = outcome * low_bet

    shape = (n_sessions, n_decks * n_checks)
    return BlackjackSimulation(
        true_counts=true_counts,
        adaptive_cumulative=adaptive_profits.reshape(shape).cumsum(axis=1),
        flat_cumulative=flat_profits.reshape(shape).cumsum(axis=1),
    )


def black_scholes_closed_form_call(
    spot: float, strike: float, rate: float, volatility: float, maturity: float
) -> float:
    """Closed-form European call value without dividends."""
    if spot < 0 or strike <= 0 or volatility <= 0 or maturity < 0:
        raise ValueError("Invalid option parameters")
    if maturity == 0:
        return max(spot - strike, 0.0)
    if spot == 0:
        return 0.0
    sigma_sqrt_t = volatility * sqrt(maturity)
    d1 = (log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    norm_cdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return spot * norm_cdf(d1) - strike * exp(-rate * maturity) * norm_cdf(d2)


def black_scholes_finite_difference_call(
    strike: float = 100.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    maturity: float = 1.0,
    *,
    max_spot: float = 200.0,
    price_points: int = 401,
    time_steps: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit PDE solver, evolving time-to-maturity from payoff to today.

    dV/dtau = 0.5*sigma**2*S**2*V_SS + r*S*V_S - r*V
    with V(0,tau)=0 and V(Smax,tau) ~= Smax-K*exp(-r*tau).
    """
    if strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("strike, volatility and maturity must be positive")
    if max_spot <= strike or price_points < 3 or time_steps < 1:
        raise ValueError("Invalid spatial domain or grid resolution")

    spots = np.linspace(0.0, max_spot, price_points)
    ds = spots[1] - spots[0]
    dt = maturity / time_steps
    # Conservative stability condition for this explicit grid.
    bound = volatility**2 * max_spot**2 / ds**2 + abs(rate) * max_spot / ds + abs(rate)
    if dt * bound >= 1.0:
        raise ValueError("Increase time_steps: explicit finite-difference grid is unstable")

    values = np.maximum(spots - strike, 0.0)
    interior = spots[1:-1]
    diffusion = 0.5 * volatility**2 * interior**2
    drift = rate * interior

    for step in range(1, time_steps + 1):
        first = (values[2:] - values[:-2]) / (2.0 * ds)
        second = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / ds**2
        updated = values.copy()
        updated[1:-1] = values[1:-1] + dt * (
            diffusion * second + drift * first - rate * values[1:-1]
        )
        updated[0] = 0.0
        updated[-1] = max_spot - strike * exp(-rate * step * dt)
        values = updated

    return spots, values


def prices_to_daily_returns(prices: np.ndarray) -> np.ndarray:
    """Convert positive daily prices to simple daily returns."""
    values = np.asarray(prices, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("prices must be a 1D array with at least three observations")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("prices must be finite and strictly positive")
    return values[1:] / values[:-1] - 1.0


def annualized_sharpe(
    daily_returns: np.ndarray,
    *,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized sample Sharpe ratio; NaN if volatility is effectively zero.

    Daily returns should already be net of fees, slippage and financing costs.
    """
    values = np.asarray(daily_returns, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Provide at least two finite daily strategy returns")
    if annual_risk_free_rate <= -1.0 or periods_per_year <= 0:
        raise ValueError("Invalid annualization parameters")
    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = values - daily_rf
    volatility = float(np.std(excess, ddof=1))
    if volatility <= np.finfo(float).eps:
        return float("nan")
    return float(np.mean(excess) / volatility * sqrt(periods_per_year))


def chronological_split(
    daily_returns: np.ndarray, cut: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split ordered returns with no overlap; both parts need >=2 observations."""
    values = np.asarray(daily_returns, dtype=np.float64)
    if values.ndim != 1 or cut < 2 or cut > values.size - 2:
        raise ValueError("Both chronological segments need at least two returns")
    return values[:cut].copy(), values[cut:].copy()


StrategyRule = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BacktestResult:
    selected_rule: str
    training_sharpe: float
    holdout_sharpe: float
    all_training_scores: dict[str, float]
    training_periods: int
    holdout_periods: int


def select_and_evaluate_rule(
    daily_asset_returns: np.ndarray,
    rules: Mapping[str, StrategyRule],
    cut: int,
    *,
    annual_risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Select a strategy using training ONLY, then test the winner ONCE.

    Each rule must be causal, must avoid future information, and must output
    one daily strategy return for each daily asset return. Rules should
    handle transaction costs internally when used with real data.
    """
    if not rules:
        raise ValueError("Provide at least one candidate rule")
    training, holdout = chronological_split(daily_asset_returns, cut)

    def score_rule(rule: StrategyRule, segment: np.ndarray) -> float:
        daily_strategy_returns = np.asarray(rule(segment.copy()), dtype=np.float64)
        if daily_strategy_returns.shape != segment.shape:
            raise ValueError("Each rule must return the same number of daily returns")
        return annualized_sharpe(
            daily_strategy_returns, annual_risk_free_rate=annual_risk_free_rate
        )

    training_scores = {
        name: score_rule(rule, training) for name, rule in rules.items()
    }
    valid = {name: score for name, score in training_scores.items() if np.isfinite(score)}
    if not valid:
        raise ValueError("No candidate has a finite in-sample Sharpe ratio")
    selected = max(valid, key=valid.get)

    # No unselected candidate is evaluated on the holdout.
    holdout_sharpe = score_rule(rules[selected], holdout)
    return BacktestResult(
        selected_rule=selected,
        training_sharpe=training_scores[selected],
        holdout_sharpe=holdout_sharpe,
        all_training_scores=training_scores,
        training_periods=training.size,
        holdout_periods=holdout.size,
    )


def run_demo(model: str, *, sessions: int, decks: int) -> None:
    if model in ("blackjack", "all"):
        results = simulate_blackjack(n_sessions=sessions, n_decks=decks)
        print("\nToy blackjack simulation (not real casino odds)")
        print(f"Sessions: {sessions}, decks per session: {decks}")
        print(f"Adaptive mean final profit (USD): {np.mean(results.adaptive_final):,.2f}")
        print(f"Flat mean final profit (USD):     {np.mean(results.flat_final):,.2f}")
        print(
            f"Paired mean difference (USD):    "
            f"{np.mean(results.adaptive_final - results.flat_final):,.2f}"
        )

    if model in ("black-scholes", "all"):
        spots, values = black_scholes_finite_difference_call()
        numerical = float(np.interp(100.0, spots, values))
        analytical = black_scholes_closed_form_call(100.0, 100.0, 0.05, 0.2, 1.0)
        print("\nEuropean call option: S=K=100, r=5%, sigma=20%, T=1")
        print(f"Finite difference (USD): {numerical:.4f}")
        print(f"Closed form (USD):       {analytical:.4f}")
        print(f"Absolute error (USD):    {abs(numerical - analytical):.4f}")

    if model in ("backtest", "all"):
        rng = np.random.default_rng(6)
        daily_asset_returns = rng.normal(0.0002, 0.01, size=1_200)

        def buy_and_hold(r: np.ndarray) -> np.ndarray:
            return r

        def lagged_momentum(r: np.ndarray) -> np.ndarray:
            return np.r_[0.0, np.sign(r[:-1])] * r

        def lagged_contrarian(r: np.ndarray) -> np.ndarray:
            return -np.r_[0.0, np.sign(r[:-1])] * r

        rules = {
            "buy_and_hold": buy_and_hold,
            "lagged_momentum": lagged_momentum,
            "lagged_contrarian": lagged_contrarian,
        }
        chosen = select_and_evaluate_rule(daily_asset_returns, rules, cut=800)
        print("\nSynthetic daily-return backtest (no fees, no market data)")
        print(f"Selected rule: {chosen.selected_rule}")
        print(f"In-sample Sharpe: {chosen.training_sharpe:.3f}")
        print(f"Holdout Sharpe:   {chosen.holdout_sharpe:.3f}")

        # Pure noise has zero expected edge, but the best of many in-sample
        # strategies will often show an impressive training Sharpe.
        noise = rng.normal(0.0, 0.01, size=(1_200, 100))
        train_scores = np.array(
            [annualized_sharpe(noise[:800, j]) for j in range(noise.shape[1])]
        )
        winner = int(np.argmax(train_scores))
        print("Selection bias among 100 independent zero-edge strategies:")
        print(f"Best training Sharpe:   {train_scores[winner]:.3f}")
        print(f"Same strategy holdout: {annualized_sharpe(noise[800:, winner]):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", choices=("blackjack", "black-scholes", "backtest", "all"),
        default="all",
    )
    parser.add_argument("--sessions", type=int, default=1_000)
    parser.add_argument("--decks", type=int, default=430)
    args = parser.parse_args()
    run_demo(args.model, sessions=args.sessions, decks=args.decks)


if __name__ == "__main__":
    main()
