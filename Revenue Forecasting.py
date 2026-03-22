"""
Monthly Sales Forecasting System
=================================
Predicts month-end results based on daily actuals + historical patterns.

HOW TO READ THIS FILE:
  The code is organized in the order things actually happen:
  
  Section 1 → Your data (the only part you edit)
  Section 2 → The data container (where results are stored)
  Section 3 → The main function that runs everything (start here to understand the flow)
  Section 4 → Each helper function, in the order it gets called
  Section 5 → The report printer
  Section 6 → The entry point (the single line that starts it all)
"""

import datetime
import calendar
from dataclasses import dataclass


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: YOUR DATA (edit this part only)                      ║
# ╚══════════════════════════════════════════════════════════════════╝

# What month are we forecasting?
YEAR = 2026
MONTH = 3

# What are your targets for this month?
MONTHLY_TARGET = {
    "revenue": 500_000,       # total revenue goal
    "orders": 2_000,          # total orders goal
    "ad_spend": 80_000,       # ad budget cap
    "crm_clicks": 15_000,     # CRM engagement goal
}

# What does a "normal" day look like for each weekday?
# (Calculate this from your past 3-6 months of data)
# Keys: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday,
#        4=Friday, 5=Saturday, 6=Sunday
HISTORICAL_DAILY_AVG = {
    "revenue": {
        0: 15_200, 1: 14_800, 2: 16_500, 3: 17_100,
        4: 19_800, 5: 21_500, 6: 18_000,
    },
    "orders": {
        0: 62, 1: 58, 2: 67, 3: 70,
        4: 80, 5: 88, 6: 73,
    },
    "ad_spend": {
        0: 2_600, 1: 2_500, 2: 2_700, 3: 2_800,
        4: 3_000, 5: 3_200, 6: 2_900,
    },
    "crm_clicks": {
        0: 480, 1: 450, 2: 510, 3: 530,
        4: 600, 5: 650, 6: 550,
    },
}

# Your actual daily numbers — add one new row each day
# Format: (day_of_month, revenue, orders, ad_spend, crm_clicks)
DAILY_ACTUALS = [
    (1,    18_200,   75,     2_800,    520),
    (2,    16_500,   68,     2_600,    490),
    (3,    15_800,   63,     2_500,    470),
    (4,    17_900,   72,     2_750,    530),
    (5,    20_100,   82,     3_100,    610),
    (6,    22_300,   90,     3_300,    680),
    (7,    19_000,   77,     2_900,    560),
    (8,    18_800,   76,     2_850,    540),
    (9,    17_200,   69,     2_650,    500),
    (10,   16_900,   67,     2_600,    485),
    (11,   19_500,   79,     2_950,    570),
    (12,   21_800,   88,     3_200,    650),
]

# The 4 metrics we track
METRIC_KEYS = ["revenue", "orders", "ad_spend", "crm_clicks"]

# How many recent days get extra weight (70%)
RECENT_WINDOW = 4


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: THE DATA CONTAINER                                   ║
# ║  This is where we store the forecast results for each metric     ║
# ╚══════════════════════════════════════════════════════════════════╝

@dataclass
class MetricForecast:
    name: str               # which metric (e.g. "revenue")
    actual_so_far: float    # sum of all daily actuals entered
    forecasted_total: float # predicted month-end number
    target: float           # what you want to hit
    pacing_score: float     # are you on track? (100% = exactly on pace)
    trend: str              # "↑ improving", "→ flat", or "↓ declining"
    status: str             # 🟢 🟡 🟠 🔴
    action: str             # what to do (continue / push / scale back / stop)
    confidence: float       # how much to trust this forecast (30-100%)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: THE MAIN ENGINE                                      ║
# ║  This is the boss function — read it first to see the full flow  ║
# ╚══════════════════════════════════════════════════════════════════╝

def forecast_metric(
    metric_name: str,
    daily_values: list[float],
    target: float,
    historical: dict[int, float],
    year: int,
    month: int,
    last_day: int,
) -> MetricForecast:
    """
    The main engine. Takes one metric's data and produces a full forecast.
    
    It calls these functions in order:
      1. weighted_daily_rate()      → get smart daily average
      2. build_seasonality_index()  → get weekday multipliers
      3. get_remaining_weekdays()   → which days are left?
      4. detect_trend()             → getting better or worse?
      5. pacing_status()            → traffic light + action
      6. compute_confidence()       → how much to trust this?
    """
    total_days = calendar.monthrange(year, month)[1]
    actual_so_far = sum(daily_values)

    # STEP 1+2: Calculate weighted daily rate, then adjust for seasonality
    w_rate = weighted_daily_rate(daily_values)
    season_index = build_seasonality_index(historical)
    remaining_weekdays = get_remaining_weekdays(year, month, last_day)

    # For each remaining day, predict: daily_rate × that day's multiplier
    remaining_forecast = sum(
        w_rate * season_index.get(wd, 1.0) for wd in remaining_weekdays
    )
    forecasted_total = actual_so_far + remaining_forecast

    # STEP 3: Calculate pacing — where should we be by today?
    # (adjusted for which weekdays have already passed)
    days_so_far_wd = [
        datetime.date(year, month, d).weekday() for d in range(1, last_day + 1)
    ]
    all_month_wd = [
        datetime.date(year, month, d).weekday() for d in range(1, total_days + 1)
    ]
    full_month_weight = sum(season_index.get(wd, 1.0) for wd in all_month_wd)
    so_far_weight = sum(season_index.get(wd, 1.0) for wd in days_so_far_wd)
    expected_by_today = target * (so_far_weight / full_month_weight)

    pacing_score = (actual_so_far / expected_by_today * 100) if expected_by_today else 100

    # STEP 4: Get status, action, trend, and confidence
    status, action = pacing_status(pacing_score, metric_name)

    return MetricForecast(
        name=metric_name,
        actual_so_far=actual_so_far,
        forecasted_total=forecasted_total,
        target=target,
        pacing_score=pacing_score,
        trend=detect_trend(daily_values),
        status=status,
        action=action,
        confidence=compute_confidence(last_day, total_days),
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: HELPER FUNCTIONS                                     ║
# ║  Each one does a single job. Listed in the order they're called. ║
# ╚══════════════════════════════════════════════════════════════════╝

# --- Called 1st: Get the smart daily average ---

def weighted_daily_rate(daily_values: list[float], recent_window: int = RECENT_WINDOW) -> float:
    """
    Instead of a flat average, gives 70% weight to the last few days
    and 30% to earlier days. This captures recent momentum.
    
    Example:
      Last 4 days averaged 19,000
      Earlier days averaged 17,000
      Result: (19,000 × 0.70) + (17,000 × 0.30) = 18,400
    """
    if not daily_values:
        return 0.0
    if len(daily_values) <= recent_window:
        return sum(daily_values) / len(daily_values)

    recent = daily_values[-recent_window:]
    earlier = daily_values[:-recent_window]
    return (sum(recent) / len(recent)) * 0.70 + (sum(earlier) / len(earlier)) * 0.30


# --- Called 2nd: Build weekday multipliers ---

def build_seasonality_index(historical: dict[int, float]) -> dict[int, float]:
    """
    Converts historical daily averages into multipliers.
    1.0 = average day. 1.22 = 22% above average. 0.84 = 16% below.
    
    Example:
      Saturday avg = 21,500, Overall avg = 17,557
      Saturday index = 21,500 / 17,557 = 1.22
    """
    overall_avg = sum(historical.values()) / len(historical)
    if overall_avg == 0:
        return {d: 1.0 for d in range(7)}
    return {day: val / overall_avg for day, val in historical.items()}


# --- Called 3rd: Figure out which days are left ---

def get_remaining_weekdays(year: int, month: int, last_completed_day: int) -> list[int]:
    """
    Returns a list of weekday numbers for the remaining days in the month.
    
    Example: If today is March 12 (Thursday), returns the weekday numbers
    for March 13, 14, 15, ... 31 so we know which multipliers to apply.
    """
    total_days = calendar.monthrange(year, month)[1]
    return [
        datetime.date(year, month, d).weekday()
        for d in range(last_completed_day + 1, total_days + 1)
    ]


# --- Called 4th: Are things getting better or worse? ---

def detect_trend(daily_values: list[float], window: int = 4) -> str:
    """
    Compares the last 4 days vs the 4 days before that.
    If change > +5%  → "↑ improving"
    If change < -5%  → "↓ declining"
    Otherwise        → "→ flat"
    """
    if len(daily_values) < window * 2:
        if len(daily_values) >= 4:
            mid = len(daily_values) // 2
            avg_first = sum(daily_values[:mid]) / mid
            avg_second = sum(daily_values[mid:]) / (len(daily_values) - mid)
            change = (avg_second - avg_first) / avg_first if avg_first else 0
        else:
            return "→ flat (not enough data)"
    else:
        prev = daily_values[-(window * 2):-window]
        recent = daily_values[-window:]
        avg_prev = sum(prev) / len(prev)
        avg_recent = sum(recent) / len(recent)
        change = (avg_recent - avg_prev) / avg_prev if avg_prev else 0

    if change > 0.05:
        return f"↑ improving (+{change:.0%})"
    elif change < -0.05:
        return f"↓ declining ({change:.0%})"
    else:
        return f"→ flat ({change:+.0%})"


# --- Called 5th: Traffic light + action recommendation ---

def pacing_status(score: float, metric_name: str) -> tuple[str, str]:
    """
    Turns a pacing score into a status + action.
    
    For revenue/orders/clicks (higher = better):
      > 105%  → 🟢 Ahead     → continue
      95-105% → 🟡 On Track  → stay consistent
      85-95%  → 🟠 Behind    → push harder
      < 85%   → 🔴 At Risk   → urgent action
    
    For ad spend (lower = better — flipped logic):
      < 95%   → 🟢 Under Budget → good
      > 115%  → 🔴 Over Budget  → bad
    """
    if metric_name == "ad_spend":
        if score < 95:
            return "🟢 Under Budget", "continue — efficient spending"
        elif score <= 105:
            return "🟡 On Budget", "continue — monitor closely"
        elif score <= 115:
            return "🟠 Over Budget", "scale back — optimize targeting"
        else:
            return "🔴 Over Budget", "reduce spend — check ROAS immediately"
    else:
        if score > 105:
            return "🟢 Ahead", "continue — maintain momentum"
        elif score >= 95:
            return "🟡 On Track", "continue — stay consistent"
        elif score >= 85:
            return "🟠 Behind", "push harder — increase effort"
        else:
            return "🔴 At Risk", "urgent action — re-evaluate strategy"


# --- Called 6th: How much should you trust this forecast? ---

def compute_confidence(day_number: int, total_days: int) -> float:
    """
    More days passed = more data = higher confidence.
    
    Day  1-7:   30-50%  (directional only — don't act on this)
    Day  8-15:  50-75%  (start making decisions)
    Day 16-25:  75-90%  (forecast is reliable)
    Day 26-31:  90-100% (almost certain)
    """
    p = day_number / total_days
    if p <= 0.25:
        return 30 + (p / 0.25) * 20
    elif p <= 0.50:
        return 50 + ((p - 0.25) / 0.25) * 25
    elif p <= 0.85:
        return 75 + ((p - 0.50) / 0.35) * 15
    else:
        return 90 + ((p - 0.85) / 0.15) * 10


# --- Called after all metrics are forecasted: Business KPIs ---

def compute_derived_kpis(forecasts: dict[str, MetricForecast]) -> dict:
    """
    Combines the 4 individual forecasts into business KPIs:
    
    AOV  = Revenue / Orders        (are order values healthy?)
    ROAS = Revenue / Ad Spend      (are ads profitable?)
    CPA  = Ad Spend / Orders       (cost per customer?)
    Conv = Orders / CRM Clicks     (is the funnel working?)
    """
    rev = forecasts["revenue"].forecasted_total
    orders = forecasts["orders"].forecasted_total
    spend = forecasts["ad_spend"].forecasted_total
    clicks = forecasts["crm_clicks"].forecasted_total

    return {
        "AOV (Avg Order Value)": round(rev / orders, 2) if orders else 0,
        "ROAS (Return on Ad Spend)": round(rev / spend, 2) if spend else 0,
        "CPA (Cost per Acquisition)": round(spend / orders, 2) if orders else 0,
        "Conv% (Orders / CRM Clicks)": round((orders / clicks) * 100, 2) if clicks else 0,
    }


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: THE REPORT                                           ║
# ║  Orchestrates everything and prints the results                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_forecast():
    """
    The orchestrator. This is what actually runs when you execute the script.
    
    Flow:
      1. Parse DAILY_ACTUALS into separate lists per metric
      2. Call forecast_metric() for each of the 4 metrics
      3. Print per-metric results
      4. Calculate and print derived KPIs
      5. Print overall health verdict
    """

    # STEP 1: Split the daily data into separate lists per metric
    metric_daily = {k: [] for k in METRIC_KEYS}
    for row in DAILY_ACTUALS:
        day, rev, orders, spend, clicks = row
        metric_daily["revenue"].append(rev)
        metric_daily["orders"].append(orders)
        metric_daily["ad_spend"].append(spend)
        metric_daily["crm_clicks"].append(clicks)

    last_day = DAILY_ACTUALS[-1][0]
    total_days = calendar.monthrange(YEAR, MONTH)[1]
    month_name = calendar.month_name[MONTH]

    # Print header
    print("=" * 70)
    print(f"  📊 MONTHLY FORECAST REPORT — {month_name} {YEAR}")
    print(f"  Day {last_day} of {total_days} ({last_day/total_days:.0%} of month complete)")
    print("=" * 70)

    # STEP 2: Run the forecast for each metric
    forecasts = {}
    for metric in METRIC_KEYS:
        forecasts[metric] = forecast_metric(
            metric_name=metric,
            daily_values=metric_daily[metric],
            target=MONTHLY_TARGET[metric],
            historical=HISTORICAL_DAILY_AVG[metric],
            year=YEAR,
            month=MONTH,
            last_day=last_day,
        )

    # STEP 3: Print each metric's results
    for metric in METRIC_KEYS:
        f = forecasts[metric]
        vs_target = ((f.forecasted_total - f.target) / f.target) * 100
        label = f.name.upper().replace("_", " ")

        print(f"\n  ┌─── {label} {'─' * (50 - len(label))}┐")
        print(f"  │  Actual so far:     {f.actual_so_far:>12,.0f}")
        print(f"  │  Forecasted total:  {f.forecasted_total:>12,.0f}")
        print(f"  │  Target:            {f.target:>12,.0f}")
        print(f"  │  vs Target:         {vs_target:>+11.1f}%")
        print(f"  │  Pacing:            {f.pacing_score:>11.1f}%  {f.status}")
        print(f"  │  Trend:             {f.trend}")
        print(f"  │  Confidence:        {f.confidence:>11.0f}%")
        print(f"  │  ➤ Action:          {f.action}")
        print(f"  └{'─' * 58}┘")

    # STEP 4: Print derived KPIs
    kpis = compute_derived_kpis(forecasts)
    print(f"\n  {'─' * 58}")
    print("  📈 FORECASTED KPIs (Month-End)")
    print(f"  {'─' * 58}")
    for name, value in kpis.items():
        if "%" in name:
            print(f"    {name:<35} {value:>8.2f}%")
        elif "ROAS" in name:
            print(f"    {name:<35} {value:>8.2f}x")
        else:
            print(f"    {name:<35} {value:>8,.2f}")

    # STEP 5: Overall health verdict
    print(f"\n  {'─' * 58}")
    print("  🎯 OVERALL HEALTH")
    print(f"  {'─' * 58}")
    rev_f = forecasts["revenue"]
    spend_f = forecasts["ad_spend"]

    if rev_f.pacing_score >= 95 and spend_f.pacing_score <= 105:
        print("    ✅ Healthy — Revenue on track, spend under control")
    elif rev_f.pacing_score >= 95 and spend_f.pacing_score > 105:
        print("    ⚠️  Revenue OK but overspending — check ROAS & optimize")
    elif rev_f.pacing_score < 95 and spend_f.pacing_score <= 105:
        print("    ⚠️  Revenue behind but spend OK — push CRM & organic")
    else:
        print("    🚨 Revenue behind AND overspending — pause & reassess")

    print(f"\n{'=' * 70}")
    print(f"  Confidence: {rev_f.confidence:.0f}% | Next: Enter Day {last_day + 1} and re-run")
    print(f"{'=' * 70}\n")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: START HERE                                           ║
# ║  This single line runs the whole system                          ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    run_forecast()
