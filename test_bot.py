import numpy as np
import pandas as pd
import pytest

from config import Config
from state import State
from strategies import Signal, adjust_score, evaluate
from trader import Trader

CFG = Config(mode="paper")
H = 3_600_000
M15 = 900_000
DAY = 86_400_000
T0 = 1_700_006_400_000  # múltiplo de 1 día UTC (00:00)


def mk(ts, o, h, l, c, v):
    return pd.DataFrame({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})


def daily_trend(n=60, g=1.008, rng=0.03):
    c = 1.0 * g ** np.arange(n)
    return mk(T0 + np.arange(n) * DAY, c, c * (1 + rng), c * (1 - rng), c, np.full(n, 1e6))


def hourly(n_pre=108, base=1.6, rng=0.01, consol=True):
    ts = T0 + np.arange(n_pre + 12) * H
    c = np.full(n_pre + 12, base)
    hi = c * (1 + rng / 2)
    lo = c * (1 - rng / 2)
    if consol:
        for k in range(12):
            hi[n_pre + k] = base * (1.004 - 0.0001 * k)
            lo[n_pre + k] = base * (0.996 + 0.0004 * k)
    return mk(ts, c, hi, lo, c, np.full(len(c), 1e4))


def m15(n=129, base=1.6, last_open=None, last_close=None, last_vol=400, vol=100):
    ts = T0 + np.arange(n) * M15
    c = np.full(n, base)
    o, h, l = c.copy(), c * 1.0007, c * 0.9993
    v = np.full(n, float(vol))
    o[-1], c[-1] = last_open, last_close
    h[-1], l[-1] = max(o[-1], c[-1]) * 1.0003, min(o[-1], c[-1]) * 0.9997
    v[-1] = last_vol
    return mk(ts, o, h, l, c, v)


def test_breakout_long_fires():
    d, h = daily_trend(), hourly()
    m = m15(last_open=1.6 * 1.002, last_close=1.6 * 1.0045)
    sigs = evaluate("TEST/USDT:USDT", d, h, m, CFG)
    s = [x for x in sigs if x.setup == "Breakout"]
    assert s and s[0].side == "long" and s[0].stop < s[0].entry
    assert s[0].score >= 55


def test_breakout_needs_volume():
    d, h = daily_trend(), hourly()
    m = m15(last_open=1.6 * 1.002, last_close=1.6 * 1.0045, last_vol=100)
    assert not [x for x in evaluate("T", d, h, m, CFG) if x.setup == "Breakout"]


def test_breakout_rejects_loose_consolidation():
    d, h = daily_trend(), hourly(consol=False, rng=0.12)   # velas de 12% de rango: nada tenso
    m = m15(last_open=1.6 * 1.002, last_close=1.6 * 1.0045)
    assert not [x for x in evaluate("T", d, h, m, CFG) if x.setup == "Breakout"]


def test_episodic_pivot_long():
    n = 60
    c = np.full(n, 1.0)
    d = mk(T0 + np.arange(n) * DAY, c, c * 1.03, c * 0.97, c, np.full(n, 1e6))
    h = hourly(base=1.0, rng=0.02, consol=False)
    m = m15(base=1.0, last_open=1.0, last_close=1.04, last_vol=900)
    s = [x for x in evaluate("T", d, h, m, CFG) if x.setup == "Episodic Pivot"]
    assert s and s[0].side == "long"


def test_parabolic_short():
    n = 60
    c = np.full(n, 1.0)
    c[-6:] = [1.0, 1.1, 1.2, 1.3, 1.45, 1.6]
    d = mk(T0 + np.arange(n) * DAY, c, c * 1.04, c * 0.97, c, np.full(n, 1e6))
    h = hourly(base=1.6, rng=0.03, consol=False)
    # 15m: la sesión de hoy termina a las 20:00 UTC
    nb = 129
    end = T0 + 20 * H
    ts = end - (nb - 1) * M15 + np.arange(nb) * M15
    cl = np.full(nb, 1.55)
    cl[-30:-2] = np.linspace(1.55, 1.64, 28)
    cl[-2] = 1.62
    cl[-1] = 1.55
    op = np.roll(cl, 1)
    op[0] = 1.55
    op[-1] = 1.61
    hi = np.maximum(op, cl) * 1.001
    lo = np.minimum(op, cl) * 0.999
    hi[-40] = 1.66   # máximo del día
    m = mk(ts, op, hi, lo, cl, np.full(nb, 100.0))
    s = [x for x in evaluate("T", d, h, m, CFG) if x.setup == "Parabolic Short"]
    assert s and s[0].side == "short" and s[0].stop > s[0].entry


def test_regime_adjustment():
    s = Signal("X", "long", "Breakout", 1.0, 0.98, 70.0, [])
    assert adjust_score(s, "bull", None).score > 70
    s = Signal("X", "long", "Breakout", 1.0, 0.98, 70.0, [])
    assert adjust_score(s, "bear", None).score < 70


class FakeEx:
    def __init__(self):
        self.price = {"TEST/USDT:USDT": 100.0}

    def amount_to_precision(self, s, q): return f"{q:.3f}"
    def price_to_precision(self, s, p): return f"{p:.4f}"
    def market(self, s): return {"limits": {"amount": {"min": 0.001}}}
    def fetch_ticker(self, s): return {"last": self.price[s]}
    def fetch_tickers(self, syms): return {s: {"last": self.price[s]} for s in syms}
    def fetch_ohlcv(self, *a, **k): raise RuntimeError("sin datos")


def make_trader(tmp_path):
    st = State(str(tmp_path), 1000.0)
    msgs = []
    return Trader(FakeEx(), CFG, st, msgs.append), st, msgs


def test_paper_lifecycle_tp1_then_breakeven(tmp_path):
    tr, st, msgs = make_trader(tmp_path)
    sig = Signal("TEST/USDT:USDT", "long", "Breakout", 100.0, 98.0, 80.0, [])
    ok, why = tr.open(sig)
    assert ok, why
    pos = st.data["positions"]["TEST/USDT:USDT"]
    # riesgo 0.5% de 1000 = $5 ; distancia 2 -> qty 2.5
    assert pos["qty"] == pytest.approx(2.5, rel=1e-3)
    tr.ex.price["TEST/USDT:USDT"] = 103.5          # 1.75R -> TP1
    tr.manage()
    pos = st.data["positions"]["TEST/USDT:USDT"]
    assert pos["tp1_hit"] and pos["qty"] == pytest.approx(1.25, rel=1e-3)
    assert pos["stop"] >= 100.0                     # break-even
    tr.ex.price["TEST/USDT:USDT"] = 99.0            # vuelve al stop
    tr.manage()
    assert "TEST/USDT:USDT" not in st.data["positions"]
    assert st.data["paper_equity"] > 1000.0         # cerró en positivo gracias al parcial


def test_paper_stop_loss_and_limits(tmp_path):
    tr, st, _ = make_trader(tmp_path)
    sig = Signal("TEST/USDT:USDT", "short", "Breakdown", 100.0, 102.0, 80.0, [])
    assert tr.open(sig)[0]
    assert not tr.open(sig)[0]                      # ya abierta
    tr.ex.price["TEST/USDT:USDT"] = 102.5
    tr.manage()
    assert not st.data["positions"]
    assert st.data["paper_equity"] < 1000.0
    st.data["paused"] = True
    assert not tr.can_open(sig)[0]


# ───────────── candidatas (watchlist) ─────────────
from collections import Counter

from strategies import daily_ctx, trend_gate, watch_candidates


def _cands(d, h, m, cfg=CFG, stats=None):
    ctx = daily_ctx(d)
    priors = {s: trend_gate(s, ctx, cfg) for s in ("long", "short")}
    priors = {s: p for s, p in priors.items() if p is not None}
    return watch_candidates("T", priors, ctx, h, m, cfg, stats)


def test_candidate_near_trigger_has_plan():
    d, h = daily_trend(), hourly()
    m = m15(last_open=1.6, last_close=1.6, last_vol=100)        # sin volumen, 0.4% bajo el gatillo
    c = _cands(d, h, m)
    assert len(c) == 1 and c[0].side == "long" and c[0].setup == "Breakout"
    assert 0 < c[0].dist_pct <= CFG.watch_dist_pct
    assert c[0].trigger > c[0].price > c[0].stop
    # el gatillo de la candidata es exactamente el nivel que luego dispara la señal
    sig = [x for x in evaluate("T", d, h, m15(last_open=1.6 * 1.002, last_close=1.6 * 1.0045), CFG)
           if x.setup == "Breakout"]
    assert sig and sig[0].entry > c[0].trigger


def test_candidate_far_or_broken_is_not_listed():
    d, h = daily_trend(), hourly()
    stats = Counter()
    assert not _cands(d, h, m15(last_open=1.55, last_close=1.55, last_vol=100), stats=stats)
    assert stats["caja válida pero lejos del gatillo"] == 1
    stats = Counter()
    assert not _cands(d, h, m15(last_open=1.6 * 1.002, last_close=1.6 * 1.0045), stats=stats)
    assert stats["caja válida ya rota"] == 1             # eso ya es señal, no candidata


def test_candidate_needs_tight_box():
    d, h = daily_trend(), hourly(consol=False, rng=0.12)
    stats = Counter()
    assert not _cands(d, h, m15(last_open=1.6, last_close=1.6, last_vol=100), stats=stats)
    assert stats["caja 1h demasiado ancha"] == 1


# ───────────── escaneo completo con exchange simulado ─────────────
from scanner import Scanner


class FakeScanEx:
    def __init__(self, data):
        self.data = data                                     # {sym: (daily, hourly, m15)}
        self.markets = {s: {"swap": True, "linear": True, "quote": "USDT", "active": True, "id": s.split("/")[0] + "-USDT"}
                        for s in data}

    def fetch_tickers(self):
        return {s: {"quoteVolume": 5e6, "percentage": 1.0} for s in self.data}

    def fetch_ohlcv(self, sym, tf, limit=100):
        d, h, m = self.data.get(sym) or next(iter(self.data.values()))
        df = {"1d": d, "1h": h, "15m": m}[tf]
        return df.values.tolist()

    def fetch_funding_rate(self, sym):
        raise RuntimeError("sin funding")


def _with_open_candle(m):
    """El escáner descarta la última vela (en curso): añadimos una de relleno."""
    return pd.concat([m, m.iloc[[-1]].assign(ts=m["ts"].iloc[-1] + M15)], ignore_index=True)


def _with_open_hour(h):
    return pd.concat([h, h.iloc[[-1]].assign(ts=h["ts"].iloc[-1] + H)], ignore_index=True)


def test_scan_separates_candidates_and_signals():
    d, h = daily_trend(), _with_open_hour(hourly())
    flat = daily_trend(g=1.0, rng=0.01)                       # ADR bajo: ni candidata ni señal
    data = {
        "WAIT/USDT:USDT": (d, h, _with_open_candle(m15(last_open=1.6, last_close=1.6, last_vol=100))),
        "GO/USDT:USDT": (d, h, _with_open_candle(m15(last_open=1.6 * 1.002, last_close=1.6 * 1.0045))),
        "DEAD/USDT:USDT": (flat, h, _with_open_candle(m15(last_open=1.6, last_close=1.6, last_vol=100))),
    }
    res = Scanner(FakeScanEx(data), CFG).scan()
    assert [c.symbol for c in res.candidates] == ["WAIT/USDT:USDT"]
    assert [s.symbol for s in res.signals if s.setup == "Breakout"] == ["GO/USDT:USDT"]
    assert res.stats["analizados"] == 3 and res.stats["errores de datos"] == 0
    assert res.stats["no pasa filtro diario: ADR diario bajo"] == 1
    assert res.movers and res.movers[0][0] == "GO/USDT:USDT"


def test_scan_reports_data_errors_instead_of_hiding_them():
    class Broken(FakeScanEx):
        def fetch_ohlcv(self, sym, tf, limit=100):
            raise ConnectionError("403 forbidden")
    d, h = daily_trend(), _with_open_hour(hourly())
    ex = Broken({"WAIT/USDT:USDT": (d, h, _with_open_candle(m15(last_open=1.6, last_close=1.6)))})
    res = Scanner(ex, CFG).scan()
    assert res.stats["errores de datos"] == 1 and "ConnectionError" in res.errors[0]


# ───────────── avisos ─────────────
import main as botmain
import telegram as tgmod


def test_formatters_do_not_crash_and_escape_html():
    d, h = daily_trend(), _with_open_hour(hourly())
    data = {"WAIT/USDT:USDT": (d, h, _with_open_candle(m15(last_open=1.6, last_close=1.6, last_vol=100)))}
    res = Scanner(FakeScanEx(data), CFG).scan()
    txt = botmain.fmt_digest(res, CFG)
    assert "Candidatas: 1" in txt and "WAIT/USDT:USDT" in txt
    assert "ENTRAR si una vela 15m CIERRA &gt;" in botmain.fmt_watch(res.candidates, CFG)
    assert "aún no" in botmain.fmt_digest(None, CFG).lower()


def test_watch_dedupe(tmp_path):
    st = State(str(tmp_path), 1000.0)
    assert st.watch_due("A|long", 360)
    st.watch_mark(["A|long"])
    assert not st.watch_due("A|long", 360) and st.watch_due("B|long", 360)
    assert not State(str(tmp_path), 1000.0).watch_due("A|long", 360)      # sobrevive a reinicios


def test_telegram_retries_as_plain_text_when_html_is_rejected(monkeypatch):
    calls = []

    class R:
        def __init__(self, code): self.status_code, self.text = code, "x"

    def fake_post(url, json=None, timeout=0):
        calls.append(json)
        return R(400 if "parse_mode" in json else 200)

    monkeypatch.setattr(tgmod.requests, "post", fake_post)
    n = tgmod.Notifier(Config(tg_token="t", tg_chat_id="1"))
    assert n.send("<b>x</b> a &lt; b") is True
    assert "parse_mode" in calls[0] and "parse_mode" not in calls[1]
    assert calls[1]["text"] == "x a < b"


# ───────────── límite de BingX (100410), distancia por ADR, /debug ─────────────
RATE_ERR = 'bingx {"code":100410,"msg":"请稍后重试"}'


def test_scan_retries_when_bingx_rate_limits():
    d, h = daily_trend(), _with_open_hour(hourly())
    m = _with_open_candle(m15(last_open=1.6, last_close=1.6, last_vol=100))

    class Flaky(FakeScanEx):
        fails = 2
        def fetch_ohlcv(self, sym, tf, limit=100):
            if tf == "15m" and Flaky.fails > 0:
                Flaky.fails -= 1
                raise Exception(RATE_ERR)
            return super().fetch_ohlcv(sym, tf, limit)

    sc = Scanner(Flaky({"WAIT/USDT:USDT": (d, h, m)}), CFG)
    waits = []
    sc._sleep = waits.append                                   # sin esperar de verdad
    res = sc.scan()
    assert res.stats["errores de datos"] == 0                  # se recuperó solo
    assert res.stats["peticiones reintentadas por límite de BingX"] == 2 and waits == [2.0, 5.0]
    assert [c.symbol for c in res.candidates] == ["WAIT/USDT:USDT"]


def test_scan_gives_up_after_retries_and_reports():
    class AlwaysLimited(FakeScanEx):
        def fetch_ohlcv(self, sym, tf, limit=100):
            raise Exception(RATE_ERR)
    d, h = daily_trend(), _with_open_hour(hourly())
    sc = Scanner(AlwaysLimited({"WAIT/USDT:USDT": (d, h, _with_open_candle(m15(last_open=1.6, last_close=1.6)))}), CFG)
    sc._sleep = lambda s: None
    res = sc.scan()
    assert res.stats["errores de datos"] == 1 and "100410" in res.errors[0]


def test_watch_distance_scales_with_adr():
    d, h = daily_trend(rng=0.05), hourly()                     # ADR ≈ 10%
    # precio ~2.3% bajo el gatillo: fuera del 1.5% fijo, dentro de 0.25 x ADR (≈2.5%)
    m = m15(last_open=1.57, last_close=1.57, last_vol=100)
    c = _cands(d, h, m)
    assert c and 1.5 < c[0].dist_pct <= 0.25 * c[0].adr
    far = []
    ctx = daily_ctx(d)
    strict = Config(mode="paper", watch_dist_adr=0.0)
    assert not watch_candidates("T", {"long": 30.0}, ctx, h, m, strict, None, far)
    assert far and far[0][0] == "T" and far[0][1] == "long" and far[0][2] > 1.5


def test_debug_command_shows_raw_volumes():
    d, h = daily_trend(), _with_open_hour(hourly())
    sc = Scanner(FakeScanEx({"TRX/USDT:USDT": (d, h, _with_open_candle(m15(last_open=1.6, last_close=1.6, last_vol=100)))}), CFG)
    txt = botmain.fmt_debug(sc, "trx")
    assert "TRX/USDT:USDT" in txt and "RVOL calculado" in txt and "UTC" in txt
    assert "Uso:" in botmain.fmt_debug(sc, "")


# ───────────── SWING (método Kullamägi en velas diarias) ─────────────
from strategies import swing_box, swing_watch


def swing_daily(side="long", wave=True, rising_lows=True):
    """Ola de +60% hasta un pico de 1.62, base de 30 días (3 tramos) bajo el pico y vela de hoy en curso."""
    n = 91
    c, hi, lo = np.empty(n), np.empty(n), np.empty(n)
    for i in range(60):
        px_ = (1.0 + 0.58 * i / 59) if wave else 1.58
        c[i], hi[i], lo[i] = px_, px_ * 1.015, px_ * 0.985
    hi[59] = 1.62                                              # máximo de la ola = nivel de ruptura
    centers = [1.50, 1.535, 1.565] if rising_lows else [1.565, 1.535, 1.50]
    ranges = [0.10, 0.07, 0.05]                                # volatilidad contrayéndose
    for k in range(3):
        for j in range(10):
            i = 60 + k * 10 + j
            c[i], hi[i], lo[i] = centers[k], centers[k] * (1 + ranges[k] / 2), centers[k] * (1 - ranges[k] / 2)
    c[90], hi[90], lo[90] = 1.57, 1.57 * 1.015, 1.57 * 0.985   # hoy (en curso)
    return mk(T0 + np.arange(n) * DAY, c, hi, lo, c, np.full(n, 1e6))


def _adr(d):
    return daily_ctx(d)["adr"]


def test_swing_box_detects_staircase_base():
    d = swing_daily()
    b = swing_box("long", d, _adr(d), CFG)
    assert b and b["level"] == pytest.approx(1.62) and b["n"] == 30 and b["wave"] > 55


def test_swing_box_rejects_when_rules_fail():
    for kw, motivo in (({"wave": False}, "sin ola previa"), ({"rising_lows": False}, "sin mínimos crecientes")):
        d, st = swing_daily(**kw), Counter()
        assert swing_box("long", d, _adr(d), CFG, st) is None
        assert any(motivo in k for k in st), (kw, dict(st))
    d = swing_daily()
    assert swing_box("long", d.tail(30).reset_index(drop=True), _adr(d), CFG) is None      # poco historial
    young = Config(mode="paper", swing_min_bars=40)                                       # base de 2 semanas no basta
    assert swing_box("long", d, _adr(d), young) is None


def test_swing_breakout_signal_stop_is_low_of_day_and_within_adr():
    d, h = swing_daily(), hourly(base=1.6)
    m = m15(base=1.60, last_open=1.60, last_close=1.625, last_vol=300)      # rompe 1.62 con volumen x3
    sig = [x for x in evaluate("T", d, h, m, CFG) if x.setup == "Swing Breakout"]
    assert sig and sig[0].mgmt == "swing" and sig[0].side == "long"
    s = sig[0]
    assert s.stop < s.entry and (s.entry - s.stop) / s.entry * 100 <= _adr(d)
    assert s.score >= CFG.min_score


def test_swing_breakout_needs_volume_and_real_break():
    d, h = swing_daily(), hourly(base=1.6)
    assert not [x for x in evaluate("T", d, h, m15(base=1.60, last_open=1.60, last_close=1.625, last_vol=100), CFG)
                if x.setup == "Swing Breakout"]                              # RVOL 1x
    assert not [x for x in evaluate("T", d, h, m15(base=1.60, last_open=1.60, last_close=1.61, last_vol=300), CFG)
                if x.setup == "Swing Breakout"]                              # no llega al nivel


def test_swing_candidate_near_level_and_plan_text():
    d = swing_daily()
    m = m15(base=1.60, last_open=1.60, last_close=1.60, last_vol=100)        # 1.25% bajo el máximo de la base
    ctx = daily_ctx(d)
    c = swing_watch("T", ("long", "short"), d, ctx, m, CFG)
    assert len(c) == 1 and c[0].setup == "Swing Breakout" and c[0].mgmt == "swing"
    assert c[0].trigger == pytest.approx(1.62) and 0 < c[0].dist_pct <= max(CFG.watch_dist_pct, 0.25 * ctx["adr"])
    txt = botmain.fmt_cand(c[0], CFG)
    assert "máximo de la base" in txt and "mínimo del día" in txt and "base de 30 días" in txt
    far = []
    m_far = m15(base=1.40, last_open=1.40, last_close=1.40, last_vol=100)
    assert not swing_watch("T", ("long",), d, ctx, m_far, CFG, None, far) and far and far[0][4] == "Swing Breakout"


def test_swing_signal_message_explains_the_plan():
    d, h = swing_daily(), hourly(base=1.6)
    sig = [x for x in evaluate("T", d, h, m15(base=1.60, last_open=1.60, last_close=1.625, last_vol=300), CFG)
           if x.setup == "Swing Breakout"][0]
    txt = botmain.fmt_signal(sig, "bull", CFG)
    assert "Plan swing" in txt and "CIERRE diario" in txt and "TP1" not in txt


def test_scan_finds_swing_candidate_and_reports_swing_section():
    d, h = swing_daily(), _with_open_hour(hourly(base=1.6))
    m = _with_open_candle(m15(base=1.60, last_open=1.60, last_close=1.60, last_vol=100))
    res = Scanner(FakeScanEx({"ARM/USDT:USDT": (d, h, m)}), CFG).scan()
    assert [c.setup for c in res.candidates if c.mgmt == "swing"] == ["Swing Breakout"]
    assert res.stats["cajas diarias swing válidas"] == 1
    txt = botmain.fmt_digest(res, CFG)
    assert "Swing diario" in txt and "Swing Breakout" in txt


# ── gestión de la posición swing ──
class SwingEx(FakeEx):
    def __init__(self, daily_closes=None):
        super().__init__()
        self.daily_closes = daily_closes

    def fetch_ohlcv(self, sym, tf, limit=60, **k):
        if tf != "1d" or self.daily_closes is None:
            raise RuntimeError("sin datos")
        now_day = int(time.time() * 1000) // DAY * DAY
        n = len(self.daily_closes)
        ts = [now_day - (n - i) * DAY for i in range(n)]                       # último cerrado = ayer
        return [[t, c, c * 1.01, c * 0.99, c, 1e6] for t, c in zip(ts, self.daily_closes)]


import time


def _open_swing(tmp_path, ex):
    st = State(str(tmp_path), 1000.0)
    tr = Trader(ex, CFG, st, [].append)
    sig = Signal("TEST/USDT:USDT", "long", "Swing Breakout", 100.0, 96.0, 80.0, [], 1.0, mgmt="swing")
    assert tr.open(sig)[0]
    return tr, st


def test_swing_position_partial_after_3_days_then_breakeven(tmp_path):
    tr, st = _open_swing(tmp_path, SwingEx())
    pos = st.data["positions"]["TEST/USDT:USDT"]
    assert pos["mgmt"] == "swing"
    tr.ex.price["TEST/USDT:USDT"] = 106.0                  # +1.5R pero solo han pasado minutos: NO hay parcial
    tr.manage()
    assert not pos["tp1_hit"] and pos["qty"] == pytest.approx(pos["qty0"])
    pos["opened"] -= 3.2 * 86400                           # pasan 3.2 días
    tr.manage()
    pos = st.data["positions"]["TEST/USDT:USDT"]
    assert pos["tp1_hit"] and pos["qty"] == pytest.approx(pos["qty0"] * 0.6, rel=0.02)
    assert pos["stop"] >= 100.0                            # stop a break-even
    tr.ex.price["TEST/USDT:USDT"] = 99.0
    tr.manage()
    assert "TEST/USDT:USDT" not in st.data["positions"]


def test_swing_position_exits_on_first_daily_close_below_ema10(tmp_path):
    closes = list(np.linspace(80, 110, 38)) + [112.0, 96.0]           # tendencia + último cierre diario bajo la EMA10
    tr, st = _open_swing(tmp_path, SwingEx(closes))
    st.data["positions"]["TEST/USDT:USDT"]["opened"] -= 2 * 86400     # entró hace 2 días (antes del último cierre)
    tr.last_trail = 0
    tr.ex.price["TEST/USDT:USDT"] = 101.0                             # el precio intradía aún no toca el stop
    tr.manage()
    assert "TEST/USDT:USDT" not in st.data["positions"]


def test_swing_position_holds_while_daily_closes_above_ema10(tmp_path):
    closes = list(np.linspace(80, 112, 40))                           # sube cada día: sin salida
    tr, st = _open_swing(tmp_path, SwingEx(closes))
    st.data["positions"]["TEST/USDT:USDT"]["opened"] -= 2 * 86400
    tr.ex.price["TEST/USDT:USDT"] = 101.0
    tr.manage()
    assert "TEST/USDT:USDT" in st.data["positions"]
