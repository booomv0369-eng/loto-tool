# app.py
# ロト6・ロト7分析ツール（ボーナス対応 / 自動取り込み / バックテスト / 見た目改善）
from __future__ import annotations

import re
import time
import itertools
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# ページ設定 + CSS
# =========================
st.set_page_config(
    page_title="ロト6・ロト7分析ツール",
    page_icon="🎯",
    layout="wide",
)

CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; max-width: 1150px; }
h1, h2, h3 { letter-spacing: 0.02em; }
small { color:#64748b; }

.topline{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
.badge{
  display:inline-block; padding:6px 10px; border-radius:999px;
  background:#f0f7ff; border:1px solid #dbeafe; color:#0f172a;
  font-weight:700; font-size:0.85rem;
}
.badge2{
  display:inline-block; padding:6px 10px; border-radius:999px;
  background:#f8fafc; border:1px solid #e2e8f0; color:#0f172a;
  font-weight:700; font-size:0.85rem;
}

.card{
  background:#ffffff; border:1px solid #e6e8ee; border-radius:16px;
  padding:16px 18px; box-shadow:0 2px 10px rgba(0,0,0,0.03);
  margin: 10px 0 16px 0;
}
.hr{ height:1px; background:#eef2f7; margin:14px 0; }

.notice{
  border-left:6px solid #0ea5e9; background:#f0f9ff;
  padding:12px 14px; border-radius:10px; color:#0f172a;
  margin:10px 0 14px 0;
}
.warn{
  border-left:6px solid #f59e0b; background:#fffbeb;
  padding:12px 14px; border-radius:10px; color:#0f172a;
  margin:10px 0 14px 0;
}

.chips{ display:flex; flex-wrap:wrap; gap:8px; }
.chip{
  display:inline-flex; align-items:center; justify-content:center;
  min-width:38px; height:32px; padding:0 10px;
  border-radius:999px; border:1px solid #e2e8f0;
  background:#f8fafc; color:#0f172a; font-weight:800;
}
.chip.main{ background:#0f172a; color:#ffffff; border-color:#0f172a; }
.chip.bonus{ background:#f59e0b; color:#111827; border-color:#f59e0b; }

thead tr th { background:#f8fafc !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================
# ゲーム設定
# =========================
@dataclass(frozen=True)
class GameSpec:
    name: str
    n_main: int
    n_bonus: int
    max_num: int


LOTO6 = GameSpec("ロト6", n_main=6, n_bonus=1, max_num=43)
LOTO7 = GameSpec("ロト7", n_main=7, n_bonus=2, max_num=37)
GAME_MAP = {"ロト6": LOTO6, "ロト7": LOTO7}


# =========================
# セッション初期化
# =========================
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = {"ロト6": [], "ロト7": []}
    if "history_keyset" not in st.session_state:
        st.session_state.history_keyset = {"ロト6": set(), "ロト7": set()}
    if "paste_text" not in st.session_state:
        st.session_state.paste_text = ""
    if "last_processed_text" not in st.session_state:
        st.session_state.last_processed_text = ""
    if "public_url" not in st.session_state:
        st.session_state.public_url = "https://xxxxxxxxxxxxxxxxxxxx.streamlit.app/"
    if "generated" not in st.session_state:
        st.session_state.generated = []


init_state()


# =========================
# パース：貼り付け
# =========================
@dataclass
class Draw:
    round_no: Optional[int]
    main: Tuple[int, ...]
    bonus: Tuple[int, ...]


def parse_draw_line(line: str, spec: GameSpec) -> Optional[Draw]:
    raw = line.strip()
    if not raw:
        return None

    m = re.search(r"(?:第)?\s*(\d+)\s*(?:回)?", raw)
    round_no = int(m.group(1)) if m else None

    bonus_part = ""
    main_part = raw

    bm = re.search(r"\b(B|BONUS|ボーナス)\b[:：]?", raw, flags=re.IGNORECASE)
    if bm:
        idx = bm.start()
        main_part = raw[:idx]
        bonus_part = raw[idx:]
        bonus_part = re.sub(r"\b(B|BONUS|ボーナス)\b[:：]?", " ", bonus_part, flags=re.IGNORECASE)

    main_tokens = re.findall(r"\d+", main_part)
    bonus_tokens = re.findall(r"\d+", bonus_part)

    if len(main_tokens) >= 1:
        maybe_round = int(main_tokens[0])
        rest = main_tokens[1:]
        if len(rest) >= spec.n_main:
            if round_no is None or round_no == maybe_round:
                round_no = maybe_round
                main_tokens = rest

    nums_main = [int(x) for x in main_tokens]
    nums_bonus = [int(x) for x in bonus_tokens]

    if len(nums_main) < spec.n_main:
        return None
    nums_main = nums_main[: spec.n_main]

    if any(n < 1 or n > spec.max_num for n in nums_main):
        return None
    if len(set(nums_main)) != len(nums_main):
        return None

    nums_bonus = [n for n in nums_bonus if 1 <= n <= spec.max_num and n not in nums_main]
    nums_bonus = nums_bonus[: spec.n_bonus]

    return Draw(round_no=round_no, main=tuple(sorted(nums_main)), bonus=tuple(sorted(nums_bonus)))


def parse_paste(text: str, spec: GameSpec) -> List[Draw]:
    draws = []
    for line in text.splitlines():
        d = parse_draw_line(line, spec)
        if d:
            draws.append(d)
    return draws


def draw_key(d: Draw) -> Tuple[Optional[int], Tuple[int, ...], Tuple[int, ...]]:
    return (d.round_no, d.main, d.bonus)


def add_draws(game_name: str, new_draws: List[Draw]) -> int:
    keyset: Set = st.session_state.history_keyset[game_name]
    hist: List[Draw] = st.session_state.history[game_name]
    added = 0
    for d in new_draws:
        k = draw_key(d)
        if k in keyset:
            continue
        keyset.add(k)
        hist.append(d)
        added += 1
    hist.sort(key=lambda x: (x.round_no if x.round_no is not None else 10**18))
    return added


# =========================
# 統計
# =========================
@dataclass
class Stats:
    freq_all: pd.Series
    freq_recent: pd.Series
    hot_score: pd.Series
    last_seen_gap: pd.Series
    streak_info: Dict[int, str]


def compute_stats(draws: List[Draw], spec: GameSpec, recent_n: int = 30) -> Stats:
    idx = pd.Index(range(1, spec.max_num + 1), name="num")

    if len(draws) == 0:
        z = pd.Series([0] * spec.max_num, index=idx, dtype=float)
        return Stats(z, z, z, z, {})

    all_nums = list(itertools.chain.from_iterable([d.main for d in draws]))
    freq_all = pd.Series(all_nums).value_counts().reindex(idx, fill_value=0).astype(float)

    recent_draws = draws[-recent_n:]
    recent_nums = list(itertools.chain.from_iterable([d.main for d in recent_draws]))
    freq_recent = pd.Series(recent_nums).value_counts().reindex(idx, fill_value=0).astype(float)

    total_all = max(1, len(all_nums))
    total_recent = max(1, len(recent_nums))
    rate_all = freq_all / total_all
    rate_recent = freq_recent / total_recent
    hot_score = (rate_recent - rate_all)

    gaps = {}
    for n in range(1, spec.max_num + 1):
        gap = None
        for i, d in enumerate(reversed(draws), start=0):
            if n in d.main:
                gap = i
                break
        gaps[n] = gap if gap is not None else len(draws)
    last_seen_gap = pd.Series(gaps, index=idx).astype(float)

    streak_info = {}
    short_n = min(20, len(draws))
    short = draws[-short_n:]
    count_short = pd.Series(list(itertools.chain.from_iterable([d.main for d in short]))).value_counts()
    for n in range(1, spec.max_num + 1):
        c = int(count_short.get(n, 0))
        if c >= 3:
            streak_info[n] = f"直近{short_n}回で{c}回出現（短期で多め）"

    return Stats(freq_all=freq_all, freq_recent=freq_recent, hot_score=hot_score, last_seen_gap=last_seen_gap, streak_info=streak_info)


@st.cache_data(show_spinner=False)
def _cached_stats(draws_serialized: List[Tuple[Optional[int], Tuple[int, ...], Tuple[int, ...]]], spec: GameSpec, recent_n: int):
    draws = [Draw(r, m, b) for (r, m, b) in draws_serialized]
    return compute_stats(draws, spec, recent_n=recent_n)


def get_stats_cached(draws: List[Draw], spec: GameSpec, recent_n: int) -> Stats:
    ser = [(d.round_no, d.main, d.bonus) for d in draws]
    return _cached_stats(ser, spec, recent_n)


def decade(n: int) -> int:
    return (n - 1) // 10


def has_3_consecutive(nums: List[int]) -> bool:
    s = sorted(nums)
    run = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    return False


# =========================
# 候補生成（KeyError潰し込み）
# =========================
def weighted_sample_without_replacement(items: List[int], weights: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    w = np.asarray(weights, dtype=float).copy()
    if np.all(w <= 0):
        w = np.ones_like(w, dtype=float)
    w = np.maximum(w, 1e-12)
    w = w / w.sum()

    chosen = []
    pool = items.copy()
    w_pool = w.copy()
    for _ in range(k):
        idx = rng.choice(len(pool), p=w_pool)
        chosen.append(pool.pop(idx))
        w_pool = np.delete(w_pool, idx)
        if len(pool) == 0:
            break
        w_pool = w_pool / w_pool.sum()
    return chosen


def generate_candidates(
    draws: List[Draw],
    spec: GameSpec,
    k_candidates: int,
    recent_n: int,
    bias_hot: float,
    avoid_3consec: bool,
    avoid_single_decade: bool,
    rng_seed: Optional[int] = None,
) -> List[Dict]:
    stats = get_stats_cached(draws, spec, recent_n=recent_n)
    nums = list(range(1, spec.max_num + 1))

    # ここが重要：Seriesのままにしない（必ずnumpy配列へ）
    base = (stats.freq_all + 1.0).to_numpy(dtype=float)  # shape=(max_num,)
    hot = stats.hot_score.to_numpy(dtype=float)

    hot_min, hot_max = float(np.min(hot)), float(np.max(hot))
    denom = (hot_max - hot_min) + 1e-9
    hot_norm = (hot - hot_min) / denom
    cold_norm = 1.0 - hot_norm

    mix = bias_hot * hot_norm + (1.0 - bias_hot) * cold_norm
    weights = base * (0.65 + 0.70 * mix)

    # 念のため二重にnumpy固定
    weights = np.asarray(weights, dtype=float)

    rng = np.random.default_rng(rng_seed if rng_seed is not None else int(time.time()))

    out = []
    tries = 0
    max_tries = k_candidates * 120

    while len(out) < k_candidates and tries < max_tries:
        tries += 1
        picked = sorted(weighted_sample_without_replacement(nums, weights, spec.n_main, rng))

        if avoid_3consec and has_3_consecutive(picked):
            continue
        if avoid_single_decade:
            if len({decade(x) for x in picked}) == 1:
                continue

        remaining = [n for n in nums if n not in picked]
        rem_w = np.array([weights[n - 1] for n in remaining], dtype=float)
        bonus = sorted(weighted_sample_without_replacement(remaining, rem_w, spec.n_bonus, rng))

        reasons = []
        for n in picked:
            if n in stats.streak_info:
                reasons.append(f"{n}: {stats.streak_info[n]}")
        if not reasons:
            reasons.append("直近傾向（ホット/コールド）と全体頻度のバランスから生成")

        out.append({"main": picked, "bonus": bonus, "reason": " / ".join(reasons[:3])})

    return out


# =========================
# バックテスト（簡易）
# =========================
def backtest(
    draws: List[Draw],
    spec: GameSpec,
    test_last_n: int = 50,
    train_window: int = 80,
    candidates_per_round: int = 30,
    recent_n: int = 30,
    bias_hot: float = 0.6,
    avoid_3consec: bool = True,
    avoid_single_decade: bool = True,
) -> pd.DataFrame:
    if len(draws) < (test_last_n + 5):
        return pd.DataFrame()

    start = max(0, len(draws) - test_last_n)
    rows = []

    for idx in range(start, len(draws)):
        target = draws[idx]
        train_start = max(0, idx - train_window)
        train = draws[train_start:idx]

        cands = generate_candidates(
            train, spec,
            k_candidates=candidates_per_round,
            recent_n=recent_n,
            bias_hot=bias_hot,
            avoid_3consec=avoid_3consec,
            avoid_single_decade=avoid_single_decade,
            rng_seed=idx + 12345
        )

        best_hit = 0
        best_main = None
        for c in cands:
            hit = len(set(c["main"]) & set(target.main))
            if hit > best_hit:
                best_hit = hit
                best_main = c["main"]

        rows.append({
            "round": target.round_no,
            "target_main": " ".join(map(str, target.main)),
            "target_bonus": " ".join(map(str, target.bonus)) if target.bonus else "",
            "best_hit_main": best_hit,
            "best_candidate_main": " ".join(map(str, best_main)) if best_main else "",
        })

    return pd.DataFrame(rows)


# =========================
# ヘッダー
# =========================
st.markdown(
    """
    <div class="topline">
      <div class="badge">ロト6・ロト7分析ツール</div>
      <div class="badge2">貼り付け → 自動取り込み → 分析 → 候補生成（ボーナス対応）</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='notice'>当せんや利益を保証するものではありません。分析結果は「買い方のルール化」や「記録の手間削減」のためにご利用ください。</div>",
    unsafe_allow_html=True,
)


# =========================
# UI
# =========================
tabs = st.tabs(["入力", "分析", "生成", "バックテスト", "設定メモ"])

with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1.0], gap="large")

    with left:
        game_name = st.selectbox("ゲーム", ["ロト6", "ロト7"], index=0)
        spec = GAME_MAP[game_name]

        auto_import = st.toggle("貼り付けを自動で履歴に取り込み（おすすめ）", value=True)
        st.caption("貼り付け例（ボーナスは B を付ける）")
        if spec == LOTO6:
            st.code("第2067回 3,4,12,15,32,33 B34\n第2068回 5 7 8 9 10 11 B13", language="text")
        else:
            st.code("第600回 1,5,7,12,18,21,33 B2 35\n第601回 3 6 9 11 17 24 31 B:1 22", language="text")

        paste = st.text_area(
            "抽選結果を貼り付け（複数行OK）",
            value=st.session_state.paste_text,
            height=160,
            placeholder="ここに貼り付け…",
        )
        st.session_state.paste_text = paste

        if auto_import and (st.session_state.paste_text != st.session_state.last_processed_text):
            new_draws = parse_paste(st.session_state.paste_text, spec)
            added = add_draws(game_name, new_draws)
            st.session_state.last_processed_text = st.session_state.paste_text
            if added > 0:
                st.success(f"履歴に追加しました：{added}行（重複は自動で除外）")
            elif len(new_draws) == 0 and st.session_state.paste_text.strip():
                st.warning("追加できる行が見つかりませんでした。形式と数字個数を確認してください。")

        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("貼り付け内容を履歴に追加（手動）"):
                new_draws = parse_paste(st.session_state.paste_text, spec)
                added = add_draws(game_name, new_draws)
                if added > 0:
                    st.success(f"履歴に追加しました：{added}行（重複は自動で除外）")
                else:
                    st.warning("追加できる行が見つかりませんでした。形式と数字個数を確認してください。")
        with b2:
            if st.button("貼り付け欄をクリア"):
                st.session_state.paste_text = ""
                st.session_state.last_processed_text = ""
                st.rerun()
        with b3:
            if st.button("履歴をリセット（このゲームのみ）"):
                st.session_state.history[game_name] = []
                st.session_state.history_keyset[game_name] = set()
                st.rerun()

    with right:
        hist = st.session_state.history[game_name]
        st.subheader("現在の履歴")
        st.caption(f"{game_name}: {len(hist)}行")
        if len(hist) == 0:
            st.info("まだ履歴がありません。左側に貼り付けてください。")
        else:
            df = pd.DataFrame([{
                "回号": d.round_no,
                "本数字": " ".join(map(str, d.main)),
                "ボーナス": " ".join(map(str, d.bonus)) if d.bonus else ""
            } for d in hist[-50:]])
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(
            "<div class='warn'>ボーナス数字の入れ方：末尾に <b>B</b> を付けてください。例：<br>"
            "ロト6 → … <b>B34</b><br>ロト7 → … <b>B2 35</b></div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    game_name = st.selectbox("ゲーム（分析）", ["ロト6", "ロト7"], index=0, key="game_analysis")
    spec = GAME_MAP[game_name]
    hist = st.session_state.history[game_name]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("分析")
    if len(hist) == 0:
        st.info("分析するには、まず履歴を追加してください。")
    else:
        recent_n = st.slider("直近N回（分析）", 10, 80, 30, step=5, key="recent_analysis")
        stats = get_stats_cached(hist, spec, recent_n=recent_n)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Hot（直近で増加）")
            hot = stats.hot_score.sort_values(ascending=False).head(10)
            st.dataframe(pd.DataFrame({"num": hot.index, "hot_score": hot.values}), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("### Cold（直近で減少）")
            cold = stats.hot_score.sort_values(ascending=True).head(10)
            st.dataframe(pd.DataFrame({"num": cold.index, "hot_score": cold.values}), hide_index=True, use_container_width=True)
        with c3:
            st.markdown("### しばらく出てない（ギャップ）")
            gap = stats.last_seen_gap.sort_values(ascending=False).head(10)
            st.dataframe(pd.DataFrame({"num": gap.index, "gap": gap.values}), hide_index=True, use_container_width=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### 傾向の説明（自動）")
        lines = []
        short_keys = sorted(stats.streak_info.keys())[:10]
        if short_keys:
            lines.append("短期で複数回出ている数字： " + "、".join([f"{n}（{stats.streak_info[n]}）" for n in short_keys]))
        if not lines:
            lines = ["履歴が少ない、または偏りが弱いため、強い傾向は検出されませんでした。"]
        st.write("\n".join(lines))

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    game_name = st.selectbox("ゲーム（生成）", ["ロト6", "ロト7"], index=0, key="game_generate")
    spec = GAME_MAP[game_name]
    hist = st.session_state.history[game_name]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("候補生成（ボーナス対応）")

    if len(hist) == 0:
        st.info("生成するには、まず履歴を追加してください。")
    else:
        c1, c2, c3 = st.columns([1.0, 1.0, 1.2])
        with c1:
            recent_n = st.slider("直近N回（ホット/コールド）", 10, 80, 30, step=5, key="recent_gen")
            k_candidates = st.number_input("候補数", min_value=5, max_value=200, value=30, step=5)
        with c2:
            bias_hot = st.slider("ホット寄り ↔ コールド寄り", 0.0, 1.0, 0.65, 0.05)
            avoid_3consec = st.checkbox("3つ以上の連番を避ける", value=True)
            avoid_single_decade = st.checkbox("同じ10番代だけを避ける", value=True)
        with c3:
            st.markdown(
                "<div class='notice'>候補のボーナスは「残り数字から重み上位」を自動で提案します。"
                "購入時のボーナスは運営側が決めるため、ここでは“補助”として扱います。</div>",
                unsafe_allow_html=True,
            )

        if st.button("候補を生成する"):
            with st.spinner("候補を生成中…"):
                time.sleep(0.15)
                cands = generate_candidates(
                    hist, spec,
                    k_candidates=int(k_candidates),
                    recent_n=int(recent_n),
                    bias_hot=float(bias_hot),
                    avoid_3consec=bool(avoid_3consec),
                    avoid_single_decade=bool(avoid_single_decade),
                )
                st.session_state.generated = cands

        cands = st.session_state.get("generated", [])
        if cands:
            rows = []
            for i, c in enumerate(cands, start=1):
                rows.append({
                    "No.": i,
                    "本数字": " ".join(map(str, c["main"])),
                    "ボーナス提案": " ".join(map(str, c["bonus"])) if c["bonus"] else "",
                    "理由（簡易）": c["reason"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.markdown("### 見やすい表示（本数字＝黒 / ボーナス＝黄）")
            for i, c in enumerate(cands[:10], start=1):
                chips_html = "<div class='chips'>"
                for n in c["main"]:
                    chips_html += f"<span class='chip main'>{n}</span>"
                for b in c["bonus"]:
                    chips_html += f"<span class='chip bonus'>B{b}</span>"
                chips_html += "</div>"
                st.markdown(
                    f"<div class='card'><b>候補 {i}</b><br>{chips_html}"
                    f"<div style='margin-top:8px; color:#64748b; font-size:0.92rem;'>{c['reason']}</div></div>",
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    game_name = st.selectbox("ゲーム（バックテスト）", ["ロト6", "ロト7"], index=0, key="game_bt")
    spec = GAME_MAP[game_name]
    hist = st.session_state.history[game_name]

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("バックテスト（簡易）")
    st.caption("過去N回を順に「直前の履歴だけ」で学習→候補生成→一致数を記録します。")

    if len(hist) < 30:
        st.info("履歴が少ないため、バックテストは十分に動きません。まず履歴を増やしてください。")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            test_last_n = st.number_input("検証する直近回数（N）", 20, 200, 50, step=10)
            train_window = st.number_input("学習に使う過去回数", 30, 300, 80, step=10)
        with col2:
            candidates_per_round = st.number_input("各回の生成候補数", 10, 200, 30, step=10)
            recent_n = st.number_input("直近N回（ホット/コールド）", 10, 80, 30, step=5)
        with col3:
            bias_hot = st.slider("ホット寄り ↔ コールド寄り", 0.0, 1.0, 0.65, 0.05, key="bt_bias")
            avoid_3consec = st.checkbox("3連番回避", True, key="bt_c1")
            avoid_single_decade = st.checkbox("同一10番代回避", True, key="bt_c2")

        if st.button("バックテストを実行"):
            with st.spinner("バックテスト中…"):
                dfbt = backtest(
                    hist, spec,
                    test_last_n=int(test_last_n),
                    train_window=int(train_window),
                    candidates_per_round=int(candidates_per_round),
                    recent_n=int(recent_n),
                    bias_hot=float(bias_hot),
                    avoid_3consec=bool(avoid_3consec),
                    avoid_single_decade=bool(avoid_single_decade),
                )
            if dfbt.empty:
                st.warning("バックテスト結果を作れませんでした（履歴不足の可能性）。")
            else:
                st.success("完了しました。")
                st.dataframe(dfbt, use_container_width=True, hide_index=True)
                csv = dfbt.to_csv(index=False).encode("utf-8-sig")
                st.download_button("CSVでダウンロード", data=csv, file_name=f"backtest_{game_name}.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("設定メモ（購入者向け案内用）")
    st.session_state.public_url = st.text_input("公開URL（ここに貼る）", value=st.session_state.public_url)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### iPhone（Safari）での使い方（商品説明にコピペ用）")
    howto = f"""
1. Safariで下のURLを開きます  
{st.session_state.public_url}

2. 画面下の「共有」→「ホーム画面に追加」  
3. ホーム画面のアイコンから開くと、アプリ風に使えます

ボーナス数字の入れ方  
・ロト6：本数字6個のあとに「B ボーナス」を1つ  
例：第2067回 3,4,12,15,32,33 B34  
・ロト7：本数字7個のあとに「B ボーナス」を最大2つ  
例：第600回 1,5,7,12,18,21,33 B2 35
"""
    st.code(howto, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("© ロト6・ロト7分析ツール（個人利用向け）")
