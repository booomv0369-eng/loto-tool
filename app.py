# -*- coding: utf-8 -*-
"""
ロト6・ロト7分析ツール（Streamlit） v4
- 貼り付け自動取り込み（任意）
- ボーナス数字の保存・生成・見やすい表示
- 詳細分析（短期偏り・10番台トレンド・直近出現密度など）
- 簡易バックテスト（過去データで候補生成を再現し、ヒット度を集計）

注意：当せんや利益を保証するものではありません。娯楽の範囲でご利用ください。
"""
from __future__ import annotations

import io
import re
import time
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# ページ設定 & スタイル
# -----------------------------
st.set_page_config(page_title="ロト6・ロト7分析ツール", layout="wide")

CSS = """
<style>
/* 全体 */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: 0.02em;}
/* カード */
.card {
  background: #ffffff;
  border: 1px solid #e6e8ee;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}
.mini {
  color: #6b7280;
  font-size: 0.92rem;
}
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  background: #0f172a;
  color: #ffffff;
  font-size: 0.80rem;
}
.badge-bonus {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: #f59e0b;
  color: #0f172a;
  font-weight: 700;
  font-size: 0.80rem;
}
.hr {height: 1px; background:#eef0f5; margin: 10px 0 14px 0;}
/* 重要メッセージ */
.notice {
  border-left: 6px solid #0ea5e9;
  background: #f0f9ff;
  padding: 12px 14px;
  border-radius: 10px;
  color: #0f172a;
}
/* 小さめテキスト */
.small {font-size: 0.9rem; color: #475569;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

APP_TITLE = "ロト6・ロト7分析ツール"
APP_SUB = "貼り付け → 自動取り込み（任意）→ 分析 → 候補生成（ボーナス対応）→ バックテスト"

# -----------------------------
# ゲーム定義
# -----------------------------
@dataclass(frozen=True)
class GameSpec:
    name: str
    pick: int
    max_n: int
    has_bonus: bool = True

LOTO6 = GameSpec("ロト6", pick=6, max_n=43, has_bonus=True)
LOTO7 = GameSpec("ロト7", pick=7, max_n=37, has_bonus=True)

# -----------------------------
# 解析・生成ユーティリティ
# -----------------------------
def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None

def normalize_main(nums: List[int], spec: GameSpec) -> List[int]:
    nums = [int(n) for n in nums if n is not None]
    nums = [n for n in nums if 1 <= n <= spec.max_n]
    nums = sorted(nums)
    # ロトは同一数字が出ない想定だが、入力ゆれに備えて重複排除
    nums = sorted(set(nums))
    return nums

def parse_paste_block(text: str, spec: GameSpec) -> List[Tuple[Tuple[int, ...], Optional[int]]]:
    """
    コピペされた行から本数字と（あれば）ボーナスを抽出。
    例：
      第2067回 3,4,12,15,32,33,34
      2068回 5 7 8 9 10 11 B 13
    ルール：
      - 数字が pick 個以上あれば本数字に採用
      - 数字が pick+1 個以上なら最後の1つをボーナスとして採用（範囲チェックあり）
    """
    if not text or not text.strip():
        return []

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    out: List[Tuple[Tuple[int, ...], Optional[int]]] = []
    for ln in lines:
        # 回号っぽい表現を削除（第2067回、2067回 など）
        ln2 = re.sub(r"第?\s*\d+\s*回", " ", ln)
        ln2 = re.sub(r"[\(\)（）]", " ", ln2)
        ln2 = ln2.replace("、", ",").replace("　", " ")
        # B/ボーナス等の表記は区切りに
        ln2 = re.sub(r"[Bb]\s*[:=]?\s*", " ", ln2)
        ln2 = ln2.replace("ボーナス", " ").replace("bonus", " ")
        parts = re.split(r"[\s,]+", ln2.strip())
        nums = [_safe_int(p) for p in parts]
        nums = [n for n in nums if n is not None]

        if len(nums) < spec.pick:
            continue

        main = nums[:spec.pick]
        main = normalize_main(main, spec)
        if len(main) != spec.pick:
            # 重複などで本数字が不足した場合はスキップ
            continue

        bonus = None
        if spec.has_bonus and len(nums) >= spec.pick + 1:
            b = nums[spec.pick]
            if b is not None and 1 <= b <= spec.max_n and b not in main:
                bonus = int(b)

        out.append((tuple(main), bonus))
    return out

def detect_encoding_and_read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc_jp"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def standardize_draw_csv(df: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    """
    CSV列を内部形式へ正規化：n1..nK + bonus
    想定：抽選履歴CSV（銀行明細CSVなどは対象外）
    """
    if spec.pick == 6:
        candidates = [
            [f"抽せん数字{i}" for i in range(1, 7)],
            [f"抽選数字{i}" for i in range(1, 7)],
            [f"本数字{i}" for i in range(1, 7)],
        ]
    else:
        candidates = [
            [f"本数字{i}" for i in range(1, 8)],
            [f"抽せん数字{i}" for i in range(1, 8)],
            [f"抽選数字{i}" for i in range(1, 8)],
        ]

    num_cols = None
    for arr in candidates:
        if all(a in df.columns for a in arr):
            num_cols = arr
            break

    if num_cols is None:
        # 推測：数値列から上位K列
        numeric_cols = []
        for c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(1, len(df) // 3):
                numeric_cols.append(c)
        if len(numeric_cols) >= spec.pick:
            num_cols = numeric_cols[: spec.pick]
        else:
            raise ValueError("抽選履歴CSVの列構造ではありません（数字列が不足）。")

    bonus_col = None
    for bc in ["ボーナス数字", "ﾎﾞｰﾅｽ数字", "BONUS", "bonus"]:
        if bc in df.columns:
            bonus_col = bc
            break

    out = pd.DataFrame()
    for i, c in enumerate(num_cols, start=1):
        out[f"n{i}"] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    if spec.has_bonus and bonus_col:
        out["bonus"] = pd.to_numeric(df[bonus_col], errors="coerce").astype("Int64")
    else:
        out["bonus"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    out = out.dropna(subset=[f"n{i}" for i in range(1, spec.pick + 1)], how="any").reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def calc_freq_main(hist: pd.DataFrame, spec: GameSpec) -> pd.Series:
    vals = hist[[f"n{i}" for i in range(1, spec.pick + 1)]].to_numpy().flatten()
    vals = vals[~pd.isna(vals)].astype(int)
    s = pd.Series(vals).value_counts().sort_index()
    return s.reindex(range(1, spec.max_n + 1), fill_value=0)

@st.cache_data(show_spinner=False)
def calc_freq_bonus(hist: pd.DataFrame, spec: GameSpec) -> pd.Series:
    if "bonus" not in hist.columns:
        return pd.Series([0]*spec.max_n, index=range(1, spec.max_n+1))
    vals = hist["bonus"].dropna().astype(int).to_numpy()
    s = pd.Series(vals).value_counts().sort_index()
    return s.reindex(range(1, spec.max_n + 1), fill_value=0)

def violates_consecutive(nums: List[int], k: int = 3) -> bool:
    run = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            run += 1
            if run >= k:
                return True
        else:
            run = 1
    return False

def decade_bucket(n: int) -> int:
    # 1-9:0, 10-19:1, 20-29:2...
    return n // 10

def decade_skew(nums: List[int]) -> int:
    buckets: Dict[int, int] = {}
    for n in nums:
        b = decade_bucket(n)
        buckets[b] = buckets.get(b, 0) + 1
    return max(buckets.values()) if buckets else 0

def is_birthday_heavy(nums: List[int], threshold: int) -> bool:
    return sum(1 <= n <= 31 for n in nums) >= threshold

def weighted_choice(numbers: np.ndarray, weights: np.ndarray, k: int, forbid: Optional[set]=None) -> List[int]:
    forbid = forbid or set()
    mask = np.array([n not in forbid for n in numbers])
    nums2 = numbers[mask]
    w2 = weights[mask]
    w2 = w2 / w2.sum()
    pick = np.random.choice(nums2, size=k, replace=False, p=w2)
    return [int(x) for x in pick]

def gen_candidates(hist: pd.DataFrame, spec: GameSpec, n: int, bias: float,
                   avoid_consec: bool, avoid_bday: bool, max_decade: int,
                   gen_bonus: bool, seed: Optional[int] = None) -> pd.DataFrame:
    """
    本数字 +（任意で）ボーナス を生成。
    bias: -1(cold) ... 0 ... +1(hot)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    freq = calc_freq_main(hist, spec).astype(float) + 1.0  # 平滑化
    p = 1.2 + abs(bias) * 1.8
    if bias >= 0:
        w = np.power(freq.to_numpy(), p)
    else:
        w = np.power(1.0 / freq.to_numpy(), p)
    w = w / w.sum()
    numbers = np.arange(1, spec.max_n + 1)

    # ボーナス用の重み（あれば）
    b_w = None
    if gen_bonus and spec.has_bonus:
        bfreq = calc_freq_bonus(hist, spec).astype(float) + 1.0
        if bias >= 0:
            b_w = np.power(bfreq.to_numpy(), p)
        else:
            b_w = np.power(1.0 / bfreq.to_numpy(), p)
        b_w = b_w / b_w.sum()

    rows = []
    tries = 0
    max_tries = max(7000, n * 250)

    seen = set()
    while len(rows) < n and tries < max_tries:
        tries += 1
        main = weighted_choice(numbers, w, spec.pick)
        main = sorted(main)

        if avoid_consec and violates_consecutive(main, 3):
            continue
        if avoid_bday and is_birthday_heavy(main, threshold=max(5, spec.pick - 1)):
            continue
        if decade_skew(main) > max_decade:
            continue

        bonus = None
        if b_w is not None:
            # ボーナスは本数字と重複させない
            bonus = weighted_choice(numbers, b_w, 1, forbid=set(main))[0]

        key = (tuple(main), bonus)
        if key in seen:
            continue
        seen.add(key)

        row = {f"n{i}": main[i-1] for i in range(1, spec.pick+1)}
        row["bonus"] = bonus
        rows.append(row)

    return pd.DataFrame(rows)

def explain_trends(hist: pd.DataFrame, spec: GameSpec, recent_n: int = 30) -> Dict[str, str]:
    """
    “納得しやすい説明”を作るための簡易コメント生成。
    注意：抽選は独立のため確率が変わるわけではない。その前提で「買い方のルール化」材料として提示。
    """
    freq_all = calc_freq_main(hist, spec)
    hist_recent = hist.tail(min(recent_n, len(hist)))
    freq_recent = calc_freq_main(hist_recent, spec)

    # 10番台トレンド（直近と全体の差）
    def decade_counts(freq: pd.Series):
        counts = {}
        for n, c in freq.items():
            b = decade_bucket(int(n))
            counts[b] = counts.get(b, 0) + int(c)
        return counts

    dc_all = decade_counts(freq_all)
    dc_recent = decade_counts(freq_recent)

    # 直近の比率が高い10番台を抽出
    messages = {}

    total_all = sum(dc_all.values()) or 1
    total_recent = sum(dc_recent.values()) or 1
    ratios = []
    for b in sorted(set(dc_all) | set(dc_recent)):
        r_all = dc_all.get(b, 0) / total_all
        r_rec = dc_recent.get(b, 0) / total_recent
        ratios.append((b, r_rec - r_all, r_rec, r_all))
    ratios.sort(key=lambda x: x[1], reverse=True)

    if ratios:
        top = ratios[0]
        b = top[0]
        label = f"{b*10}番台" if b > 0 else "1〜9"
        messages["decade"] = (
            f"直近{recent_n}回では、全期間に比べて「{label}」の比率がやや高めです。"
            "ただし抽選は独立なので、確率が変化するわけではありません。"
            "買い方のルール化として、該当帯を少し厚めにする判断材料にできます。"
        )

    # “短期間に3回以上” 出ている数字（過密）
    # 直近W回での出現回数を見て抽出
    W = min(30, len(hist))
    freq_w = calc_freq_main(hist.tail(W), spec)
    crowded = freq_w[freq_w >= 3].index.tolist()
    if crowded:
        s = "、".join(map(str, crowded[:10]))
        messages["crowded"] = (
            f"直近{W}回の中で3回以上出ている数字があります（例：{s}）。"
            "『短期で出すぎだから次は控える』という買い方ルールを作るなら、ここが候補になります。"
        )
    else:
        messages["crowded"] = f"直近{W}回で3回以上出ている数字は見当たりませんでした。"

    # “長期間出ていない”数字（空白）
    # 最終出現からの距離を計算
    last_seen = {n: None for n in range(1, spec.max_n+1)}
    main_cols = [f"n{i}" for i in range(1, spec.pick+1)]
    for idx, row in hist[main_cols].iterrows():
        for n in row.tolist():
            last_seen[int(n)] = idx
    cur = len(hist)-1
    gaps = []
    for n, idx in last_seen.items():
        if idx is None:
            gaps.append((n, 10**9))
        else:
            gaps.append((n, cur-idx))
    gaps.sort(key=lambda x: x[1], reverse=True)
    topg = gaps[:10]
    messages["gaps"] = "最近出ていない順（回数差）：" + " / ".join([f"{n}({g})" for n,g in topg])

    return messages

def style_candidates(df: pd.DataFrame, spec: GameSpec) -> pd.io.formats.style.Styler:
    df2 = df.copy()
    # 表示用に bonus を "B:xx" に
    def fmt_bonus(x):
        if pd.isna(x) or x is None:
            return ""
        return f"B:{int(x)}"
    df2["bonus"] = df2["bonus"].apply(fmt_bonus)

    def highlight_bonus(val):
        if isinstance(val, str) and val.startswith("B:"):
            return "background-color: #fef3c7; color:#0f172a; font-weight:700;"
        return ""

    sty = df2.style.applymap(highlight_bonus, subset=["bonus"])
    return sty

def match_score(draw_main: List[int], draw_bonus: Optional[int], cand_main: List[int], cand_bonus: Optional[int]) -> Tuple[int, bool]:
    main_hit = len(set(draw_main) & set(cand_main))
    bonus_hit = (draw_bonus is not None and cand_bonus is not None and int(draw_bonus) == int(cand_bonus))
    return main_hit, bool(bonus_hit)

def run_backtest(hist: pd.DataFrame, spec: GameSpec, train_window: int, test_last: int, n_cands: int,
                 bias: float, avoid_consec: bool, avoid_bday: bool, max_decade: int, gen_bonus: bool, seed: Optional[int]) -> pd.DataFrame:
    main_cols = [f"n{i}" for i in range(1, spec.pick+1)]
    rows = []
    N = len(hist)
    start = max(train_window, N - test_last)
    for t in range(start, N):
        train = hist.iloc[t-train_window:t].copy()
        test_row = hist.iloc[t]
        draw_main = [int(test_row[c]) for c in main_cols]
        draw_bonus = _safe_int(test_row.get("bonus", None))

        cands = gen_candidates(train, spec, n=n_cands, bias=bias,
                               avoid_consec=avoid_consec, avoid_bday=avoid_bday,
                               max_decade=max_decade, gen_bonus=gen_bonus, seed=seed)

        best_main = -1
        best_bonus = False
        for _, r in cands.iterrows():
            cand_main = [int(r[c]) for c in main_cols]
            cand_bonus = _safe_int(r.get("bonus", None))
            m, b = match_score(draw_main, draw_bonus, cand_main, cand_bonus)
            if (m > best_main) or (m == best_main and b and not best_bonus):
                best_main, best_bonus = m, b

        rows.append({
            "対象Index": t,
            "本数字ヒット最大": best_main,
            "ボーナス一致": best_bonus,
        })
    return pd.DataFrame(rows)

# -----------------------------
# セッション初期化
# -----------------------------
def ensure_state():
    if "spec_name" not in st.session_state:
        st.session_state.spec_name = LOTO6.name
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame()
    if "paste_box" not in st.session_state:
        st.session_state.paste_box = ""
    if "auto_import" not in st.session_state:
        st.session_state.auto_import = True
    if "seen_keys" not in st.session_state:
        st.session_state.seen_keys = set()

ensure_state()

# -----------------------------
# ヘッダー
# -----------------------------
st.markdown(f"<div class='badge'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<h2 style='margin-top:10px'>{APP_SUB}</h2>", unsafe_allow_html=True)
st.markdown("<div class='notice'>当せんや利益を保証するものではありません。分析結果は「買い方のルール化」や「記録の手間削減」のためにご利用ください。</div>", unsafe_allow_html=True)

# -----------------------------
# サイドバー
# -----------------------------
spec = LOTO6 if st.session_state.spec_name == LOTO6.name else LOTO7

with st.sidebar:
    st.subheader("対象ゲーム")
    game = st.radio("ロト", [LOTO6.name, LOTO7.name], index=0 if spec == LOTO6 else 1)
    st.session_state.spec_name = game
    spec = LOTO6 if game == LOTO6.name else LOTO7

    st.divider()
    st.subheader("入力オプション")
    st.session_state.auto_import = st.toggle("貼り付け自動取り込み", value=st.session_state.auto_import,
                                             help="ONなら、貼り付け欄の内容を検知して自動で履歴に追加します（重複は追加しません）。")

    st.divider()
    st.subheader("候補生成")
    bias = st.slider("ホット / コールド寄り", -1.0, 1.0, 0.3, 0.1,
                     help="-1=コールド寄り / +1=ホット寄り（傾向を“強める”だけで確率を変えるものではありません）")
    n_cands = st.number_input("生成数", min_value=1, max_value=300, value=30, step=1)
    gen_bonus = st.checkbox("ボーナスも生成", value=True, help="ONで候補にボーナス（B）も付けます。")
    avoid_consec = st.checkbox("3連番以上を除外", value=True)
    avoid_bday = st.checkbox("誕生日数字の偏りを除外", value=True)
    max_dec = st.slider("同じ10番台の最大個数", min_value=2, max_value=spec.pick, value=min(4, spec.pick), step=1)
    seed_txt = st.text_input("再現用シード（任意）", value="", help="数値を入れると同じ結果を再現しやすいです。")

# -----------------------------
# 自動取り込みコールバック
# -----------------------------
def import_from_paste():
    text = st.session_state.paste_box
    parsed = parse_paste_block(text, spec)
    if not parsed:
        return
    # 履歴に追加（重複除外）
    add_rows = []
    for main, bonus in parsed:
        key = (spec.name, main, bonus)
        if key in st.session_state.seen_keys:
            continue
        st.session_state.seen_keys.add(key)
        row = {f"n{i}": main[i-1] for i in range(1, spec.pick+1)}
        row["bonus"] = bonus
        add_rows.append(row)

    if add_rows:
        df_new = pd.DataFrame(add_rows)
        # spec切替時に列が変わるので整合をとる
        if st.session_state.history is None or len(st.session_state.history) == 0:
            st.session_state.history = df_new.copy()
        else:
            # 列が違う場合は上書き（ゲーム切替に追従）
            if set(st.session_state.history.columns) != set(df_new.columns):
                st.session_state.history = df_new.copy()
            else:
                st.session_state.history = pd.concat([st.session_state.history, df_new], ignore_index=True)

# -----------------------------
# タブ
# -----------------------------
tabs = st.tabs(["入力", "分析", "生成", "バックテスト", "設定メモ"])

# -----------------------------
# 入力
# -----------------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("貼り付け（おすすめ）")
    st.markdown("<div class='mini'>例：第2067回 3,4,12,15,32,33,34（最後がボーナス） / 2068回 5 7 8 9 10 11 B 13</div>", unsafe_allow_html=True)

    # on_change で自動取り込み
    st.text_area(
        "抽選結果を貼り付け",
        key="paste_box",
        height=120,
        on_change=(import_from_paste if st.session_state.auto_import else None),
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    if col1.button("手動で履歴に追加", use_container_width=True):
        import_from_paste()
        st.success("追加しました（重複は自動で除外）。")

    if col2.button("貼り付け欄をクリア", use_container_width=True):
        st.session_state.paste_box = ""

    if col3.button("履歴をリセット", use_container_width=True):
        st.session_state.history = pd.DataFrame()
        st.session_state.seen_keys = set()
        st.success("履歴をリセットしました。")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("CSV取り込み（任意）")
    up = st.file_uploader("抽選履歴CSVをアップロード", type=["csv"])
    if up is not None:
        try:
            df_raw = detect_encoding_and_read_csv(up)
            df_std = standardize_draw_csv(df_raw, spec)
            st.session_state.history = df_std.copy()
            # seen_keys を再構築（重複除外のため）
            keys = set()
            main_cols = [f"n{i}" for i in range(1, spec.pick+1)]
            for _, r in df_std.iterrows():
                main = tuple(int(r[c]) for c in main_cols)
                bonus = _safe_int(r.get("bonus", None))
                keys.add((spec.name, main, bonus))
            st.session_state.seen_keys = keys
            st.success(f"CSVを読み込みました（{len(df_std)}行）。")
        except Exception as e:
            st.error(
                "CSVの読み込みに失敗しました。\n\n"
                "・このツールは『抽選履歴CSV』を想定しています（銀行明細CSVなどは対象外）。\n"
                f"・必要：{spec.pick}個の数字列（本数字/抽せん数字など）。\n\n"
                f"エラー：{e}"
            )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("現在の履歴（最新10行）")
    hist = st.session_state.history
    if hist is None or len(hist) == 0:
        st.info("履歴がまだありません。貼り付けかCSVで追加してください。")
    else:
        st.dataframe(hist.tail(10), use_container_width=True, height=320)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 分析
# -----------------------------
with tabs[1]:
    hist = st.session_state.history
    if hist is None or len(hist) == 0:
        st.info("分析するには履歴を追加してください。")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("サマリー")

        freq = calc_freq_main(hist, spec)
        bfq = calc_freq_bonus(hist, spec) if spec.has_bonus else None
        total_draws = len(hist)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("履歴行数", f"{total_draws}")
        c2.metric("最大数字", f"{spec.max_n}")
        c3.metric("ホット上位の最大回数", f"{int(freq.max())}")
        c4.metric("コールド最小回数", f"{int(freq.min())}")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        st.subheader("ホット / コールド（本数字）")
        df_freq = pd.DataFrame({"数字": freq.index, "回数": freq.values})
        colA, colB = st.columns(2)
        with colA:
            st.caption("ホット上位（回数が多い）")
            st.table(df_freq.sort_values("回数", ascending=False).head(12))
        with colB:
            st.caption("コールド上位（回数が少ない）")
            st.table(df_freq.sort_values("回数", ascending=True).head(12))

        if spec.has_bonus:
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.subheader("ボーナス（B）の頻度")
            df_b = pd.DataFrame({"数字": bfq.index, "回数": bfq.values})
            col1, col2 = st.columns(2)
            with col1:
                st.caption("ボーナスのホット上位")
                st.table(df_b.sort_values("回数", ascending=False).head(10))
            with col2:
                st.caption("ボーナスのコールド上位")
                st.table(df_b.sort_values("回数", ascending=True).head(10))

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.subheader("納得しやすい説明（自動コメント）")
        recent_n = st.slider("直近何回で見る？", 10, 200, 30, 5)
        msgs = explain_trends(hist, spec, recent_n=recent_n)
        st.markdown(f"<div class='small'>・{msgs.get('decade','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>・{msgs.get('crowded','')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>・{msgs.get('gaps','')}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 生成
# -----------------------------
with tabs[2]:
    hist = st.session_state.history
    if hist is None or len(hist) == 0:
        st.info("候補生成には履歴が必要です。まずは入力タブで履歴を追加してください。")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("候補生成")

        seed_val = None
        if seed_txt.strip():
            try:
                seed_val = int(seed_txt.strip())
            except Exception:
                seed_val = None

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<div class='mini'>ボーナスは <span class='badge-bonus'>B</span> として表示します。</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='mini'>重いときは「生成数」を下げると体感が改善します。</div>", unsafe_allow_html=True)

        if st.button("候補を生成", type="primary"):
            with st.spinner("生成中…"):
                t0 = time.time()
                df_c = gen_candidates(
                    hist=hist, spec=spec, n=int(n_cands), bias=float(bias),
                    avoid_consec=bool(avoid_consec), avoid_bday=bool(avoid_bday),
                    max_decade=int(max_dec), gen_bonus=bool(gen_bonus), seed=seed_val
                )
                time.sleep(0.05)
                t1 = time.time()

            if df_c is None or len(df_c) == 0:
                st.warning("条件が厳しすぎて候補が生成できませんでした。条件を緩めて再実行してください。")
            else:
                st.success(f"{len(df_c)}件を生成しました（{(t1 - t0):.2f}秒）")

                # 表示（ボーナスを目立たせる）
                sty = style_candidates(df_c, spec)
                st.dataframe(sty, use_container_width=True, height=420)

                # コピペ用
                st.subheader("コピペ用（本数字 + ボーナス）")
                main_cols = [f"n{i}" for i in range(1, spec.pick+1)]
                lines = []
                for _, r in df_c.iterrows():
                    main = [str(int(r[c])) for c in main_cols]
                    b = r.get("bonus", None)
                    if pd.isna(b) or b is None:
                        lines.append(",".join(main))
                    else:
                        lines.append(",".join(main) + f",B:{int(b)}")
                st.text_area("ここをコピー", value="\n".join(lines), height=180)

        st.caption("注意：抽選は独立です。ここでの“納得”は「買い方のルール化」に役立つ説明として提示しています。")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# バックテスト
# -----------------------------
with tabs[3]:
    hist = st.session_state.history
    if hist is None or len(hist) == 0:
        st.info("バックテストには履歴が必要です。まずは入力タブで履歴を追加してください。")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("バックテスト（簡易）")
        st.markdown("<div class='mini'>過去の各回を“未来”と見立て、直前の履歴から候補生成→最大ヒット数を集計します。</div>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        train_window = colA.number_input("学習窓（直前何回で作る？）", min_value=30, max_value=800, value=200, step=10)
        test_last = colB.number_input("直近何回を検証？", min_value=10, max_value=300, value=50, step=10)
        n_c = colC.number_input("各回で生成する候補数", min_value=5, max_value=300, value=min(30, int(n_cands)), step=5)

        seed_val = None
        if seed_txt.strip():
            try:
                seed_val = int(seed_txt.strip())
            except Exception:
                seed_val = None

        if st.button("バックテストを実行", type="primary"):
            with st.spinner("集計中…"):
                bt = run_backtest(
                    hist=hist, spec=spec,
                    train_window=int(train_window), test_last=int(test_last), n_cands=int(n_c),
                    bias=float(bias), avoid_consec=bool(avoid_consec), avoid_bday=bool(avoid_bday),
                    max_decade=int(max_dec), gen_bonus=bool(gen_bonus), seed=seed_val
                )

            if bt is None or len(bt) == 0:
                st.warning("バックテスト対象が不足しています。学習窓や検証範囲を調整してください。")
            else:
                st.success(f"{len(bt)}回分の検証が完了しました。")
                st.dataframe(bt.tail(50), use_container_width=True, height=320)

                # 集計
                st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
                st.subheader("集計（本数字ヒット最大）")
                dist = bt["本数字ヒット最大"].value_counts().sort_index()
                dist_df = dist.rename_axis("ヒット数").reset_index(name="回数")
                st.dataframe(dist_df, use_container_width=True, height=260)
                st.bar_chart(dist_df.set_index("ヒット数")["回数"])

                if spec.has_bonus and gen_bonus:
                    st.subheader("ボーナス一致（最大ヒットの候補の中で一致した回数）")
                    bcnt = int(bt["ボーナス一致"].sum())
                    st.write(f"ボーナス一致：{bcnt} / {len(bt)} 回")

        st.caption("注意：バックテスト結果が良くても、将来の当せんを保証するものではありません。")
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# 設定メモ
# -----------------------------
with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("設定メモ（あなた用）")
    st.markdown("""
- まずは「生成数：30」「学習窓：200」「検証：50」でバックテストし、重くなるなら数値を下げる  
- ボーナスは <span class='badge-bonus'>B</span> で表示。候補にも付けたい場合は「ボーナスも生成」をON  
- 自動取り込みは便利ですが、貼り付け直後に重い場合はOFFにして「手動で履歴に追加」でもOK  
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
