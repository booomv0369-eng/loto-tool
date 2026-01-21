# app.py
# 宝くじ分析ツール（ロト6・ロト7対応）
# 入力：貼り付け / CSVアップロード（任意）
# 分析：直近N回の出現頻度など
# 生成：ホット/コールド寄りスライダー＋除外/固定＋重なり制限＋高速化

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 0) 基本設定
# =========================
st.set_page_config(page_title="宝くじ分析ツール", layout="wide")

st.title("宝くじ分析ツール")
st.caption("ロト6・ロト7対応。履歴は貼り付け方式が最速です。CSVは任意です。")


@dataclass(frozen=True)
class GameSpec:
    name: str
    pick: int
    min_n: int
    max_n: int


SPECS: Dict[str, GameSpec] = {
    "ロト6": GameSpec(name="ロト6", pick=6, min_n=1, max_n=43),
    "ロト7": GameSpec(name="ロト7", pick=7, min_n=1, max_n=37),
}


def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = {}  # game -> DataFrame
    if "generated" not in st.session_state:
        st.session_state.generated = {}  # game -> list[list[int]]


ensure_state()


# =========================
# 1) ユーティリティ
# =========================
def normalize_history_df(df: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    """内部表現に正規化: draw_no(Int64|NA), date(str|NA), n1..nK(Int64)"""
    df = df.copy()

    num_cols = [f"n{i}" for i in range(1, spec.pick + 1)]
    for c in num_cols:
        if c not in df.columns:
            df[c] = pd.NA

    if "draw_no" not in df.columns:
        df["draw_no"] = pd.NA
    if "date" not in df.columns:
        df["date"] = pd.NA

    df["draw_no"] = pd.to_numeric(df["draw_no"], errors="coerce").astype("Int64")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in num_cols:
        bad = df[c].notna() & ((df[c] < spec.min_n) | (df[c] > spec.max_n))
        df.loc[bad, c] = pd.NA

    df = df.dropna(subset=num_cols).copy()

    if df["draw_no"].notna().any():
        df = df.sort_values(["draw_no"], kind="stable")
        df = df.drop_duplicates(subset=["draw_no"], keep="last")
    else:
        df = df.drop_duplicates(subset=num_cols, keep="last")

    cols = ["draw_no", "date"] + num_cols
    df = df[cols].reset_index(drop=True)
    return df


def merge_history(old: pd.DataFrame, new: pd.DataFrame, spec: GameSpec) -> pd.DataFrame:
    if old is None or len(old) == 0:
        return normalize_history_df(new, spec)
    old_n = normalize_history_df(old, spec)
    new_n = normalize_history_df(new, spec)
    merged = pd.concat([old_n, new_n], ignore_index=True)
    merged = normalize_history_df(merged, spec)
    return merged


def parse_paste_lines(text: str, spec: GameSpec) -> pd.DataFrame:
    """
    貼り付けテキストから抽選履歴を抽出。
    例:
      第2067回 3,4,12,15,32,42
      2068 2 10 13 14 29 33
      3 4 12 15 32 42
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        nums = [int(x) for x in re.findall(r"\d+", line)]
        if len(nums) < spec.pick:
            continue

        draw_no = None
        picks = None

        if "回" in line and len(nums) >= spec.pick + 1:
            draw_no = nums[0]
            picks = nums[1 : 1 + spec.pick]
        else:
            if len(nums) >= spec.pick + 1 and nums[0] > spec.max_n:
                draw_no = nums[0]
                picks = nums[1 : 1 + spec.pick]
            else:
                picks = nums[:spec.pick]

        if picks is None or len(picks) != spec.pick:
            continue
        if len(set(picks)) != spec.pick:
            continue
        if any((n < spec.min_n) or (n > spec.max_n) for n in picks):
            continue

        row = {"draw_no": draw_no, "date": pd.NA}
        for i, n in enumerate(picks, start=1):
            row[f"n{i}"] = n
        rows.append(row)

    df = pd.DataFrame(rows)
    return normalize_history_df(df, spec)


def try_read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """CSVの文字コードをいくつか試して読む（utf-8以外も吸う）。"""
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis", "euc-jp", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"CSVの読み込みに失敗しました: {last_err}")


def find_number_columns(df: pd.DataFrame, spec: GameSpec) -> Optional[List[str]]:
    cols = list(df.columns)

    ncols = [f"n{i}" for i in range(1, spec.pick + 1)]
    if all(c in cols for c in ncols):
        return ncols

    patterns = [
        [f"数字{i}" for i in range(1, spec.pick + 1)],
        [f"抽せん数字{i}" for i in range(1, spec.pick + 1)],
        [f"抽選数字{i}" for i in range(1, spec.pick + 1)],
        [f"number{i}" for i in range(1, spec.pick + 1)],
    ]
    for p in patterns:
        if all(c in cols for c in p):
            return p

    def find_like(target: str) -> Optional[str]:
        for c in cols:
            if str(c).replace(" ", "") == target.replace(" ", ""):
                return c
        return None

    cand = []
    for i in range(1, spec.pick + 1):
        hit = None
        for base in ["数字", "抽せん数字", "抽選数字"]:
            hit = find_like(f"{base}{i}")
            if hit is not None:
                break
        if hit is None:
            cand = []
            break
        cand.append(hit)
    if cand:
        return cand

    return None


def find_drawno_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        s = str(c).replace(" ", "")
        if s in ["draw_no", "回号", "抽選回", "抽せん回", "抽選回号", "抽せん回号", "第何回"]:
            return c
    return None


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        s = str(c).replace(" ", "")
        if s in ["date", "抽選日", "抽せん日", "日付"]:
            return c
    return None


@st.cache_data(show_spinner=False)
def calc_freq(df: pd.DataFrame, spec: GameSpec, recent_n: int) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series([0] * (spec.max_n - spec.min_n + 1), index=range(spec.min_n, spec.max_n + 1))

    recent_n = int(max(1, recent_n))
    recent_n = min(recent_n, len(df))

    d = df.tail(recent_n)
    cols = [f"n{i}" for i in range(1, spec.pick + 1)]
    s = pd.concat([d[c] for c in cols], ignore_index=True).dropna().astype(int)
    freq = s.value_counts().reindex(range(spec.min_n, spec.max_n + 1), fill_value=0).sort_index()
    return freq


def parse_int_list_csvlike(text: str) -> List[int]:
    if not text:
        return []
    parts = re.split(r"[,\s、]+", text.strip())
    out = []
    for p in parts:
        if not p:
            continue
        if re.fullmatch(r"\d+", p):
            out.append(int(p))
    return out


def weighted_pick_k(numbers: np.ndarray, weights: np.ndarray, k: int, rng: np.random.Generator) -> List[int]:
    w = weights.astype(float).copy()
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    picked = rng.choice(numbers, size=k, replace=False, p=w)
    return sorted(picked.tolist())


def generate_candidates(
    freq: pd.Series,
    spec: GameSpec,
    n_sets: int,
    exclude: Set[int],
    fixed: Set[int],
    overlap_max: int,
    bias: float,
    role_split: bool,
    seed: int,
) -> Tuple[List[List[int]], int]:
    rng = np.random.default_rng(int(seed))

    nums = freq.index.to_numpy()
    f = freq.to_numpy().astype(float)

    if exclude:
        mask = ~np.isin(nums, list(exclude))
        nums = nums[mask]
        f = f[mask]

    if fixed & exclude:
        return ([], 0)

    fixed_list = sorted(list(fixed))
    need = spec.pick - len(fixed_list)
    if need < 0:
        return ([], 0)

    pool_mask = ~np.isin(nums, fixed_list)
    pool_nums = nums[pool_mask]
    pool_f = f[pool_mask]
    if need > len(pool_nums):
        return ([], 0)

    base_w_hot = (pool_f + 1.0) ** float(max(-2.0, min(2.0, bias)))
    base_w_cold = (pool_f + 1.0) ** float(max(-2.0, min(2.0, -bias)))

    results: List[List[int]] = []
    tries = 0
    max_tries = max(4000, n_sets * 400)

    hot_k = need
    cold_k = 0
    if role_split and need > 0:
        hot_k = (need + 1) // 2
        cold_k = need - hot_k

    while len(results) < n_sets and tries < max_tries:
        tries += 1

        if need == 0:
            cand = fixed_list.copy()
        else:
            if role_split and cold_k > 0:
                hot_part = weighted_pick_k(pool_nums, base_w_hot, hot_k, rng)
                remain_mask = ~np.isin(pool_nums, hot_part)
                remain_nums = pool_nums[remain_mask]
                remain_w = base_w_cold[remain_mask]
                cold_part = weighted_pick_k(remain_nums, remain_w, cold_k, rng)
                cand = sorted(fixed_list + hot_part + cold_part)
            else:
                picked = weighted_pick_k(pool_nums, base_w_hot, need, rng)
                cand = sorted(fixed_list + picked)

        ok = True
        s_cand = set(cand)
        for prev in results:
            if len(set(prev) & s_cand) > overlap_max:
                ok = False
                break
        if ok:
            results.append(cand)

    return (results, tries)


def make_download_csv(df: pd.DataFrame, spec: GameSpec) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# =========================
# 2) UI
# =========================
tab_in, tab_an, tab_gen = st.tabs(["入力", "分析", "生成"])

with tab_in:
    st.subheader("ゲームと履歴")

    game = st.selectbox("ゲーム", list(SPECS.keys()), index=0, key="game_select")
    spec = SPECS[game]

    hist_df: pd.DataFrame = st.session_state.history.get(game, pd.DataFrame())
    hist_df = normalize_history_df(hist_df, spec) if len(hist_df) else hist_df
    st.session_state.history[game] = hist_df

    st.write("履歴の入れ方は2つ。おすすめはコピペ方式です。")

    st.markdown("### 1) 抽選結果を貼り付け（おすすめ）")
    example = "第2067回 3,4,12,15,32,42" if spec.pick == 6 else "第620回 1,7,13,19,23,31,37"
    paste = st.text_area(
        "ここに貼り付け",
        value="",
        height=120,
        placeholder=f"例）\n{example}\n（複数行OK）",
        key=f"paste_{game}",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("貼り付けを履歴に追加", key=f"btn_add_paste_{game}"):
            if not paste.strip():
                st.warning("貼り付けが空です。")
            else:
                new_df = parse_paste_lines(paste, spec)
                if len(new_df) == 0:
                    st.warning("追加できる行が見つかりませんでした。形式と数字個数を確認してください。")
                else:
                    before = len(hist_df)
                    hist_df = merge_history(hist_df, new_df, spec)
                    st.session_state.history[game] = hist_df
                    st.success(f"{len(hist_df) - before} 行を追加しました。")
    with c2:
        if st.button("履歴を消す", key=f"btn_clear_{game}"):
            st.session_state.history[game] = pd.DataFrame()
            st.session_state.generated[game] = []
            st.success("履歴を消しました。")

    st.markdown("### 2) CSVを読み込む（任意）")
    up = st.file_uploader("CSVアップロード", type=["csv"], key=f"uploader_{game}")

    if up is not None:
        try:
            raw = up.getvalue()
            df_csv = try_read_csv_bytes(raw)

            num_cols = find_number_columns(df_csv, spec)
            if not num_cols:
                st.error(
                    "このCSVは抽選履歴の列構造ではありません。\n\n"
                    f"必要: {spec.pick}個の数字列（例: n1..n{spec.pick} / 数字1..数字{spec.pick} / 抽選数字1..）。\n"
                    f"検出した列: {list(df_csv.columns)}"
                )
            else:
                draw_col = find_drawno_column(df_csv)
                date_col = find_date_column(df_csv)

                rows = []
                for _, r in df_csv.iterrows():
                    row = {
                        "draw_no": r[draw_col] if draw_col else pd.NA,
                        "date": r[date_col] if date_col else pd.NA,
                    }
                    for i, c in enumerate(num_cols, start=1):
                        row[f"n{i}"] = r[c]
                    rows.append(row)

                new_df = pd.DataFrame(rows)
                new_df = normalize_history_df(new_df, spec)

                if len(new_df) == 0:
                    st.warning("読み込みはできましたが、有効な行がありませんでした。")
                else:
                    before = len(hist_df)
                    hist_df = merge_history(hist_df, new_df, spec)
                    st.session_state.history[game] = hist_df
                    st.success(f"CSVを読み込みました: {len(new_df)} 行（統合後: {len(hist_df)} 行）")

        except Exception as e:
            st.error(f"CSVの読み込みに失敗しました: {e}")

    st.markdown("### 現在の履歴（最新10行）")
    hist_df = st.session_state.history.get(game, pd.DataFrame())
    if hist_df is None or len(hist_df) == 0:
        st.info("履歴がまだありません。貼り付けかCSVで追加してください。")
    else:
        st.dataframe(hist_df.tail(10), use_container_width=True, height=260)
        st.download_button(
            "履歴CSVをダウンロード",
            data=make_download_csv(hist_df, spec),
            file_name=f"{game}_history.csv",
            mime="text/csv",
            key=f"dl_hist_{game}",
        )


with tab_an:
    st.subheader("分析（見やすさ優先）")

    game_a = st.selectbox("ゲーム", list(SPECS.keys()), index=list(SPECS.keys()).index(game), key="game_select_an")
    spec_a = SPECS[game_a]
    hist_a: pd.DataFrame = st.session_state.history.get(game_a, pd.DataFrame())
    hist_a = normalize_history_df(hist_a, spec_a) if len(hist_a) else hist_a
    st.session_state.history[game_a] = hist_a

    if hist_a is None or len(hist_a) == 0:
        st.warning("履歴が空です。入力タブで貼り付けてください。")
    else:
        max_recent = len(hist_a)
        default_recent = min(200, max_recent)

        recent_n = st.number_input(
            "直近の回（分析）",
            min_value=1,
            max_value=max_recent,
            value=default_recent,
            step=10,
            key=f"recent_an_{game_a}",
        )

        freq = calc_freq(hist_a, spec_a, int(recent_n))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### 出現回数（直近）")
            df_freq = pd.DataFrame({"数字": freq.index, "出現回数": freq.values})
            st.dataframe(df_freq, use_container_width=True, height=420)

        with c2:
            st.markdown("#### ホット/コールド（上位・下位）")
            top_k = min(10, len(freq))
            hot = freq.sort_values(ascending=False).head(top_k)
            cold = freq.sort_values(ascending=True).head(top_k)
            st.write("ホット上位")
            st.dataframe(pd.DataFrame({"数字": hot.index, "回数": hot.values}), use_container_width=True, height=220)
            st.write("コールド上位（出てない側）")
            st.dataframe(pd.DataFrame({"数字": cold.index, "回数": cold.values}), use_container_width=True, height=220)


with tab_gen:
    st.subheader("候補生成（ネット購入向け）")

    game_g = st.selectbox("ゲーム", list(SPECS.keys()), index=list(SPECS.keys()).index(game), key="game_select_gen")
    spec_g = SPECS[game_g]
    hist_g: pd.DataFrame = st.session_state.history.get(game_g, pd.DataFrame())
    hist_g = normalize_history_df(hist_g, spec_g) if len(hist_g) else hist_g
    st.session_state.history[game_g] = hist_g

    if hist_g is None or len(hist_g) == 0:
        st.warning("履歴が空です。入力タブで履歴を入れてください。")
    else:
        max_recent = len(hist_g)
        default_recent = min(200, max_recent)

        with st.form(f"gen_form_{game_g}"):
            cL, cR = st.columns([1, 1])

            with cL:
                n_sets = st.number_input("口数（生成数）", min_value=1, max_value=200, value=10, step=1, key=f"gen_nsets_{game_g}")
                recent_n = st.number_input("直近の回を分析", min_value=1, max_value=max_recent, value=default_recent, step=10, key=f"gen_recent_{game_g}")
                overlap_max_default = 2 if spec_g.pick == 6 else 3
                overlap_max = st.number_input("重なり上限", min_value=0, max_value=spec_g.pick, value=overlap_max_default, step=1, key=f"gen_overlap_{game_g}")
                seed = st.number_input("乱数シード（同じ設定で同じ結果）", min_value=0, value=0, step=1, key=f"gen_seed_{game_g}")

            with cR:
                bias = st.slider("ホット/コールド", min_value=-2.0, max_value=2.0, value=0.0, step=0.1, key=f"gen_bias_{game_g}")
                exclude_text = st.text_input("除外したい数字（例: 1,2,10）", value="", key=f"gen_ex_{game_g}")
                fixed_text = st.text_input("固定したい数字（例: 7,23）", value="", key=f"gen_fx_{game_g}")
                role_split = st.toggle("役割分担（おすすめ）", value=True, key=f"gen_roles_{game_g}")

            submitted = st.form_submit_button("候補を生成する")

        if submitted:
            exclude = set(parse_int_list_csvlike(exclude_text))
            fixed = set(parse_int_list_csvlike(fixed_text))

            if any((n < spec_g.min_n or n > spec_g.max_n) for n in exclude | fixed):
                st.error(f"数字の範囲が不正です（{spec_g.min_n}〜{spec_g.max_n}）。")
            elif len(fixed) > spec_g.pick:
                st.error(f"固定が多すぎます（最大 {spec_g.pick} 個）。")
            elif fixed & exclude:
                st.error("固定した数字が除外にも入っています。どちらかから外してください。")
            else:
                freq = calc_freq(hist_g, spec_g, int(recent_n))
                with st.spinner("候補を生成中…"):
                    cands, tries = generate_candidates(
                        freq=freq,
                        spec=spec_g,
                        n_sets=int(n_sets),
                        exclude=exclude,
                        fixed=fixed,
                        overlap_max=int(overlap_max),
                        bias=float(bias),
                        role_split=bool(role_split),
                        seed=int(seed),
                    )
                st.session_state.generated[game_g] = cands

                if len(cands) == 0:
                    st.warning("候補が作れませんでした。重なり上限を増やすか、除外/固定を減らしてください。")
                elif len(cands) < int(n_sets):
                    st.info(f"{len(cands)} 口だけ生成できました（目標 {int(n_sets)} 口）。試行回数: {tries}。条件を緩めると増えます。")
                else:
                    st.success(f"{len(cands)} 口を生成しました。（試行回数: {tries}）")

        cands_show: List[List[int]] = st.session_state.generated.get(game_g, [])
        if cands_show:
            st.markdown("### 購入用（まとめ）")
            lines = [" ".join(map(str, c)) for c in cands_show]
            st.code("\n".join(lines))
            df_show = pd.DataFrame(cands_show, columns=[f"n{i}" for i in range(1, spec_g.pick + 1)])
            st.dataframe(df_show, use_container_width=True, height=320)

        st.caption("注意: 当たりやすさを保証するものではありません。ここは「入力のラクさ」と「買い方のルール化」を支援します。")
