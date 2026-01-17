#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ScoutDota — Universal Draft Scout (FULL MONOLITH)

PART 1 / 4
- Imports
- Constants
- HTTP helpers
- Steam ID utils
- OpenDota API
- STRATZ GraphQL
- Hero map + portraits (with fallbacks)

DO NOT RUN UNTIL ALL PARTS ARE PASTED
"""

# ============================================================
# ======================= IMPORTS ============================
# ============================================================

import re
import io
import time
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Set

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# ============================================================
# ======================= CONSTANTS ==========================
# ============================================================

APP_VERSION = "v2-full-monolith"
USER_AGENT = f"scoutdota/{APP_VERSION}"

OPENDOTA_BASE = "https://api.opendota.com/api"
STRATZ_GQL = "https://api.stratz.com/graphql"

STEAM64_BASE = 76561197960265728

# Manual mode draft heuristic
MANUAL_MIN_OVERLAP = 3  # <<< YOU REQUESTED THIS

# Default RD2L leagues (editable later)
DEFAULT_RD2L_LEAGUES = [17804, 17805]

# ============================================================
# ===================== HTTP HELPERS =========================
# ============================================================

def http_get(url: str, params=None, timeout=30):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def http_post(url: str, json_body=None, headers=None, timeout=40):
    h = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        h.update(headers)
    r = requests.post(url, json=json_body, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


# ============================================================
# ===================== STEAM UTILS ==========================
# ============================================================

def to_steam32(val: str) -> int:
    n = int(str(val).strip())
    return n - STEAM64_BASE if n >= STEAM64_BASE else n


# ============================================================
# ===================== OPENDOTA API =========================
# ============================================================

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def od_heroes_map() -> Dict[int, Dict[str, str]]:
    """
    hero_id -> {name, slug, img}
    """
    js = http_get(f"{OPENDOTA_BASE}/heroes").json()
    heroes = {}
    for h in js:
        hid = int(h["id"])
        name = h.get("localized_name", "")
        code = h.get("name", "npc_dota_hero_")
        slug = code.replace("npc_dota_hero_", "")
        img = h.get("img", "")
        heroes[hid] = {
            "name": name,
            "slug": slug,
            "img": img,
        }
    return heroes


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def od_match(match_id: int) -> dict:
    try:
        return http_get(f"{OPENDOTA_BASE}/matches/{int(match_id)}").json()
    except Exception:
        return {}


def od_recent_matches(steam32: int, limit=100) -> pd.DataFrame:
    js = http_get(
        f"{OPENDOTA_BASE}/players/{steam32}/matches",
        params={"limit": limit},
    ).json()
    return pd.DataFrame(js)


def od_player_profile(steam32: int) -> dict:
    return http_get(f"{OPENDOTA_BASE}/players/{steam32}").json()


def od_player_heroes(steam32: int) -> pd.DataFrame:
    js = http_get(f"{OPENDOTA_BASE}/players/{steam32}/heroes").json()
    return pd.DataFrame(js)


# ============================================================
# ===================== HERO PORTRAITS =======================
# ============================================================

_IMG_CACHE: Dict[int, bytes] = {}

def hero_portrait_urls(hero_id: int, heroes: Dict[int, Dict[str, str]]) -> List[str]:
    h = heroes.get(hero_id, {})
    urls = []

    if h.get("img"):
        urls.append("https://api.opendota.com" + h["img"])

    slug = h.get("slug")
    if slug:
        urls.append(
            f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{slug}.png"
        )
        urls.append(
            f"https://cdn.stratz.com/images/dota2/heroes/{slug}_icon.png"
        )

    return urls


def get_hero_image(hero_id: int, heroes: Dict[int, Dict[str, str]]) -> Optional[bytes]:
    if hero_id in _IMG_CACHE:
        return _IMG_CACHE[hero_id]

    for url in hero_portrait_urls(hero_id, heroes):
        try:
            r = http_get(url, timeout=20)
            if len(r.content) > 2000:
                _IMG_CACHE[hero_id] = r.content
                return r.content
        except Exception:
            continue

    return None


def hero_img_data_uri(hero_id: int, heroes: Dict[int, Dict[str, str]], size=(120, 65)) -> str:
    b = get_hero_image(hero_id, heroes)
    if not b:
        return ""

    try:
        img = Image.open(io.BytesIO(b)).convert("RGB").resize(size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


# ============================================================
# ===================== STRATZ API ===========================
# ============================================================

STRATZ_MATCH_QUERY = """
query($steam32: Long!, $leagues: [Int!]) {
  player(steamAccountId: $steam32) {
    matches(orderBy: MATCH_DATE_DESC, leagueIds: $leagues, take: 300) {
      endDateTime
      leagueId
      players(steamAccountId: $steam32) {
        heroId
        isVictory
        position
        role
        lane
      }
    }
  }
}
"""

def stratz_fetch_matches(
    steam32: int,
    token: str,
    leagues: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, str]:
    if not token:
        return pd.DataFrame(), "Missing STRATZ token"

    try:
        r = http_post(
            STRATZ_GQL,
            json_body={
                "query": STRATZ_MATCH_QUERY,
                "variables": {
                    "steam32": steam32,
                    "leagues": leagues,
                },
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        js = r.json()

        if js.get("errors"):
            return pd.DataFrame(), js["errors"][0].get("message", "STRATZ error")

        rows = []
        matches = js["data"]["player"]["matches"]
        for m in matches:
            p = m["players"][0]
            rows.append({
                "hero_id": p["heroId"],
                "is_victory": p["isVictory"],
                "position": p.get("position"),
                "role": p.get("role"),
                "lane": p.get("lane"),
                "leagueId": m.get("leagueId"),
                "endDateTime": m.get("endDateTime"),
            })

        return pd.DataFrame(rows), ""

    except Exception as e:
        return pd.DataFrame(), str(e)
# ============================================================
# ===================== RD2L SCRAPING ========================
# ============================================================

TEAM_ID_RE = re.compile(r"/teams/([A-Za-z0-9_-]+)")
MATCH_ID_RE = re.compile(r"/matches/(\d+)")

def rd2l_team_page(team_url: str) -> Tuple[List[int], Dict[str, str], str]:
    """
    Scrape RD2L team page.
    Returns:
      - match_ids
      - player profile links {name: url}
      - team_name
    """
    html = http_get(team_url).text
    soup = BeautifulSoup(html, "html.parser")

    # Team name
    team_name = ""
    h = soup.find(["h1", "h2"])
    if h:
        team_name = h.get_text(strip=True)
    if not team_name and soup.title:
        team_name = soup.title.get_text(strip=True).split(" – ")[0]

    match_ids: Set[int] = set()
    player_links: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Match links
        if "opendota.com/matches" in href or "dotabuff.com/matches" in href:
            m = MATCH_ID_RE.search(href)
            if m:
                match_ids.add(int(m.group(1)))

        # Player profile links
        if re.search(r"/profile/\d+$", href):
            name = a.get_text(strip=True)
            if name:
                player_links[name] = requests.compat.urljoin(team_url, href)

    # Loose match IDs
    text = soup.get_text(" ", strip=True)
    for m in re.finditer(r"\b(\d{9,10})\b", text):
        match_ids.add(int(m.group(1)))

    return sorted(match_ids), player_links, team_name or "RD2L Team"


def rd2l_profile_to_steam32(profile_url: str) -> Optional[int]:
    """
    Resolve Steam32 from RD2L profile page.
    """
    try:
        html = http_get(profile_url).text
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        h = a["href"]
        if "opendota.com/players/" in h:
            return int(h.split("/")[-1])
        if "steamcommunity.com/profiles/" in h:
            return int(int(h.split("/")[-1]) - STEAM64_BASE)
        if "dotabuff.com/players/" in h:
            return int(h.split("/")[-1])

    return None


# ============================================================
# ================ MATCH / DRAFT DETECTION ==================
# ============================================================

def side_from_slot(slot: int) -> int:
    return 0 if slot < 128 else 1  # 0 = Radiant, 1 = Dire


def detect_team_side(match: dict, roster_ids: Set[int]) -> Optional[int]:
    r, d = 0, 0
    for p in match.get("players", []):
        acc = p.get("account_id")
        if acc in roster_ids:
            if side_from_slot(p.get("player_slot", 0)) == 0:
                r += 1
            else:
                d += 1
    if r == 0 and d == 0:
        return None
    return 0 if r >= d else 1


# ============================================================
# ================= MANUAL MODE MATCH FIND ==================
# ============================================================

def manual_collect_candidate_matches(
    players: List[dict],
    recent_per_player: int,
) -> Tuple[List[int], List[str]]:
    """
    Manual mode:
    - Pull recent matches per player
    - Count match_id occurrences
    - Keep matches with >= MANUAL_MIN_OVERLAP players
    """
    warnings = []
    counter: Dict[int, int] = {}

    for p in players:
        try:
            df = od_recent_matches(p["steam32"], limit=recent_per_player)
            for mid in df.get("match_id", []):
                mid = int(mid)
                counter[mid] = counter.get(mid, 0) + 1
        except Exception as e:
            warnings.append(f"Failed recent matches for {p['name']}: {e}")

    valid = [
        mid for mid, count in counter.items()
        if count >= MANUAL_MIN_OVERLAP
    ]

    if valid:
        warnings.append(
            f"Manual mode heuristic used: drafts shown only where "
            f"{MANUAL_MIN_OVERLAP}+ players overlapped."
        )
    else:
        warnings.append(
            f"No matches found with ≥{MANUAL_MIN_OVERLAP} player overlap."
        )

    return sorted(valid), warnings


# ============================================================
# ================= MATCH FILTER + DRAFT ====================
# ============================================================

def filter_matches_with_drafts(
    match_ids: List[int],
    roster_ids: Set[int],
    max_matches: int,
) -> Tuple[pd.DataFrame, Dict[int, List[dict]]]:
    """
    Fetch matches, ensure:
    - roster side detected
    - picks/bans exist
    """
    rows = []
    drafts = {}

    for mid in match_ids[-max_matches:]:
        m = od_match(mid)
        if not m or "players" not in m:
            continue

        side = detect_team_side(m, roster_ids)
        if side is None:
            continue

        seq = m.get("picks_bans") or []
        if not seq:
            continue

        drafts[mid] = seq

        radiant_win = bool(m.get("radiant_win"))
        our_win = radiant_win if side == 0 else not radiant_win

        ts = m.get("start_time", 0)
        start = (
            datetime.utcfromtimestamp(ts).isoformat()
            if ts else ""
        )

        rows.append({
            "match_id": mid,
            "side": "Radiant" if side == 0 else "Dire",
            "side_idx": side,
            "win": our_win,
            "start": start,
            "duration": m.get("duration", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("start")

    return df, drafts
# ============================================================
# ===================== DRAFT PARSING ========================
# ============================================================

def hero_name(hero_id: Optional[int], heroes: Dict[int, Dict[str, str]]) -> str:
    if not hero_id:
        return ""
    return heroes.get(hero_id, {}).get("name", f"id{hero_id}")


def parse_draft_sequence(
    seq: List[dict],
    our_side_idx: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Returns:
      our_picks, our_bans, opp_picks, opp_bans
    """
    our_p, our_b, opp_p, opp_b = [], [], [], []

    for x in sorted(seq, key=lambda y: y.get("order", 0)):
        hid = x.get("hero_id")
        is_pick = x.get("is_pick")
        team = x.get("team")

        if team == our_side_idx:
            if is_pick:
                our_p.append(hid)
            else:
                our_b.append(hid)
        else:
            if is_pick:
                opp_p.append(hid)
            else:
                opp_b.append(hid)

    return our_p, our_b, opp_p, opp_b


# ============================================================
# ===================== DRAFT RENDER =========================
# ============================================================

def render_draft_html(
    seq: List[dict],
    our_side_idx: int,
    heroes: Dict[int, Dict[str, str]],
    show_portraits: bool = True,
) -> Tuple[str, str]:
    """
    Returns (our_html, opp_html)
    """

    our_cards = []
    opp_cards = []

    pick_n = 0
    ban_n = 0

    for x in sorted(seq, key=lambda y: y.get("order", 0)):
        hero_id = x.get("hero_id")
        is_pick = x.get("is_pick")
        team = x.get("team")

        if is_pick:
            pick_n += 1
            label = f"PICK {pick_n}"
        else:
            ban_n += 1
            label = f"BAN {ban_n}"

        img_uri = ""
        if show_portraits and hero_id:
            img_uri = hero_img_data_uri(hero_id, heroes)

        name = hero_name(hero_id, heroes)

        grayscale = "" if is_pick else "filter:grayscale(100%);"

        cross = ""
        if not is_pick:
            cross = (
                "<svg width='120' height='65' "
                "style='position:absolute;top:0;left:0'>"
                "<line x1='5' y1='5' x2='115' y2='60' stroke='gray' stroke-width='4'/>"
                "<line x1='115' y1='5' x2='5' y2='60' stroke='gray' stroke-width='4'/>"
                "</svg>"
            )

        card = f"""
        <div style="
            display:inline-block;
            background:#111;
            border:1px solid #555;
            border-radius:8px;
            padding:6px;
            margin-right:8px;
            width:130px;
            text-align:center;
            position:relative;
        ">
            <div style="font-size:11px;color:#aaa;margin-bottom:4px">{label}</div>
            <div style="position:relative;width:120px;height:65px;margin:auto">
                <img src="{img_uri}" style="width:120px;height:65px;{grayscale}">
                {cross}
            </div>
            <div style="font-size:12px;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                {name}
            </div>
        </div>
        """

        if team == our_side_idx:
            our_cards.append(card)
        else:
            opp_cards.append(card)

    our_html = (
        "<div style='white-space:nowrap;overflow-x:auto;padding-bottom:6px'>"
        + "".join(our_cards)
        + "</div>"
    )
    opp_html = (
        "<div style='white-space:nowrap;overflow-x:auto;padding-bottom:6px'>"
        + "".join(opp_cards)
        + "</div>"
    )

    return our_html, opp_html
# ============================================================
# ===================== PLAYER SCOUT =========================
# ============================================================

def rank_label(steam32: int) -> str:
    try:
        p = od_player_profile(steam32)
        if p.get("leaderboard_rank"):
            return f"Immortal #{p['leaderboard_rank']}"
        rt = p.get("rank_tier")
        if rt:
            major, minor = rt // 10, rt % 10
            names = ["Herald","Guardian","Crusader","Archon","Legend","Ancient","Divine","Immortal"]
            if 1 <= major <= 8:
                return names[major-1] + (f" {minor}" if major < 8 else "")
    except Exception:
        pass
    return "Unranked"


def aggregate_heroes(df: pd.DataFrame, heroes, topn: int):
    if df.empty or "hero_id" not in df:
        return []
    g = df.groupby("hero_id")["is_victory"]
    games = g.count()
    wins = g.sum()
    out = []
    for hid in games.sort_values(ascending=False).head(topn).index:
        pct = int(round(100 * wins[hid] / max(1, games[hid])))
        out.append(f"{hero_name(hid, heroes)} ({games[hid]} – {pct}%)")
    return out


def role_mask(df: pd.DataFrame, pos: int):
    if "position" in df and df["position"].notna().any():
        return df["position"] == pos
    return pd.Series([True] * len(df))


def player_scout(
    players: List[dict],
    topn: int,
    months: int,
    leagues: List[int],
    stratz_token: str,
):
    heroes = od_heroes_map()
    results = {"overall": [], "lastx": [], "tourn": [], "ranks": []}
    warnings = []

    cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months)
    cutoff_ts = int(cutoff.timestamp())

    for p in players:
        s32 = p["steam32"]
        pos = p["pos"]
        results["ranks"].append(rank_label(s32))

        # Overall (OpenDota)
        df_overall = od_player_heroes(s32)
        results["overall"].append(aggregate_heroes(df_overall, heroes, topn))

        # Last X months
        try:
            df, err = stratz_fetch_matches(s32, stratz_token)
            if df.empty:
                raise RuntimeError(err)
            ts = pd.to_numeric(df["endDateTime"], errors="coerce").fillna(0)
            ts = ts.where(ts < 10_000_000_000, ts / 1000)
            df = df[ts >= cutoff_ts]
            df = df[role_mask(df, pos)]
            results["lastx"].append(aggregate_heroes(df, heroes, topn))
        except Exception:
            warnings.append("STRATZ failed for Last X months; OpenDota used (not role-specific)")
            df = od_recent_matches(s32, 100)
            results["lastx"].append(aggregate_heroes(df, heroes, topn))

        # Tournament
        try:
            df, err = stratz_fetch_matches(s32, stratz_token, leagues)
            if df.empty:
                raise RuntimeError(err)
            df = df[role_mask(df, pos)]
            results["tourn"].append(aggregate_heroes(df, heroes, topn))
        except Exception:
            warnings.append("STRATZ failed for Tournament; OpenDota used")
            df = od_recent_matches(s32, 100)
            results["tourn"].append(aggregate_heroes(df, heroes, topn))

    return results, warnings


# ============================================================
# ========================= UI ===============================
# ============================================================

st.set_page_config("ScoutDota", layout="wide")
st.title("ScoutDota — Universal Draft Scout")

with st.sidebar:
    mode = st.radio("Mode", ["RD2L", "Manual"])
    stratz_token = st.text_input("STRATZ Token", type="password")
    show_portraits = st.checkbox("Show portraits", True)
    max_matches = st.slider("Max matches", 10, 100, 30)
    topN = st.slider("Top N heroes", 5, 25, 15)
    months = st.slider("Last X months", 1, 12, 3)
    leagues_txt = st.text_input("League IDs (comma)", ",".join(map(str, DEFAULT_RD2L_LEAGUES)))
    leagues = [int(x) for x in leagues_txt.split(",") if x.strip().isdigit()]

# ============================================================
# ========================= LOAD =============================
# ============================================================

players = []
warnings = []

if mode == "RD2L":
    team_url = st.text_input("RD2L Team URL")
    if st.button("Load RD2L"):
        mids, links, team = rd2l_team_page(team_url)
        roster = set()
        for name, url in links.items():
            s32 = rd2l_profile_to_steam32(url)
            if s32:
                players.append({"name": name, "steam32": s32, "pos": 0})
                roster.add(s32)
        df, drafts = filter_matches_with_drafts(mids, roster, max_matches)
        st.session_state["data"] = (team, players, df, drafts)

else:
    st.subheader("Manual Players")
    for i in range(5):
        c1, c2, c3 = st.columns([2,2,1])
        name = c1.text_input(f"name{i}")
        sid = c2.text_input(f"sid{i}")
        pos = c3.selectbox(f"pos{i}", ["1","2","3","4","5"])
        if name and sid:
            players.append({"name": name, "steam32": to_steam32(sid), "pos": int(pos)})

    recent = st.slider("Recent matches per player", 20, 200, 80)
    if st.button("Load Manual"):
        mids, w = manual_collect_candidate_matches(players, recent)
        warnings.extend(w)
        roster = {p["steam32"] for p in players}
        df, drafts = filter_matches_with_drafts(mids, roster, max_matches)
        st.session_state["data"] = ("Manual Team", players, df, drafts)

# ============================================================
# ========================= TABS =============================
# ============================================================

if "data" in st.session_state:
    team, players, df, drafts = st.session_state["data"]
    heroes = od_heroes_map()

    if warnings:
        st.warning(" / ".join(set(warnings)))

    tabs = st.tabs(["Overview", "Drafts", "Player Scout", "Raw Matches"])

    with tabs[0]:
        st.metric("Team", team)
        st.metric("Matches Found", len(df))
        st.metric("Players", len(players))

    with tabs[1]:
        for _, r in df.iterrows():
            mid = r["match_id"]
            with st.expander(f"Match {mid} — {r['side']} — {'Win' if r['win'] else 'Loss'}"):
                our, opp = render_draft_html(
                    drafts[mid],
                    r["side_idx"],
                    heroes,
                    show_portraits,
                )
                st.markdown("**Our Draft**", unsafe_allow_html=True)
                st.markdown(our, unsafe_allow_html=True)
                st.markdown("**Opponent Draft**", unsafe_allow_html=True)
                st.markdown(opp, unsafe_allow_html=True)

    with tabs[2]:
        scout, warn = player_scout(players, topN, months, leagues, stratz_token)
        if warn:
            st.warning(" / ".join(set(warn)))

        headers = [""] + [f"{p['name']} ({rank})" for p, rank in zip(players, scout["ranks"])]

        def table(title, cols):
            rows = [[title] + [""] * len(cols)]
            for i in range(topN):
                rows.append([""] + [(c[i] if i < len(c) else "") for c in cols])
            st.dataframe(pd.DataFrame(rows, columns=headers), use_container_width=True)

        table("Overall", scout["overall"])
        table("Last X Months", scout["lastx"])
        table("Tournament", scout["tourn"])

    with tabs[3]:
        st.dataframe(df, use_container_width=True)
