#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ScoutDota — Universal Draft Scout (PART 1 OF 4)
- Imports
- Constants  
- HTTP helpers
- Steam utils
- OpenDota API
- Hero portraits
- STRATZ API

PASTE ALL 4 PARTS BEFORE RUNNING
"""

# ============================================================
# IMPORTS
# ============================================================

import re
import io
import time
import base64
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Set

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# ============================================================
# CONSTANTS
# ============================================================

APP_VERSION = "v3-unified"
USER_AGENT = f"scoutdota/{APP_VERSION}"
OPENDOTA_BASE = "https://api.opendota.com/api"
STRATZ_GQL = "https://api.stratz.com/graphql"
STEAM64_BASE = 76561197960265728
DEFAULT_RD2L_LEAGUES = [17804, 17805]
MAX_TAKE = 100
MIN_INTERVAL = 1.0
_LAST_GQL_CALL = 0.0
_IMG_CACHE: Dict[int, bytes] = {}

# ============================================================
# HTTP HELPERS
# ============================================================

def http_get(url: str, params=None, timeout=30):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

def http_post(url: str, json_body=None, headers=None, timeout=40):
    h = {"User-Agent": USER_AGENT, "Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    r = requests.post(url, json=json_body, headers=h, timeout=timeout)
    r.raise_for_status()
    return r

# ============================================================
# STEAM UTILS
# ============================================================

def to_steam32(val: str) -> int:
    try:
        n = int(str(val).strip())
        return n - STEAM64_BASE if n >= STEAM64_BASE else n
    except:
        return 0

# ============================================================
# OPENDOTA API
# ============================================================

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def od_heroes_map() -> Dict[int, Dict[str, str]]:
    js = http_get(f"{OPENDOTA_BASE}/heroes").json()
    heroes = {}
    for h in js:
        hid = int(h["id"])
        heroes[hid] = {
            "name": h.get("localized_name", ""),
            "slug": h.get("name", "npc_dota_hero_").replace("npc_dota_hero_", ""),
            "img": h.get("img", ""),
        }
    return heroes

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def od_match(match_id: int) -> dict:
    try:
        return http_get(f"{OPENDOTA_BASE}/matches/{int(match_id)}").json()
    except:
        return {}

def od_recent_matches(steam32: int, limit=100) -> pd.DataFrame:
    try:
        js = http_get(f"{OPENDOTA_BASE}/players/{steam32}/matches", params={"limit": limit}).json()
        return pd.DataFrame(js)
    except:
        return pd.DataFrame()

def od_player_profile(steam32: int) -> dict:
    try:
        return http_get(f"{OPENDOTA_BASE}/players/{steam32}").json()
    except:
        return {}

def od_player_heroes(steam32: int) -> pd.DataFrame:
    try:
        js = http_get(f"{OPENDOTA_BASE}/players/{steam32}/heroes").json()
        return pd.DataFrame(js)
    except:
        return pd.DataFrame()

# ============================================================
# HERO PORTRAITS
# ============================================================

def hero_portrait_urls(hero_id: int, heroes: Dict[int, Dict[str, str]]) -> List[str]:
    h = heroes.get(hero_id, {})
    urls = []
    if h.get("img"):
        urls.append("https://api.opendota.com" + h["img"])
    slug = h.get("slug")
    if slug:
        urls.append(f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{slug}.png")
        urls.append(f"https://cdn.stratz.com/images/dota2/heroes/{slug}_icon.png")
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
        except:
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
    except:
        return ""

# ============================================================
# STRATZ API
# ============================================================

def _respect_rate_limit():
    global _LAST_GQL_CALL
    dt = time.time() - _LAST_GQL_CALL
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _LAST_GQL_CALL = time.time()

def gql_post(query: str, variables: dict, headers: dict) -> Tuple[dict, str]:
    backoff = 1.0
    for attempt in range(6):
        _respect_rate_limit()
        try:
            r = http_post(STRATZ_GQL, json_body={"query": query, "variables": variables}, headers=headers)
            data = r.json()
            if isinstance(data, dict) and data.get("errors"):
                return {}, data["errors"][0].get("message", "GraphQL error")
            return data, ""
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                wait = float(e.response.headers.get("Retry-After", backoff))
                time.sleep(wait)
                backoff = min(backoff * 2, 16)
                continue
            if e.response.status_code >= 500:
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            return {}, str(e)
        except Exception as e:
            if attempt >= 2:
                return {}, str(e)
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
    return {}, "Exceeded retry budget"

GQL_FORMS = [
    """query($steam32: Long!, $take: Int!, $skip: Int, $leagues: [Int!]){
      player(steamAccountId: $steam32){
        matches(orderBy: MATCH_DATE_DESC, leagueIds: $leagues, take: $take, skip: $skip){
          endDateTime leagueId
          players(steamAccountId:$steam32){ heroId isVictory position role lane }
        }
      }
    }""",
    """query($steam32: Long!, $take: Int!, $skip: Int, $leagues: [Int!]){
      player(steamAccountId: $steam32){
        matches(orderBy: { field: MATCH_DATE, sort: DESC }, leagueIds: $leagues, take: $take, skip: $skip){
          endDateTime leagueId
          players(steamAccountId:$steam32){ heroId isVictory position role lane }
        }
      }
    }""",
]

def stratz_fetch_matches(steam32: int, token: str, t_from=None, t_to=None, leagues=None, want: int = 220) -> Tuple[pd.DataFrame, str]:
    if not token or not token.strip():
        return pd.DataFrame(), "Missing STRATZ token"
    
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": USER_AGENT}
    target_pages = max(1, min(3, math.ceil(want / MAX_TAKE)))
    base_vars = {"steam32": int(steam32)}
    if leagues:
        base_vars["leagues"] = [int(x) for x in leagues]
    
    gathered = []
    last_err = ""
    
    for query in GQL_FORMS:
        gathered.clear()
        last_err = ""
        for p in range(target_pages):
            vars_ = dict(base_vars)
            vars_["take"] = MAX_TAKE
            vars_["skip"] = p * MAX_TAKE
            data, err = gql_post(query, vars_, headers)
            if err:
                last_err = err
                break
            
            matches = data.get("data", {}).get("player", {}).get("matches") or []
            batch = []
            for m in matches:
                ts = m.get("endDateTime")
                if ts is not None:
                    ts_sec = int(ts) // 1000 if int(ts) > 10_000_000_000 else int(ts)
                    if t_from is not None and ts_sec < int(t_from):
                        matches = []
                        break
                    if t_to is not None and ts_sec > int(t_to):
                        continue
                
                ps = m.get("players") or []
                if not ps:
                    continue
                p0 = ps[0]
                batch.append({
                    "hero_id": p0.get("heroId"),
                    "is_victory": bool(p0.get("isVictory", False)),
                    "position": p0.get("position"),
                    "role": p0.get("role") or "",
                    "lane": p0.get("lane"),
                    "leagueId": m.get("leagueId"),
                    "endDateTime": m.get("endDateTime"),
                })
            
            if batch:
                gathered.extend(batch)
                if len(batch) < MAX_TAKE:
                    break
            else:
                break
        
        if gathered:
            return pd.DataFrame(gathered), ""
    
    return pd.DataFrame(), (last_err or "No matches returned")

# ============================================================
# RD2L SCRAPING
# ============================================================

MATCH_ID_RE = re.compile(r"/matches/(\d+)")

def rd2l_team_page(team_url: str) -> Tuple[List[int], Dict[str, str], str]:
    html = http_get(team_url).text
    soup = BeautifulSoup(html, "html.parser")
    
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
        if "opendota.com/matches" in href or "dotabuff.com/matches" in href:
            m = MATCH_ID_RE.search(href)
            if m:
                match_ids.add(int(m.group(1)))
        if re.search(r"/profile/\d+$", href):
            name = a.get_text(strip=True)
            if name:
                player_links[name] = requests.compat.urljoin(team_url, href)
    
    text = soup.get_text(" ", strip=True)
    for m in re.finditer(r"\b(\d{9,10})\b", text):
        match_ids.add(int(m.group(1)))
    
    return sorted(match_ids), player_links, team_name or "RD2L Team"

def rd2l_profile_to_steam32(profile_url: str) -> Optional[int]:
    try:
        html = http_get(profile_url).text
    except:
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
# MATCH / DRAFT DETECTION
# ============================================================

def side_from_slot(slot: int) -> int:
    return 0 if slot < 128 else 1

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
# MANUAL MODE MATCH FIND
# ============================================================

def manual_collect_candidate_matches(players: List[dict], recent_per_player: int, min_overlap: int) -> Tuple[List[int], List[str]]:
    warnings = []
    counter: Dict[int, int] = {}
    
    for p in players:
        try:
            df = od_recent_matches(p["steam32"], limit=recent_per_player)
            for mid in df.get("match_id", []):
                counter[int(mid)] = counter.get(int(mid), 0) + 1
        except Exception as e:
            warnings.append(f"Failed recent matches for {p['name']}: {e}")
    
    valid = [mid for mid, count in counter.items() if count >= min_overlap]
    
    if valid:
        warnings.append(f"Manual mode: found {len(valid)} matches where {min_overlap}+ players overlapped.")
    else:
        warnings.append(f"No matches found with ≥{min_overlap} player overlap.")
    
    return sorted(valid), warnings

# ============================================================
# MATCH FILTER + DRAFT
# ============================================================

def filter_matches_with_drafts(match_ids: List[int], roster_ids: Set[int], max_matches: int) -> Tuple[pd.DataFrame, Dict[int, List[dict]]]:
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
        start = datetime.utcfromtimestamp(ts).isoformat() if ts else ""
        
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
# DRAFT RENDERING
# ============================================================

def hero_name(hero_id: Optional[int], heroes: Dict[int, Dict[str, str]]) -> str:
    if not hero_id:
        return ""
    return heroes.get(hero_id, {}).get("name", f"id{hero_id}")

def render_draft_html(seq: List[dict], our_side_idx: int, heroes: Dict[int, Dict[str, str]], show_portraits: bool = True) -> Tuple[str, str]:
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
                "<svg width='120' height='65' style='position:absolute;top:0;left:0'>"
                "<line x1='5' y1='5' x2='115' y2='60' stroke='gray' stroke-width='4'/>"
                "<line x1='115' y1='5' x2='5' y2='60' stroke='gray' stroke-width='4'/>"
                "</svg>"
            )
        
        card = f"""
        <div style="display:inline-block;background:#111;border:1px solid #555;border-radius:8px;padding:6px;margin-right:8px;width:130px;text-align:center;position:relative;">
            <div style="font-size:11px;color:#aaa;margin-bottom:4px">{label}</div>
            <div style="position:relative;width:120px;height:65px;margin:auto">
                <img src="{img_uri}" style="width:120px;height:65px;{grayscale}">
                {cross}
            </div>
            <div style="font-size:12px;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{name}</div>
        </div>"""
        
        if team == our_side_idx:
            our_cards.append(card)
        else:
            opp_cards.append(card)
    
    our_html = "<div style='white-space:nowrap;overflow-x:auto;padding-bottom:6px'>" + "".join(our_cards) + "</div>"
    opp_html = "<div style='white-space:nowrap;overflow-x:auto;padding-bottom:6px'>" + "".join(opp_cards) + "</div>"
    
    return our_html, opp_html

# ============================================================
# PLAYER SCOUT
# ============================================================

def rank_label(steam32: int) -> str:
    try:
        p = od_player_profile(steam32)
        if p.get("leaderboard_rank"):
            return f"Immortal #{p['leaderboard_rank']}"
        rt = p.get("rank_tier")
        if rt:
            major, minor = rt // 10, rt % 10
            names = ["Herald", "Guardian", "Crusader", "Archon", "Legend", "Ancient", "Divine", "Immortal"]
            if 1 <= major <= 8:
                return names[major - 1] + (f" {minor}" if major < 8 else "")
    except:
        pass
    return "Unranked"

def role_mask(df: pd.DataFrame, pos: int, any_role: bool) -> pd.Series:
    if df.empty or any_role:
        return pd.Series([True] * len(df), index=df.index)
    
    mask = pd.Series([False] * len(df), index=df.index)
    
    if "position" in df and df["position"].notna().any():
        mask |= (df["position"] == pos)
    
    if "role" in df and df["role"].astype(str).str.len().gt(0).any():
        R = df["role"].astype(str).str.upper()
        is_sup = (R == "SUPPORT") | (R == "2")
        is_core = (R == "CORE") | (R == "1")
        mask |= (is_sup if pos in (4, 5) else is_core)
    
    if not mask.any():
        return pd.Series([True] * len(df), index=df.index)
    
    return mask

def aggregate_heroes(df: pd.DataFrame, heroes, topn: int) -> List[str]:
    if df.empty or "hero_id" not in df:
        return []
    
    g = df.groupby("hero_id", dropna=True)["is_victory"]
    games = g.count().astype(int)
    wins = g.sum().astype(int)
    
    out = []
    for hid in games.sort_values(ascending=False).head(topn).index:
        pct = int(round(100 * wins[hid] / max(1, games[hid])))
        out.append(f"{hero_name(hid, heroes)} ({games[hid]} – {pct}%)")
    return out

def od_last_months_fallback(steam32: int, months: int, heroes: dict, topn: int) -> List[str]:
    days = 30 * months
    try:
        ms = http_get(f"{OPENDOTA_BASE}/players/{steam32}/matches", params={"date": days}).json()
        if not ms:
            return []
        
        df = pd.DataFrame(ms)
        if df.empty or "hero_id" not in df.columns:
            return []
        
        if "radiant_win" not in df:
            df["radiant_win"] = False
        if "player_slot" not in df:
            df["player_slot"] = 0
        
        you_won = (((df["radiant_win"]) & (df["player_slot"] < 128)) | ((~df["radiant_win"]) & (df["player_slot"] >= 128)))
        df["is_victory"] = you_won
        
        return aggregate_heroes(df, heroes, topn)
    except:
        return []

def od_league_fallback(steam32: int, leagues: List[int], heroes: dict, topn: int) -> List[str]:
    rows = []
    if not leagues:
        return []
    
    for L in leagues:
        try:
            part = http_get(f"{OPENDOTA_BASE}/players/{steam32}/matches", params={"league_id": L}).json()
            rows.extend(part)
        except:
            pass
    
    if not rows:
        return []
    
    df = pd.DataFrame(rows)
    if df.empty or "hero_id" not in df.columns:
        return []
    
    if "radiant_win" not in df:
        df["radiant_win"] = False
    if "player_slot" not in df:
        df["player_slot"] = 0
    
    you_won = (((df["radiant_win"]) & (df["player_slot"] < 128)) | ((~df["radiant_win"]) & (df["player_slot"] >= 128)))
    df["is_victory"] = you_won
    
    return aggregate_heroes(df, heroes, topn)

def player_scout(players: List[dict], topn: int, months: int, leagues: List[int], stratz_token: str, any_role: bool = False):
    heroes = od_heroes_map()
    results = {
        "overall": [],
        "lastx": [],
        "tourn": [],
        "ranks": [],
        "stats_overall": [],
        "stats_lastx": [],
        "stats_tourn": [],
    }
    warnings = []
    
    now = datetime.now(timezone.utc)
    t_to = int(now.timestamp())
    t_from = int((now - timedelta(days=30 * months)).timestamp())
    
    for p in players:
        s32 = p["steam32"]
        pos = p.get("pos", 0)
        
        results["ranks"].append(rank_label(s32))
        
        # Overall (OpenDota)
        try:
            df_overall = od_player_heroes(s32)
            results["overall"].append(aggregate_heroes(df_overall, heroes, topn))
            
            total = df_overall["games"].sum() if "games" in df_overall else 0
            wins = df_overall["win"].sum() if "win" in df_overall else 0
            wr = int(round(100 * wins / max(1, total)))
            results["stats_overall"].append(f"{total} games, {wr}% WR")
        except Exception as e:
            results["overall"].append([])
            results["stats_overall"].append("N/A")
            warnings.append(f"Overall failed for {p['name']}: {e}")
        
        # Last X months
        try:
            df, err = stratz_fetch_matches(s32, stratz_token, t_from, t_to, [], want=200)
            if df.empty:
                raise RuntimeError(err or "STRATZ returned no matches")
            mask = role_mask(df, pos, any_role)
            df_filtered = df[mask]
            results["lastx"].append(aggregate_heroes(df_filtered, heroes, topn))
            
            total = len(df_filtered)
            wins = df_filtered["is_victory"].sum() if "is_victory" in df_filtered else 0
            wr = int(round(100 * wins / max(1, total)))
            results["stats_lastx"].append(f"{total} games, {wr}% WR")
        except Exception as e:
            results["lastx"].append(od_last_months_fallback(s32, months, heroes, topn))
            results["stats_lastx"].append("OpenDota (fallback)")
            warnings.append(f"Last {months}mo STRATZ failed for {p['name']}: {e}")
        
        # Tournament
        try:
            df, err = stratz_fetch_matches(s32, stratz_token, None, None, leagues, want=200)
            if df.empty:
                raise RuntimeError(err or "STRATZ returned no league matches")
            mask = role_mask(df, pos, any_role)
            df_filtered = df[mask]
            results["tourn"].append(aggregate_heroes(df_filtered, heroes, topn))
            
            total = len(df_filtered)
            wins = df_filtered["is_victory"].sum() if "is_victory" in df_filtered else 0
            wr = int(round(100 * wins / max(1, total)))
            results["stats_tourn"].append(f"{total} games, {wr}% WR")
        except Exception as e:
            results["tourn"].append(od_league_fallback(s32, leagues, heroes, topn))
            results["stats_tourn"].append("OpenDota (fallback)")
            warnings.append(f"Tournament STRATZ failed for {p['name']}: {e}")
        
        time.sleep(0.4)
    
    return results, warnings

# ============================================================
# UI
# ============================================================

st.set_page_config("ScoutDota", layout="wide")
st.title("ScoutDota — Universal Draft Scout")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Mode", ["RD2L", "Manual"])
    
    st.divider()
    stratz_token = st.text_input("STRATZ Token (optional)", type="password", 
                                  help="For role-specific filtering. Leave empty to use OpenDota only.")
    
    st.divider()
    show_portraits = st.checkbox("Show hero portraits", True)
    max_matches = st.slider("Max matches to analyze", 10, 100, 30)
    
    st.divider()
    st.subheader("Player Scout Settings")
    topN = st.slider("Top N heroes", 5, 25, 15)
    months = st.slider("Last X months", 1, 12, 3)
    any_role = st.checkbox("Any role (ignore position)", False, help="If unchecked, filters by player position")
    
    st.divider()
    st.subheader("Tournament Leagues")
    leagues_txt = st.text_input("League IDs (comma-separated)", ",".join(map(str, DEFAULT_RD2L_LEAGUES)))
    leagues = [int(x.strip()) for x in leagues_txt.split(",") if x.strip().isdigit()]

# ============================================================
# LOAD DATA
# ============================================================

players = []
warnings = []

if mode == "RD2L":
    st.subheader("🏆 RD2L Mode")
    team_url = st.text_input("RD2L Team URL", placeholder="https://rd2l.gg/teams/...")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        load_btn = st.button("🔍 Load RD2L Team", type="primary", use_container_width=True)
    
    if load_btn and team_url:
        with st.spinner("Scraping RD2L team page..."):
            try:
                mids, links, team = rd2l_team_page(team_url)
                
                if not mids:
                    st.error("❌ No matches found on this RD2L page. The page may have zero matches recorded.")
                    st.session_state.pop("data", None)
                elif not links:
                    st.error("❌ No player profiles found on this RD2L page.")
                    st.session_state.pop("data", None)
                else:
                    roster = set()
                    for name, url in links.items():
                        s32 = rd2l_profile_to_steam32(url)
                        if s32:
                            players.append({"name": name, "steam32": s32, "pos": 0})
                            roster.add(s32)
                    
                    if not roster:
                        st.error("❌ Could not resolve any Steam IDs from player profiles.")
                        st.session_state.pop("data", None)
                    else:
                        df, drafts = filter_matches_with_drafts(mids, roster, max_matches)
                        
                        if df.empty:
                            st.warning("⚠️ No valid matches with drafts found for this roster.")
                        
                        st.session_state["data"] = (team, players, df, drafts)
                        st.success(f"✅ Loaded {len(players)} players and {len(df)} matches with drafts!")
            except Exception as e:
                st.error(f"❌ Error loading RD2L team: {e}")
                st.session_state.pop("data", None)

else:  # Manual mode
    st.subheader("✋ Manual Mode")
    st.write("Enter up to 5 players manually:")
    
    for i in range(5):
        c1, c2, c3 = st.columns([2, 2, 1])
        name = c1.text_input(f"Player {i+1} name", key=f"name{i}", placeholder="Player name")
        sid = c2.text_input(f"Player {i+1} Steam ID", key=f"sid{i}", placeholder="Steam32 or Steam64")
        pos = c3.selectbox(f"Pos {i+1}", ["—", "1", "2", "3", "4", "5"], key=f"pos{i}")
        
        if name and sid and pos != "—":
            players.append({"name": name, "steam32": to_steam32(sid), "pos": int(pos)})
    
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        recent_per_player = st.slider("Recent matches per player", 20, 200, 80)
    with col2:
        min_overlap = st.slider("Min player overlap", 2, 5, 3)
    
    load_btn = st.button("🔍 Find Matches", type="primary", use_container_width=True)
    
    if load_btn:
        if len(players) < 2:
            st.error("❌ Please enter at least 2 players with positions.")
        else:
            with st.spinner("Searching for matches..."):
                try:
                    mids, w = manual_collect_candidate_matches(players, recent_per_player, min_overlap)
                    warnings.extend(w)
                    roster = {p["steam32"] for p in players}
                    df, drafts = filter_matches_with_drafts(mids, roster, max_matches)
                    
                    st.session_state["data"] = ("Manual Team", players, df, drafts)
                    
                    if df.empty:
                        st.warning("⚠️ No valid matches with drafts found.")
                    else:
                        st.success(f"✅ Found {len(df)} matches with drafts!")
                except Exception as e:
                    st.error(f"❌ Error finding matches: {e}")

# ============================================================
# DISPLAY TABS
# ============================================================

if "data" in st.session_state:
    team, players, df, drafts = st.session_state["data"]
    heroes = od_heroes_map()
    
    if warnings:
        st.warning(" | ".join(set(warnings)))
    
    tabs = st.tabs(["📊 Overview", "📋 Drafts", "🔍 Player Scout", "📈 Raw Matches"])
    
    # TAB 1: Overview
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        col1.metric("Team", team)
        col2.metric("Matches Found", len(df))
        col3.metric("Players", len(players))
        
        if not df.empty:
            wins = df["win"].sum()
            col1.metric("Wins", wins)
            col2.metric("Losses", len(df) - wins)
            col3.metric("Win Rate", f"{int(100 * wins / len(df))}%")
            
            st.subheader("Players")
            player_data = []
            for p in players:
                rank = rank_label(p["steam32"])
                player_data.append({
                    "Name": p["name"],
                    "Steam32": p["steam32"],
                    "Position": p.get("pos", "—"),
                    "Rank": rank,
                })
            st.dataframe(pd.DataFrame(player_data), use_container_width=True, hide_index=True)
    
    # TAB 2: Drafts
    with tabs[1]:
        if df.empty:
            st.info("No draft data available.")
        else:
            for _, r in df.iterrows():
                mid = r["match_id"]
                result = "✅ Win" if r["win"] else "❌ Loss"
                with st.expander(f"Match {mid} — {r['side']} — {result} — {r['start'][:10]}"):
                    our, opp = render_draft_html(drafts[mid], r["side_idx"], heroes, show_portraits)
                    st.markdown("**Our Draft**", unsafe_allow_html=True)
                    st.markdown(our, unsafe_allow_html=True)
                    st.markdown("**Opponent Draft**", unsafe_allow_html=True)
                    st.markdown(opp, unsafe_allow_html=True)
    
    # TAB 3: Player Scout
    with tabs[2]:
        if not players:
            st.info("No players to scout.")
        else:
            with st.spinner("Scouting players..."):
                scout, warn = player_scout(players, topN, months, leagues, stratz_token, any_role)
                
                if warn:
                    st.warning(" | ".join(set(warn)))
                
                # Build headers
                headers = ["Category"] + [f"{p['name']} (P{p.get('pos', '?')}) — {rank}" 
                                          for p, rank in zip(players, scout["ranks"])]
                
                def build_table(title: str, data_cols: List[List[str]], stats_cols: List[str] = None):
                    rows = []
                    rows.append([f"**{title}**"] + [""] * len(players))
                    
                    if stats_cols:
                        rows.append(["Stats"] + stats_cols)
                    
                    for i in range(topN):
                        rows.append([""] + [(col[i] if i < len(col) else "") for col in data_cols])
                    
                    df_table = pd.DataFrame(rows, columns=headers)
                    st.dataframe(df_table, use_container_width=True, hide_index=True)
                
                build_table(f"Overall (Top {topN})", scout["overall"], scout["stats_overall"])
                st.divider()
                build_table(f"Last {months} Months ({'any role' if any_role else 'role-specific'})", 
                           scout["lastx"], scout["stats_lastx"])
                st.divider()
                leagues_label = ",".join(map(str, leagues)) if leagues else "none"
                build_table(f"Tournament ({leagues_label}) ({'any role' if any_role else 'role-specific'})", 
                           scout["tourn"], scout["stats_tourn"])
    
    # TAB 4: Raw Matches
    with tabs[3]:
        if df.empty:
            st.info("No match data available.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info("👆 Configure settings in the sidebar and click a Load button to begin.")



