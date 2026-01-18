#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RD2L Draft Scout — v7.3 (no OpenDota key; STRATZ only; faster)
- RD2L scrape (Dotabuff/OpenDota links + free-text match IDs)
- Drafts with portraits (bans greyed + X), 2-row layout
- Overview (4×1), Summary (side / FP–SP / phases), Picks-by-Position
- Player Scout (no “last N”): Overall (OD), Last X months (STRATZ→OD), Tournament (STRATZ→OD)
- Full PDF export
- Performance: capped matches, caching (heroes/matches), portrait toggle, shorter OD reparse loop, default no sleeps

Requires: streamlit, requests, beautifulsoup4, pillow, reportlab, pandas
Run: streamlit run rd2l_scout_app.py
"""

import os, re, io, time, base64, math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont, ImageOps
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
)

APP_VERSION = "v8.0"
OPENDOTA_BASE = "https://api.opendota.com/api"
USER_AGENT = "scoutdota/8.0"
STEAM64_BASE = 76561197960265728

# STRATZ GraphQL (Bearer token required)
GQL_URL = "https://api.stratz.com/graphql"
MAX_TAKE = 100      # STRATZ polite page size
MIN_INTERVAL = 1.0  # min spacing between GraphQL calls
_last_gql = 0.0

# Default leagues (RD2L): Season 35 & 36
DEFAULT_LEAGUES = [17804, 17805]

# ============================================================
# CUSTOM CSS - Dota 2 Inspired Dark Theme
# ============================================================
CUSTOM_CSS = """
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%); border-right: 1px solid #e94560; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: rgba(26, 26, 46, 0.8); padding: 8px 12px; border-radius: 12px; border: 1px solid #e94560; }
    .stTabs [data-baseweb="tab"] { background-color: #16213e; color: #8b949e; border-radius: 8px; padding: 12px 24px; font-weight: 600; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #1f4068; color: #edf2f4; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%) !important; color: #ffffff !important; box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4); }
    [data-testid="stMetricValue"] { font-size: 36px; font-weight: 700; color: #ff6b6b; text-shadow: 0 0 20px rgba(233, 69, 96, 0.5); }
    [data-testid="stMetricLabel"] { color: #8b949e; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; font-size: 12px; }
    .stDataFrame { border-radius: 12px; border: 1px solid #e94560; }
    .streamlit-expanderHeader { background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%) !important; color: #edf2f4 !important; border: 1px solid #e94560 !important; }
    h1, h2, h3 { color: #edf2f4 !important; }
    .stButton > button { background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%) !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    .stButton > button:hover { background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%) !important; box-shadow: 0 6px 20px rgba(233, 69, 96, 0.5); }
    .stDownloadButton > button { background: linear-gradient(135deg, #4ecca3 0%, #45b393 100%) !important; color: #0f0f23 !important; }
    .stSuccess { background: rgba(78, 204, 163, 0.2) !important; border: 1px solid #4ecca3 !important; color: #4ecca3 !important; }
    .stError { background: rgba(233, 69, 96, 0.2) !important; border: 1px solid #e94560 !important; color: #ff6b6b !important; }
    .stWarning { background: rgba(255, 193, 7, 0.2) !important; border: 1px solid #ffc107 !important; color: #ffc107 !important; }
    .stInfo { background: rgba(78, 204, 163, 0.15) !important; border: 1px solid #4ecca3 !important; color: #4ecca3 !important; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #0f0f23; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%); border-radius: 5px; }
</style>
"""

# ------------------------------------------------------------
# HTTP helpers (default: no sleeps)
# ------------------------------------------------------------
def http_get(url: str, params=None, headers=None, timeout=25, sleep: float = 0.0):
    """Generic GET with backoff; never appends OpenDota api_key."""
    if headers is None: headers = {}
    headers.setdefault("User-Agent", USER_AGENT)
    for a in range(4):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                if sleep: time.sleep(sleep)
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**a); continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if a == 3: raise
            time.sleep(2**a)

def http_post(url: str, json_body=None, headers=None, timeout=25):
    """Generic POST with backoff; no OpenDota api_key usage."""
    if headers is None: headers = {}
    headers.setdefault("User-Agent", USER_AGENT)
    headers.setdefault("Content-Type", "application/json")
    for a in range(4):
        try:
            r = requests.post(url, json=json_body, headers=headers, timeout=timeout)
            if r.status_code in (200, 201, 202):
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2**a); continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if a == 3: raise
            time.sleep(2**a)

# ------------------------------------------------------------
# STRATZ GraphQL helpers (Bearer token; rate limits; retries)
# ------------------------------------------------------------
def _rate_limit_gql():
    global _last_gql
    d = time.time() - _last_gql
    if d < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - d)
    _last_gql = time.time()

def gql_post(query: str, variables: dict, token: str) -> Tuple[dict, str]:
    if not token:
        return {}, "Missing STRATZ token (Authorization: Bearer <token> required)"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,  # STRATZ sees a UA
    }
    backoff = 1.0
    for attempt in range(6):
        _rate_limit_gql()
        try:
            r = requests.post(GQL_URL, json={"query": query, "variables": variables},
                              headers=headers, timeout=40)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra else backoff
                time.sleep(wait); backoff = min(backoff*2, 16); continue
            if r.status_code >= 500:
                time.sleep(backoff); backoff = min(backoff*2, 16); continue
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and js.get("errors"):
                return {}, js["errors"][0].get("message","GraphQL error")
            return js, ""
        except requests.RequestException as e:
            if attempt >= 2: return {}, str(e)
            time.sleep(backoff); backoff = min(backoff*2, 16)
    return {}, "Exceeded retry budget"

GQL_FORMS = [
    # enum orderBy (preferred)
    ("""
    query($steam32: Long!, $take: Int!, $skip: Int, $leagues: [Int!]){
      player(steamAccountId: $steam32){
        matches(
          orderBy: MATCH_DATE_DESC,
          leagueIds: $leagues,
          take: $take,
          skip: $skip
        ){
          endDateTime
          leagueId
          players(steamAccountId:$steam32){ heroId isVictory position role lane }
        }
      }
    }"""),
    # object orderBy fallback
    ("""
    query($steam32: Long!, $take: Int!, $skip: Int, $leagues: [Int!]){
      player(steamAccountId: $steam32){
        matches(
          orderBy: { field: MATCH_DATE, sort: DESC },
          leagueIds: $leagues,
          take: $take,
          skip: $skip
        ){
          endDateTime
          leagueId
          players(steamAccountId:$steam32){ heroId isVictory position role lane }
        }
      }
    }"""),
]

def months_window_utc(months:int):
    now = datetime.now(timezone.utc)
    t_to = int(now.timestamp())
    t_fr = int((now - timedelta(days=30*months)).timestamp())
    return t_fr, t_to

def stratz_fetch_matches(steam32:int, token:str, want:int=300, leagues=None, t_from=None, t_to=None) -> Tuple[pd.DataFrame,str]:
    """Fetch pages of matches; STRATZ has no date args; we filter locally by endDateTime."""
    target_pages = max(1, min(4, math.ceil(want / MAX_TAKE)))
    last_err = ""
    for query in GQL_FORMS:
        gathered=[]
        last_err=""
        for page in range(target_pages):
            js, err = gql_post(query, {"steam32": int(steam32),
                                       "take": MAX_TAKE, "skip": page*MAX_TAKE,
                                       "leagues": leagues}, token)
            if err: last_err=err; break
            matches = (((js.get("data") or {}).get("player") or {}).get("matches")) or []
            batch=[]
            for m in matches:
                ts = m.get("endDateTime")
                if ts is not None:
                    ts_sec = int(ts)
                    if ts_sec > 10_000_000_000: ts_sec//=1000
                    if t_from is not None and ts_sec < int(t_from):
                        matches=[]; break
                    if t_to is not None and ts_sec > int(t_to):
                        continue
                ps = m.get("players") or []
                if not ps: continue
                p0=ps[0]
                batch.append({
                    "hero_id": p0.get("heroId"),
                    "is_victory": bool(p0.get("isVictory", False)),
                    "position": p0.get("position"),
                    "role": (p0.get("role") or ""),
                    "lane": p0.get("lane"),
                    "leagueId": m.get("leagueId"),
                    "endDateTime": ts
                })
            if batch:
                gathered.extend(batch)
                if len(batch) < MAX_TAKE: break
            else:
                break
        if gathered:
            return pd.DataFrame(gathered), ""
    return pd.DataFrame(), (last_err or "No matches from STRATZ")

# ------------------------------------------------------------
# RD2L scraping (robust)
# ------------------------------------------------------------
def extract_team_id(url: str) -> Optional[str]:
    m = re.search(r"/teams/([A-Za-z0-9_\-]+)", url)
    return m.group(1) if m else None

def parse_rd2l_team_page(team_url: str, sleep: float = 0.0):
    r = http_get(team_url, sleep=sleep)
    soup = BeautifulSoup(r.text, "html.parser")

    # team name
    team_name=None
    h=soup.find(["h1","h2"])
    if h: team_name=h.get_text(strip=True)
    if not team_name and soup.title: team_name=soup.title.get_text(strip=True)
    if team_name and " – " in team_name: team_name=team_name.split(" – ")[0]

    our_team_id = extract_team_id(team_url)

    # All links
    match_links=[]; match_to_opp={}
    for a in soup.find_all("a", href=True):
        href=a["href"]
        if "dotabuff.com/matches" in href or "opendota.com/matches" in href:
            match_links.append(href)
            m=re.search(r"/matches/(\d+)", href)
            if not m: continue
            mid=int(m.group(1))
            # opponent: look up/down for other /teams/ anchor not equal ours
            opp=None; node=a
            for _ in range(4):
                if node is None: break
                t_as=[x for x in node.find_all("a", href=True) if "/teams/" in x["href"]]
                for ta in t_as:
                    tid=extract_team_id(ta["href"])
                    if tid and tid != our_team_id:
                        nm=ta.get_text(strip=True)
                        if nm: opp=nm; break
                if opp: break
                node=node.parent
            match_to_opp[mid]=opp

    # also look for plain 9-10 digit IDs near “Match”
    cand_ids=set()
    text = soup.get_text(" ", strip=True)
    for m in re.finditer(r"\b(?:Match\s*)?(\d{9,10})\b", text):
        cand_ids.add(int(m.group(1)))

    # player profile links (for roster & names)
    player_links={}
    for a in soup.find_all("a", href=True):
        if re.search(r"/profile/\d+$", a["href"]):
            nm=a.get_text(strip=True)
            if nm:
                player_links[nm]=requests.compat.urljoin(team_url, a["href"])

    return match_links, list(cand_ids), player_links, team_name, match_to_opp, soup

def extract_match_ids(dotabuff_links: List[str], cand_ids: List[int]) -> List[int]:
    ids=[]
    for url in dotabuff_links:
        m=re.search(r"/matches/(\d+)", url)
        if m: ids.append(int(m.group(1)))
    ids.extend(list(cand_ids or []))
    seen=set(); out=[]
    for x in ids:
        if x not in seen:
            out.append(x); seen.add(x)
    return sorted(out)

# ------------------------------------------------------------
# Profiles → account_id (steam32)
# ------------------------------------------------------------
def get_roster_account_id(profile_url: str, sleep: float = 0.0) -> Optional[int]:
    try:
        r = http_get(profile_url, sleep=sleep)
    except Exception:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        m = re.search(r"opendota\.com/players/(\d+)", a["href"])
        if m: return int(m.group(1))
    for a in soup.find_all("a", href=True):
        m = re.search(r"steamcommunity\.com/profiles/(\d+)", a["href"])
        if m: return int(int(m.group(1)) - STEAM64_BASE)
    for a in soup.find_all("a", href=True):
        m = re.search(r"dotabuff\.com/players/(\d+)", a["href"])
        if m: return int(m.group(1))
    return None

# ------------------------------------------------------------
# OpenDota helpers (heroes, portraits, draft parsing, positions)
# ------------------------------------------------------------
def fetch_heroes_map(sleep: float = 0.0):
    r = http_get(f"{OPENDOTA_BASE}/heroes", sleep=sleep)
    data = r.json()
    out={}
    for h in data:
        hid=int(h["id"])
        loc=h.get("localized_name","")
        img=h.get("img")
        code=h.get("name","npc_dota_hero_")
        slug=code.replace("npc_dota_hero_","")
        out[hid]={"name":loc,"img":img,"slug":slug}
    return out

def hero_name(hid: Optional[int], heroes: Dict[int,dict]) -> str:
    return heroes.get(hid, {}).get("name","") if hid else ""

def hero_img_primary(hid: int, heroes) -> Optional[str]:
    path=heroes.get(hid,{}).get("img")
    if path:
        return "https://api.opendota.com"+path if path.startswith("/") else path
    return None

def hero_img_fallbacks(hid: int, heroes) -> List[str]:
    # Prefer Valve CDN first (fast), then STRATZ
    slug=heroes.get(hid,{}).get("slug")
    if not slug: return []
    return [
        f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{slug}.png",
        f"https://cdn.stratz.com/images/dota2/heroes/{slug}_icon.png",
    ]

_IMG_BYTES: Dict[int, bytes] = {}
def _get_hero_bytes(hid: Optional[int], heroes) -> Optional[bytes]:
    # Optional portraits toggle
    if not st.session_state.get("show_portraits", True):
        return None
    if not hid: return None
    if hid in _IMG_BYTES: return _IMG_BYTES[hid]
    urls=[]
    p=hero_img_primary(hid, heroes)
    if p: urls.append(p)
    urls += hero_img_fallbacks(hid, heroes)
    for u in urls:
        try:
            b=http_get(u, headers={"Referer":"","User-Agent":USER_AGENT}, timeout=20).content
            if b and len(b)>2000:
                _IMG_BYTES[hid]=b
                return b
        except Exception:
            continue
    return None

def _portrait_img(hid: Optional[int], heroes):
    b=_get_hero_bytes(hid, heroes)
    if not b: return None
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception:
        return None

def portrait_data_uri(hid: Optional[int], heroes, size=(120,65)) -> str:
    im=_portrait_img(hid, heroes)
    if im is None: return ""
    im2=im.resize(size)
    bio=io.BytesIO(); im2.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

def _font(sz=14):
    try: return ImageFont.truetype("arial.ttf", sz)
    except: return ImageFont.load_default()

def side_from_slot(slot: int) -> int:  # 0=radiant, 1=dire
    return 0 if slot < 128 else 1

def gold_at_minute(p: dict, minute: int) -> Optional[int]:
    gt=p.get("gold_t")
    if isinstance(gt,list) and len(gt)>minute: return gt[minute]
    gpm=p.get("gold_per_min")
    if gpm is not None: return int(gpm*minute)
    total=p.get("total_gold"); dur=p.get("duration") or 1800
    if total is not None: return int(total*(minute*60/max(dur,60)))
    return None

def lane_role_to_pos(lr: Optional[int]) -> Optional[int]:
    if lr == 2: return 2
    if lr == 1: return 1
    if lr == 3: return 3
    return None

def support_signal(p: dict):
    wards=(p.get("obs_placed",0) or 0) + (p.get("sen_placed",0) or 0)
    purchase=p.get("purchase") or {}
    wards += purchase.get("ward_observer",0)+purchase.get("ward_sentry",0)
    smokes=purchase.get("smoke_of_deceit",0)
    dusts = purchase.get("dust",0) + purchase.get("dust_of_appearance",0)
    stacks=p.get("camps_stacked",0) or 0
    g10=gold_at_minute(p,10) or 0
    score5=wards*3 + stacks*1.5 + smokes + 0.5*dusts
    return score5, g10

def assign_positions(match: dict, our_side: int):
    players=[p for p in match.get("players",[]) if side_from_slot(p.get("player_slot",0))==our_side]
    if len(players)!=5: return {}, players
    dur=match.get("duration")
    for p in players:
        if dur is not None: p.setdefault("duration", dur)

    mids=[p for p in players if lane_role_to_pos(p.get("lane_role"))==2 or p.get("lane")==2]
    if len(mids)==1: mid=mids[0]
    elif len(mids)>1: mid=max(mids, key=lambda x: (gold_at_minute(x,10) or 0))
    else:
        def xp10(p): xt=p.get("xp_t"); return xt[10] if isinstance(xt,list) and len(xt)>10 else 0
        mid=max(players, key=lambda x: (xp10(x), gold_at_minute(x,10) or 0))
    rest=[p for p in players if p is not mid]

    carries=[p for p in rest if lane_role_to_pos(p.get("lane_role"))==1]
    carry=max(carries, key=lambda x: (gold_at_minute(x,10) or 0)) if carries else max(rest, key=lambda x:(gold_at_minute(x,10) or 0))
    rest2=[p for p in rest if p is not carry]

    offs=[p for p in rest2 if lane_role_to_pos(p.get("lane_role"))==3]
    off=max(offs, key=lambda x: (gold_at_minute(x,10) or 0)) if offs else max(rest2, key=lambda x:(gold_at_minute(x,10) or 0))
    supports=[p for p in rest2 if p is not off]
    s_sorted=sorted(supports, key=lambda p: (-(support_signal(p)[0]), support_signal(p)[1]))
    pos5=s_sorted[0] if s_sorted else None
    pos4=s_sorted[1] if len(s_sorted)>1 else None

    pos_map={}
    pos_map[mid.get("hero_id")]   = 2
    pos_map[carry.get("hero_id")] = 1
    pos_map[off.get("hero_id")]   = 3
    if pos5: pos_map[pos5.get("hero_id")] = 5
    if pos4: pos_map[pos4.get("hero_id")] = 4
    return pos_map, players

def extract_picks_bans(match: dict, our_side: int):
    seq=sorted(match.get("picks_bans") or [], key=lambda x: x.get("order",0))
    our_p=[x["hero_id"] for x in seq if x.get("is_pick") and x.get("team")==our_side]
    our_b=[x["hero_id"] for x in seq if not x.get("is_pick") and x.get("team")==our_side]
    op_p =[x["hero_id"] for x in seq if x.get("is_pick") and x.get("team")!=our_side]
    op_b =[x["hero_id"] for x in seq if not x.get("is_pick") and x.get("team")!=our_side]
    return our_p, our_b, op_p, op_b, seq

def detect_our_side(match: dict, roster_ids: set) -> Optional[int]:
    r_ct=sum(1 for p in match.get("players",[]) if p.get("account_id") in roster_ids and side_from_slot(p.get("player_slot",0))==0)
    d_ct=sum(1 for p in match.get("players",[]) if p.get("account_id") in roster_ids and side_from_slot(p.get("player_slot",0))==1)
    if r_ct==0 and d_ct==0: return None
    return 0 if r_ct>=d_ct else 1

def request_parse_and_refetch(mid: int, sleep: float):
    """Ask OD to parse, then poll a couple of times (short)."""
    try:
        http_post(f"{OPENDOTA_BASE}/request/{mid}")
    except Exception:
        return None
    for _ in range(2):          # shorter loop
        time.sleep(1.2)         # shorter wait
        try:
            m=http_get(f"{OPENDOTA_BASE}/matches/{mid}", sleep=sleep).json()
            if m and m.get("picks_bans"):
                return m
        except Exception:
            pass
    return None

def first_pick_team(seq: List[dict]) -> Optional[int]:
    for it in sorted(seq, key=lambda x: x.get("order",0)):
        if it.get("is_pick"): return it.get("team")
    return None

def pick_phase_from_index(p_idx: int) -> str:
    if 1<=p_idx<=2: return "Pick phase 1"
    if 3<=p_idx<=6: return "Pick phase 2"
    return "Pick phase 3"

def ban_phase_from_index(b_idx: int) -> str:
    if 1<=b_idx<=6: return "Ban phase 1"
    if 7<=b_idx<=10: return "Ban phase 2"
    return "Ban phase 3"

# ------------------------------------------------------------
# Streamlit caches (HUGE speedups on reloads)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24*3600)
def cached_fetch_heroes_map():
    return fetch_heroes_map(sleep=0.0)

@st.cache_data(show_spinner=False, ttl=6*3600)
def cached_match(mid: int):
    try:
        return http_get(f"{OPENDOTA_BASE}/matches/{mid}", sleep=0.0).json()
    except Exception:
        return {}

# ------------------------------------------------------------
# Collect all team data
# ------------------------------------------------------------
def collect(team_url: str, sleep: float, request_parse: bool, max_matches: int):
    mlinks, cand_ids, plinks, team_name, match_to_opp, _ = parse_rd2l_team_page(team_url, sleep=sleep)
    mids = extract_match_ids(mlinks, cand_ids)
    # Keep only the most recent N
    mids = mids[-max_matches:]

    # roster
    roster_ids=set()
    for _, url in plinks.items():
        acc=get_roster_account_id(url, sleep=sleep)
        if acc is not None: roster_ids.add(acc)

    heroes = cached_fetch_heroes_map()

    rows=[]; seq_by_match={}
    pick_ctr=Counter(); ban_ctr=Counter(); opp_pick_ctr=Counter(); opp_ban_ctr=Counter()

    for mid in mids:
        m = cached_match(mid)
        if not m or "players" not in m:
            if request_parse:
                m2=request_parse_and_refetch(mid, sleep)
                if m2: m=m2
            if not m or "players" not in m:
                continue

        if not m.get("picks_bans") and request_parse:
            m2=request_parse_and_refetch(mid, sleep)
            if m2: m=m2

        our_side=detect_our_side(m, roster_ids)
        if our_side is None: continue

        our_p, our_b, op_p, op_b, seq=extract_picks_bans(m, our_side)
        seq_by_match[mid]=seq
        radiant_win=bool(m.get("radiant_win"))
        our_win = radiant_win if our_side==0 else (not radiant_win)

        for hid in our_p: pick_ctr[hero_name(hid, heroes)] += 1
        for hid in our_b: ban_ctr[hero_name(hid, heroes)]  += 1
        for hid in op_p:  opp_pick_ctr[hero_name(hid, heroes)] += 1
        for hid in op_b:  opp_ban_ctr[hero_name(hid, heroes)]  += 1

        pos_map,_=assign_positions(m, our_side)

        ts=m.get("start_time")
        start_iso=(datetime.utcfromtimestamp(ts).isoformat() if isinstance(ts,(int,float)) and ts>0 else "")
        rows.append({
            "match_id": mid,
            "start_time_utc": start_iso,
            "duration_s": m.get("duration",""),
            "our_side": "Radiant" if our_side==0 else "Dire",
            "our_side_idx": our_side,
            "our_win": our_win,
            "our_picks_ids": our_p, "our_bans_ids": our_b,
            "opp_picks_ids": op_p,  "opp_bans_ids": op_b,
            "pos_map_base": pos_map,
            "opponent": match_to_opp.get(mid) or ""
        })

    df=pd.DataFrame(rows).sort_values("start_time_utc")

    def winrate_table(counter: Counter, lists_series: pd.Series, wins_series: pd.Series) -> pd.DataFrame:
        out=[]
        for hero, total in counter.most_common():
            n=w=0
            for picks_ids, won in zip(lists_series, wins_series):
                if not picks_ids: continue
                names_in_match={hero_name(h, heroes) for h in picks_ids}
                if hero in names_in_match:
                    n+=1; w+=1 if won else 0
            wr=round(100.0*w/n,1) if n else 0.0
            out.append({"hero":hero,"count":total,"winrate_%":wr})
        return pd.DataFrame(out)

    if not df.empty:
        picks_df     = winrate_table(pick_ctr,     df["our_picks_ids"], df["our_win"])
        bans_df      = winrate_table(ban_ctr,      df["our_bans_ids"],  df["our_win"])
        opp_picks_df = winrate_table(opp_pick_ctr, df["opp_picks_ids"], df["our_win"])
        opp_bans_df  = winrate_table(opp_ban_ctr,  df["opp_bans_ids"],  df["our_win"])
    else:
        picks_df=bans_df=opp_picks_df=opp_bans_df=pd.DataFrame()

    return df, heroes, picks_df, bans_df, opp_picks_df, opp_bans_df, seq_by_match, team_name, plinks

# ------------------------------------------------------------
# Collect Custom Tournament Data (Manual Players + League IDs)
# ------------------------------------------------------------
def collect_custom_tournament(players: list, league_ids: list, team_name: str, stratz_token: str, max_matches: int):
    """
    Collect match data for a custom team in specific tournaments.
    players: List of {name, steam32, pos}
    league_ids: List of tournament league IDs
    """
    heroes = cached_fetch_heroes_map()
    roster_ids = {p["steam32"] for p in players}
    
    # Collect match IDs from all players in specified leagues
    match_id_counts = Counter()
    
    for p in players:
        # Try STRATZ first
        if stratz_token:
            try:
                df_matches, err = stratz_fetch_matches(p["steam32"], stratz_token, want=200, leagues=league_ids)
                if not df_matches.empty:
                    for _, row in df_matches.iterrows():
                        mid = row.get("matchId") or row.get("match_id")
                        if mid:
                            match_id_counts[int(mid)] += 1
                    continue
            except Exception:
                pass
        
        # Fallback to OpenDota
        for lid in league_ids:
            try:
                matches = http_get(f"{OPENDOTA_BASE}/players/{p['steam32']}/matches", 
                                  params={"league_id": lid}).json()
                for m in matches:
                    mid = m.get("match_id")
                    if mid:
                        match_id_counts[int(mid)] += 1
            except Exception:
                pass
    
    # Filter matches where 3+ team members played (team matches)
    team_match_ids = sorted([mid for mid, count in match_id_counts.items() if count >= 3])[-max_matches:]
    
    if not team_match_ids:
        return pd.DataFrame(), heroes, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, team_name, {}
    
    rows = []
    seq_by_match = {}
    pick_ctr = Counter()
    ban_ctr = Counter()
    opp_pick_ctr = Counter()
    opp_ban_ctr = Counter()
    
    for mid in team_match_ids:
        m = cached_match(mid)
        if not m or "players" not in m:
            continue
        
        our_side = detect_our_side(m, roster_ids)
        if our_side is None:
            continue
        
        our_p, our_b, op_p, op_b, seq = extract_picks_bans(m, our_side)
        if not seq:
            continue
        
        seq_by_match[mid] = seq
        radiant_win = bool(m.get("radiant_win"))
        our_win = radiant_win if our_side == 0 else (not radiant_win)
        
        for hid in our_p:
            pick_ctr[hero_name(hid, heroes)] += 1
        for hid in our_b:
            ban_ctr[hero_name(hid, heroes)] += 1
        for hid in op_p:
            opp_pick_ctr[hero_name(hid, heroes)] += 1
        for hid in op_b:
            opp_ban_ctr[hero_name(hid, heroes)] += 1
        
        pos_map, _ = assign_positions(m, our_side)
        
        ts = m.get("start_time")
        start_iso = datetime.utcfromtimestamp(ts).isoformat() if isinstance(ts, (int, float)) and ts > 0 else ""
        rows.append({
            "match_id": mid,
            "start_time_utc": start_iso,
            "duration_s": m.get("duration", ""),
            "our_side": "Radiant" if our_side == 0 else "Dire",
            "our_side_idx": our_side,
            "our_win": our_win,
            "our_picks_ids": our_p,
            "our_bans_ids": our_b,
            "opp_picks_ids": op_p,
            "opp_bans_ids": op_b,
            "pos_map_base": pos_map,
            "opponent": ""
        })
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("start_time_utc")
    
    def winrate_table(counter, lists_series, wins_series):
        out = []
        for hero, total in counter.most_common():
            n = w = 0
            for picks_ids, won in zip(lists_series, wins_series):
                if not picks_ids:
                    continue
                names_in_match = {hero_name(h, heroes) for h in picks_ids}
                if hero in names_in_match:
                    n += 1
                    w += 1 if won else 0
            wr = round(100.0 * w / n, 1) if n else 0.0
            out.append({"hero": hero, "count": total, "winrate_%": wr})
        return pd.DataFrame(out)
    
    if not df.empty:
        picks_df = winrate_table(pick_ctr, df["our_picks_ids"], df["our_win"])
        bans_df = winrate_table(ban_ctr, df["our_bans_ids"], df["our_win"])
        opp_picks_df = winrate_table(opp_pick_ctr, df["opp_picks_ids"], df["our_win"])
        opp_bans_df = winrate_table(opp_ban_ctr, df["opp_bans_ids"], df["our_win"])
    else:
        picks_df = bans_df = opp_picks_df = opp_bans_df = pd.DataFrame()
    
    plinks = {p["name"]: f"steam32:{p['steam32']}" for p in players}
    return df, heroes, picks_df, bans_df, opp_picks_df, opp_bans_df, seq_by_match, team_name, plinks

# ------------------------------------------------------------
# Summary / positions
# ------------------------------------------------------------
def wr_pct(wins:int, total:int)->float:
    return round(100.0*wins/total,1) if total else 0.0

def compute_summary(df: pd.DataFrame, seq_by_match: Dict[int,List[dict]], heroes: Dict[int,dict]):
    summary={}

    # Row1: Radiant vs Dire
    side_overall={}
    for side in ["Radiant","Dire"]:
        dsub=df[df["our_side"]==side]
        wins=int(dsub["our_win"].sum()); total=len(dsub)
        side_overall[side]={"wins":wins,"total":total,"wr":wr_pct(wins,total)}
    hero_side_us={"Radiant":defaultdict(lambda:{"c":0,"w":0}),
                  "Dire":   defaultdict(lambda:{"c":0,"w":0})}
    hero_side_vs={"Radiant":defaultdict(lambda:{"c":0,"w":0}),
                  "Dire":   defaultdict(lambda:{"c":0,"w":0})}
    for _,row in df.iterrows():
        side=row["our_side"]; opp="Dire" if side=="Radiant" else "Radiant"
        win=bool(row["our_win"])
        for hid in row["our_picks_ids"]:
            hero_side_us[side][hid]["c"]+=1
            if win: hero_side_us[side][hid]["w"]+=1
        for hid in row["opp_picks_ids"]:
            hero_side_vs[opp][hid]["c"]+=1
            if win: hero_side_vs[opp][hid]["w"]+=1
    def hm(mp):
        rows=[]
        for hid,agg in sorted(mp.items(), key=lambda kv:(-kv[1]["c"], heroes.get(kv[0],{}).get("name",""))):
            c=agg["c"]; w=agg["w"]
            rows.append({"hero":hero_name(hid,heroes),"count":c,"winrate_%":wr_pct(w,c)})
        return pd.DataFrame(rows)
    summary["row1"]={
        "overall": side_overall,
        "by_us": {"Radiant":hm(hero_side_us["Radiant"]), "Dire":hm(hero_side_us["Dire"])},
        "vs_us": {"Radiant":hm(hero_side_vs["Radiant"]), "Dire":hm(hero_side_vs["Dire"])},
    }

    # Row2: FP/SP
    fp_overall={"FP":{"wins":0,"total":0},"SP":{"wins":0,"total":0}}
    hero_fp_us={"FP":defaultdict(lambda:{"c":0,"w":0}),"SP":defaultdict(lambda:{"c":0,"w":0})}
    hero_fp_vs={"FP":defaultdict(lambda:{"c":0,"w":0}),"SP":defaultdict(lambda:{"c":0,"w":0})}
    for _,row in df.iterrows():
        seq=seq_by_match.get(int(row["match_id"])) or []
        fp_t=first_pick_team(seq)
        if fp_t is None: continue
        our_fp=(fp_t==(0 if row["our_side"]=="Radiant" else 1))
        grp_us="FP" if our_fp else "SP"
        grp_vs="SP" if our_fp else "FP"
        win=bool(row["our_win"])
        fp_overall[grp_us]["total"]+=1
        if win: fp_overall[grp_us]["wins"]+=1
        for hid in row["our_picks_ids"]:
            hero_fp_us[grp_us][hid]["c"]+=1
            if win: hero_fp_us[grp_us][hid]["w"]+=1
        for hid in row["opp_picks_ids"]:
            hero_fp_vs[grp_vs][hid]["c"]+=1
            if win: hero_fp_vs[grp_vs][hid]["w"]+=1
    def dfd(mp):
        rows=[]
        for hid,agg in sorted(mp.items(), key=lambda kv:(-kv[1]["c"], heroes.get(kv[0],{}).get("name",""))):
            c=agg["c"]; w=agg["w"]
            rows.append({"hero":hero_name(hid,heroes),"count":c,"winrate_%":wr_pct(w,c)})
        return pd.DataFrame(rows)
    summary["row2"]={
        "overall":{"FP":{"wins":fp_overall["FP"]["wins"],"total":fp_overall["FP"]["total"],"wr":wr_pct(fp_overall["FP"]["wins"],fp_overall["FP"]["total"])},
                   "SP":{"wins":fp_overall["SP"]["wins"],"total":fp_overall["SP"]["total"],"wr":wr_pct(fp_overall["SP"]["wins"],fp_overall["SP"]["total"])}} ,
        "by_us":{"FP":dfd(hero_fp_us["FP"]), "SP":dfd(hero_fp_us["SP"])},
        "vs_us":{"FP":dfd(hero_fp_vs["FP"]), "SP":dfd(hero_fp_vs["SP"])},
    }

    # Rows3 & 4: phases
    def phase_acc():
        return {"Pick phase 1":defaultdict(lambda:{"c":0,"w":0}),
                "Pick phase 2":defaultdict(lambda:{"c":0,"w":0}),
                "Pick phase 3":defaultdict(lambda:{"c":0,"w":0}),
                "Ban  phase 1":defaultdict(lambda:{"c":0,"w":0}),
                "Ban  phase 2":defaultdict(lambda:{"c":0,"w":0}),
                "Ban  phase 3":defaultdict(lambda:{"c":0,"w":0})}
    by_us=phase_acc(); vs_us=phase_acc()
    for _,row in df.iterrows():
        seq=seq_by_match.get(int(row["match_id"])) or []
        our_idx=0 if row["our_side"]=="Radiant" else 1
        win=bool(row["our_win"])
        p_idx=b_idx=0
        for it in sorted(seq, key=lambda x:x.get("order",0)):
            if it.get("is_pick"):
                p_idx+=1; ph=pick_phase_from_index(p_idx)
                if it.get("team")==our_idx:
                    by_us[ph][it.get("hero_id")]["c"]+=1
                    if win: by_us[ph][it.get("hero_id")]["w"]+=1
                else:
                    vs_us[ph][it.get("hero_id")]["c"]+=1
                    if win: vs_us[ph][it.get("hero_id")]["w"]+=1
            else:
                b_idx+=1; ph=ban_phase_from_index(b_idx).replace("Ban","Ban ")
                if it.get("team")==our_idx:
                    by_us[ph][it.get("hero_id")]["c"]+=1
                    if win: by_us[ph][it.get("hero_id")]["w"]+=1
                else:
                    vs_us[ph][it.get("hero_id")]["c"]+=1
                    if win: vs_us[ph][it.get("hero_id")]["w"]+=1
    def phase_to_dfs(mp):
        out={}
        for ph, heroes_mp in mp.items():
            rows=[]
            for hid,agg in sorted(heroes_mp.items(), key=lambda kv:(-kv[1]["c"], heroes.get(kv[0],{}).get("name",""))):
                c=agg["c"]; w=agg["w"]
                rows.append({"hero":hero_name(hid,heroes),"count":c,"winrate_%":wr_pct(w,c)})
            out[ph]=pd.DataFrame(rows)
        return out
    summary["row3"]=phase_to_dfs(by_us)
    summary["row4"]=phase_to_dfs(vs_us)
    return summary

def compute_pos_tables(df: pd.DataFrame, heroes: Dict[int,dict]) -> Dict[int,pd.DataFrame]:
    stats=defaultdict(lambda: defaultdict(lambda:{"count":0,"wins":0}))
    for _,row in df.iterrows():
        for hid,pos in dict(row["pos_map_base"]).items():
            if pos in (1,2,3,4,5):
                nm=hero_name(hid, heroes)
                stats[pos][nm]["count"]+=1
                if row["our_win"]: stats[pos][nm]["wins"]+=1
    out={}
    for pos in [1,2,3,4,5]:
        rows=[]
        for h,s in sorted(stats[pos].items(), key=lambda kv:(-kv[1]["count"], kv[0])):
            rows.append({"hero":h,"count":s["count"],"winrate_%":wr_pct(s["wins"], s["count"])})
        out[pos]=pd.DataFrame(rows)
    return out

# ------------------------------------------------------------
# Player Scout (no “last-N”)
# ------------------------------------------------------------
def to_steam32(v) -> int:
    n=int(v); return n - STEAM64_BASE if n >= STEAM64_BASE else n

def od_get(path:str):
    url=f"{OPENDOTA_BASE}/{path.lstrip('/')}"
    r=requests.get(url, headers={"User-Agent":USER_AGENT}, timeout=30); r.raise_for_status(); return r.json()

def load_heroes_map_od() -> Dict[int,str]:
    js=od_get("constants/heroes")
    out={}
    for _,v in js.items():
        out[int(v["id"])]=v.get("localized_name") or v.get("name", f"id{v['id']}")
    return out

def fetch_rank_label(steam32:int) -> str:
    try:
        P=od_get(f"players/{steam32}")
        if P.get("leaderboard_rank"): return f"Immortal {P['leaderboard_rank']}"
        rt=P.get("rank_tier")
        if rt:
            major=rt//10; minor=rt%10
            names=["Herald","Guardian","Crusader","Archon","Legend","Ancient","Divine","Immortal"]
            if 1<=major<=8:
                if major<8 and 1<=minor<=5: return f"{names[major-1]} {minor}"
                return names[major-1]
    except Exception: pass
    return "Unranked"

def aggregate_pairs_from_df(df: pd.DataFrame, heroes_map:dict, topn:int) -> List[str]:
    if df.empty or "hero_id" not in df: return []
    g=df.groupby("hero_id", dropna=True)["is_victory"]
    counts=g.count().astype(int); wins=g.sum().astype(int)
    order=counts.sort_values(ascending=False)
    out=[]
    for hid,games in order.head(topn).items():
        w=int(wins.get(hid,0)); pct=int(round(100*w/max(1,games)))
        name=heroes_map.get(int(hid), f"id{hid}")
        out.append(f"{name} ({games} – {pct}%)")
    return out

def role_mask(df: pd.DataFrame, pos:int, any_role:bool) -> pd.Series:
    if df.empty or any_role:
        return pd.Series([True]*len(df), index=df.index)
    m=pd.Series([False]*len(df), index=df.index)
    if "position" in df and df["position"].notna().any(): m |= (df["position"]==pos)
    if "role" in df and df["role"].astype(str).str.len().gt(0).any():
        R=df["role"].astype(str).str.upper()
        is_sup=(R=="SUPPORT") | (R=="2")
        is_core=(R=="CORE") | (R=="1")
        m |= (is_sup if pos in (4,5) else is_core)
    if not m.any(): return pd.Series([True]*len(df), index=df.index)
    return m

def od_last_months_pairs(steam32:int, months:int, heroes:dict, topn:int) -> List[str]:
    days=30*months
    try:
        ms=od_get(f"players/{steam32}/matches?date={days}")
    except Exception:
        return []
    df=pd.DataFrame(ms)
    if df.empty or "hero_id" not in df.columns: return []
    if "radiant_win" not in df: df["radiant_win"]=False
    if "player_slot" not in df: df["player_slot"]=0
    you_won=( (df["radiant_win"]) & (df["player_slot"]<128) ) | ( (~df["radiant_win"]) & (df["player_slot"]>=128) )
    df2=pd.DataFrame({"hero_id":df["hero_id"], "is_victory":you_won})
    return aggregate_pairs_from_df(df2, heroes, topn)

def od_league_pairs(steam32:int, leagues:List[int], heroes:dict, topn:int) -> List[str]:
    rows=[]
    for L in leagues or []:
        try:
            part=od_get(f"players/{steam32}/matches?leagueid={int(L)}")
            rows.extend(part)
        except Exception:
            pass
    if not rows: return []
    df=pd.DataFrame(rows)
    if df.empty or "hero_id" not in df.columns: return []
    if "radiant_win" not in df: df["radiant_win"]=False
    if "player_slot" not in df: df["player_slot"]=0
    you_won=( (df["radiant_win"]) & (df["player_slot"]<128) ) | ( (~df["radiant_win"]) & (df["player_slot"]>=128) )
    df2=pd.DataFrame({"hero_id":df["hero_id"], "is_victory":you_won})
    return aggregate_pairs_from_df(df2, heroes, topn)

def player_scout(players: List[dict], topN:int, months:int, leagues:List[int], any_role:bool, token:str):
    """
    players: [{name, steam32, pos}]
    returns dict with 3 matrices (lists of columns):
      - overall
      - lastx
      - tourn
    """
    heroes_map = load_heroes_map_od()
    overall=[]; lastx=[]; tourn=[]; ranks=[]
    t_from, t_to = months_window_utc(months)

    for p in players:
        s32=int(p["steam32"]); pos=int(p["pos"])
        ranks.append(fetch_rank_label(s32))

        # overall (OD)
        try:
            js=od_get(f"players/{s32}/heroes")
            rows=[]
            for row in js:
                hid=row.get("hero_id")
                games=row.get("games") or row.get("matches") or row.get("games_played") or 0
                wins=row.get("win") or row.get("wins") or 0
                if hid is None: continue
                name=heroes_map.get(int(hid), f"id{hid}")
                rows.append((name,int(games),int(wins)))
            rows.sort(key=lambda x:x[1], reverse=True)
            col=[]
            for name,g,w in rows[:topN]:
                pct=int(round(100*w/max(1,g)))
                col.append(f"{name} ({g} – {pct}%)")
            overall.append(col)
        except Exception:
            overall.append([])

        # last X months (STRATZ→OD)
        try:
            df, err = stratz_fetch_matches(s32, token, want=400, leagues=None, t_from=t_from, t_to=t_to)
            if df.empty: raise RuntimeError(err or "STRATZ empty")
            mask=role_mask(df, pos, any_role)
            lastx.append(aggregate_pairs_from_df(df[mask], heroes_map, topN))
        except Exception:
            lastx.append(od_last_months_pairs(s32, months, heroes_map, topN))

        # tournament (STRATZ→OD)
        try:
            dfL, errL = stratz_fetch_matches(s32, token, want=400, leagues=leagues or DEFAULT_LEAGUES)
            if dfL.empty: raise RuntimeError(errL or "STRATZ empty")
            maskL=role_mask(dfL, pos, any_role)
            tourn.append(aggregate_pairs_from_df(dfL[maskL], heroes_map, topN))
        except Exception:
            tourn.append(od_league_pairs(s32, leagues or DEFAULT_LEAGUES, heroes_map, topN))

        time.sleep(0.25)  # small courtesy gap

    return {"overall": overall, "lastx": lastx, "tourn": tourn, "ranks": ranks}

# ------------------------------------------------------------
# Draft PNG + PDF helpers
# ------------------------------------------------------------
def draft_png_two_rows(seq: List[dict], our_side: int, heroes, team_name: str, scale: float=1.0) -> bytes:
    b=p=0; our=[]; opp=[]
    for it in sorted(seq, key=lambda x: x.get("order",0)):
        if it.get("is_pick"): p+=1; lab=f"PICK {p}"
        else: b+=1; lab=f"BAN {b}"
        row={"hero_id":it.get("hero_id"),"label":lab,"team":it.get("team"),"is_pick":bool(it.get("is_pick"))}
        (our if it.get("team")==our_side else opp).append(row)

    cw=int(120*scale); ch=int(120*scale); img_w=int(96*scale); img_h=int(56*scale)
    pad=int(16*scale); gap=int(10*scale); title_h=int(28*scale); label_h=int(18*scale)
    cols=12
    W=pad*2 + cols*cw + gap*(cols-0)
    H=pad*2 + title_h + ch + gap + ch + label_h

    im=Image.new("RGB",(W,H),(255,255,255))
    d=ImageDraw.Draw(im)
    f_title=_font(int(16*scale)); f_step=_font(int(12*scale)); f_name=_font(int(11*scale)); f_small=_font(int(12*scale))
    d.text((pad,pad), f"{team_name} — draft", fill=(0,0,0), font=f_title)

    def draw_row(items, y0):
        x=pad
        for i in range(cols):
            it = items[i] if i<len(items) else {"hero_id":None,"label":"","team":our_side,"is_pick":True}
            side_fill=(230,245,234) if it.get("team")==0 else (247,229,229)
            side_border=(52,132,63) if it.get("team")==0 else (176,46,46)
            d.rounded_rectangle([x,y0,x+cw-6,y0+ch-6], radius=10, outline=side_border, width=2, fill=side_fill)
            d.text((x+6,y0+6), it.get("label",""), fill=(0,0,0), font=f_step)
            port=_portrait_img(it.get("hero_id"), heroes)
            if port:
                port=port.resize((img_w,img_h))
                if not it.get("is_pick", True): port=ImageOps.grayscale(port).convert("RGB")
                im.paste(port,(x+10,y0+26))
                nm=hero_name(it.get("hero_id"), heroes)
                d.text((x+10, y0+26+img_h+2), nm, fill=(0,0,0), font=f_name)
                if not it.get("is_pick", True):
                    x0,y0i=x+10,y0+26; x1,y1i=x0+img_w,y0i+img_h
                    w=max(2,int(4*scale)); col=(120,120,120)
                    d.line([(x0,y0i),(x1,y1i)], fill=col, width=w); d.line([(x1,y0i),(x0,y1i)], fill=col, width=w)
            x+=cw+gap

    y_top=pad+title_h; y_bot=y_top+ch+gap
    draw_row(our, y_top); draw_row(opp, y_bot)
    d.text((pad, H-label_h), team_name, fill=(0,0,0), font=f_small)
    d.text((W-pad-d.textlength("Opponents", font=f_small), H-label_h), "Opponents", fill=(0,0,0), font=f_small)
    b_io=io.BytesIO(); im.save(b_io, format="PNG"); return b_io.getvalue()

def flowable_from_png(png_bytes: bytes, max_width_pts: int = 780) -> RLImage:
    im=Image.open(io.BytesIO(png_bytes)); w,h=im.size
    if w>max_width_pts: scale=max_width_pts/float(w); w2,h2=max_width_pts, int(h*scale)
    else: w2,h2=w,h
    bio=io.BytesIO(); im.save(bio, format="PNG"); bio.seek(0)
    return RLImage(bio, width=w2, height=h2)

def pdf_table(df: pd.DataFrame):
    df=(df if not df.empty else pd.DataFrame({"info":["(empty)"]})).astype(str)
    data=[df.columns.tolist()] + df.values.tolist()
    t=Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#e6e6e6")),
        ("TEXTCOLOR",(0,0),(-1,0), colors.black),
        ("GRID",(0,0),(-1,-1), 0.25, colors.HexColor("#bbbbbb")),
        ("BACKGROUND",(0,1),(-1,-1), colors.white),
        ("TEXTCOLOR",(0,1),(-1,-1), colors.black),
        ("FONTSIZE",(0,0),(-1,-1), 8),
        ("ALIGN",(0,0),(-1,-1), "LEFT"),
        ("LEFTPADDING",(0,0),(-1,-1), 3), ("RIGHTPADDING",(0,0),(-1,-1), 3),
        ("TOPPADDING",(0,0),(-1,-1), 1), ("BOTTOMPADDING",(0,0),(-1,-1), 1),
    ]))
    return t

def _df_from_overall_map(m: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows=[]
    for key in ["Radiant","Dire"]:
        if key in m: rows.append({"side":key, "wins":m[key]["wins"], "total":m[key]["total"], "winrate_%":m[key]["wr"]})
    for key in ["FP","SP"]:
        if key in m: rows.append({"group":key, "wins":m[key]["wins"], "total":m[key]["total"], "winrate_%":m[key]["wr"]})
    return pd.DataFrame(rows)

def build_pdf(team_name: str,
              picks_df: pd.DataFrame, bans_df: pd.DataFrame,
              opp_picks_df: pd.DataFrame, opp_bans_df: pd.DataFrame,
              pos_tables: Dict[int,pd.DataFrame],
              df: pd.DataFrame, seq_by_match: Dict[int,list],
              heroes: Dict[int,dict],
              summary: Dict,
              scout: Optional[dict]=None,
              players_meta: Optional[List[dict]]=None) -> bytes:

    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf, pagesize=landscape(A4),
                          rightMargin=16,leftMargin=16,topMargin=16,bottomMargin=16)
    styles=getSampleStyleSheet()
    elems=[]
    elems.append(Paragraph(f"<b>RD2L Draft Scout — {team_name}</b> (app {APP_VERSION})", styles["Title"]))
    elems.append(Spacer(0,8))

    # Overview 4×1
    TOP_N=15
    a=picks_df.head(TOP_N); b=opp_picks_df.head(TOP_N)
    c=bans_df.head(TOP_N);  d=opp_bans_df.head(TOP_N)
    titles=Table([[Paragraph(f"<b>Most Picked — {team_name}</b>", styles["Heading4"]),
                    Paragraph(f"<b>Picks vs {team_name}</b>", styles["Heading4"]),
                    Paragraph(f"<b>Most Banned — {team_name}</b>", styles["Heading4"]),
                    Paragraph(f"<b>Bans vs {team_name}</b>", styles["Heading4"])]],
                 colWidths=[195,195,195,195])
    grid=Table([[pdf_table(a), pdf_table(b), pdf_table(c), pdf_table(d)]],
               colWidths=[195,195,195,195])
    grid.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP")]))
    elems += [titles, Spacer(0,4), grid,
              Paragraph("<i>(Top 15 shown to fit one page)</i>", styles["Normal"]),
              PageBreak()]

    # Summary Row1
    elems.append(Paragraph("<b>Summary — Row 1: Radiant vs Dire</b>", styles["Heading2"]))
    over1=_df_from_overall_map(summary["row1"]["overall"])
    elems += [Paragraph("<b>Overall by us</b>", styles["Heading4"]), pdf_table(over1), Spacer(0,6)]
    elems += [Table([[Paragraph("<b>By Us — Radiant</b>", styles["Heading4"]),
                      Paragraph("<b>By Us — Dire</b>", styles["Heading4"])]],
                    colWidths=[390,390])]
    elems += [Table([[pdf_table(summary["row1"]["by_us"]["Radiant"]),
                      pdf_table(summary["row1"]["by_us"]["Dire"])]],
                    colWidths=[390,390], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(Spacer(0,6))
    elems += [Table([[Paragraph("<b>Against Us — Radiant</b>", styles["Heading4"]),
                      Paragraph("<b>Against Us — Dire</b>", styles["Heading4"])]],
                    colWidths=[390,390])]
    elems += [Table([[pdf_table(summary["row1"]["vs_us"]["Radiant"]),
                      pdf_table(summary["row1"]["vs_us"]["Dire"])]],
                    colWidths=[390,390], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(PageBreak())

    # Summary Row2
    elems.append(Paragraph("<b>Summary — Row 2: First Pick vs Second Pick</b>", styles["Heading2"]))
    over2=_df_from_overall_map(summary["row2"]["overall"])
    elems += [Paragraph("<b>Overall by us</b>", styles["Heading4"]), pdf_table(over2), Spacer(0,6)]
    elems += [Table([[Paragraph("<b>By Us — First Pick</b>", styles["Heading4"]),
                      Paragraph("<b>By Us — Second Pick</b>", styles["Heading4"])]],
                    colWidths=[390,390])]
    elems += [Table([[pdf_table(summary["row2"]["by_us"]["FP"]),
                      pdf_table(summary["row2"]["by_us"]["SP"])]],
                    colWidths=[390,390], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(Spacer(0,6))
    elems += [Table([[Paragraph("<b>Against Us — First Pick</b>", styles["Heading4"]),
                      Paragraph("<b>Against Us — Second Pick</b>", styles["Heading4"])]],
                    colWidths=[390,390])]
    elems += [Table([[pdf_table(summary["row2"]["vs_us"]["FP"]),
                      pdf_table(summary["row2"]["vs_us"]["SP"])]],
                    colWidths=[390,390], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(PageBreak())

    # Summary Row3 (By us)
    elems.append(Paragraph("<b>Summary — Row 3: Draft Phases (By Us)</b>", styles["Heading2"]))
    elems += [Table([[Paragraph("<b>Pick phase 1</b>", styles["Heading4"]),
                      Paragraph("<b>Pick phase 2</b>", styles["Heading4"]),
                      Paragraph("<b>Pick phase 3</b>", styles["Heading4"])]],
                    colWidths=[260,260,260])]
    elems += [Table([[pdf_table(summary["row3"]["Pick phase 1"]),
                      pdf_table(summary["row3"]["Pick phase 2"]),
                      pdf_table(summary["row3"]["Pick phase 3"])]],
                    colWidths=[260,260,260], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(Spacer(0,6))
    elems += [Table([[Paragraph("<b>Ban phase 1</b>", styles["Heading4"]),
                      Paragraph("<b>Ban phase 2</b>", styles["Heading4"]),
                      Paragraph("<b>Ban phase 3</b>", styles["Heading4"])]],
                    colWidths=[260,260,260])]
    elems += [Table([[pdf_table(summary["row3"]["Ban  phase 1"]),
                      pdf_table(summary["row3"]["Ban  phase 2"]),
                      pdf_table(summary["row3"]["Ban  phase 3"])]],
                    colWidths=[260,260,260], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(PageBreak())

    # Summary Row4 (Vs us)
    elems.append(Paragraph("<b>Summary — Row 4: Draft Phases (Against Us)</b>", styles["Heading2"]))
    elems += [Table([[Paragraph("<b>Pick phase 1</b>", styles["Heading4"]),
                      Paragraph("<b>Pick phase 2</b>", styles["Heading4"]),
                      Paragraph("<b>Pick phase 3</b>", styles["Heading4"])]],
                    colWidths=[260,260,260])]
    elems += [Table([[pdf_table(summary["row4"]["Pick phase 1"]),
                      pdf_table(summary["row4"]["Pick phase 2"]),
                      pdf_table(summary["row4"]["Pick phase 3"])]],
                    colWidths=[260,260,260], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(Spacer(0,6))
    elems += [Table([[Paragraph("<b>Ban phase 1</b>", styles["Heading4"]),
                      Paragraph("<b>Ban phase 2</b>", styles["Heading4"]),
                      Paragraph("<b>Ban phase 3</b>", styles["Heading4"])]],
                    colWidths=[260,260,260])]
    elems += [Table([[pdf_table(summary["row4"]["Ban  phase 1"]),
                      pdf_table(summary["row4"]["Ban  phase 2"]),
                      pdf_table(summary["row4"]["Ban  phase 3"])]],
                    colWidths=[260,260,260], style=[("VALIGN",(0,0),(-1,-1),"TOP")])]
    elems.append(PageBreak())

    # Picks by Position
    MAX_ROWS_POS=14
    cols=[]; heads=[]
    for pos in [1,2,3,4,5]:
        dfp=pos_tables.get(pos, pd.DataFrame(columns=["hero","count","winrate_%"])).head(MAX_ROWS_POS)
        cols.append(pdf_table(dfp)); heads.append(Paragraph(f"<b>Pos{pos}</b>", styles["Heading4"]))
    elems += [Table([heads], colWidths=[160]*5), Spacer(0,4), Table([cols], colWidths=[160]*5)]
    elems.append(PageBreak())

    # Per-match drafts
    for _,row in df.sort_values("start_time_utc").iterrows():
        mid=int(row["match_id"]); date=row["start_time_utc"].replace("T"," ") if isinstance(row["start_time_utc"],str) else ""
        res="Win" if row["our_win"] else "Loss"; opp=row.get("opponent") or "Opponent"
        head=f"Match {mid} — {row['our_side']} — {res} — {date} — vs {opp}"
        elems.append(Paragraph(f"<b>{head}</b>", styles["Heading3"]))
        elems.append(Spacer(0,6))
        seq=seq_by_match.get(mid) or []; our_idx=0 if row["our_side"]=="Radiant" else 1
        png_bytes=draft_png_two_rows(seq, our_idx, heroes, team_name, scale=1.0)
        elems.append(flowable_from_png(png_bytes, max_width_pts=780))
        elems.append(Spacer(0,10))

    # Player Scout in PDF
    if scout and players_meta:
        elems.append(PageBreak())
        elems.append(Paragraph("<b>Player Scout — Last X months (role-specific)</b>", styles["Heading2"]))
        headers=[]; cols=[]
        for i,p in enumerate(players_meta):
            nm=p["name"]; pos=p["pos"]; rank=(scout.get("ranks") or [""]*5)[i]
            headers.append(Paragraph(f"<b>{nm} (P{pos}) — {rank}</b>", styles["Heading4"]))
            col = pd.DataFrame({ "Top heroes": (scout["lastx"][i] or []) })
            cols.append(pdf_table(col))
        elems += [Table([headers], colWidths=[156]*len(headers)), Spacer(0,4),
                  Table([cols], colWidths=[156]*len(cols))]
        elems.append(PageBreak())

        elems.append(Paragraph(f"<b>Player Scout — Tournament heroes (leagues: {', '.join(map(str,DEFAULT_LEAGUES))})</b>", styles["Heading2"]))
        headers=[]; cols=[]
        for i,p in enumerate(players_meta):
            nm=p["name"]; pos=p["pos"]; rank=(scout.get("ranks") or [""]*5)[i]
            headers.append(Paragraph(f"<b>{nm} (P{pos}) — {rank}</b>", styles["Heading4"]))
            col = pd.DataFrame({ "Top heroes": (scout["tourn"][i] or []) })
            cols.append(pdf_table(col))
        elems += [Table([headers], colWidths=[156]*len(headers)), Spacer(0,4),
                  Table([cols], colWidths=[156]*len(cols))]

    doc.build(elems)
    return buf.getvalue()

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title=f"ScoutDota {APP_VERSION}", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title(f"🎮 ScoutDota - Universal Draft Scout {APP_VERSION}")
st.caption("Draft analysis for RD2L teams or any custom tournament. Features: Overview, Summary, Drafts, Picks by Position, Player Scout, PDF Export.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection
    mode = st.radio("📋 Select Mode", ["🏆 RD2L Mode", "🎯 Custom Tournament"],
                   help="RD2L: Scrape team from RD2L page | Custom: Enter players manually with league IDs")
    
    st.divider()
    
    # STRATZ token
    stratz_token = st.text_input("🔑 STRATZ Token", type="password",
                                 value=st.session_state.get("stratz_token",""),
                                 help="Get your token at stratz.com")
    
    st.divider()
    st.subheader("🔧 Settings")
    sleep_s = st.number_input("Request delay (s)", 0.0, 5.0, 0.0, 0.1)
    req_parse = st.checkbox("Request parse for missing matches", value=False)
    max_matches = st.slider("Max matches to fetch", 10, 120, 30, 5)
    
    st.divider()
    if st.button("🗑️ Clear Cache"):
        try: st.cache_data.clear()
        except: pass
        st.success("Cache cleared!")

# ============================================================
# MODE: RD2L
# ============================================================
if mode == "🏆 RD2L Mode":
    st.header("🏆 RD2L Mode")
    default_url="https://rd2l.gg/seasons/QSj0aP5YM/divisions/U-ZTEMOBg/teams/KGghuA-8c"
    team_url = st.text_input("📎 RD2L Team URL", value=st.session_state.get("loaded_url", default_url))
    
    col1, col2 = st.columns([1, 4])
    with col1:
        load_rd2l = st.button("🚀 Load Team", type="primary", use_container_width=True)
    
    if load_rd2l and team_url:
        with st.spinner("Fetching RD2L page + OpenDota..."):
            try:
                (df, heroes, picks_df, bans_df, opp_picks_df, opp_bans_df,
                 seq_by_match, team_name, plinks) = collect(team_url, sleep_s, req_parse, max_matches)
                st.session_state.update({
                    "loaded_url": team_url,
                    "stratz_token": stratz_token,
                    "mode": "rd2l",
                    "data": {
                        "df": df, "heroes": heroes, "picks_df": picks_df, "bans_df": bans_df,
                        "opp_picks_df": opp_picks_df, "opp_bans_df": opp_bans_df,
                        "seq": seq_by_match, "team_name": team_name, "plinks": plinks,
                        "leagues": DEFAULT_LEAGUES
                    }
                })
                st.success(f"✅ Loaded {len(df)} matches for **{team_name}**!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ============================================================
# MODE: Custom Tournament
# ============================================================
else:
    st.header("🎯 Custom Tournament Mode")
    st.info("💡 Enter your team's players and the tournament league ID(s) to analyze.")
    
    custom_team_name = st.text_input("🏷️ Team Name", value=st.session_state.get("custom_team_name", "My Team"))
    leagues_input = st.text_input("🏆 League ID(s)", value="17804,17805", 
                                  help="Comma-separated. Find IDs on OpenDota/Dotabuff")
    
    st.subheader("👥 Team Roster")
    players_input = []
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(f"**P{i+1}**")
            name = st.text_input("Name", key=f"cname_{i}", placeholder="Name")
            steam = st.text_input("Steam ID", key=f"csteam_{i}", placeholder="Steam32/64")
            pos = st.selectbox("Pos", ["", "1", "2", "3", "4", "5"], key=f"cpos_{i}", index=i+1 if i<5 else 0)
            if name and steam and pos:
                try:
                    s32 = int(steam)
                    if s32 >= 76561197960265728:
                        s32 = s32 - 76561197960265728
                    players_input.append({"name": name, "steam32": s32, "pos": int(pos)})
                except:
                    st.warning(f"Invalid Steam ID")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        load_custom = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    if load_custom:
        if len(players_input) < 3:
            st.error("❌ Enter at least 3 valid players")
        else:
            league_ids = [int(x.strip()) for x in leagues_input.split(",") if x.strip().isdigit()]
            if not league_ids:
                st.error("❌ Enter at least one valid league ID")
            else:
                with st.spinner("Fetching tournament data..."):
                    try:
                        (df, heroes, picks_df, bans_df, opp_picks_df, opp_bans_df,
                         seq_by_match, team_name, plinks) = collect_custom_tournament(
                            players_input, league_ids, custom_team_name, stratz_token, max_matches)
                        st.session_state.update({
                            "stratz_token": stratz_token,
                            "mode": "custom",
                            "custom_team_name": custom_team_name,
                            "custom_players": players_input,
                            "data": {
                                "df": df, "heroes": heroes, "picks_df": picks_df, "bans_df": bans_df,
                                "opp_picks_df": opp_picks_df, "opp_bans_df": opp_bans_df,
                                "seq": seq_by_match, "team_name": team_name, "plinks": plinks,
                                "leagues": league_ids
                            }
                        })
                        if df.empty:
                            st.warning("⚠️ No matches found")
                        else:
                            st.success(f"✅ Found {len(df)} matches for **{custom_team_name}**!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

# ============================================================
# Data Display (shared by both modes)
# ============================================================
data = st.session_state.get("data")
if not data:
    st.info("👆 Configure settings above and click Load/Analyze to begin.")
    st.stop()

df=data["df"]; heroes=data["heroes"]; team_name=data["team_name"] or "Our Team"
picks_df=data["picks_df"]; bans_df=data["bans_df"]
opp_picks_df=data["opp_picks_df"]; opp_bans_df=data["opp_bans_df"]
seq_by_match=data["seq"]; plinks=data["plinks"]
leagues = data.get("leagues", DEFAULT_LEAGUES)

if df.empty:
    st.warning("No match data available.")
    st.stop()

st.divider()
st.subheader(f"📊 Team: **{team_name}**")

tabs = st.tabs(["Overview", "Summary", "Drafts (2-row)", "Picks by Position", "Raw Data", "Player Scout", "Export PDF"])

# ---- Overview (4×1)
with tabs[0]:
    c1,c2,c3,c4=st.columns(4)
    with c1:
        st.markdown("**Most Picked**"); st.dataframe(picks_df, use_container_width=True)
    with c2:
        st.markdown(f"**Picks vs {team_name}**"); st.dataframe(opp_picks_df, use_container_width=True)
    with c3:
        st.markdown("**Most Banned**"); st.dataframe(bans_df, use_container_width=True)
    with c4:
        st.markdown(f"**Bans vs {team_name}**"); st.dataframe(opp_bans_df, use_container_width=True)

# ---- Summary
with tabs[1]:
    st.subheader("Summary")
    summary=compute_summary(df, seq_by_match, heroes)

    st.markdown("### Row 1 — Radiant vs Dire")
    c1,c2=st.columns(2)
    with c1:
        o=summary["row1"]["overall"]; m1,m2=st.columns(2)
        m1.metric("Radiant WR", f'{o["Radiant"]["wr"]}%', f'{o["Radiant"]["wins"]}/{o["Radiant"]["total"]}')
        m2.metric("Dire WR",    f'{o["Dire"]["wr"]}%',    f'{o["Dire"]["wins"]}/{o["Dire"]["total"]}')
        t=st.tabs(["Radiant (by us)","Dire (by us)"])
        with t[0]: st.dataframe(summary["row1"]["by_us"]["Radiant"], use_container_width=True)
        with t[1]: st.dataframe(summary["row1"]["by_us"]["Dire"],    use_container_width=True)
    with c2:
        t=st.tabs(["Radiant (vs us)","Dire (vs us)"])
        with t[0]: st.dataframe(summary["row1"]["vs_us"]["Radiant"], use_container_width=True)
        with t[1]: st.dataframe(summary["row1"]["vs_us"]["Dire"],    use_container_width=True)

    st.markdown("### Row 2 — First Pick vs Second Pick")
    c1,c2=st.columns(2)
    with c1:
        o=summary["row2"]["overall"]; m1,m2=st.columns(2)
        m1.metric("First Pick WR",  f'{o["FP"]["wr"]}%', f'{o["FP"]["wins"]}/{o["FP"]["total"]}')
        m2.metric("Second Pick WR", f'{o["SP"]["wr"]}%', f'{o["SP"]["wins"]}/{o["SP"]["total"]}')
        t=st.tabs(["FP (by us)","SP (by us)"])
        with t[0]: st.dataframe(summary["row2"]["by_us"]["FP"], use_container_width=True)
        with t[1]: st.dataframe(summary["row2"]["by_us"]["SP"], use_container_width=True)
    with c2:
        t=st.tabs(["FP (vs us)","SP (vs us)"])
        with t[0]: st.dataframe(summary["row2"]["vs_us"]["FP"], use_container_width=True)
        with t[1]: st.dataframe(summary["row2"]["vs_us"]["SP"], use_container_width=True)

    st.markdown("### Row 3 — Draft Phases (By Us)")
    c1,c2=st.columns(2)
    with c1:
        t=st.tabs(["Pick 1","Pick 2","Pick 3"])
        with t[0]: st.dataframe(summary["row3"]["Pick phase 1"], use_container_width=True)
        with t[1]: st.dataframe(summary["row3"]["Pick phase 2"], use_container_width=True)
        with t[2]: st.dataframe(summary["row3"]["Pick phase 3"], use_container_width=True)
    with c2:
        t=st.tabs(["Ban 1","Ban 2","Ban 3"])
        with t[0]: st.dataframe(summary["row3"]["Ban  phase 1"], use_container_width=True)
        with t[1]: st.dataframe(summary["row3"]["Ban  phase 2"], use_container_width=True)
        with t[2]: st.dataframe(summary["row3"]["Ban  phase 3"], use_container_width=True)

    st.markdown("### Row 4 — Draft Phases (Against Us)")
    c1,c2=st.columns(2)
    with c1:
        t=st.tabs(["Pick 1","Pick 2","Pick 3"])
        with t[0]: st.dataframe(summary["row4"]["Pick phase 1"], use_container_width=True)
        with t[1]: st.dataframe(summary["row4"]["Pick phase 2"], use_container_width=True)
        with t[2]: st.dataframe(summary["row4"]["Pick phase 3"], use_container_width=True)
    with c2:
        t=st.tabs(["Ban 1","Ban 2","Ban 3"])
        with t[0]: st.dataframe(summary["row4"]["Ban  phase 1"], use_container_width=True)
        with t[1]: st.dataframe(summary["row4"]["Ban  phase 2"], use_container_width=True)
        with t[2]: st.dataframe(summary["row4"]["Ban  phase 3"], use_container_width=True)

# ---- Drafts
with tabs[2]:
    # NEW: portrait toggle (major speed if off)
    use_portraits = st.checkbox("Show portraits", value=True, key="show_portraits")
    scale=st.slider("PNG scale", 0.8, 2.0, 1.1, 0.1, key="scale_drafts")
    for _,row in df.sort_values("start_time_utc").iterrows():
        mid=int(row["match_id"])
        time_str=row["start_time_utc"].replace("T"," ") if isinstance(row["start_time_utc"],str) else ""
        outcome="Win ✅" if row["our_win"] else "Loss ❌"
        opp=row.get("opponent") or "Opponent"
        with st.expander(f"Match {mid} — {row['our_side']} — {outcome} — {time_str} — vs {opp}"):
            seq=seq_by_match.get(mid) or []; our_idx=0 if row["our_side"]=="Radiant" else 1
            # HTML row (greyed bans + cross)
            b=p=0; our_html=[]; opp_html=[]
            for it in sorted(seq, key=lambda x:x.get("order",0)):
                is_pick=bool(it.get("is_pick")); b+= (0 if is_pick else 1); p+= (1 if is_pick else 0)
                lab=f"{'PICK' if is_pick else 'BAN'} {p if is_pick else b}"
                hid=it.get("hero_id"); nm=hero_name(hid, heroes); data_uri=portrait_data_uri(hid, heroes)
                filter_css="" if is_pick else "filter: grayscale(100%);"
                tag=(f"<div style='font-size:12px;padding:2px 6px;border:1px solid #666;border-radius:6px;display:inline-block;margin-bottom:6px'>{lab}</div>")
                cross_svg=(""
                           if is_pick else
                           "<svg width='120' height='65' style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none'>"
                           "<line x1='6' y1='6' x2='114' y2='59' stroke='rgba(120,120,120,0.95)' stroke-width='4'/>"
                           "<line x1='114' y1='6' x2='6' y2='59' stroke='rgba(120,120,120,0.95)' stroke-width='4'/></svg>")
                card=(f"<div style='display:inline-block;border:1px solid #555;border-radius:8px;padding:6px;margin-right:8px;background:#111;text-align:center;position:relative'>"
                      f"{tag}<div style='position:relative;width:120px;height:65px;margin:0 auto'>"
                      f"<img src='{data_uri}' style='width:120px;height:65px;border-radius:6px;object-fit:cover;{filter_css}'>{cross_svg}</div>"
                      f"<div style='font-size:11px;color:#ddd;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:2px'>{nm}</div></div>")
                (our_html if it.get("team")==our_idx else opp_html).append(card)
            st.markdown("<div style='white-space:nowrap;overflow-x:auto'>"+"".join(our_html)+"</div>"
                        "<div style='color:#bbb;margin-top:6px'>"+team_name+"</div>", unsafe_allow_html=True)
            st.markdown("<div style='white-space:nowrap;overflow-x:auto;margin-top:8px'>"+"".join(opp_html)+"</div>"
                        "<div style='color:#bbb;margin-top:6px'>Opponents</div>", unsafe_allow_html=True)
            png_bytes=draft_png_two_rows(seq, our_idx, heroes, team_name, scale=st.session_state.get("scale_drafts",1.1))
            st.download_button(f"Download Match {mid} (PNG)", png_bytes, file_name=f"draft_{mid}.png", mime="image/png")

# ---- Picks by Position
with tabs[3]:
    pos_tables=compute_pos_tables(df, heroes)
    cols=st.columns(5)
    for i,pos in enumerate([1,2,3,4,5]):
        with cols[i]:
            st.markdown(f"**Pos{pos}**"); st.dataframe(pos_tables[pos], use_container_width=True)

# ---- Raw Data
with tabs[4]:
    st.dataframe(df, use_container_width=True)

# ---- Player Scout (role-specific; STRATZ token recommended)
with tabs[5]:
    st.subheader("Player Scout — role-specific (no last-N cut-off)")

    # Defaults from RD2L roster (prefill positions 1..5)
    default_players = []
    for i, (nm, url) in enumerate(list(plinks.items())[:5]):
        acc = get_roster_account_id(url) or 0
        default_players.append({"name": nm, "steam": str(acc), "pos": str(i + 1)})
    while len(default_players) < 5:
        i = len(default_players)
        default_players.append({"name": "", "steam": "", "pos": str((i % 5) + 1)})

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            default_players[i]["name"] = st.text_input(
                f"Player {i+1} name",
                value=default_players[i]["name"],
                key=f"ps_name_{i}",
            )
            default_players[i]["steam"] = st.text_input(
                f"SteamID (32 or 64) — P{i+1}",
                value=default_players[i]["steam"],
                key=f"ps_steam_{i}",
            )
            options = ["", "1", "2", "3", "4", "5"]
            idx = options.index(default_players[i]["pos"]) if default_players[i]["pos"] in options else 0
            default_players[i]["pos"] = st.selectbox(
                f"Pos (P{i+1})",
                options,
                index=idx,
                key=f"ps_pos_{i}",
            )

    topN = st.number_input("Show top heroes (per column)", 5, 30, 15, 1, key="ps_topN")
    months = st.number_input("Last X months (for STRATZ window)", 1, 12, 3, 1, key="ps_months")
    leagues_str = st.text_input(
        "Tournament leagues (comma-separated)",
        ",".join(map(str, DEFAULT_LEAGUES)),
        key="ps_leagues",
    )
    any_role = st.checkbox("Any role (ignore position filter)", value=False, key="ps_any_role")

    if st.button("Fetch Player Scout", key="ps_fetch"):
        players = []
        for i in range(5):
            name = st.session_state.get(f"ps_name_{i}", "").strip()
            steam = st.session_state.get(f"ps_steam_{i}", "").strip()
            pos_s = st.session_state.get(f"ps_pos_{i}", "").strip()
            if not (name and steam and pos_s):
                continue
            try:
                s32 = to_steam32(int(steam))
                pos = int(pos_s)
            except Exception:
                continue
            players.append({"name": name, "steam32": s32, "pos": pos})

        if not players:
            st.warning("Please enter at least one valid player (name + steam + pos).")
        else:
            leagues = [int(x.strip()) for x in leagues_str.split(",") if x.strip().isdigit()]
            token = st.session_state.get("stratz_token") or stratz_token
            with st.spinner("Fetching STRATZ/OD…"):
                scout = player_scout(players, int(topN), int(months), leagues, bool(any_role), token)

            # Preview tables
            st.markdown("### Top heroes last X months (role-specific)")
            cols = st.columns(len(players))
            for i, p in enumerate(players):
                with cols[i]:
                    rank = (scout["ranks"][i] if i < len(scout["ranks"]) else "")
                    st.markdown(f"**{p['name']} (P{p['pos']}) — {rank}**")
                    dfcol = pd.DataFrame({
                        "#": [j+1 for j in range(topN)],
                        f"{p['name']} — last {months} mo": (scout['lastx'][i] + [''] * topN)[:topN],
                    })
                    st.dataframe(dfcol, use_container_width=True, hide_index=True)

            st.markdown(f"### Tournament heroes (leagues: {', '.join(map(str, [int(x) for x in leagues or DEFAULT_LEAGUES]))})")
            cols = st.columns(len(players))
            for i, p in enumerate(players):
                with cols[i]:
                    rank = (scout["ranks"][i] if i < len(scout["ranks"]) else "")
                    st.markdown(f"**{p['name']} (P{p['pos']}) — {rank}**")
                    dfcol = pd.DataFrame({
                        "#": [j+1 for j in range(topN)],
                        f"{p['name']} — tournament": (scout['tourn'][i] + [''] * topN)[:topN],
                    })
                    st.dataframe(dfcol, use_container_width=True, hide_index=True)

            # store for Export PDF
            st.session_state["scout_payload"] = {"players": players, "scout": scout}

# ---- Export PDF
with tabs[6]:
    st.subheader("Full Report (PDF)")
    pos_tables_pdf=compute_pos_tables(df, heroes)
    summary_pdf=compute_summary(df, seq_by_match, heroes)

    scout_payload=st.session_state.get("scout_payload")
    scout=None; players_meta=None
    if scout_payload:
        scout=scout_payload["scout"]; players_meta=scout_payload["players"]
    else:
        # Auto-build from current UI fields if possible (if STRATZ token exists)
        token = st.session_state.get("stratz_token") or stratz_token
        if token:
            players = []
            for i in range(5):
                name = st.session_state.get(f"ps_name_{i}", "").strip()
                steam = st.session_state.get(f"ps_steam_{i}", "").strip()
                pos_s = st.session_state.get(f"ps_pos_{i}", "").strip()
                if not (name and steam and pos_s):
                    continue
                try:
                    s32 = to_steam32(int(steam))
                    pos = int(pos_s)
                except Exception:
                    continue
                players.append({"name": name, "steam32": s32, "pos": pos})
            if players:
                topN = int(st.session_state.get("ps_topN", 15))
                months = int(st.session_state.get("ps_months", 3))
                leagues_str = st.session_state.get("ps_leagues", ",".join(map(str, DEFAULT_LEAGUES)))
                leagues = [int(x.strip()) for x in leagues_str.split(",") if x.strip().isdigit()]
                any_role = bool(st.session_state.get("ps_any_role", False))
                scout = player_scout(players, topN, months, leagues, any_role, token)
                players_meta = players

    pdf_bytes=build_pdf(team_name, picks_df, bans_df, opp_picks_df, opp_bans_df,
                        pos_tables_pdf, df, seq_by_match, heroes, summary_pdf,
                        scout=scout, players_meta=players_meta)
    st.download_button("Download FULL REPORT (PDF)", pdf_bytes, file_name="rd2l_report.pdf", mime="application/pdf")
