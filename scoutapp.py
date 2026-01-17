#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RD2L / Universal Draft Scout (Streamlit)
- Mode 1: RD2L Team URL (scrape match IDs + roster IDs)
- Mode 2: Manual / Universal (enter players, tournament label; optional league IDs)
- Drafts: match-by-match picks/bans with portraits (bans greyed)
- Player Scout: Overall (OpenDota), Last X months (STRATZ -> OpenDota fallback), Tournament (STRATZ -> OpenDota fallback)
- Debug mode: shows API errors + optional JSON previews (no token logging)

Run locally:
  streamlit run app.py

Deploy:
  Push to GitHub, deploy in Streamlit Community Cloud
"""

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
from PIL import Image, ImageOps

# =========================
# ====== CONSTANTS ========
# =========================
APP_VERSION = "universal-v1"
USER_AGENT = f"rd2l-universal-scout/{APP_VERSION}"
OPENDOTA_BASE = "https://api.opendota.com/api"
STRATZ_GQL = "https://api.stratz.com/graphql"
STEAM64_BASE = 76561197960265728

DEFAULT_RD2L_TEAM_URL = "https://rd2l.gg/seasons/QSj0aP5YM/divisions/U-ZTEMOBg/teams/KGghuA-8c"
DEFAULT_LEAGUES_RD2L = [17804, 17805]  # RD2L S35/S36 (edit anytime)


# =========================
# ===== HTTP HELPERS ======
# =========================
def http_get(url: str, params=None, timeout=25) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

def http_post(url: str, json_body=None, headers=None, timeout=40) -> requests.Response:
    h = {"User-Agent": USER_AGENT, "Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    r = requests.post(url, json=json_body, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


# =========================
# ===== STEAM UTILS =======
# =========================
def to_steam32(v: str) -> int:
    n = int(str(v).strip())
    return n - STEAM64_BASE if n >= STEAM64_BASE else n


# =========================
# ===== OPENDOTA API ======
# =========================
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def od_heroes_map() -> Dict[int, Dict[str, str]]:
    """
    Returns {hero_id: {"name": "...", "slug": "...", "img": "/apps/..."}}
    Uses /heroes (fast) for names + img paths.
    """
    js = http_get(f"{OPENDOTA_BASE}/heroes").json()
    out = {}
    for h in js:
        hid = int(h["id"])
        name = h.get("localized_name") or ""
        img = h.get("img")
        code = h.get("name", "npc_dota_hero_")
        slug = code.replace("npc_dota_hero_", "")
        out[hid] = {"name": name, "slug": slug, "img": img or "",}
    return out

def hero_name(hid: Optional[int], heroes: Dict[int, Dict[str, str]]) -> str:
    if not hid:
        return ""
    return heroes.get(int(hid), {}).get("name", f"id{hid}")

def hero_portrait_urls(hid: int, heroes: Dict[int, Dict[str, str]]) -> List[str]:
    """
    Try: OpenDota img path -> Valve CDN -> Stratz CDN fallback
    """
    h = heroes.get(int(hid), {})
    urls = []
    img = h.get("img") or ""
    if img:
        urls.append("https://api.opendota.com" + img if img.startswith("/") else img)
    slug = h.get("slug") or ""
    if slug:
        urls.append(f"https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/{slug}.png")
        urls.append(f"https://cdn.stratz.com/images/dota2/heroes/{slug}_icon.png")
    return urls

_IMG_BYTES: Dict[int, bytes] = {}

def get_hero_image_bytes(hid: Optional[int], heroes: Dict[int, Dict[str, str]], allow: bool) -> Optional[bytes]:
    if not allow or not hid:
        return None
    hid = int(hid)
    if hid in _IMG_BYTES:
        return _IMG_BYTES[hid]
    for u in hero_portrait_urls(hid, heroes):
        try:
            b = http_get(u, timeout=20).content
            if b and len(b) > 2000:
                _IMG_BYTES[hid] = b
                return b
        except Exception:
            continue
    return None

def portrait_data_uri(hid: Optional[int], heroes: Dict[int, Dict[str, str]], allow: bool, size=(120, 65)) -> str:
    b = get_hero_image_bytes(hid, heroes, allow=allow)
    if not b:
        return ""
    try:
        im = Image.open(io.BytesIO(b)).convert("RGB").resize(size)
        bio = io.BytesIO()
        im.save(bio, format="PNG")
        return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")
    except Exception:
        return ""

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def od_match(match_id: int) -> dict:
    try:
        return http_get(f"{OPENDOTA_BASE}/matches/{int(match_id)}", timeout=30).json()
    except Exception:
        return {}

def od_recent_matches(steam32: int, limit: int = 100) -> pd.DataFrame:
    js = http_get(f"{OPENDOTA_BASE}/players/{int(steam32)}/matches", params={"limit": int(limit)}, timeout=30).json()
    return pd.DataFrame(js)

def od_overall_heroes(steam32: int) -> pd.DataFrame:
    js = http_get(f"{OPENDOTA_BASE}/players/{int(steam32)}/heroes", timeout=30).json()
    return pd.DataFrame(js)

def od_player_profile(steam32: int) -> dict:
    return http_get(f"{OPENDOTA_BASE}/players/{int(steam32)}", timeout=30).json()


# =========================
# ===== STRATZ GQL ========
# =========================
def gql_post(query: str, variables: dict, token: str) -> Tuple[dict, str]:
    """
    Returns (json, error_string). Never logs token.
    """
    if not token:
        return {}, "Missing STRATZ token"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = http_post(STRATZ_GQL, json_body={"query": query, "variables": variables}, headers=headers, timeout=40)
        js = r.json()
        if isinstance(js, dict) and js.get("errors"):
            return {}, js["errors"][0].get("message", "GraphQL error")
        return js, ""
    except requests.HTTPError as e:
        return {}, f"HTTPError: {str(e)}"
    except Exception as e:
        return {}, f"Exception: {str(e)}"

STRATZ_MATCHES_QUERY = """
query($steam32: Long!, $take: Int!, $skip: Int, $leagues: [Int!]) {
  player(steamAccountId: $steam32) {
    matches(orderBy: MATCH_DATE_DESC, leagueIds: $leagues, take: $take, skip: $skip) {
      endDateTime
      leagueId
      players(steamAccountId:$steam32) { heroId isVictory position role lane }
    }
  }
}
"""

def stratz_fetch_matches(steam32: int, token: str, want: int = 300, leagues: Optional[List[int]] = None) -> Tuple[pd.DataFrame, str]:
    """
    Fetch up to 'want' match rows (paged). Each row contains hero_id/is_victory/position/role/lane/leagueId/endDateTime.
    """
    take = 100
    pages = max(1, min(5, (want + take - 1) // take))
    rows = []
    for page in range(pages):
        js, err = gql_post(
            STRATZ_MATCHES_QUERY,
            {"steam32": int(steam32), "take": int(take), "skip": int(page * take), "leagues": leagues},
            token=token
        )
        if err:
            return pd.DataFrame(), err
        matches = (((js.get("data") or {}).get("player") or {}).get("matches")) or []
        if not matches:
            break
        for m in matches:
            ps = m.get("players") or []
            if not ps:
                continue
            p0 = ps[0]
            rows.append({
                "hero_id": p0.get("heroId"),
                "is_victory": bool(p0.get("isVictory", False)),
                "position": p0.get("position"),
                "role": (p0.get("role") or ""),
                "lane": p0.get("lane"),
                "leagueId": m.get("leagueId"),
                "endDateTime": m.get("endDateTime"),
            })
        if len(matches) < take:
            break
        time.sleep(0.2)  # courtesy
    return pd.DataFrame(rows), ""

def role_mask(df: pd.DataFrame, pos: int, any_role: bool) -> pd.Series:
    if df.empty or any_role:
        return pd.Series([True] * len(df), index=df.index)
    m = pd.Series([False] * len(df), index=df.index)

    # If STRATZ returns numeric position (often does), use it:
    if "position" in df and df["position"].notna().any():
        try:
            m |= (df["position"].astype("Int64") == int(pos))
        except Exception:
            pass

    # Fallback by CORE/SUPPORT
    if "role" in df and df["role"].astype(str).str.len().gt(0).any():
        R = df["role"].astype(str).str.upper()
        is_sup = (R == "SUPPORT") | (R == "2")
        is_core = (R == "CORE") | (R == "1")
        m |= (is_sup if pos in (4, 5) else is_core)

    # If still nothing, allow all rather than blank
    if not m.any():
        return pd.Series([True] * len(df), index=df.index)
    return m


# =========================
# ===== RD2L SCRAPE =======
# =========================
TEAM_ID_RE = re.compile(r"/teams/([A-Za-z0-9_\-]+)")
MATCH_ID_RE = re.compile(r"/matches/(\d+)")

def extract_team_id(url: str) -> Optional[str]:
    m = TEAM_ID_RE.search(url)
    return m.group(1) if m else None

def rd2l_team_page(team_url: str) -> Tuple[List[int], Dict[str, str], str]:
    """
    Returns:
      - match_ids found on the team page (dotabuff/opendota links + loose IDs)
      - player profile links {player_name: profile_url}
      - team_name
    """
    html = http_get(team_url, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")

    team_name = ""
    h = soup.find(["h1", "h2"])
    if h:
        team_name = h.get_text(strip=True)
    if not team_name and soup.title:
        team_name = soup.title.get_text(strip=True).split(" – ")[0].strip()

    match_ids: Set[int] = set()
    player_links: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # matches links
        if "dotabuff.com/matches" in href or "opendota.com/matches" in href:
            mm = MATCH_ID_RE.search(href)
            if mm:
                match_ids.add(int(mm.group(1)))

        # player profile links
        if re.search(r"/profile/\d+$", href):
            nm = a.get_text(strip=True)
            if nm:
                player_links[nm] = requests.compat.urljoin(team_url, href)

    # loose match ids in text
    text = soup.get_text(" ", strip=True)
    for m in re.finditer(r"\b(\d{9,10})\b", text):
        match_ids.add(int(m.group(1)))

    return sorted(match_ids), player_links, team_name or "Team"

def rd2l_profile_to_steam32(profile_url: str) -> Optional[int]:
    """
    Attempts to find OpenDota /players/<steam32> or Steam64 or Dotabuff player id.
    """
    try:
        html = http_get(profile_url, timeout=25).text
    except Exception:
        return None
    soup = BeautifulSoup(html, "html.parser")

    for a in soup.find_all("a", href=True):
        m = re.search(r"opendota\.com/players/(\d+)", a["href"])
        if m:
            return int(m.group(1))

    for a in soup.find_all("a", href=True):
        m = re.search(r"steamcommunity\.com/profiles/(\d+)", a["href"])
        if m:
            return int(int(m.group(1)) - STEAM64_BASE)

    for a in soup.find_all("a", href=True):
        m = re.search(r"dotabuff\.com/players/(\d+)", a["href"])
        if m:
            return int(m.group(1))

    return None


# =========================
# ===== DRAFT PARSING =====
# =========================
def side_from_slot(slot: int) -> int:
    return 0 if slot < 128 else 1  # 0=Radiant, 1=Dire

def detect_our_side(match: dict, roster_ids: Set[int]) -> Optional[int]:
    """
    Determine if roster appears on Radiant or Dire.
    """
    r_ct = 0
    d_ct = 0
    for p in match.get("players", []) or []:
        acc = p.get("account_id")
        if acc in roster_ids:
            if side_from_slot(p.get("player_slot", 0)) == 0:
                r_ct += 1
            else:
                d_ct += 1
    if r_ct == 0 and d_ct == 0:
        return None
    return 0 if r_ct >= d_ct else 1

def extract_picks_bans(match: dict, our_side_idx: int) -> Tuple[List[int], List[int], List[int], List[int], List[dict]]:
    seq = sorted(match.get("picks_bans") or [], key=lambda x: x.get("order", 0))
    our_p = [x["hero_id"] for x in seq if x.get("is_pick") and x.get("team") == our_side_idx]
    our_b = [x["hero_id"] for x in seq if (not x.get("is_pick")) and x.get("team") == our_side_idx]
    op_p  = [x["hero_id"] for x in seq if x.get("is_pick") and x.get("team") != our_side_idx]
    op_b  = [x["hero_id"] for x in seq if (not x.get("is_pick")) and x.get("team") != our_side_idx]
    return our_p, our_b, op_p, op_b, seq

def render_draft_html(seq: List[dict], our_side_idx: int, heroes: Dict[int, Dict[str, str]], allow_portraits: bool) -> Tuple[str, str]:
    """
    Returns (our_html, opp_html) as HTML strings of cards in order.
    """
    b = p = 0
    our_cards = []
    opp_cards = []

    for it in sorted(seq, key=lambda x: x.get("order", 0)):
        is_pick = bool(it.get("is_pick"))
        if is_pick:
            p += 1
            lab = f"PICK {p}"
        else:
            b += 1
            lab = f"BAN {b}"

        hid = it.get("hero_id")
        nm = hero_name(hid, heroes)
        data_uri = portrait_data_uri(hid, heroes, allow=allow_portraits)

        filter_css = "" if is_pick else "filter: grayscale(100%);"
        cross_svg = (
            "" if is_pick else
            "<svg width='120' height='65' style='position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none'>"
            "<line x1='6' y1='6' x2='114' y2='59' stroke='rgba(120,120,120,0.95)' stroke-width='4'/>"
            "<line x1='114' y1='6' x2='6' y2='59' stroke='rgba(120,120,120,0.95)' stroke-width='4'/>"
            "</svg>"
        )

        tag = (
            f"<div style='font-size:12px;padding:2px 6px;border:1px solid #666;border-radius:6px;"
            f"display:inline-block;margin-bottom:6px;background:#1b1b1b;color:#ddd'>{lab}</div>"
        )

        card = (
            "<div style='display:inline-block;border:1px solid #555;border-radius:8px;padding:6px;margin-right:8px;"
            "background:#111;text-align:center;position:relative'>"
            f"{tag}"
            "<div style='position:relative;width:120px;height:65px;margin:0 auto'>"
            f"<img src='{data_uri}' style='width:120px;height:65px;border-radius:6px;object-fit:cover;{filter_css}'>"
            f"{cross_svg}"
            "</div>"
            f"<div style='font-size:11px;color:#ddd;max-width:120px;overflow:hidden;text-overflow:ellipsis;"
            f"white-space:nowrap;margin-top:2px'>{nm}</div>"
            "</div>"
        )

        (our_cards if it.get("team") == our_side_idx else opp_cards).append(card)

    our_html = "<div style='white-space:nowrap;overflow-x:auto;padding-bottom:8px'>" + "".join(our_cards) + "</div>"
    opp_html = "<div style='white-space:nowrap;overflow-x:auto;padding-bottom:8px'>" + "".join(opp_cards) + "</div>"
    return our_html, opp_html


# =========================
# ===== PLAYER SCOUT ======
# =========================
def rank_label(steam32: int) -> str:
    try:
        P = od_player_profile(steam32)
        if P.get("leaderboard_rank"):
            return f"Immortal {P['leaderboard_rank']}"
        rt = P.get("rank_tier")
        if rt:
            major, minor = rt // 10, rt % 10
            names = ["Herald", "Guardian", "Crusader", "Archon", "Legend", "Ancient", "Divine", "Immortal"]
            if 1 <= major <= 8:
                if major < 8 and 1 <= minor <= 5:
                    return f"{names[major-1]} {minor}"
                return names[major-1]
    except Exception:
        pass
    return "Unranked"

def overall_top_heroes(steam32: int, heroes: Dict[int, Dict[str, str]], topn: int) -> List[str]:
    try:
        df = od_overall_heroes(steam32)
        if df.empty:
            return []
        # OD columns: hero_id, games, win
        out = []
        df = df.copy()
        df["games"] = pd.to_numeric(df.get("games", 0), errors="coerce").fillna(0).astype(int)
        df["win"] = pd.to_numeric(df.get("win", 0), errors="coerce").fillna(0).astype(int)
        df = df.sort_values("games", ascending=False)
        for _, r in df.head(topn).iterrows():
            hid = int(r["hero_id"])
            g = int(r["games"])
            w = int(r["win"])
            pct = int(round(100 * w / max(1, g)))
            out.append(f"{hero_name(hid, heroes)} ({g} – {pct}%)")
        return out
    except Exception:
        return []

def od_last_months_pairs(steam32: int, months: int, heroes: Dict[int, Dict[str, str]], topn: int) -> List[str]:
    """
    OpenDota fallback: /players/{id}/matches?date=days (not role-specific)
    """
    days = 30 * int(months)
    try:
        js = http_get(f"{OPENDOTA_BASE}/players/{int(steam32)}/matches", params={"date": days}, timeout=30).json()
        df = pd.DataFrame(js)
        if df.empty or "hero_id" not in df.columns:
            return []
        if "radiant_win" not in df:
            df["radiant_win"] = False
        if "player_slot" not in df:
            df["player_slot"] = 0
        you_won = ((df["radiant_win"]) & (df["player_slot"] < 128)) | ((~df["radiant_win"]) & (df["player_slot"] >= 128))
        df2 = pd.DataFrame({"hero_id": df["hero_id"], "is_victory": you_won})
        return aggregate_pairs(df2, heroes, topn)
    except Exception:
        return []

def od_league_pairs(steam32: int, leagues: List[int], heroes: Dict[int, Dict[str, str]], topn: int) -> List[str]:
    rows = []
    for L in leagues or []:
        try:
            part = http_get(f"{OPENDOTA_BASE}/players/{int(steam32)}/matches", params={"leagueid": int(L)}, timeout=30).json()
            rows.extend(part)
        except Exception:
            continue
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if df.empty or "hero_id" not in df.columns:
        return []
    if "radiant_win" not in df:
        df["radiant_win"] = False
    if "player_slot" not in df:
        df["player_slot"] = 0
    you_won = ((df["radiant_win"]) & (df["player_slot"] < 128)) | ((~df["radiant_win"]) & (df["player_slot"] >= 128))
    df2 = pd.DataFrame({"hero_id": df["hero_id"], "is_victory": you_won})
    return aggregate_pairs(df2, heroes, topn)

def aggregate_pairs(df: pd.DataFrame, heroes: Dict[int, Dict[str, str]], topn: int) -> List[str]:
    if df.empty or "hero_id" not in df:
        return []
    g = df.groupby("hero_id", dropna=True)["is_victory"]
    counts = g.count().astype(int)
    wins = g.sum().astype(int)
    out = []
    for hid, games in counts.sort_values(ascending=False).head(topn).items():
        w = int(wins.get(hid, 0))
        pct = int(round(100 * w / max(1, int(games))))
        out.append(f"{hero_name(int(hid), heroes)} ({int(games)} – {pct}%)")
    return out

def player_scout(players: List[dict], topn: int, months: int, leagues: List[int], any_role: bool, stratz_token: str,
                debug: bool = False) -> Tuple[dict, List[str]]:
    """
    Returns (scout_data, warnings)
    scout_data keys: overall, lastx, tourn, ranks
    """
    heroes = od_heroes_map()
    warnings = []
    overall_cols = []
    lastx_cols = []
    tourn_cols = []
    ranks = []

    for p in players:
        s32 = int(p["steam32"])
        pos = int(p["pos"])
        ranks.append(rank_label(s32))

        overall_cols.append(overall_top_heroes(s32, heroes, topn))

        # Last X months: STRATZ -> OD fallback
        try:
            if not stratz_token:
                raise RuntimeError("No STRATZ token")
            df, err = stratz_fetch_matches(s32, stratz_token, want=300, leagues=None)
            if df.empty:
                raise RuntimeError(err or "STRATZ empty")
            # time window filter
            t_from = int((datetime.now(timezone.utc) - timedelta(days=30 * months)).timestamp())
            # normalize timestamp
            ts = pd.to_numeric(df.get("endDateTime"), errors="coerce").fillna(0)
            ts = ts.where(ts < 10_000_000_000, ts / 1000.0)  # ms -> sec
            df = df.loc[ts >= t_from].copy()
            mask = role_mask(df, pos, any_role)
            lastx_cols.append(aggregate_pairs(df[mask], heroes, topn))
        except Exception as e:
            warnings.append("STRATZ failed for Last X months; using OpenDota (NOT role-specific).")
            if debug:
                warnings.append(f"[debug] LastX STRATZ exception: {e}")
            lastx_cols.append(od_last_months_pairs(s32, months, heroes, topn))

        # Tournament: STRATZ -> OD fallback
        try:
            if not stratz_token:
                raise RuntimeError("No STRATZ token")
            dfL, errL = stratz_fetch_matches(s32, stratz_token, want=300, leagues=leagues)
            if dfL.empty:
                raise RuntimeError(errL or "STRATZ empty")
            maskL = role_mask(dfL, pos, any_role)
            tourn_cols.append(aggregate_pairs(dfL[maskL], heroes, topn))
        except Exception as e:
            warnings.append("STRATZ failed for Tournament slice; using OpenDota fallback.")
            if debug:
                warnings.append(f"[debug] Tournament STRATZ exception: {e}")
            tourn_cols.append(od_league_pairs(s32, leagues, heroes, topn))

        time.sleep(0.15)

    return {"overall": overall_cols, "lastx": lastx_cols, "tourn": tourn_cols, "ranks": ranks}, warnings


# =========================
# ===== MANUAL DRAFTS =====
# =========================
def manual_collect_match_ids_from_players(players: List[dict], recent_per_player: int, debug: bool = False) -> Tuple[List[int], List[str]]:
    """
    Universal method (no RD2L page):
    - Pull recent matches per player from OpenDota
    - Count match_ids; keep those that appear >= 2 times (tuneable) to approximate "team matches"
    """
    warnings = []
    counter = {}
    for p in players:
        try:
            df = od_recent_matches(int(p["steam32"]), limit=recent_per_player)
            mids = df["match_id"].dropna().astype(int).tolist() if "match_id" in df.columns else []
            for mid in mids:
                counter[mid] = counter.get(mid, 0) + 1
        except Exception as e:
            warnings.append(f"OpenDota recent matches failed for {p.get('name','player')}: {e}")
            if debug:
                warnings.append(f"[debug] {e}")

    # heuristic: match appears in >=2 players' recent lists
    # (set to 3 if you want stricter, but 2 is more forgiving)
    candidates = [mid for mid, c in counter.items() if c >= 2]
    candidates = sorted(candidates)
    return candidates, warnings

def filter_matches_by_roster_presence(match_ids: List[int], roster_ids: Set[int], max_matches: int, debug: bool = False) -> Tuple[pd.DataFrame, Dict[int, List[dict]], List[str]]:
    """
    Fetch matches by ID, keep those where we can detect our side, and picks_bans exist.
    Returns:
      df rows with match metadata
      seq_by_match dict {match_id: picks_bans_sequence}
      warnings
    """
    warnings = []
    heroes = od_heroes_map()
    rows = []
    seq_by_match = {}

    # take most recent-ish by numeric id heuristic (not perfect)
    match_ids = match_ids[-max_matches:]

    for mid in match_ids:
        m = od_match(int(mid))
        if not m or "players" not in m:
            continue

        our_side = detect_our_side(m, roster_ids)
        if our_side is None:
            continue

        seq = m.get("picks_bans") or []
        if not seq:
            continue

        seq_by_match[int(mid)] = seq

        # win/loss
        radiant_win = bool(m.get("radiant_win"))
        our_win = radiant_win if our_side == 0 else (not radiant_win)

        ts = m.get("start_time")
        start_iso = datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc).isoformat() if isinstance(ts, (int, float)) and ts > 0 else ""

        rows.append({
            "match_id": int(mid),
            "start_time_utc": start_iso,
            "duration_s": m.get("duration", ""),
            "our_side": "Radiant" if our_side == 0 else "Dire",
            "our_side_idx": our_side,
            "our_win": our_win,
        })

    df = pd.DataFrame(rows).sort_values("start_time_utc") if rows else pd.DataFrame()
    return df, seq_by_match, warnings


# =========================
# ========== UI ===========
# =========================
st.set_page_config(page_title=f"Draft Scout ({APP_VERSION})", layout="wide")
st.title(f"Draft Scout — {APP_VERSION}")

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("Scouting mode", ["Option 1: RD2L Team URL", "Option 2: Manual / Universal"], index=0)

    st.markdown("### STRATZ")
    stratz_token = st.text_input("STRATZ token (Bearer)", type="password", help="Users paste their own token. Not stored.")

    debug_mode = st.checkbox("Debug mode", value=False, help="Shows extra error detail (never prints token).")

    st.markdown("### Performance")
    show_portraits = st.checkbox("Show portraits", value=True)
    max_matches = st.slider("Max matches to show", 10, 120, 30, 5)

    st.markdown("---")

# Common settings for Player Scout
st.sidebar.markdown("### Player Scout settings")
topN = st.sidebar.slider("Top N heroes", 5, 25, 15, 1)
months = st.sidebar.slider("Last X months", 1, 12, 3, 1)
any_role = st.sidebar.checkbox("Any role (ignore position)", value=False)

# Tournament config
st.sidebar.markdown("### Tournament filter")
leagues_text = st.sidebar.text_input("League IDs (comma-separated, optional)", value=",".join(map(str, DEFAULT_LEAGUES_RD2L)))
league_ids = []
if leagues_text.strip():
    for x in leagues_text.split(","):
        x = x.strip()
        if x.isdigit():
            league_ids.append(int(x))

# =========================
# ===== MODE 1: RD2L ======
# =========================
if mode.startswith("Option 1"):
    st.subheader("Option 1 — RD2L Team URL")

    team_url = st.text_input("RD2L Team URL", value=DEFAULT_RD2L_TEAM_URL)
    tournament_label = "RD2L"  # used in headers; you can enhance this later

    if st.button("Load / Reload (RD2L)", type="primary"):
        with st.spinner("Scraping RD2L + building roster + loading matches..."):
            warn = []

            try:
                match_ids, plinks, team_name = rd2l_team_page(team_url)
            except Exception as e:
                st.error(f"Failed to load RD2L page: {e}")
                st.stop()

            players_meta = []
            roster_ids = set()
            
            for nm, url in plinks.items():
                s32 = rd2l_profile_to_steam32(url)
                if s32 is not None:
                    s32 = int(s32)
                    roster_ids.add(s32)
                    players_meta.append({
                        "name": nm,
                        "steam32": s32,
                        "pos": 0,  # 0 means "not set yet" (we'll show as —)
                    })
            
            # Keep a stable order for UI (by name)
            players_meta = sorted(players_meta, key=lambda x: x["name"].lower())


            if not roster_ids:
                warn.append("Could not resolve roster Steam IDs from RD2L profiles. Draft side detection may fail.")

            # Fetch matches and filter those that belong to roster
            df, seq_by_match, w2 = filter_matches_by_roster_presence(match_ids, roster_ids, max_matches=max_matches, debug=debug_mode)
            warn.extend(w2)

            st.session_state["loaded"] = {
                "mode": "rd2l",
                "team_name": team_name,
                "tournament_label": tournament_label,
                "roster_ids": roster_ids,
                "players_meta": [],  # optional to fill later
                "df": df,
                "seq": seq_by_match,
                "warnings": warn,
            }

# =========================
# == MODE 2: MANUAL / UNI ==
# =========================
else:
    st.subheader("Option 2 — Manual / Universal")

    tournament_label = st.text_input("Tournament name (label shown in drafts/PDF)", value="My Tournament")

    st.markdown("### Enter players (name, SteamID, position)")
    manual_players = []
    cols = st.columns([2, 2, 1])
    cols[0].markdown("**Name**")
    cols[1].markdown("**Steam ID (32 or 64)**")
    cols[2].markdown("**Pos**")

    for i in range(5):
        c1, c2, c3 = st.columns([2, 2, 1])
        name = c1.text_input(f"name_{i}", label_visibility="collapsed", placeholder=f"Player {i+1}")
        sid  = c2.text_input(f"sid_{i}", label_visibility="collapsed", placeholder="Steam32 or Steam64")
        pos  = c3.selectbox(f"pos_{i}", ["—", "1", "2", "3", "4", "5"], label_visibility="collapsed")
        if name and sid and pos != "—":
            try:
                manual_players.append({"name": name, "steam32": to_steam32(sid), "pos": int(pos)})
            except Exception:
                st.warning(f"Invalid Steam ID for {name}")

    recent_per_player = st.slider("Recent matches per player to scan (manual mode)", 20, 200, 80, 10)

    if st.button("Load / Reload (Manual)", type="primary"):
        if len(manual_players) < 2:
            st.error("Enter at least 2 players (name + steam + position) for manual mode.")
            st.stop()

        with st.spinner("Collecting candidate matches from players (OpenDota) ..."):
            roster_ids = {int(p["steam32"]) for p in manual_players}
            match_ids, warn1 = manual_collect_match_ids_from_players(manual_players, recent_per_player=recent_per_player, debug=debug_mode)
            df, seq_by_match, warn2 = filter_matches_by_roster_presence(match_ids, roster_ids, max_matches=max_matches, debug=debug_mode)

            st.session_state["loaded"] = {
                "mode": "manual",
                "team_name": "Manual Team",
                "tournament_label": tournament_label,
                "roster_ids": roster_ids,
                "players_meta": manual_meta,
                "df": df,
                "seq": seq_by_match,
                "warnings": warn1 + warn2,
            }

# =========================
# ===== SHOW OUTPUTS ======
# =========================
loaded = st.session_state.get("loaded")
if not loaded:
    st.info("Choose an option in the sidebar, fill inputs, then click **Load / Reload**.")
    st.stop()

heroes = od_heroes_map()
df = loaded["df"]
seq_by_match = loaded["seq"]
team_name = loaded["team_name"]
tournament_label = loaded["tournament_label"]
players_meta = loaded.get("players_meta") or []

if loaded.get("warnings"):
    st.warning(" / ".join(dict.fromkeys(loaded["warnings"]))[:2000])

tabs = st.tabs(["Drafts (match-by-match)", "Raw Matches", "Player Scout"])

# ---- Drafts tab
with tabs[0]:
    st.subheader(f"Drafts — {team_name} ({tournament_label})")

    if df.empty:
        st.info("No matches with picks/bans were found (or roster side couldn’t be detected).")
    else:
        for _, row in df.sort_values("start_time_utc").iterrows():
            mid = int(row["match_id"])
            tstr = (row["start_time_utc"].replace("T", " ") if isinstance(row["start_time_utc"], str) else "")
            outcome = "Win ✅" if bool(row["our_win"]) else "Loss ❌"
            side = row["our_side"]

            with st.expander(f"Match {mid} — {tournament_label} — {side} — {outcome} — {tstr}"):
                seq = seq_by_match.get(mid) or []
                our_idx = int(row["our_side_idx"])
                our_html, opp_html = render_draft_html(seq, our_idx, heroes, allow_portraits=show_portraits)

                st.markdown("**Our draft**", help="Top row = our team picks/bans in order")
                st.markdown(our_html, unsafe_allow_html=True)

                st.markdown("**Opponent draft**", help="Bottom row = opponent picks/bans in order")
                st.markdown(opp_html, unsafe_allow_html=True)

# ---- Raw matches tab
with tabs[1]:
    st.subheader("Raw match list")
    if df.empty:
        st.write("(empty)")
    else:
        st.dataframe(df, use_container_width=True)

# ---- Player Scout tab
with tabs[2]:
    st.subheader("Player Scout")

    # Determine players list: manual has explicit; rd2l mode doesn't (yet).
      if loaded["mode"] == "rd2l":
          if not players_meta:
              st.warning("Could not resolve any Steam IDs from RD2L roster links. Use Manual mode or enter players below.")
              players_for_scout = []
          else:
              st.success("RD2L roster auto-filled ✅  (set positions below)")
              st.markdown("### Confirm positions (required for role-specific STRATZ)")
              temp_players = []
              for i, p in enumerate(players_meta[:5]):  # RD2L teams are 5; slice for safety
                  c1, c2, c3 = st.columns([2, 2, 1])
                  name = c1.text_input(f"ps_name_{i}", value=p["name"])
                  sid  = c2.text_input(f"ps_sid_{i}", value=str(p["steam32"]))
                  # show "—" when pos == 0
                  pos_options = ["—", "1", "2", "3", "4", "5"]
                  default_idx = 0 if int(p.get("pos", 0)) == 0 else int(p["pos"])
                  pos = c3.selectbox(f"ps_pos_{i}", pos_options, index=default_idx)
      
                  if name and sid and pos != "—":
                      try:
                          temp_players.append({"name": name, "steam32": to_steam32(sid), "pos": int(pos)})
                      except Exception:
                          st.warning(f"Invalid Steam ID for {name}")
      
              players_for_scout = temp_players
      else:
          players_for_scout = players_meta


    if not players_for_scout:
        st.warning("No players available for Player Scout.")
    else:
        scout, warns = player_scout(
            players_for_scout, topn=int(topN), months=int(months),
            leagues=league_ids if league_ids else DEFAULT_LEAGUES_RD2L,
            any_role=any_role, stratz_token=stratz_token,
            debug=debug_mode
        )

        # Explain STRATZ fallback
        if warns:
            st.warning(
                "STRATZ did not work for some parts. Falling back to OpenDota where needed. "
                "OpenDota fallback is **NOT role-specific**."
            )
            if debug_mode:
                st.code("\n".join(warns[:50]))

        headers = [""] + [f"{p['name']} (P{p['pos']}) — {scout['ranks'][i]}" for i, p in enumerate(players_for_scout)]

        def build_matrix(title: str, cols: List[List[str]]) -> pd.DataFrame:
            rows = [[title] + [""] * len(cols)]
            for i in range(int(topN)):
                rows.append([""] + [(c[i] if i < len(c) else "") for c in cols])
            return pd.DataFrame(rows, columns=headers)

        df_overall = build_matrix(f"Most played overall (Top {topN})", scout["overall"])
        df_lastx   = build_matrix(f"Most played last {months} months ({'any role' if any_role else 'role-aware (STRATZ)'} )", scout["lastx"])
        df_tourn   = build_matrix(f"Most played tournament ({'any role' if any_role else 'role-aware (STRATZ)'} ; leagues: {','.join(map(str, league_ids or DEFAULT_LEAGUES_RD2L))})", scout["tourn"])

        st.markdown("### Overall")
        st.dataframe(df_overall, use_container_width=True)
        st.markdown("### Last X months")
        st.dataframe(df_lastx, use_container_width=True)
        st.markdown("### Tournament")
        st.dataframe(df_tourn, use_container_width=True)

