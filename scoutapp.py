#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ==================================================
# CONFIG
# ==================================================
APP_VERSION = "v1-clean"
OPENDOTA = "https://api.opendota.com/api"
STRATZ_GQL = "https://api.stratz.com/graphql"
STEAM64_BASE = 76561197960265728

# ==================================================
# HELPERS
# ==================================================
def to_steam32(v: str) -> int:
    n = int(v.strip())
    return n - STEAM64_BASE if n >= STEAM64_BASE else n

def http_get(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# ==================================================
# OPENDOTA
# ==================================================
@st.cache_data(ttl=86400)
def heroes_map():
    heroes = {}
    for h in http_get(f"{OPENDOTA}/heroes"):
        heroes[h["id"]] = h["localized_name"]
    return heroes

def recent_matches(pid, limit=50):
    return http_get(f"{OPENDOTA}/players/{pid}/matches", {"limit": limit})

def match_details(mid):
    return http_get(f"{OPENDOTA}/matches/{mid}")

def player_heroes(pid):
    return http_get(f"{OPENDOTA}/players/{pid}/heroes")

# ==================================================
# STRATZ
# ==================================================
def stratz_matches(pid, token, leagues=None):
    if not token:
        return None

    query = """
    query ($id: Long!, $leagues: [Int!]) {
      player(steamAccountId: $id) {
        matches(orderBy: MATCH_DATE_DESC, leagueIds: $leagues, take: 200) {
          players(steamAccountId: $id) {
            heroId
            isVictory
            position
          }
        }
      }
    }
    """

    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(
        STRATZ_GQL,
        json={"query": query, "variables": {"id": pid, "leagues": leagues}},
        headers=headers,
        timeout=40,
    )

    if r.status_code != 200:
        return None

    data = r.json()
    if "errors" in data:
        return None

    rows = []
    for m in data["data"]["player"]["matches"]:
        p = m["players"][0]
        rows.append({
            "hero_id": p["heroId"],
            "win": p["isVictory"],
            "pos": p["position"],
        })

    return pd.DataFrame(rows)

# ==================================================
# UI
# ==================================================
st.set_page_config("ScoutDota", layout="wide")
st.title("ScoutDota — Universal Draft Scout")

heroes = heroes_map()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    mode = st.radio("Mode", ["RD2L URL", "Manual"])
    token = st.text_input("STRATZ Token", type="password")
    debug = st.checkbox("Debug")

# ==================================================
# INPUT
# ==================================================
players = []

if mode == "Manual":
    st.subheader("Manual Player Entry")
    for i in range(5):
        c1, c2, c3 = st.columns([2, 2, 1])
        name = c1.text_input(f"name{i}")
        sid = c2.text_input(f"sid{i}")
        pos = c3.selectbox(f"pos{i}", ["—", "1", "2", "3", "4", "5"])
        if name and sid and pos != "—":
            players.append({
                "name": name,
                "id": to_steam32(sid),
                "pos": int(pos),
            })

if st.button("Load"):
    st.session_state.players = players

players = st.session_state.get("players", [])

# ==================================================
# TABS
# ==================================================
tabs = st.tabs(["Overview", "Drafts", "Player Scout", "Raw Matches"])

# ==================================================
# OVERVIEW TAB
# ==================================================
with tabs[0]:
    st.subheader("Overview")
    if not players:
        st.info("No players loaded.")
    else:
        st.success(f"{len(players)} players loaded")
        for p in players:
            st.write(f"- {p['name']} (Pos {p['pos']})")

# ==================================================
# DRAFTS TAB
# ==================================================
with tabs[1]:
    st.subheader("Drafts")
    if not players:
        st.info("Load players first.")
    else:
        mids = []
        for p in players:
            for m in recent_matches(p["id"], 30):
                mids.append(m["match_id"])

        mids = list(set(mids))[:10]
        for mid in mids:
            m = match_details(mid)
            if not m.get("picks_bans"):
                continue
            with st.expander(f"Match {mid}"):
                for pb in m["picks_bans"]:
                    hero = heroes.get(pb["hero_id"], "Unknown")
                    st.write(("PICK" if pb["is_pick"] else "BAN"), hero)

# ==================================================
# PLAYER SCOUT TAB
# ==================================================
with tabs[2]:
    st.subheader("Player Scout")

    if not players:
        st.info("Load players first.")
    else:
        for p in players:
            st.markdown(f"### {p['name']} (Pos {p['pos']})")

            df = stratz_matches(p["id"], token)
            if df is None:
                st.warning("STRATZ failed → OpenDota fallback (NOT role-specific)")
                data = player_heroes(p["id"])
                for h in data[:10]:
                    st.write(heroes.get(h["hero_id"]), h["games"])
            else:
                df = df[df["pos"] == p["pos"]]
                counts = df.groupby("hero_id").size().sort_values(ascending=False)
                for hid, g in counts.head(10).items():
                    st.write(heroes.get(hid), g)

# ==================================================
# RAW MATCHES TAB
# ==================================================
with tabs[3]:
    st.subheader("Raw Matches")
    if not players:
        st.info("Load players first.")
    else:
        rows = []
        for p in players:
            for m in recent_matches(p["id"], 20):
                rows.append(m)
        st.dataframe(pd.DataFrame(rows))
