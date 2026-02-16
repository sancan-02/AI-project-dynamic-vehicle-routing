import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hybrid_controller import HybridController
from src.traffic_api import TrafficAPI

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import requests
import time
import torch

st.set_page_config(page_title="Dynamic Routing", layout="wide", page_icon="🚛")

if "locations" not in st.session_state:
    st.session_state.locations = [[12.9716, 77.5946]]
if "route_data" not in st.session_state:
    st.session_state.route_data = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "api" not in st.session_state:
    st.session_state.api = TrafficAPI()

def clear_data():
    st.session_state.locations = [[12.9716, 77.5946]]
    st.session_state.route_data = None
    st.session_state.metrics = None

def generate_route_logic():
    locs = st.session_state.locations
    if len(locs) < 2:
        st.error("Need at least 2 locations")
        return

    n_nodes = len(locs)
    locs_np = np.array(locs)
    traffic_matrix = np.ones((n_nodes, n_nodes))
    
    use_dqn = (5 <= n_nodes <= 10)
    dqn_agent = None
    
    if use_dqn:
        try:
            import warnings
            warnings.filterwarnings('ignore')
            controller = HybridController(mode="adaptive", n_nodes=n_nodes)
            
            if controller.agent:
                try:
                    model_path = "models/dqn_agent.pth"
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    controller.agent.q_network.load_state_dict(checkpoint['q_network'])
                    controller.agent.target_network.load_state_dict(checkpoint['target_network'])
                    controller.agent.q_network.to('cpu')
                    controller.agent.target_network.to('cpu')
                    controller.agent.q_network.eval()
                    dqn_agent = controller.agent
                except:
                    dqn_agent = controller.agent
        except:
            use_dqn = False
    
    current = 0
    unvisited = set(range(1, n_nodes))
    route = [0]
    path_traffic_build = []
    
    while unvisited:
        best_node = None
        min_cost = float('inf')
        
        if dqn_agent and len(unvisited) > 1:
            try:
                state = []
                for i in range(n_nodes):
                    if i == current:
                        state.extend([1, 0, 0])
                    elif i in unvisited:
                        state.extend([0, 1, 0])
                    else:
                        state.extend([0, 0, 1])
                state.append(len(unvisited) / n_nodes)
                state.append(len(route) / n_nodes)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = dqn_agent.q_network(state_tensor).squeeze()
                
                for i in range(n_nodes):
                    if i not in unvisited:
                        q_values[i] = -float('inf')
                
                dqn_choice = q_values.argmax().item()
                if dqn_choice in unvisited:
                    best_node = dqn_choice
            except:
                pass
        
        if best_node is None:
            for candidate in unvisited:
                dist_km = 111 * np.linalg.norm(locs_np[current] - locs_np[candidate])
                mult = st.session_state.api.get_traffic_multiplier(
                    tuple(locs[current]), tuple(locs[candidate])
                )
                traffic_matrix[current, candidate] = mult
                cost = (dist_km / 30.0) * 60 * mult
                if cost < min_cost:
                    min_cost = cost
                    best_node = candidate
        
        route.append(best_node)
        unvisited.remove(best_node)
        
        mult = st.session_state.api.get_traffic_multiplier(
            tuple(locs[current]), tuple(locs[best_node])
        )
        traffic_matrix[current, best_node] = mult
        path_traffic_build.append(mult)
        current = best_node
    
    if len(route) > 3:
        def route_distance(r):
            return sum(
                np.linalg.norm(locs_np[r[i]] - locs_np[r[i+1]])
                for i in range(len(r) - 1)
            )
        
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if route_distance(new_route) < route_distance(route) - 0.001:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
    
    total_time = 0
    total_dist = 0
    path_traffic = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        dist = 111 * np.linalg.norm(locs_np[u] - locs_np[v])
        mult = st.session_state.api.get_traffic_multiplier(
            tuple(locs[u]), tuple(locs[v])
        )
        traffic_matrix[u, v] = mult
        path_traffic.append(mult)
        total_time += (dist / (30.0 / mult)) * 60
        total_dist += dist
    
    avg_traffic = np.mean(path_traffic) if path_traffic else 1.0
    avg_speed = 30.0 / avg_traffic
    safety_status = "🔴 RISK" if avg_traffic > 1.8 else ("🟡 CAUTION" if avg_traffic > 1.5 else "🟢 SAFE")
    
    low_count = sum(1 for t in path_traffic if t <= 1.2)
    medium_count = sum(1 for t in path_traffic if 1.2 < t <= 1.5)
    heavy_count = sum(1 for t in path_traffic if t > 1.5)
    total_segments = len(path_traffic) if path_traffic else 1
    
    traffic_variance = float(np.var(path_traffic)) if len(path_traffic) > 0 else 0.0
    
    if traffic_variance > 0.08:
        policy_text = "🔴 Hybrid: DQN Active (High Variance)"
        policy_color = "#E74C3C"
    elif avg_traffic > 1.3:
        policy_text = "🟡 Hybrid: Adaptive Mode"
        policy_color = "#F39C12"
    else:
        policy_text = "🟢 Hybrid: Heuristic (Stable)"
        policy_color = "#2ECC71"
    
    st.session_state.metrics = {
        "total_stops": n_nodes - 1,
        "policy": policy_text,
        "policy_color": policy_color,
        "avg_speed": f"{avg_speed:.1f} km/h",
        "safety": safety_status,
        "eta": f"{total_time:.1f} min",
        "distance": f"{total_dist:.1f} km",
        "traffic_variance": traffic_variance,
        "traffic_breakdown": {
            "low": int(100 * low_count / total_segments),
            "medium": int(100 * medium_count / total_segments),
            "heavy": int(100 * heavy_count / total_segments)
        }
    }
    
    st.session_state.route_data = {
        "route": route,
        "traffic": traffic_matrix
    }

st.title("🚛 Dynamic Routing Dashboard")

with st.expander("🧠 **RL Approach: Deep Q-Learning**", expanded=False):
    st.markdown("""
    **Method:** DQN with Dueling Architecture
    
    **State:** Current position + visited mask + progress + traffic  
    **Action:** Select next delivery location  
    **Reward:** `-travel_time - delay_penalties`
    
    **Q-Update:** `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`
    
    **Hybrid Controller:**
    - Variance < 0.02 → Heuristic
    - Variance 0.02-0.08 → Adaptive
    - Variance > 0.08 → DQN
    """)

with st.sidebar:
    st.header("📍 Add Locations")
    
    address_input = st.text_input("Search Address", placeholder="e.g., Koramangala, Bangalore")
    if st.button("📍 Add", use_container_width=True):
        if address_input:
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": address_input + " Bangalore India", "format": "json", "limit": 1},
                    headers={"User-Agent": "routing-demo-1.0"},
                    timeout=5
                )
                if r.ok and r.json():
                    res = r.json()[0]
                    pt = [float(res["lat"]), float(res["lon"])]
                    st.session_state.locations.append(pt)
                    st.toast(f"✅ Added: {res.get('display_name', '')[:35]}...")
                    st.rerun()
                else:
                    st.error("No results found")
            except Exception as e:
                st.error(f"Search failed: {str(e)[:50]}")
    
    st.caption("Or click directly on the map")
    st.divider()
    
    c1, c2 = st.columns(2)
    if c1.button("🗑️ Reset", use_container_width=True):
        clear_data()
        st.rerun()
        
    if c2.button("🚀 Route", type="primary", use_container_width=True):
        if len(st.session_state.locations) < 2:
            st.error("Need 2+ locations")
        else:
            with st.spinner("Calculating..."):
                try:
                    generate_route_logic()
                    st.toast("Route Optimized!", icon="✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    # JSON File Uploader
    st.subheader("📂 Upload Orders")
    uploaded_file = st.file_uploader("Upload JSON", type=["json"])
    
    if uploaded_file is not None:
        if st.button("Process File", use_container_width=True):
            try:
                import json
                data = json.load(uploaded_file)
                
                if not isinstance(data, list):
                    st.error("JSON must be a list of objects")
                else:
                    added_count = 0
                    with st.status("Processing addresses...") as status:
                        for item in data:
                            addr = item.get("location")
                            if addr:
                                try:
                                    # Geocoding logic
                                    r = requests.get(
                                        "https://nominatim.openstreetmap.org/search",
                                        params={"q": addr + " Bangalore India", "format": "json", "limit": 1},
                                        headers={"User-Agent": "routing-demo-1.0"},
                                        timeout=5
                                    )
                                    if r.ok and r.json():
                                        res = r.json()[0]
                                        pt = [float(res["lat"]), float(res["lon"])]
                                        st.session_state.locations.append(pt)
                                        status.write(f"✅ Found: {addr}")
                                        added_count += 1
                                        time.sleep(1.1) # Respect Nominatim rate limits (1 per sec)
                                    else:
                                        status.write(f"❌ Not found: {addr}")
                                except Exception as e:
                                    status.write(f"⚠️ Error {addr}: {e}")
                        
                        if added_count > 0:
                            status.update(label="Optimization starting...", state="running")
                            generate_route_logic()
                            status.update(label="Complete!", state="complete")
                            st.toast(f"Imported {added_count} locations!", icon="🚀")
                            time.sleep(1)
                            st.rerun()
                        else:
                            status.update(label="No valid locations found", state="error")
                            
            except Exception as e:
                st.error(f"File error: {e}")

    st.divider()
    
    if st.session_state.metrics:
        st.subheader("📊 Route Metrics")
        m = st.session_state.metrics
        
        st.metric("⏱️ ETA", m["eta"])
        
        policy_color = m.get("policy_color", "white")
        st.markdown(
            f'<div style="padding:10px; border-radius:6px; '
            f'background:#1a1a2e; border-left: 4px solid {policy_color};">' 
            f'<small>🤖 Active Policy</small><br>'
            f'<b>{m["policy"]}</b></div>',
            unsafe_allow_html=True
        )
        st.metric("📍 Distance", m["distance"])
        st.metric("Avg Speed", m["avg_speed"])
        st.metric("🛑 Stops", m["total_stops"])
        st.metric("🛡️ Safety", m["safety"])
        
        st.divider()
        
        st.markdown("**🧠 Stability Trace**")
        var = m.get("traffic_variance", 0)
        
        if var < 0.02:
            stab_label = "LOW ✅"
            stab_color = "#2ECC71"
            decision = "→ Heuristic (efficient)"
        elif var < 0.08:
            stab_label = "MEDIUM ⚠️"
            stab_color = "#F39C12"
            decision = "→ Adaptive Mode"
        else:
            stab_label = "HIGH 🔴"
            stab_color = "#E74C3C"
            decision = "→ DQN (adaptive)"
        
        st.markdown(
            f'<div style="background:#0d1117; padding:14px; '
            f'border-radius:8px; font-family:monospace; '
            f'font-size:15px; line-height:1.8;">'
            f'<span style="color:#888">Variance:</span> '
            f'<b style="color:white">{var:.4f}</b><br>'
            f'<span style="color:#888">Stability:</span> '
            f'<b style="color:{stab_color}; font-size:17px">'
            f'{stab_label}</b><br>'
            f'<span style="color:#888">Decision:</span> '
            f'<b style="color:white">{decision}</b>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if "traffic_breakdown" in m:
            st.divider()
            st.markdown("**🚦 Traffic Breakdown**")
            breakdown = m["traffic_breakdown"]
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("🟢", f"{breakdown['low']}%", help="Smooth segments")
            col_b.metric("🟡", f"{breakdown['medium']}%", help="Moderate segments") 
            col_c.metric("🔴", f"{breakdown['heavy']}%", help="Heavy segments")

m = folium.Map(location=st.session_state.locations[0], zoom_start=12)

if st.session_state.route_data:
    route = st.session_state.route_data["route"]
    traffic = st.session_state.route_data["traffic"]
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        loc_u = st.session_state.locations[u]
        loc_v = st.session_state.locations[v]
        
        segment = st.session_state.api.get_route_geometry([tuple(loc_u), tuple(loc_v)])
        
        mult = traffic[u, v]
        if mult <= 1.2:
            color = '#2ECC71'  # Green - smooth
        elif mult <= 1.5:
            color = '#F39C12'  # Orange - moderate
        else:
            color = '#E74C3C'  # Red - heavy
        
        folium.PolyLine(locations=segment, color=color, weight=6, opacity=0.85).add_to(m)
    
    folium.Marker(st.session_state.locations[route[0]], popup="Depot", icon=folium.Icon(color="green", icon="home")).add_to(m)
    
    for idx in range(1, len(route)):
        folium.Marker(st.session_state.locations[route[idx]], popup=f"Stop {idx}", icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
    
    all_locs = [st.session_state.locations[i] for i in route]
    lats = [loc[0] for loc in all_locs]
    lons = [loc[1] for loc in all_locs]
    lat_pad = max((max(lats) - min(lats)) * 0.4, 0.02)
    lon_pad = max((max(lons) - min(lons)) * 0.4, 0.02)
    m.fit_bounds([
        [min(lats) - lat_pad, min(lons) - lon_pad],
        [max(lats) + lat_pad, max(lons) + lon_pad]
    ])
else:
    for i, loc in enumerate(st.session_state.locations):
        icon_color = "green" if i == 0 else "lightgray"
        icon_name = "home" if i == 0 else "info-sign"
        folium.Marker(loc, popup=f"Point {i}", icon=folium.Icon(color=icon_color, icon=icon_name)).add_to(m)

st_data = st_folium(m, width=None, height=600)

if st_data and st_data.get("last_clicked"):
    clicked = st_data["last_clicked"]
    new_loc = [clicked["lat"], clicked["lng"]]
    if new_loc not in st.session_state.locations:
        st.session_state.locations.append(new_loc)
        st.rerun()
