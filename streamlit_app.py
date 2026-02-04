from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


MODEL_PATH = Path("models/rf-idealista.pickle")
GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
YELLOW = "#FFFFFF"
MAGENTA = "#b3206e"
GRAY = "#8A8A8A"
WHITE = "#c8d936"
BLACK = "#000000"
DEFAULT_FEATURE_ORDER = [
    "ROOMNUMBER",
    "CONSTRUCTEDAREA",
    "HASLIFT",
    "HASTERRACE",
    "FLOORCLEAN",
    "HASAIRCONDITIONING",
    "HASPARKINGSPACE",
    "CADASTRALQUALITYID",
    "CADCONSTRUCTIONYEAR",
    "FLATLOCATIONID",
    "DISTANCE_TO_METRO",
    "HASSWIMMINGPOOL",
    "BUILTTYPEID_1",
    "BUILTTYPEID_2",
    "BUILTTYPEID_3",
    "LATITUDE",
    "LONGITUDE",
]


def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: {WHITE};
                color: {BLACK};
            }}
            h1, h2, h3 {{
                color: {BLACK} !important;
            }}
            .stCaption {{
                color: {GRAY} !important;
            }}
            .stButton > button {{
                background: {YELLOW};
                color: {BLACK};
                border: 2px solid {BLACK};
                border-radius: 10px;
                font-weight: 700;
            }}
            .stButton > button:hover {{
                background: {MAGENTA};
                color: {WHITE};
                border-color: {BLACK};
            }}
            [data-testid="stMetric"] {{
                border: 2px solid {BLACK};
                border-radius: 12px;
                padding: 0.5rem;
                background: {YELLOW};
            }}
            [data-testid="stMetricLabel"] {{
                color: {BLACK} !important;
                font-weight: 700;
            }}
            [data-testid="stMetricValue"] {{
                color: {MAGENTA} !important;
                font-weight: 800;
                font-size: 1.5rem !important;   /* smaller number */
                line-height: 1.1;
            }}
            .stTextInput > div > div > input,
            .stNumberInput input,
            .stSelectbox div[data-baseweb="select"] > div {{
                border: 1px solid {GRAY} !important;
            }}
            .stAlert {{
                border: 2px solid {BLACK};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(model_path: Path) -> Any:
    with model_path.open("rb") as f:
        return pickle.load(f)


def geocode_address(address: str) -> tuple[float, float] | None:
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": "ml-idealista18-streamlit-app/1.0",
    }
    response = requests.get(GEOCODE_URL, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    if not data:
        return None
    return float(data[0]["lat"]), float(data[0]["lon"])


def build_feature_row(values: dict[str, Any], model: Any) -> pd.DataFrame:
    row = {
        "ROOMNUMBER": int(values["roomnumber"]),
        "CONSTRUCTEDAREA": float(values["constructedarea"]),
        "HASLIFT": int(values["haslift"]),
        "HASTERRACE": int(values["hasterrace"]),
        "FLOORCLEAN": float(values["floorclean"]),
        "HASAIRCONDITIONING": int(values["hasairconditioning"]),
        "HASPARKINGSPACE": int(values["hasparkingspace"]),
        "CADASTRALQUALITYID": int(values["cadastralqualityid"]),
        "CADCONSTRUCTIONYEAR": int(values["cadconstructionyear"]),
        "FLATLOCATIONID": int(values["flatlocationid"]),
        "DISTANCE_TO_METRO": float(values["distance_to_metro"]),
        "HASSWIMMINGPOOL": int(values["hasswimmingpool"]),
        "BUILTTYPEID_1": int(values["buildtypeid"] == 1),
        "BUILTTYPEID_2": int(values["buildtypeid"] == 2),
        "BUILTTYPEID_3": int(values["buildtypeid"] == 3),
        "LATITUDE": float(values["latitude"]),
        "LONGITUDE": float(values["longitude"]),
    }
    feature_order = list(getattr(model, "feature_names_in_", DEFAULT_FEATURE_ORDER))
    row_df = pd.DataFrame([row])
    for col in feature_order:
        if col not in row_df.columns:
            row_df[col] = 0
    return row_df[feature_order]


def main() -> None:
    st.set_page_config(page_title="Idealista House Valuator", layout="wide")
    inject_custom_css()
    st.title("Madrid House Valuation")
    #st.caption("Random Forest valuation using your trained model.")

    if not MODEL_PATH.exists():
        st.error(f"Model not found at `{MODEL_PATH}`. Train and export the model first.")
        st.stop()

    model = load_model(MODEL_PATH)

    if "latitude" not in st.session_state:
        st.session_state.latitude = 40.4168
    if "longitude" not in st.session_state:
        st.session_state.longitude = -3.7038

    map_col, options_col = st.columns([1, 2], gap="large")

    with map_col:
        #st.subheader("Location map")
        map_df = pd.DataFrame(
            {
                "lat": [float(st.session_state.latitude)],
                "lon": [float(st.session_state.longitude)],
            }
        )
        st.map(map_df, zoom=14, use_container_width=True)
        price_container = st.container()
        if "pred_unitprice" in st.session_state and "pred_total" in st.session_state:
            p1, p2 = price_container.columns(2)
            p2.metric("Price/m²", f"{st.session_state.pred_unitprice:,.0f} €/m²")
            p2.metric("Price", f"{st.session_state.pred_total:,.0f} €")

    with options_col:
       # st.subheader("Property features")
        col1, col2, col3 = st.columns(3)
        with col1:
            roomnumber = st.number_input("Rooms", min_value=0, max_value=20, value=2, step=1)
            constructedarea = st.number_input(
                "Constructed area (m²)", min_value=15.0, max_value=1500.0, value=90.0, step=1.0
            )
            floorclean = st.number_input("Floor", min_value=-2.0, max_value=80.0, value=2.0, step=1.0)
            cadconstructionyear = st.number_input(
                "Construction year", min_value=1850, max_value=2035, value=1998, step=1
            )
            flatlocationid = st.number_input(
                "Flat location id", min_value=0, max_value=15, value=2, step=1
            )
        with col2:
            cadastralqualityid = st.number_input(
                "Cadastral quality id", min_value=1, max_value=10, value=5, step=1
            )
            distance_to_metro = st.number_input(
                "Distance to metro (meters)", min_value=0.0, max_value=10000.0, value=400.0, step=10.0
            )
            latitude = st.number_input(
                "Latitude",
                min_value=39.0,
                max_value=42.0,
                value=float(st.session_state.latitude),
                format="%.6f",
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-5.0,
                max_value=-2.0,
                value=float(st.session_state.longitude),
                format="%.6f",
            )
            buildtypeid = st.selectbox("Built type", options=[1, 2, 3], index=0)
        with col3:
            haslift = st.checkbox("Lift", value=True)
            hasterrace = st.checkbox("Terrace", value=False)
            hasairconditioning = st.checkbox("Air conditioning", value=False)
            hasparkingspace = st.checkbox("Parking space", value=False)
            hasswimmingpool = st.checkbox("Swimming pool", value=False)

        if st.button("Valuate property", type="primary", use_container_width=True):
            values = {
                "roomnumber": roomnumber,
                "constructedarea": constructedarea,
                "haslift": haslift,
                "hasterrace": hasterrace,
                "floorclean": floorclean,
                "hasairconditioning": hasairconditioning,
                "hasparkingspace": hasparkingspace,
                "cadastralqualityid": cadastralqualityid,
                "cadconstructionyear": cadconstructionyear,
                "flatlocationid": flatlocationid,
                "distance_to_metro": distance_to_metro,
                "hasswimmingpool": hasswimmingpool,
                "buildtypeid": buildtypeid,
                "latitude": latitude,
                "longitude": longitude,
            }
            features = build_feature_row(values, model)
            pred_unitprice = float(model.predict(features)[0])
            pred_total = pred_unitprice * float(constructedarea)
            st.session_state.pred_unitprice = pred_unitprice
            st.session_state.pred_total = pred_total

            p1, p2 = price_container.columns(2)
            p1.metric("Price/m²", f"{pred_unitprice:,.0f} €/m²")
            p2.metric("Price", f"{pred_total:,.0f} €")

    st.session_state.latitude = latitude
    st.session_state.longitude = longitude

    #st.subheader("Search address")
    address_col, button_col = st.columns([4, 1], gap="small")
    with address_col:
        address = st.text_input("Address", placeholder="Set the address, example: Calle de Alcalá 50, Madrid", label_visibility="collapsed")
    with button_col:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        if not address.strip():
            st.warning("Please enter an address.")
        else:
            try:
                point = geocode_address(address.strip())
                if point is None:
                    st.warning("No coordinates found for that address.")
                else:
                    st.session_state.latitude, st.session_state.longitude = point
                    st.success(
                        f"Coordinates loaded: lat={point[0]:.6f}, lon={point[1]:.6f}"
                    )
                    st.rerun()
            except requests.RequestException as exc:
                st.error(f"Geocoding service error: {exc}")


if __name__ == "__main__":
    main()
