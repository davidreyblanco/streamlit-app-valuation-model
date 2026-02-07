from __future__ import annotations

import base64
import pickle
from pathlib import Path
from typing import Any
import gzip
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors

import src.geo_data as geo_data

MODEL_PATH = Path("models/rf-idealista.pickle.gz")
RASTER_PATH = Path("raster/unitprice_grid_100x100.tif")
ICON_PATH = Path("img/icons8-home-50.png")
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
    with gzip.open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_icon_data_url(icon_path: Path) -> str | None:
    if not icon_path.exists():
        return None
    encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data
def load_osm_pois(city_name: str = "Madrid") -> Any:
    return geo_data.load_osm_data(city_name=city_name, use_geopandas=True)


def build_osm_points(osm_gdf: Any, selected_codes: list[str]) -> list[dict[str, Any]]:
    if not selected_codes:
        return []
    df = osm_gdf[osm_gdf["CODE"].isin(selected_codes)].copy()
    if df.empty:
        return []
    if "geometry" in df.columns:
        df["lat"] = df.geometry.y
        df["lon"] = df.geometry.x
    else:
        df["lat"] = df["LAT"]
        df["lon"] = df["LNG"]
    code_styles = {
        "HEALTH": {"color": "#d62728", "shape": "health"},
        "EDUCATION": {"color": "#1f77b4", "shape": "education"},
        "TRANSPORT": {"color": "#2ca02c", "shape": "transport"},
        "FOOD": {"color": "#ff7f0e", "shape": "food"},
        "SHOP": {"color": "#9467bd", "shape": "shop"},
        "SPORT": {"color": "#17becf", "shape": "sport"},
    }
    default_style = {"color": "#7f7f7f", "shape": "generic"}

    def svg_icon(color: str, shape: str) -> dict[str, Any]:
        if shape == "health":
            glyph = (
                "<rect x='17' y='9' width='6' height='22' rx='2' fill='#ffffff'/>"
                "<rect x='9' y='17' width='22' height='6' rx='2' fill='#ffffff'/>"
            )
        elif shape == "education":
            glyph = (
                "<polygon points='20,9 34,16 20,23 6,16' fill='#ffffff'/>"
                "<rect x='14' y='23' width='12' height='6' rx='2' fill='#ffffff'/>"
            )
        elif shape == "transport":
            glyph = (
                "<rect x='10' y='12' width='20' height='14' rx='3' fill='#ffffff'/>"
                "<circle cx='14' cy='28' r='2.5' fill='#ffffff'/>"
                "<circle cx='26' cy='28' r='2.5' fill='#ffffff'/>"
            )
        elif shape == "food":
            glyph = (
                "<rect x='12' y='9' width='3' height='18' rx='1' fill='#ffffff'/>"
                "<rect x='17' y='9' width='2' height='8' rx='1' fill='#ffffff'/>"
                "<rect x='21' y='9' width='2' height='8' rx='1' fill='#ffffff'/>"
                "<rect x='25' y='9' width='2' height='8' rx='1' fill='#ffffff'/>"
                "<rect x='24' y='17' width='3' height='10' rx='1' fill='#ffffff'/>"
            )
        elif shape == "shop":
            glyph = (
                "<rect x='10' y='14' width='20' height='14' rx='2' fill='#ffffff'/>"
                "<rect x='13' y='10' width='14' height='4' rx='2' fill='#ffffff'/>"
            )
        elif shape == "sport":
            glyph = (
                "<circle cx='20' cy='20' r='8' fill='none' stroke='#ffffff' stroke-width='2'/>"
                "<path d='M20 12 L20 28 M12 20 L28 20' stroke='#ffffff' stroke-width='2'/>"
            )
        else:
            glyph = "<circle cx='20' cy='20' r='6' fill='#ffffff'/>"

        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40' viewBox='0 0 40 40'>"
            f"<circle cx='20' cy='20' r='18' fill='{color}' stroke='#111' stroke-width='2'/>"
            f"{glyph}</svg>"
        )
        encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return {
            "url": f"data:image/svg+xml;base64,{encoded}",
            "width": 52,
            "height": 52,
            "anchorY": 52,
        }

    df["style"] = df["CODE"].apply(lambda c: code_styles.get(c, default_style))
    df["icon"] = df["style"].apply(lambda s: svg_icon(s["color"], s["shape"]))
    return df[["lat", "lon", "NOMBRE", "CODE", "SUBCODE", "icon"]].to_dict(orient="records")



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


def render_prices(container: Any, unit_price: float, total_price: float) -> None:
    col1, col2 = container.columns(2)
    col1.metric("Price/m²", f"{unit_price:,.0f} €/m²")
    col2.metric("Price", f"{total_price:,.0f} €")


def geocode_and_store(address: str) -> str | None:
    point = geocode_address(address.strip() + ', Madrid, Comunidad de Madrid, Spain')
    if point is None:
        return "No coordinates found for that address."
    st.session_state.latitude, st.session_state.longitude = point
    st.session_state.last_geocoded_address = address.strip()
    return None


@st.cache_data
def load_raster_overlay(raster_path: Path) -> dict[str, Any] | None:
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        return None

    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True).astype("float32")
        if np.ma.count(band) == 0:
            return None

        valid_values = band.compressed()
        positive_values = valid_values[valid_values > 0]
        if positive_values.size == 0:
            return None

        vmin, vmax = np.nanpercentile(positive_values, [1, 99.4])
        if vmin <= 0:
            vmin = float(np.nanmin(positive_values))
        if vmin >= vmax:
            vmax = float(np.nanmax(positive_values))
        if vmin >= vmax:
            vmax = vmin * 1.01

        # Emphasize high-end differences by stretching the upper range.
        high_end_gamma = 1.5
        norm = mcolors.PowerNorm(gamma=high_end_gamma, vmin=vmin, vmax=vmax, clip=True)
        cmap = cm.get_cmap("inferno")

        if src.crs is None:
            return None
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        mask = np.ma.getmaskarray(band)

        rows, cols = band.shape
        cells: list[dict[str, Any]] = []
        for row in range(rows):
            for col in range(cols):
                if mask[row, col]:
                    continue
                value = float(band[row, col])
                color_value = max(value, vmin)
                color = (np.array(cmap(norm(color_value))) * 255).astype(np.uint8)

                x_left, y_top = src.transform * (col, row)
                x_right, y_bottom = src.transform * (col + 1, row + 1)

                lon_tl, lat_tl = transformer.transform(x_left, y_top)
                lon_tr, lat_tr = transformer.transform(x_right, y_top)
                lon_br, lat_br = transformer.transform(x_right, y_bottom)
                lon_bl, lat_bl = transformer.transform(x_left, y_bottom)

                cells.append(
                    {
                        "polygon": [
                            [lon_tl, lat_tl],
                            [lon_tr, lat_tr],
                            [lon_br, lat_br],
                            [lon_bl, lat_bl],
                        ],
                        "fill_color": [int(color[0]), int(color[1]), int(color[2]), 130],
                    }
                )

    return {
        "cells": cells,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "gamma": float(high_end_gamma),
        "scale": f"power(gamma={high_end_gamma})",
    }


def render_map(
    latitude: float,
    longitude: float,
    raster_overlay: dict[str, Any] | None,
    osm_points: list[dict[str, Any]] | None,
) -> None:
    layers = []
    if raster_overlay and raster_overlay.get("cells"):
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=raster_overlay["cells"],
                get_polygon="polygon",
                get_fill_color="fill_color",
                stroked=False,
                filled=True,
                pickable=False,
            )
        )

    if osm_points:
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=osm_points,
                get_position="[lon, lat]",
                get_icon="icon",
                get_size=1,
                size_scale=6,
                pickable=True,
            )
        )

    icon_url = load_icon_data_url(ICON_PATH)
    if icon_url:
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=[
                    {
                        "lat": latitude,
                        "lon": longitude,
                        "icon": {
                            "url": icon_url,
                            "width": 10,
                            "height": 10,
                            "anchorY": 25,
                        },
                        "size": 6,
                    }
                ],
                get_position="[lon, lat]",
                get_icon="icon",
                get_size="size",
                size_scale=4,
                pickable=False,
            )
        )
    else:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": latitude, "lon": longitude}],
                get_position="[lon, lat]",
                get_fill_color=[179, 32, 110, 255],
                get_radius=50,
                radius_min_pixels=8,
                radius_max_pixels=16,
                pickable=False,
            )
        )

    tooltip = None
    if osm_points:
        tooltip = {
            "html": "<b>{NOMBRE}</b><br/>CODE: {CODE}<br/>SUBCODE: {SUBCODE}",
            "style": {
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "color": "black",
                "fontSize": "12px",
                "padding": "6px 8px",
            },
        }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=13.8,
            pitch=0,
        ),
        map_provider="carto",
        map_style="light",
        tooltip=tooltip,
    )
    st.pydeck_chart(
        deck,
        use_container_width=True,
        key="location_map",
    )


def render_raster_legend(raster_overlay: dict[str, Any]) -> None:
    vmin = float(1000.0)
    vmax = float(raster_overlay["vmax"])
    gamma = float(raster_overlay.get("gamma", 0.8))

    fig, ax = plt.subplots(figsize=(5.6, 0.55))
    fig.patch.set_alpha(0)
    ax.set_axis_off()

    norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap("inferno"))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=1.0, pad=0.0)
    cbar.set_ticks([vmin, (vmin + vmax) / 2.0, vmax])
    cbar.set_ticklabels([f"{vmin:,.0f} €/m²", f"{(vmin + vmax) / 2.0:,.0f} €/m²", f"{vmax:,.0f} €/m²"])
    cbar.ax.tick_params(labelsize=8, length=0, pad=1)
    #cbar.set_label("Unit price (€/m²)", fontsize=9, labelpad=2)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Idealista House Valuator", layout="wide")
    inject_custom_css()
    st.title("Madrid House Valuation")
    #st.caption("Random Forest valuation using your trained model.")

    if not MODEL_PATH.exists():
        st.error(f"Model not found at `{MODEL_PATH}`. Train and export the model first.")
        st.stop()

    model = load_model(MODEL_PATH)
    raster_overlay = load_raster_overlay(RASTER_PATH) if RASTER_PATH.exists() else None

    if "latitude" not in st.session_state:
        st.session_state.latitude = 40.4168
    if "longitude" not in st.session_state:
        st.session_state.longitude = -3.7038
    if "address_input" not in st.session_state:
        st.session_state.address_input = ""
    if "last_geocoded_address" not in st.session_state:
        st.session_state.last_geocoded_address = ""

    map_col, options_col = st.columns([1, 2], gap="large")

    with map_col:
        #st.subheader("Location map")
        toggle_col1, toggle_col2 = st.columns(2)
        with toggle_col1:
            show_raster = st.toggle("Show unit price raster overlay", value=True)
        with toggle_col2:
            show_pois = st.toggle("Show OSM POIs", value=False)
        osm_points = None
        if show_pois:
            osm_data_gdf = load_osm_pois("Madrid")
            codes = sorted(osm_data_gdf["CODE"].dropna().unique().tolist())
            default_codes = [code for code in ("HEALTH", "EDUCATION", "TRANSPORT") if code in codes]
            selected_codes = st.multiselect("POI codes", options=codes, default=default_codes)
            osm_points = build_osm_points(osm_data_gdf, selected_codes)
        render_map(
            float(st.session_state.latitude),
            float(st.session_state.longitude),
            raster_overlay if show_raster else None,
            osm_points,
        )
        if show_raster and RASTER_PATH.exists() and raster_overlay:
            if False:
                st.caption(
                f"Raster overlay (unit price €/m², balanced high-end scale): ~P1 {raster_overlay['vmin']:.0f} to ~P99.4 {raster_overlay['vmax']:.0f}"
                )
            render_raster_legend(raster_overlay)
        elif show_raster and RASTER_PATH.exists() and not raster_overlay:
            st.caption("Raster found but unavailable (install `rasterio` to enable overlay).")
        price_container = st.container().empty()
        if "pred_unitprice" in st.session_state and "pred_total" in st.session_state:
            render_prices(
                price_container,
                st.session_state.pred_unitprice,
                st.session_state.pred_total,
            )

    with options_col:
       # st.subheader("Property features")
        col1, col2, col3 = st.columns(3)
        with col1:
            roomnumber = st.number_input("Rooms", min_value=1, max_value=10, value=2, step=1)
            constructedarea = st.number_input(
                "Constructed area (m²)", min_value=35.0, max_value=300.0, value=90.0, step=1.0
            )
            floorclean = st.number_input("Floor", min_value=-2.0, max_value=80.0, value=2.0, step=1.0)
            cadconstructionyear = st.number_input(
                "Construction year", min_value=1850, max_value=2035, value=1998, step=1
            )
            flatlocationid = st.number_input(
                "Flat location id", min_value=0, max_value=15, value=2, step=1
            )
        with col2:
            build_type_options = {
                "1 - New / Renewed": 1,
                "2 - 2nd Hand Good Condition": 2,
                "3 - 2nd hand To Renovate": 3,
            }
            cadastral_quality_options = {
                "1 - Excellent (Best quality)": 1,
                "2 - Very high": 2,
                "3 - High": 3,
                "4 - Upper-mid": 4,
                "5 - Mid (Default)": 5,
                "6 - Lower-mid": 6,
                "7 - Low": 7,
                "8 - Very low": 8,
                "9 - Poor": 9,
                "10 - Very poor (Worst quality)": 10,
            }
            selected_cadastral_quality = st.selectbox(
                "Cadastral quality id",
                options=list(cadastral_quality_options.keys()),
                index=4,
            )
            cadastralqualityid = cadastral_quality_options[selected_cadastral_quality]
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
            selected_build_type = st.selectbox(
                "Built type", options=list(build_type_options.keys()), index=0
            )
            buildtypeid = build_type_options[selected_build_type]
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

            render_prices(price_container, pred_unitprice, pred_total)

    st.session_state.latitude = latitude
    st.session_state.longitude = longitude

    #st.subheader("Search address")
    address_col, button_col = st.columns([4, 1], gap="small")

    def geocode_from_input() -> None:
        address_value = st.session_state.address_input.strip()
        if not address_value or address_value == st.session_state.last_geocoded_address:
            return
        try:
            geocode_and_store(address_value)
        except requests.RequestException:
            # Keep UI responsive on blur updates; explicit errors are shown on manual Search.
            return

    with address_col:
        address = st.text_input(
            "Address",
            placeholder="Set the address, example: Calle de Alcalá 50, Madrid",
            label_visibility="collapsed",
            key="address_input",
            on_change=geocode_from_input,
        )
    with button_col:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        if not address.strip():
            st.warning("Please enter an address.")
        else:
            try:
                error = geocode_and_store(address)
                if error:
                    st.warning(error)
                else:
                    st.success(
                        f"Coordinates loaded: lat={st.session_state.latitude:.6f}, lon={st.session_state.longitude:.6f}"
                    )
                    st.rerun()
            except requests.RequestException as exc:
                st.error(f"Geocoding service error: {exc}")

    st.markdown("---")
    st.caption(
        "Reference: Rey-Blanco, D., Arbues, P., et al., and Paez, A. "
        "(2024). *A geo-referenced micro-data set of real estate listings for Spain's three largest cities*. "
        "Environment and Planning B: Urban Analytics and City Science, 51(6). "
        "https://doi.org/10.1177/23998083241242844"
    )


if __name__ == "__main__":
    main()
