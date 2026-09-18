#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined IMD GFS 850 hPa animations (3 products) + Indian radiosonde Skew‑T plots.

GFS products:
  - national
  - maharashtra
  - regional

Sounding plots:
  - SkewT_XXXXX.png for each configured station

Designed to run in CI (GitHub Actions) and commit outputs back to the repo.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urlencode
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units


# ======================================================================
# CONFIGURATION – GFS ANIMATIONS (3 PRODUCTS)
# ======================================================================

FORECAST_HOURS = [
    24, 48, 72, 96, 120,
    144, 168, 192, 216, 240
]

PRODUCTS = {
    "national": {
        "url_template": (
            "https://nwp.imd.gov.in/gfs/00/"
            "{hour}hGFS1534w850rf.gif"
        ),
        "output_directory": "national",
        "animation_file": "imd_gfs_national_850hPa.gif",
        "label": "IMD GFS National"
    },
    "maharashtra": {
        "url_template": (
            "https://nwp.imd.gov.in/gfs_ar/00/"
            "{hour}GFS1534w850rf_maharashtra_state.gif"
        ),
        "output_directory": "maharashtra",
        "animation_file": "imd_gfs_maharashtra_850hPa.gif",
        "label": "IMD GFS Maharashtra"
    },
    "regional": {
        "url_template": (
            "https://nwp.imd.gov.in/gfs_ar/00/"
            "{hour}hGFS1534w850rf.gif"
        ),
        "output_directory": "regional",
        "animation_file": "imd_gfs_regional_850hPa.gif",
        "label": "IMD GFS Regional"
    }
}

BASE_OUTPUT_DIRECTORY_GFS = Path("imd_gfs_downloads")

FRAME_DURATION_MS = 1000
LOOP = 0
REQUEST_TIMEOUT = 60
RETRIES = 3

ADD_LABEL = True


# ======================================================================
# CONFIGURATION – SKEW‑T SOUNDINGS
# ======================================================================

STATIONS = {
    "43003": "Mumbai",
    "43110": "Ratnagiri",
    "43063": "Pune",
    "43014": "Chhatrapati Sambhajinagar",
    "42867": "Nagpur",
}

SAVE_DIR_SOUNDING = Path("sounding_plots")


# ======================================================================
# GFS DOWNLOAD FUNCTIONS
# ======================================================================


def download_image_gfs(url: str, output_path: Path) -> Image.Image:
    """
    Download and decode one GIF image for GFS products.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; IMD-GFS-downloader/1.0; "
            "+https://nwp.imd.gov.in/)"
        )
    }

    last_error = None

    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            if not response.content:
                raise RuntimeError("Empty response received.")

            image = Image.open(BytesIO(response.content))
            image.load()

            output_path.write_bytes(response.content)

            print(
                f"Downloaded: {url} "
                f"({image.width} x {image.height})"
            )

            return image.convert("RGBA")

        except Exception as error:
            last_error = error

            print(
                f"Attempt {attempt}/{RETRIES} failed:\n"
                f"  {url}\n"
                f"  Error: {error}"
            )

            if attempt < RETRIES:
                time.sleep(2)

    raise RuntimeError(
        f"Could not download image after {RETRIES} attempts:\n"
        f"{url}\nLast error: {last_error}"
    )


def add_label(
    image: Image.Image,
    product_label: str,
    forecast_hour: int
) -> Image.Image:
    """
    Add a product and forecast-hour label to the image.
    """
    if not ADD_LABEL:
        return image

    frame = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(frame)

    text = f"{product_label} | Forecast +{forecast_hour} h"

    try:
        bbox = draw.textbbox((0, 0), text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text)

    padding = 8

    rectangle = (
        padding,
        padding,
        text_width + 3 * padding,
        text_height + 3 * padding
    )

    draw.rectangle(rectangle, fill="white")
    draw.text(
        (2 * padding, 2 * padding),
        text,
        fill="black"
    )

    return frame


def normalize_frame_sizes(frames):
    """
    Make all frames in an animation the same size.
    """
    max_width = max(frame.width for frame in frames)
    max_height = max(frame.height for frame in frames)

    normalized_frames = []

    for frame in frames:
        canvas = Image.new(
            "RGBA",
            (max_width, max_height),
            "white"
        )

        x = (max_width - frame.width) // 2
        y = (max_height - frame.height) // 2

        canvas.alpha_composite(frame, (x, y))
        normalized_frames.append(canvas)

    return normalized_frames


def create_animation(frames, output_file: Path):
    """
    Save a list of PIL images as an animated GIF.
    """
    if not frames:
        raise RuntimeError(
            f"No frames available for {output_file.name}"
        )

    frames = normalize_frame_sizes(frames)

    gif_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE)
        for frame in frames
    ]

    gif_frames[0].save(
        output_file,
        save_all=True,
        append_images=gif_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=LOOP,
        optimize=False,
        disposal=2
    )


def process_product(product_name: str, product_config: dict):
    """
    Download all frames for one GFS product and create its animation.
    """
    product_directory = (
        BASE_OUTPUT_DIRECTORY_GFS /
        product_config["output_directory"]
    )

    product_directory.mkdir(parents=True, exist_ok=True)

    frames = []
    successful_hours = []

    print()
    print("=" * 70)
    print(f"Processing: {product_config['label']}")
    print("=" * 70)

    for forecast_hour in FORECAST_HOURS:
        url = product_config["url_template"].format(
            hour=forecast_hour
        )

        image_file = (
            product_directory /
            f"{forecast_hour:03d}h.gif"
        )

        try:
            image = download_image_gfs(url, image_file)

            image = add_label(
                image=image,
                product_label=product_config["label"],
                forecast_hour=forecast_hour
            )

            frames.append(image)
            successful_hours.append(forecast_hour)

        except Exception as error:
            print(
                f"Skipping {product_name} +{forecast_hour} h:\n"
                f"{error}"
            )

    if not frames:
        print(
            f"No valid frames downloaded for {product_name}. "
            "Animation was not created."
        )
        return

    animation_file = (
        BASE_OUTPUT_DIRECTORY_GFS /
        product_config["animation_file"]
    )

    create_animation(frames, animation_file)

    print()
    print(f"Created: {animation_file}")
    print(f"Frames: {successful_hours}")


def run_gfs_animations():
    BASE_OUTPUT_DIRECTORY_GFS.mkdir(parents=True, exist_ok=True)

    for product_name, product_config in PRODUCTS.items():
        process_product(product_name, product_config)

    print()
    print("All GFS product downloads and animations are complete.")


# ======================================================================
# SKEW‑T SOUNDING FUNCTIONS
# ======================================================================


def build_url_sounding(dt: datetime, station_number: int, src: str = "BUFR", out_type: str = "TEXT:CSV") -> str:
    params = {
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "id": str(station_number),
        "type": out_type,
        "src": src,
    }

    return (
        "https://weather.uwyo.edu/wsgi/sounding?"
        + urlencode(params)
    )


def try_download_sounding(dt: datetime, station_number: int, src: str):
    url = build_url_sounding(dt, station_number, src=src)

    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        return None, url, f"Request error: {exc}"

    if not response.ok:
        return None, url, f"HTTP {response.status_code}"

    text = response.text.strip()

    if not text:
        return None, url, "Empty response"

    if "pressure_hPa" not in text:
        return None, url, text[:300]

    try:
        df = pd.read_csv(
            StringIO(text),
            skipinitialspace=True
        )
    except Exception as exc:
        return None, url, f"CSV parsing error: {exc}"

    if df.empty:
        return None, url, "Parsed empty CSV"

    return df, url, None


def fetch_with_fallbacks_sounding(station_number: int, base_date: date):
    candidate_times = [
        datetime(base_date.year, base_date.month, base_date.day, 0),
        datetime(base_date.year, base_date.month, base_date.day, 12),
        datetime(base_date.year, base_date.month, base_date.day, 0)
        - timedelta(days=1),
        datetime(base_date.year, base_date.month, base_date.day, 12)
        - timedelta(days=1),
    ]

    expanded_times = []

    for dt in candidate_times:
        expanded_times.append(dt.replace(hour=0))
        expanded_times.append(dt.replace(hour=12))

    seen = set()
    ordered_times = []

    for dt in expanded_times:
        key = dt.strftime("%Y-%m-%d %H")

        if key not in seen:
            seen.add(key)
            ordered_times.append(dt)

    for dt in ordered_times:
        for src in ["BUFR", "FM35"]:
            df, url, error = try_download_sounding(
                dt,
                station_number,
                src
            )

            status = error if error else "OK"

            print(
                f"{station_number}: "
                f"Trying {dt:%Y-%m-%d %H:%M} UTC "
                f"src={src} -> {status}"
            )

            if df is not None:
                return df, dt, src, url

    raise RuntimeError(
        f"No valid sounding found for station {station_number}."
    )


def format_quantity(value, unit, decimals=0):
    """
    Safely format a MetPy quantity. Returns '--' for missing values.
    """
    if value is None:
        return "--"

    try:
        converted = value.to(unit).magnitude
        return f"{converted:.{decimals}f} {unit}"
    except Exception:
        return "--"


def calculate_sounding_parameters(df: pd.DataFrame):
    required_cols = [
        "pressure_hPa",
        "geopotential height_m",
        "temperature_C",
        "dew point temperature_C",
        "wind direction_degree",
        "wind speed_m/s",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    df = df.dropna(subset=required_cols).copy()
    df = df[df["pressure_hPa"] > 0]
    df = df.drop_duplicates(subset="pressure_hPa")
    df = df.sort_values("pressure_hPa", ascending=False).reset_index(drop=True)

    if len(df) < 3:
        raise ValueError(
            "Not enough valid levels for sounding calculations."
        )

    p = df["pressure_hPa"].to_numpy() * units.hPa
    temperature = df["temperature_C"].to_numpy() * units.degC
    dewpoint = df["dew point temperature_C"].to_numpy() * units.degC
    wind_direction = df["wind direction_degree"].to_numpy() * units.degrees
    wind_speed = df["wind speed_m/s"].to_numpy() * units("m/s")

    u, v = mpcalc.wind_components(wind_speed, wind_direction)

    parcel_profile = mpcalc.parcel_profile(
        p,
        temperature[0],
        dewpoint[0]
    ).to("degC")

    lcl_p, lcl_t = mpcalc.lcl(p[0], temperature[0], dewpoint[0])

    try:
        lfc_p, lfc_t = mpcalc.lfc(p, temperature, dewpoint)
    except Exception:
        lfc_p, lfc_t = None, None

    try:
        el_p, el_t = mpcalc.el(p, temperature, dewpoint, parcel_profile)
    except Exception:
        el_p, el_t = None, None

    try:
        ccl_p, ccl_t, t_c = mpcalc.ccl(p, temperature, dewpoint)
        if (
            np.isnan(ccl_p.magnitude)
            or np.isnan(ccl_t.magnitude)
            or np.isnan(t_c.magnitude)
        ):
            ccl_p, ccl_t, t_c = None, None, None
    except Exception:
        ccl_p, ccl_t, t_c = None, None, None

    try:
        cape, cin = mpcalc.cape_cin(p, temperature, dewpoint, parcel_profile)
    except Exception:
        cape, cin = np.nan * units("J/kg"), np.nan * units("J/kg")

    try:
        u_shear, v_shear = mpcalc.bulk_shear(
            p,
            u,
            v,
            bottom=900 * units.hPa,
            depth=500 * units.hPa
        )
        bulk_shear = mpcalc.wind_speed(u_shear, v_shear).to("knot")
    except Exception:
        bulk_shear = np.nan * units.knot

    return {
        "df": df,
        "p": p,
        "temperature": temperature,
        "dewpoint": dewpoint,
        "u": u,
        "v": v,
        "parcel_profile": parcel_profile,
        "lcl_p": lcl_p,
        "lcl_t": lcl_t,
        "lfc_p": lfc_p,
        "lfc_t": lfc_t,
        "el_p": el_p,
        "el_t": el_t,
        "ccl_p": ccl_p,
        "ccl_t": ccl_t,
        "t_c": t_c,
        "cape": cape,
        "cin": cin,
        "bulk_shear": bulk_shear,
    }


def plot_sounding(
    station_number: str,
    station_name: str,
    used_dt: datetime,
    used_src: str,
    source_url: str,
    sounding: dict,
    save_dir: Path
):
    p = sounding["p"]
    temperature = sounding["temperature"]
    dewpoint = sounding["dewpoint"]
    parcel_profile = sounding["parcel_profile"]
    u = sounding["u"]
    v = sounding["v"]

    lcl_p = sounding["lcl_p"]
    lcl_t = sounding["lcl_t"]
    lfc_p = sounding["lfc_p"]
    lfc_t = sounding["lfc_t"]
    el_p = sounding["el_p"]
    el_t = sounding["el_t"]
    ccl_p = sounding["ccl_p"]
    ccl_t = sounding["ccl_t"]
    t_c = sounding["t_c"]
    cape = sounding["cape"]
    cin = sounding["cin"]
    bulk_shear = sounding["bulk_shear"]

    fig = plt.figure(figsize=(10, 10))
    skew = SkewT(fig, rotation=30)

    skew.plot(p, temperature, "r", linewidth=2, label="Temperature")
    skew.plot(p, dewpoint, "g", linewidth=2, label="Dewpoint")
    skew.plot(p, parcel_profile, "k", linewidth=2, label="Parcel")

    skew.plot(lcl_p, lcl_t, "ko", markersize=7, label="LCL")

    if lfc_p is not None:
        skew.plot(
            lfc_p, lfc_t, "bo",
            markerfacecolor="blue", markersize=7, label="LFC"
        )

    if el_p is not None:
        skew.plot(
            el_p, el_t, "mo",
            markerfacecolor="magenta", markersize=7, label="EL"
        )

    if ccl_p is not None:
        skew.plot(
            ccl_p, ccl_t, marker="D",
            color="orange", markersize=7, label="CCL"
        )

    step = max(1, len(p) // 60)
    skew.plot_barbs(
        p[::step],
        u[::step].to("knot"),
        v[::step].to("knot")
    )

    try:
        skew.shade_cape(p, temperature, parcel_profile)
        skew.shade_cin(p, temperature, parcel_profile, dewpoint)
    except Exception as exc:
        print(f"{station_name}: shading skipped: {exc}")

    skew.ax.set_ylim(1000, 80)
    skew.ax.set_xlim(-50, 50)
    skew.ax.axvline(0, color="c", linestyle="--", linewidth=1)

    skew.plot_dry_adiabats(alpha=0.5)
    skew.plot_moist_adiabats(alpha=0.5)
    skew.plot_mixing_lines(alpha=0.5)

    cape_text = format_quantity(cape, "J/kg", decimals=0)
    cin_text = format_quantity(cin, "J/kg", decimals=0)
    lcl_text = (
        f"{lcl_p.to('hPa').magnitude:.0f} hPa / "
        f"{lcl_t.to('degC').magnitude:.1f} °C"
    )
    lfc_text = (
        f"{lfc_p.to('hPa').magnitude:.0f} hPa / "
        f"{lfc_t.to('degC').magnitude:.1f} °C"
        if lfc_p is not None else "Not found"
    )
    el_text = (
        f"{el_p.to('hPa').magnitude:.0f} hPa / "
        f"{el_t.to('degC').magnitude:.1f} °C"
        if el_p is not None else "Not found"
    )
    ccl_text = (
        f"{ccl_p.to('hPa').magnitude:.0f} hPa / "
        f"{ccl_t.to('degC').magnitude:.1f} °C / "
        f"Tconv {t_c.to('degC').magnitude:.1f} °C"
        if ccl_p is not None else "Not found"
    )
    shear_text = format_quantity(bulk_shear, "knot", decimals=1)

    parameter_text = (
        f"CAPE: {cape_text}\n"
        f"CIN: {cin_text}\n"
        f"LCL: {lcl_text}\n"
        f"LFC: {lfc_text}\n"
        f"EL: {el_text}\n"
        f"CCL: {ccl_text}\n"
        f"0–500 hPa bulk shear: {shear_text}"
    )

    skew.ax.text(
        0.98, 0.98, parameter_text,
        transform=skew.ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=9,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black",
            alpha=0.85
        ),
        zorder=10
    )

    skew.ax.legend(loc="lower left", fontsize=9)

    plt.title(
        f"{station_name} ({station_number})\n"
        f"Skew-T: {used_dt:%Y-%m-%d %H:%M UTC} ({used_src})",
        fontsize=13
    )

    plt.tight_layout()

    outfile = save_dir / f"SkewT_{station_number}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved: {outfile}")
    print(f"Source URL: {source_url}")

    plt.close(fig)


def run_sounding_plots():
    SAVE_DIR_SOUNDING.mkdir(parents=True, exist_ok=True)

    base_date = date.today()

    for station_number, station_name in STATIONS.items():
        print("\n" + "=" * 70)
        print(f"Processing {station_name} ({station_number})")
        print("=" * 70)

        try:
            df, used_dt, used_src, source_url = fetch_with_fallbacks_sounding(
                int(station_number),
                base_date
            )

            sounding = calculate_sounding_parameters(df)

            plot_sounding(
                station_number=station_number,
                station_name=station_name,
                used_dt=used_dt,
                used_src=used_src,
                source_url=source_url,
                sounding=sounding,
                save_dir=SAVE_DIR_SOUNDING
            )

            print(f"CAPE: {sounding['cape'].to('J/kg')}")
            print(f"CIN: {sounding['cin'].to('J/kg')}")
            print(f"LCL: {sounding['lcl_p'].to('hPa')}")

            if sounding["lfc_p"] is not None:
                print(f"LFC: {sounding['lfc_p'].to('hPa')}")
            else:
                print("LFC: Not found")

            if sounding["el_p"] is not None:
                print(f"EL: {sounding['el_p'].to('hPa')}")
            else:
                print("EL: Not found")

            if sounding["ccl_p"] is not None:
                print(
                    f"CCL: {sounding['ccl_p'].to('hPa')} / "
                    f"{sounding['ccl_t'].to('degC')} / "
                    f"Convective T: {sounding['t_c'].to('degC')}"
                )
            else:
                print("CCL: Not found")

            print(f"0–500 hPa bulk shear: {sounding['bulk_shear'].to('knot')}")

        except Exception as exc:
            print(f"ERROR for {station_name} ({station_number}): {exc}")


# ======================================================================
# MAIN
# ======================================================================


def main():
    print("=== Running IMD GFS 850 hPa animations (3 products) ===")
    run_gfs_animations()

    print("\n=== Running Skew‑T sounding plots ===")
    run_sounding_plots()

    print("\nAll products generated.")


if __name__ == "__main__":
    main()
