"""PDF report generation and visualization helpers."""

from __future__ import annotations

import datetime
import io
import textwrap
from typing import Dict, Optional

import numpy as np
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .constants import MUMBAI_BOUNDS, PILLAR_DISPLAY
from .models import RegionData

_PDF_MARGIN = 54  # 0.75 inches


def radar_chart_png(scores: Dict[str, float], title: str) -> bytes:
    categories = [PILLAR_DISPLAY[key] for key in ["air", "water", "green", "built"]]
    values = [max(0.0, min(1.0, scores.get(key, 0.0))) for key in ["air", "water", "green", "built"]]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(3.6, 3.6), dpi=150)
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, color="#1d4ed8", linewidth=2)
    ax.fill(angles, values, color="#3b82f6", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.grid(color="#cbd5f5", linestyle="--", linewidth=0.6)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def map_snapshot_png(region: RegionData, scenario_region: RegionData) -> bytes:
    west, south, east, north = MUMBAI_BOUNDS
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=150)
    ax.set_facecolor("#f8fafc")
    ax.add_patch(
        plt.Rectangle(
            (west, south),
            east - west,
            north - south,
            fill=False,
            linewidth=1.2,
            edgecolor="#1f2937",
        )
    )
    ax.scatter(
        [region.lon],
        [region.lat],
        s=60,
        color="#ef4444",
        label="Selected location",
        zorder=5,
    )
    ax.text(
        region.lon,
        region.lat + 0.01,
        f"NDVI → {scenario_region.ndvi:.2f}\nNDBI → {scenario_region.ndbi:.2f}",
        fontsize=8,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", alpha=0.9, edgecolor="#94a3b8"),
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_xticks(np.linspace(west, east, 4))
    ax.set_yticks(np.linspace(south, north, 4))
    ax.tick_params(labelsize=8)
    ax.set_title("Mumbai focus area", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, color="#cbd5f5")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def _render_lines(
    can: canvas.Canvas,
    lines: list[str],
    margin: int,
    page_height: int,
    start_y: int,
    line_height: int,
) -> int:
    y = start_y
    for line in lines:
        if y < margin + line_height:
            can.showPage()
            y = page_height - margin
        can.drawString(margin, y, line)
        y -= line_height
    return y


def generate_pdf_report(
    region: RegionData,
    scores: Dict[str, float],
    recs: Dict[str, str | float],
    scenario_region: RegionData,
    scenario_scores: Dict[str, float],
    scenario_recs: Dict[str, str | float],
    scenario_hci: float,
    ai_text: Optional[str],
    radar_png: Optional[bytes] = None,
    map_png: Optional[bytes] = None,
) -> bytes:
    buffer = io.BytesIO()
    can = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = _PDF_MARGIN
    page_height = int(height)
    y = page_height - margin
    line_height = 14

    can.setFont("Helvetica-Bold", 18)
    can.drawString(margin, y, "Healthy City Scenario Report")
    can.setLineWidth(1)
    can.line(margin, y - 4, width - margin, y - 4)
    y -= 2 * line_height
    can.setFont("Helvetica", 10)
    can.drawString(
        margin,
        y,
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )
    y -= int(1.5 * line_height)

    can.setFont("Helvetica-Bold", 14)
    can.drawString(margin, y, "Section 1 · Current Snapshot")
    y -= line_height
    can.setLineWidth(0.5)
    can.line(margin, y, width - margin, y)
    y -= int(1.2 * line_height)
    can.setFont("Helvetica", 11)
    current_lines = [
        f"Location: {region.lat:.4f}, {region.lon:.4f}",
        f"Healthy City Index: {recs['hci']:.2f}",
        u"• Air quality · PM2.5 {region.air_pm25:.1f} µg/m³ · NO₂ {region.air_no2:.1f} µmol/m² · CO₂ {region.air_co2:.1f} ppm",
        u"• Water quality · SWIR ratio {region.water_pollution:.3f}",
        u"• Green assets · NDVI {region.ndvi:.3f} · SAVI {region.savi:.3f}",
        u"• Built intensity · NDBI {region.ndbi:.3f}",
        u"• Thermal comfort · LST {region.lst_c:.1f} °C",
        u"• Density · {:,.0f} residents/km²".format(region.pop_density),
        u"• Habitability · " + str(recs["habitability"]),
        u"• Parks & shade · " + str(recs["parks"]),
        u"• Waste systems · " + str(recs["waste"]),
        u"• Health resilience · " + str(recs["disease"]),
    ]
    y = _render_lines(can, current_lines, margin, page_height, y, line_height)
    y -= line_height

    can.setFont("Helvetica-Bold", 14)
    can.drawString(margin, y, "Section 2 · AI Strategy & Scenario Insights")
    y -= line_height
    can.setLineWidth(0.5)
    can.line(margin, y, width - margin, y)
    y -= int(1.2 * line_height)
    can.setFont("Helvetica", 11)
    scenario_summary = [
        "Scenario summary:",
        u"• Target NDVI: {:.3f} (Δ {:+.3f})".format(
            scenario_region.ndvi, scenario_region.ndvi - region.ndvi
        ),
        u"• Target NDBI: {:.3f} (Δ {:+.3f})".format(
            scenario_region.ndbi, scenario_region.ndbi - region.ndbi
        ),
        u"• Projected LST: {:.1f} °C (Δ {:+.1f})".format(
            scenario_region.lst_c, scenario_region.lst_c - region.lst_c
        ),
        u"• Projected PM2.5: {:.1f} µg/m³ (Δ {:+.1f})".format(
            scenario_region.air_pm25, scenario_region.air_pm25 - region.air_pm25
        ),
        u"• Projected water ratio: {:.3f} (Δ {:+.3f})".format(
            scenario_region.water_pollution,
            scenario_region.water_pollution - region.water_pollution,
        ),
        u"• Projected HCI: {:.2f} (Δ {:+.2f})".format(
            scenario_hci, scenario_hci - recs["hci"]
        ),
    ]
    y = _render_lines(can, scenario_summary, margin, page_height, y, line_height)
    y -= line_height

    can.setFont("Helvetica", 11)
    ai_lines: list[str] = []
    if ai_text:
        ai_lines.append("AI recommendation:")
        for paragraph in ai_text.splitlines():
            wrapped = textwrap.wrap(paragraph, width=90)
            ai_lines.extend(wrapped or [""])
    else:
        ai_lines.append("AI recommendation unavailable. Generate the report once the assistant is reachable.")
    y = _render_lines(can, ai_lines, margin, page_height, y, line_height)
    y -= line_height

    can.setFont("Helvetica-Bold", 14)
    can.drawString(margin, y, "Appendix · Visuals & Contacts")
    y -= line_height
    can.setLineWidth(0.5)
    can.line(margin, y, width - margin, y)
    y -= int(1.2 * line_height)
    appendix_lines = [
        "• Radar chart: Healthy City pillar scores",
        "• Map overlays: latest heatmap selection and point analysis",
        "• Municipal liaison (placeholder): +91 00000 00000 · resilience@city.gov",
        "• Additional visualizations can be embedded in future iterations.",
    ]
    y = _render_lines(can, appendix_lines, margin, page_height, y, line_height)
    y -= line_height

    if radar_png or map_png:
        if y < margin + 260:
            can.showPage()
            y = page_height - margin
        if radar_png:
            radar_reader = ImageReader(io.BytesIO(radar_png))
            img_w, img_h = radar_reader.getSize()
            max_width = (letter[0] - 2 * margin) / 2 - 12
            scale = min(max_width / img_w, 220 / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale
            can.drawImage(
                radar_reader,
                margin,
                y - draw_h,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
            )
            can.setFont("Helvetica", 9)
            can.drawString(margin, y - draw_h - 12, "Figure · HCI pillar radar chart")
        if map_png:
            map_reader = ImageReader(io.BytesIO(map_png))
            img_w, img_h = map_reader.getSize()
            max_width = (letter[0] - 2 * margin) / 2 - 12
            scale = min(max_width / img_w, 220 / img_h)
            draw_w = img_w * scale
            draw_h = img_h * scale
            x_pos = margin + ((letter[0] - 2 * margin) / 2) + 12
            can.drawImage(
                map_reader,
                x_pos,
                y - draw_h,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
            )
            can.setFont("Helvetica", 9)
            can.drawString(x_pos, y - draw_h - 12, "Figure · Mumbai focus map")
        y -= 240

    can.save()
    buffer.seek(0)
    return buffer.read()


__all__ = [
    "radar_chart_png",
    "map_snapshot_png",
    "generate_pdf_report",
]
