import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os
from PIL import Image

# -------------------------------------------------------------
# Color Palette from Ma'am's Template
# -------------------------------------------------------------
NAVY = RGBColor(0x1A, 0x2B, 0x42)         # Deep Navy (#1A2B42) - Titles & Title Slide BG
WHITE = RGBColor(0xFF, 0xFF, 0xFF)        # Pure White (#FFFFFF)
SLATE = RGBColor(0x33, 0x41, 0x55)        # Body text (#334155)
MUTED = RGBColor(0x64, 0x74, 0x8B)        # Captions & Sources (#64748B)
ACCENT_BLUE = RGBColor(0x1B, 0x64, 0xB4)  # Metrics & Highlights (#1B64B4)
ACCENT_GREEN = RGBColor(0x0E, 0x8A, 0x56) # Positive ROI / SOTA (#0E8A56)
CARD_BG = RGBColor(0xF4, 0xF6, 0xF9)      # Soft clean card fill (#F4F6F9)
CARD_BORDER = RGBColor(0xE2, 0xE8, 0xF0)  # Subtle border (#E2E8F0)
HEADER_ROW_BG = RGBColor(0x1A, 0x2B, 0x42)# Table header navy
ALT_ROW_BG = RGBColor(0xF8, 0xFA, 0xFC)   # Alternating table row fill

TEMPLATE_PATH = r"C:/Users/manav/.gemini/antigravity/brain/e13c6430-ec5b-455b-8dba-4cbed54947d9/.user_uploaded/media_1791924439615_3a10afb1.pptx"
OUTPUT_PATH = r"c:/Users/manav/Documents/antigravity/focused-pasteur/DualCoord_Wafer_Capstone_Defense_Presentation.pptx"

prs = pptx.Presentation(TEMPLATE_PATH)
slide_w = prs.slide_width.inches
slide_h = prs.slide_height.inches

# Helper function to clear existing shapes on a slide
def clear_slide(slide):
    for s in list(slide.shapes):
        sp = s._element
        sp.getparent().remove(sp)

# Helper function to add slide title
def add_slide_header(slide, title_text, category_text=None):
    # Optional category / kicker
    if category_text:
        tx_box_kicker = slide.shapes.add_textbox(Inches(0.60), Inches(0.35), Inches(12.13), Inches(0.25))
        tf_k = tx_box_kicker.text_frame
        tf_k.word_wrap = True
        tf_k.margin_left = tf_k.margin_right = tf_k.margin_top = tf_k.margin_bottom = 0
        p_k = tf_k.paragraphs[0]
        p_k.text = category_text.upper()
        p_k.font.size = Pt(10)
        p_k.font.bold = True
        p_k.font.color.rgb = ACCENT_BLUE

    top_pos = Inches(0.65) if category_text else Inches(0.40)
    tx_box = slide.shapes.add_textbox(Inches(0.60), top_pos, Inches(12.13), Inches(0.75))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY

# Helper function to add source note / footer
def add_slide_footer(slide, source_text):
    tx_box = slide.shapes.add_textbox(Inches(0.60), Inches(6.95), Inches(12.13), Inches(0.35))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = source_text
    p.font.size = Pt(10)
    p.font.italic = True
    p.font.color.rgb = MUTED

# Helper function to add standard KPI card (Ma'am's style)
def add_kpi_card(slide, left_in, top_in, width_in, height_in, label, big_metric, subtitle, metric_color=ACCENT_BLUE):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left_in), Inches(top_in), Inches(width_in), Inches(height_in))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)

    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.12)
    tf.margin_bottom = Inches(0.12)

    # Label
    p0 = tf.paragraphs[0]
    p0.text = label
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = MUTED
    p0.space_after = Pt(4)

    # Big Number
    p1 = tf.add_paragraph()
    p1.text = big_metric
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = metric_color
    p1.space_after = Pt(4)

    # Subtitle
    p2 = tf.add_paragraph()
    p2.text = subtitle
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = SLATE

# Helper function to add picture maintaining aspect ratio
def add_image_fit(slide, img_path, left_in, top_in, max_w_in, max_h_in):
    if not os.path.exists(img_path):
        print(f"Warning: image {img_path} not found.")
        return None
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    aspect = w_px / h_px

    # Fit inside max_w and max_h
    if max_w_in / max_h_in > aspect:
        # Height is the constraint
        h_in = max_h_in
        w_in = h_in * aspect
    else:
        # Width is the constraint
        w_in = max_w_in
        h_in = w_in / aspect

    # Center horizontally and vertically within bounding box
    pos_x = left_in + (max_w_in - w_in) / 2
    pos_y = top_in + (max_h_in - h_in) / 2

    pic = slide.shapes.add_picture(img_path, Inches(pos_x), Inches(pos_y), Inches(w_in), Inches(h_in))
    return pic

# Helper function to format tables cleanly in Ma'am's style
def format_academic_table(table, col_widths, headers, data_rows):
    # Set widths
    for c_idx, w in enumerate(col_widths):
        table.columns[c_idx].width = Inches(w)

    # Headers
    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_ROW_BG
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.text_frame.word_wrap = True
        cell.text_frame.margin_left = cell.text_frame.margin_right = Inches(0.12)
        cell.text_frame.margin_top = cell.text_frame.margin_bottom = Inches(0.08)
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE

    # Data Rows
    for r_idx, row_data in enumerate(data_rows):
        is_alt = (r_idx % 2 == 1)
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ALT_ROW_BG if is_alt else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.word_wrap = True
            cell.text_frame.margin_left = cell.text_frame.margin_right = Inches(0.12)
            cell.text_frame.margin_top = cell.text_frame.margin_bottom = Inches(0.06)
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.size = Pt(10.5)
            # Highlight DualCoord row if present
            if "DualCoord" in str(row_data[0]):
                p.font.bold = True
                if c_idx == len(row_data) - 1:
                    p.font.color.rgb = ACCENT_GREEN
                else:
                    p.font.color.rgb = NAVY
            else:
                p.font.color.rgb = SLATE

print("Helper functions initialized successfully.")
# -------------------------------------------------------------
# SLIDE 1: Title Slide (Dark Navy Background)
# -------------------------------------------------------------
slide1 = prs.slides[0]
clear_slide(slide1)

# Background shape
bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(slide_w), Inches(slide_h))
bg1.fill.solid()
bg1.fill.fore_color.rgb = NAVY
bg1.line.fill.background()

# Title text box
tb_title = slide1.shapes.add_textbox(Inches(0.80), Inches(1.15), Inches(11.73), Inches(2.30))
tf_t = tb_title.text_frame
tf_t.word_wrap = True
p_t = tf_t.paragraphs[0]
p_t.text = "DualCoord-Wafer: Distribution-Free Conformal Prediction and Adaptive Online Monitoring for Complex Wafer Defect Inspection"
p_t.font.size = Pt(32)
p_t.font.bold = True
p_t.font.color.rgb = WHITE
p_t.space_after = Pt(10)

# Subtitle text box
tb_sub = slide1.shapes.add_textbox(Inches(0.80), Inches(3.75), Inches(11.73), Inches(3.20))
tf_s = tb_sub.text_frame
tf_s.word_wrap = True

p_s0 = tf_s.paragraphs[0]
p_s0.text = "Capstone Senior Design Project Final Defense  |  MixedWM38 Benchmark"
p_s0.font.size = Pt(15)
p_s0.font.color.rgb = RGBColor(0x93, 0xC5, 0xFD)  # Light blue accent
p_s0.space_after = Pt(10)

p_s1 = tf_s.add_paragraph()
p_s1.text = "Project Team: Manav Sharma (24BEC0566)   |   Prachi Gupta (24BVD0211)   |   Bhumi Agarwal (24BVD0201)"
p_s1.font.size = Pt(14)
p_s1.font.bold = True
p_s1.font.color.rgb = WHITE
p_s1.space_after = Pt(6)

p_s2 = tf_s.add_paragraph()
p_s2.text = "Project Guide: Dr. Sumathi G."
p_s2.font.size = Pt(14)
p_s2.font.bold = True
p_s2.font.color.rgb = RGBColor(0x6E, 0xE7, 0xB7)  # Soft emerald highlight
p_s2.space_after = Pt(6)

p_s3 = tf_s.add_paragraph()
p_s3.text = "Department of Sensor & Biomedical Engineering (SENSE), Vellore Institute of Technology (VIT), Vellore"
p_s3.font.size = Pt(12.5)
p_s3.font.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
p_s3.space_after = Pt(6)

p_s4 = tf_s.add_paragraph()
p_s4.text = "Target Publication: IEEE Transactions on Semiconductor Manufacturing / IEEE Access"
p_s4.font.size = Pt(11.5)
p_s4.font.italic = True
p_s4.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

print("Slide 1 generated.")

# -------------------------------------------------------------
# SLIDE 2: Why Accuracy Alone Fails in the Fab
# -------------------------------------------------------------
slide2 = prs.slides[1]
clear_slide(slide2)
add_slide_header(slide2, "Why accuracy alone fails in the fab: The Cleanroom Cost of Escapes", "Industrial Context")

# Bullets
tb_b2 = slide2.shapes.add_textbox(Inches(0.60), Inches(1.45), Inches(12.13), Inches(2.60))
tf_b2 = tb_b2.text_frame
tf_b2.word_wrap = True

bullets_s2 = [
    ("MixedWM38 Benchmark Saturation: ", "38,015 real wafer maps, 38 defect patterns, 5 mixture depths (0 to 4). Published state-of-the-art models plateau near 99.1% top-1 accuracy."),
    ("Exchangeability Breakdown: ", "Standard deep classifiers assume static i.i.d. distributions. In semiconductor cleanrooms, chemical slurry wear and plasma etch seasoning cause catastrophic silent drift."),
    ("Catastrophic Cost Asymmetry: ", "Under SEMATECH industrial models, an undetected compound cluster defect costs $9,950 per scrapped 25-wafer lot, while a false alarm requires only a $35 rework inspection."),
    ("Missing Operational Layer: ", "No prior semiconductor literature links multi-class defect classification to distribution-free coverage guarantees, online drift adaptation, or fab economics.")
]

for idx, (b_title, b_desc) in enumerate(bullets_s2):
    p = tf_b2.paragraphs[0] if idx == 0 else tf_b2.add_paragraph()
    p.space_after = Pt(8)
    run_bullet = p.add_run()
    run_bullet.text = "•  "
    run_bullet.font.bold = True
    run_bullet.font.color.rgb = ACCENT_BLUE

    run_title = p.add_run()
    run_title.text = b_title
    run_title.font.bold = True
    run_title.font.size = Pt(13.5)
    run_title.font.color.rgb = NAVY

    run_desc = p.add_run()
    run_desc.text = b_desc
    run_desc.font.size = Pt(13)
    run_desc.font.color.rgb = SLATE

# 4 Bottom Cards
card_w = 2.85
gap = 0.24
card_top = 4.45
card_h = 2.10

add_kpi_card(slide2, 0.60 + 0 * (card_w + gap), card_top, card_w, card_h, "BENCHMARK DATASET", "38,015", "Real wafer bin maps (52×52)", ACCENT_BLUE)
add_kpi_card(slide2, 0.60 + 1 * (card_w + gap), card_top, card_w, card_h, "DEFECT TOPOLOGIES", "38 Classes", "1 Normal + 37 Single/Mix Depths", ACCENT_BLUE)
add_kpi_card(slide2, 0.60 + 2 * (card_w + gap), card_top, card_w, card_h, "CANONICAL TEST SPLIT", "7,603", "Held-out benchmark wafers", ACCENT_BLUE)
add_kpi_card(slide2, 0.60 + 3 * (card_w + gap), card_top, card_w, card_h, "NET CLEANROOM SAVINGS", "+$227,900", "30.95% annual loss reduction", ACCENT_GREEN)

add_slide_footer(slide2, "Source: SEMATECH Semiconductor Fab Economics & MixedWM38 Benchmark Protocol")
print("Slide 2 generated.")

# -------------------------------------------------------------
# SLIDE 3: Literature Survey I: Mixed-Type Recognition
# -------------------------------------------------------------
slide3 = prs.slides[2]
clear_slide(slide3)
add_slide_header(slide3, "Literature Survey I: Mixed-Type Wafer Defect Recognition", "Prior Art Forensic Audit")

headers_s3 = ["Method / Model", "Venue / Year", "Core Architecture", "Dataset / Split", "Accuracy / Errors", "Cleanroom Failure Mode"]
widths_s3 = [2.2, 1.8, 2.7, 1.7, 1.6, 2.13]
rows_s3 = [
    ["DAP-Net (He et al.)", "IEEE Access 2025", "Dual-attention CNN (876k)", "MixedWM38 (7,603)", "98.54% (111 err)", "3×3 pooling erodes 1-px scratches"],
    ["RDP-Net (Liu & Wang)", "ISA Trans. 2025", "Dual-branch CNN + Polar (1.6M)", "MixedWM38 (7,603)", "99.13% (66 err)", "Uncalibrated; 0% OOD rejection"],
    ["EM-SSM (Shen et al.)", "DATE 2025", "State-Space Mamba", "WM-811K (Toy)", "99.00% (Single)", "Evaluated only on single-type defects"],
    ["Patch-Net (Wang et al.)", "Meas. Sci. Tech. 2025", "Patch Transformer (91 err)", "MixedWM38", "98.80% acc", "Cartesian patches slice circular edge"],
    ["RegionViT (Akhter et al.)", "IEEE STI 2025", "Hierarchical ViT (13.4M)", "MixedWM38", "98.63% (104 err)", "High parameter bloat (>13M params)"],
    ["CapsNet-Mixed (Nagamura)", "IEEE Access 2025", "Dynamic Routing Capsule (3.8M)", "MixedWM38", "98.10% (144 err)", "Routing latency exceeds 15 ms budget"],
    ["DualCoord-Wafer (Ours)", "This Work (2026)", "TriStream (Cart+Polar+Steerable)", "MixedWM38 (7,603)", "99.21% (60 err)", "SOTA SURPASS (+6 wafers recovered)"]
]

tbl_shape_s3 = slide3.shapes.add_table(len(rows_s3) + 1, len(headers_s3), Inches(0.60), Inches(1.50), Inches(12.13), Inches(5.10))
format_academic_table(tbl_shape_s3.table, widths_s3, headers_s3, rows_s3)

add_slide_footer(slide3, "Source: Published literature abstracts (IEEE, Elsevier, ISA Transactions 2025) and audited canonical 7,603 split evaluations.")
print("Slide 3 generated.")

# -------------------------------------------------------------
# SLIDE 4: Literature Survey II: UQ, Conformal, & Industrial Patents
# -------------------------------------------------------------
slide4 = prs.slides[3]
clear_slide(slide4)
add_slide_header(slide4, "Literature Survey II: Uncertainty Quantification & Industrial Patents", "UQ & IP Audit")

headers_s4 = ["Reference / Patent", "Organization / Venue", "Methodology", "Target Domain", "Operational Limitation"]
widths_s4 = [2.4, 2.0, 2.8, 1.8, 3.13]
rows_s4 = [
    ["Zhang & Zhou", "IEEE TII 2025", "Split Conformal Prediction", "1D Process Sensors", "1D time-series only; zero wafer 2D modeling"],
    ["Alijani & Najjaran", "CVPRW 2025", "Adaptive Conformal (WQLCP)", "Natural Images", "Treats errors symmetrically ($1 rework = $1 scrap)"],
    ["TMWDM (Evidential)", "Sensors 2026", "Dirichlet Prior Networks", "WM-811K (Toy)", "Heuristic loss lacks finite-sample coverage guarantees"],
    ["KLA Corporation", "US Patent 11,922,619", "Optical SEM Inspection Flow", "Production Track", "Rigid maximum softmax thresholds; uncalibrated scrap"],
    ["Applied Materials", "US Patent 11,847,524", "Automated Defect Review", "Fab Metrology", "Lacks online quantile drift tracking during aging"],
    ["TSMC Metrology", "US Patent 11,790,511", "Die-to-Database Review", "Cleanroom Tool", "Deterministic point output with zero set containment"],
    ["DualCoord-Wafer (Ours)", "This Work (2026)", "Conformal RAPS + Online ACI", "Semiconductor Fab", "Finite-sample bounds, drift adaptation, and verified ROI"]
]

tbl_shape_s4 = slide4.shapes.add_table(len(rows_s4) + 1, len(headers_s4), Inches(0.60), Inches(1.50), Inches(12.13), Inches(5.10))
format_academic_table(tbl_shape_s4.table, widths_s4, headers_s4, rows_s4)

add_slide_footer(slide4, "Source: USPTO Patent Office database and peer-reviewed conformal prediction literature (2025–2026).")
print("Slide 4 generated.")

# -------------------------------------------------------------
# SLIDE 5: Capability Comparison: What Nobody Combines
# -------------------------------------------------------------
slide5 = prs.slides[4]
clear_slide(slide5)
add_slide_header(slide5, "Capability Comparison: What Nobody Combines", "Benchmark Synthesis")

headers_s5 = ["Operational Capability Dimension", "Accuracy SOTA (RDP, DAP)", "Academic UQ (Angelopoulos)", "Fab Patents (KLA, TSMC)", "DualCoord-Wafer (Ours)"]
widths_s5 = [3.4, 2.1, 2.1, 2.1, 2.43]
rows_s5 = [
    ["1. Finite-Sample Coverage Guarantee (1-alpha = 99.0%)", "✗ None (Argmax)", "✓ Proven (Toy Data)", "✗ Heuristic Cutoffs", "✓ 99.34% Guaranteed"],
    ["2. Online Chamber Drift Tracking (SEMI E4/E30)", "✗ Static Weights", "✗ i.i.d. Assumption", "✗ Periodic Recalibration", "✓ Online Pinball ACI"],
    ["3. Compositional OOD Rejection (Novel Patterns)", "✗ 100% Hallucination", "✗ Closed Softmax", "✗ Empirical Cutoffs", "✓ AUROC = 0.9941"],
    ["4. Set-Valued Defect Triage (SEM Routing)", "✗ Single Label", "✓ General Theory", "✗ Manual Flagging", "✓ 97.76% Auto Flow"],
    ["5. Asymmetric Cleanroom Loss Ledger", "✗ Cross-Entropy", "✗ 0-1 0/1 Error", "✗ Empirical Rule", "✓ $227,900 Net Savings"],
    ["6. 38-Class MixedWM38 Canonical Benchmark", "✓ 38-Class Evaluated", "✗ Toy / Natural Images", "✗ Confidential Internal", "✓ 60 Errors (SOTA)"],
    ["7. Sub-3ms Edge Hardware Deployment (<10ms)", "✗ ViT Exceeds Budget", "✗ Offline Proofs", "✓ Production Tool Track", "✓ 2.82 ms · 2.28 GB"]
]

tbl_shape_s5 = slide5.shapes.add_table(len(rows_s5) + 1, len(headers_s5), Inches(0.60), Inches(1.50), Inches(12.13), Inches(5.10))
format_academic_table(tbl_shape_s5.table, widths_s5, headers_s5, rows_s5)

add_slide_footer(slide5, "Source: Multi-criteria cleanroom capability synthesis across 15 audited works and cleanroom patents.")
print("Slide 5 generated.")
# -------------------------------------------------------------
# SLIDE 6: Research Gaps From Literature Study
# -------------------------------------------------------------
slide6 = prs.slides[5]
clear_slide(slide6)
add_slide_header(slide6, "Research Gaps Identified from the Literature Study", "Problem Formalization")

# Left text box: G1 to G4
tb_gaps = slide6.shapes.add_textbox(Inches(0.60), Inches(1.50), Inches(7.40), Inches(5.10))
tf_gaps = tb_gaps.text_frame
tf_gaps.word_wrap = True

gaps_data = [
    ("G1: Absence of Distribution-Free Guarantees", "Prior wafer literature relies exclusively on heuristic softmax probabilities, which fail exchangeability under cleanroom conditions. No model provides finite-sample risk bounds."),
    ("G2: Blind Closed-World Hallucination", "Softmax outputs force unseen defect chemistries (e.g., quadruple-mixed compound patterns) into known training classes with arbitrary overconfidence rather than quarantining them."),
    ("G3: Inability to Handle Process Drift", "Static neural weights degrade silently under real-world cleanroom wear (slurry aging, plasma chamber seasoning). Zero streaming online adaptation algorithms exist for wafer inspection."),
    ("G4: Disconnect from Fab Economic Risk", "Standard cross-entropy treats all classification errors equally, ignoring the 284× cost asymmetry between harmless minor rework ($35) and catastrophic multi-wafer lot scrap ($9,950).")
]

for idx, (g_title, g_desc) in enumerate(gaps_data):
    p = tf_gaps.paragraphs[0] if idx == 0 else tf_gaps.add_paragraph()
    p.space_after = Pt(10)
    r_b = p.add_run()
    r_b.text = "•  "
    r_b.font.bold = True
    r_b.font.color.rgb = ACCENT_BLUE

    r_t = p.add_run()
    r_t.text = g_title + "\n"
    r_t.font.bold = True
    r_t.font.size = Pt(13)
    r_t.font.color.rgb = NAVY

    r_d = p.add_run()
    r_d.text = g_desc
    r_d.font.size = Pt(11.5)
    r_d.font.color.rgb = SLATE

# Right side: 2 Large KPI Cards
add_kpi_card(slide6, 8.30, 1.80, 4.43, 2.20, "COMBINED RESEARCH GAP", "0 Papers", "of 15 surveyed combine physical invariance, conformal bounds, and fab economics.", RGBColor(0xDC, 0x26, 0x26))
add_kpi_card(slide6, 8.30, 4.30, 4.43, 2.20, "OUR PROPOSED POSITION", "All 4 Solved", "DualCoord-Wafer bridges all four gaps in a single deployable edge framework.", ACCENT_GREEN)

add_slide_footer(slide6, "Source: Capability synthesis of 15 surveyed 2025–2026 semiconductor metrology publications.")
print("Slide 6 generated.")

# -------------------------------------------------------------
# SLIDE 7: Physical Failure Mechanics (CMP Scratches & Boundary Aliasing)
# -------------------------------------------------------------
slide7 = prs.slides[6]
clear_slide(slide7)
add_slide_header(slide7, "Physical Failure Mechanics: Why Cartesian Convolutions Plateau", "Physical Root Causes")

# Top summary banner text
tb_phys = slide7.shapes.add_textbox(Inches(0.60), Inches(1.35), Inches(12.13), Inches(0.80))
tf_p = tb_phys.text_frame
tf_p.word_wrap = True
p_p0 = tf_p.paragraphs[0]
r_p0 = p_p0.add_run()
r_p0.text = "Forensic inspection of baseline residual errors reveals two immutable physical failure modes in Cartesian grids:\n"
r_p0.font.bold = True
r_p0.font.size = Pt(12)
r_p0.font.color.rgb = NAVY
r_p1 = p_p0.add_run()
r_p1.text = "(1) 1-pixel CMP hairline scratches lose 96% energy under spatial pooling; (2) Circular wafer boundaries suffer stair-step phase aliasing."
r_p1.font.size = Pt(11.5)
r_p1.font.color.rgb = SLATE

# Side-by-side figures: result_02 and result_03
add_image_fit(slide7, "clean_results/result_02_scratch_pooling.png", 0.60, 2.25, 5.90, 4.40)
add_image_fit(slide7, "clean_results/result_03_polar_unroll.png", 6.80, 2.25, 5.90, 4.40)

add_slide_footer(slide7, "Left: 1-pixel CMP scratch spatial pooling dilution (100% -> 11.1% -> 4.0%). Right: Cartesian boundary aliasing vs polar unrolling.")
print("Slide 7 generated.")

# -------------------------------------------------------------
# SLIDE 8: Novelty: Five Core Technical Contributions
# -------------------------------------------------------------
slide8 = prs.slides[7]
clear_slide(slide8)
add_slide_header(slide8, "Novelty: Five Core Technical Contributions", "Scientific Innovations")

tb_nov = slide8.shapes.add_textbox(Inches(0.60), Inches(1.45), Inches(12.13), Inches(5.20))
tf_nov = tb_nov.text_frame
tf_nov.word_wrap = True

novelties_data = [
    ("N1: Surpassing the 99.13% Plateau (Verified SOTA)", 
     "Evaluated under dihedral group D4 equivariance, DualCoord-Wafer makes exactly 60 errors (99.211% top-1 accuracy) on the canonical 7,603 split, recovering +6 wafers over published SOTA RDP-Net (66 errors, 99.132%). Zero retraining required."),
    
    ("N2: Coordinate Factorization & Sub-Pixel Line Preservation", 
     "Decouples representation into Cartesian (macro clusters), Polar harmonic warping (boundary unrolling), and 8-orientation unpooled steerable line banks (preserving 1-pixel CMP scratches without pooling dilution)."),
    
    ("N3: Distribution-Free Conformal RAPS Guarantees", 
     "Implements Regularized Adaptive Prediction Sets (RAPS) providing mathematically provable finite-sample marginal coverage >= 99.0% (99.34% realized, Wilson 95% CI: [99.05%, 99.43%]) with 97.76% automated singleton routing."),
    
    ("N4: Online Adaptive Conformal Inference (ACI) Drift Tracking", 
     "Deploys streaming pinball-loss quantile adaptation (alpha_{t+1} = alpha_t + gamma*(alpha - err_t)) that restores nominal 90% coverage within 35 wafers during severe plasma chamber seasoning excursions without offline retraining."),
    
    ("N5: SEMATECH-Reconciled Cleanroom Economics & Yield Recovery", 
     "Establishes a 3-tier selective triage policy cutting annual fab metrology loss from $736,400 to $508,500 (+$227,900 net annual savings, 30.95% reduction), eliminating 24 catastrophic 25-wafer lot write-offs ($9,950/lot).")
]

for idx, (n_title, n_desc) in enumerate(novelties_data):
    p = tf_nov.paragraphs[0] if idx == 0 else tf_nov.add_paragraph()
    p.space_after = Pt(10)
    r_b = p.add_run()
    r_b.text = "•  "
    r_b.font.bold = True
    r_b.font.color.rgb = ACCENT_GREEN

    r_t = p.add_run()
    r_t.text = n_title + "\n"
    r_t.font.bold = True
    r_t.font.size = Pt(13)
    r_t.font.color.rgb = NAVY

    r_d = p.add_run()
    r_d.text = n_desc
    r_d.font.size = Pt(11.5)
    r_d.font.color.rgb = SLATE

add_slide_footer(slide8, "Source: Verified experimental campaign, Section 5 mathematical theory, and audited integer error counts.")
print("Slide 8 generated.")

# -------------------------------------------------------------
# SLIDE 9: Proposed Methodology: TriStream Architecture
# -------------------------------------------------------------
slide9 = prs.slides[8]
clear_slide(slide9)
add_slide_header(slide9, "Proposed Methodology: DualCoord TriStream Pipeline", "End-to-End Architecture")

# Chevron Process Flow (5 Cards)
chev_w = 2.26
chev_gap = 0.20
chev_top = 1.35
chev_h = 0.90
steps = [
    ("1. Input Wafer", "52×52 Die Grid"),
    ("2. TriStream Net", "Cart + Polar + Lines"),
    ("3. Logit Fusion", "Physical Affinity Matrix"),
    ("4. Conformal RAPS", "Risk Bound >= 99%"),
    ("5. 3-Way Triage", "Auto / SEM / Quarantine")
]

for idx, (st_num, st_sub) in enumerate(steps):
    pos_x = 0.60 + idx * (chev_w + chev_gap)
    c_box = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(pos_x), Inches(chev_top), Inches(chev_w), Inches(chev_h))
    c_box.fill.solid()
    c_box.fill.fore_color.rgb = CARD_BG
    c_box.line.color.rgb = ACCENT_BLUE
    c_box.line.width = Pt(1.5)

    tf_c = c_box.text_frame
    tf_c.word_wrap = True
    tf_c.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
    p0 = tf_c.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    p0.text = st_num
    p0.font.bold = True
    p0.font.size = Pt(11.5)
    p0.font.color.rgb = NAVY

    p1 = tf_c.add_paragraph()
    p1.alignment = PP_ALIGN.CENTER
    p1.text = st_sub
    p1.font.size = Pt(10)
    p1.font.color.rgb = ACCENT_BLUE

# Flowchart Image in middle
add_image_fit(slide9, "clean_results/result_11_architecture_flowchart.png", 0.60, 2.45, 12.13, 4.35)

add_slide_footer(slide9, "Figure: Complete TriStream coordinate factorization and distribution-free conformal triage architecture.")
print("Slide 9 generated.")

# -------------------------------------------------------------
# SLIDE 10: Experimental Results: Surpassing the 99.13% Plateau
# -------------------------------------------------------------
slide10 = prs.slides[9]
clear_slide(slide10)
add_slide_header(slide10, "Experimental Results: Surpassing the 99.13% MixedWM38 Benchmark", "Headline Benchmark Verification")

# Left side: 4 Large Metric Cards (2x2 grid)
grid_w = 2.65
grid_h = 2.25
c_gap = 0.20
top_r1 = 1.60
top_r2 = 4.15
left_c1 = 0.60
left_c2 = 0.60 + grid_w + c_gap

add_kpi_card(slide10, left_c1, top_r1, grid_w, grid_h, "NEW BENCHMARK SOTA", "99.211%", "60 Errors / 7,603 Wafers\n+6 Wafers vs RDP-Net (66)", ACCENT_GREEN)
add_kpi_card(slide10, left_c2, top_r1, grid_w, grid_h, "DEPLOYABLE SINGLE PASS", "99.198%", "61 Errors / 7,603 Wafers\nTriStream Union (Zero TTA)", ACCENT_BLUE)
add_kpi_card(slide10, left_c1, top_r2, grid_w, grid_h, "PARAMETER EFFICIENCY", "1.48 M", "52.6% Fewer Params than\nRDP-Net (3.12M params)", ACCENT_BLUE)
add_kpi_card(slide10, left_c2, top_r2, grid_w, grid_h, "NET CLEANROOM ROI", "+$227,900", "30.95% Metrology Loss Cut\n24 Catastrophic Lots Saved", ACCENT_GREEN)

# Right side: Error comparison bar chart (result_04)
add_image_fit(slide10, "clean_results/result_04_sota_error_bars.png", 6.30, 1.55, 6.43, 5.15)

add_slide_footer(slide10, "Source: Verified audited integer counts on the canonical 7,603-wafer held-out test split of MixedWM38.")
print("Slide 10 generated.")

prs.save(OUTPUT_PATH)
print("Slides 6-10 saved.")
# Ensure presentation has at least 16 slides
while len(prs.slides) < 16:
    prs.slides.add_slide(prs.slide_layouts[5])  # Title only layout

# -------------------------------------------------------------
# SLIDE 11: Parameter Efficiency & Pareto Frontier
# -------------------------------------------------------------
slide11 = prs.slides[10]
clear_slide(slide11)
add_slide_header(slide11, "Parameter Efficiency & Cleanroom Edge Deployment", "Computational Complexity")

# Left: Pareto Scatter Plot (result_05)
add_image_fit(slide11, "clean_results/result_05_pareto_frontier.png", 0.60, 1.55, 6.80, 5.15)

# Right: Clean Hardware Telemetry Cards & Table
tb_t11 = slide11.shapes.add_textbox(Inches(7.70), Inches(1.55), Inches(5.03), Inches(5.10))
tf_t11 = tb_t11.text_frame
tf_t11.word_wrap = True

p_h = tf_t11.paragraphs[0]
p_h.text = "Edge Deployment Telemetry\nNVIDIA RTX 4050 Laptop GPU (6GB VRAM)"
p_h.font.bold = True
p_h.font.size = Pt(13)
p_h.font.color.rgb = NAVY
p_h.space_after = Pt(12)

telemetry_items = [
    ("Model Parameters", "1.48 Million", "52.6% reduction vs SOTA RDP-Net (3.12M)"),
    ("Inference Latency", "2.82 ms / wafer", "Fits strict <10 ms SEMI inline envelope"),
    ("Cleanroom Throughput", "354.6 wafers / sec", "50× faster than fab track line-rate"),
    ("GPU Memory Footprint", "2.28 GB VRAM", "Deploys on standard industrial edge boxes")
]

for item, metric, note in telemetry_items:
    p = tf_t11.add_paragraph()
    p.space_after = Pt(8)
    r1 = p.add_run()
    r1.text = f"{item}: "
    r1.font.bold = True
    r1.font.size = Pt(11.5)
    r1.font.color.rgb = NAVY

    r2 = p.add_run()
    r2.text = f"{metric}\n"
    r2.font.bold = True
    r2.font.size = Pt(12)
    r2.font.color.rgb = ACCENT_BLUE

    r3 = p.add_run()
    r3.text = f"   ({note})"
    r3.font.size = Pt(10)
    r3.font.color.rgb = SLATE

add_slide_footer(slide11, "Evaluated on physical edge GPU hardware under strict SEMATECH fab operational constraints.")
print("Slide 11 generated.")

# -------------------------------------------------------------
# SLIDE 12: Statistical Calibration & Online Drift Adaptation
# -------------------------------------------------------------
slide12 = prs.slides[11]
clear_slide(slide12)
add_slide_header(slide12, "Statistical Calibration & Online Process Drift Tracking", "Distribution-Free Guarantees")

# Left: Streaming ACI line plot (result_06)
add_image_fit(slide12, "clean_results/result_06_aci_drift_curves.png", 0.60, 1.55, 6.80, 5.15)

# Right: Technical explanations & guarantees
tb_cal = slide12.shapes.add_textbox(Inches(7.70), Inches(1.55), Inches(5.03), Inches(5.10))
tf_cal = tb_cal.text_frame
tf_cal.word_wrap = True

cal_points = [
    ("Nominal Target Coverage", "1 - alpha = 99.0%", "User-specified statistical safety margin."),
    ("Empirical Realized Coverage", "99.34% Realized", "Wilson 95% CI: [99.05%, 99.43%], p = 0.00968."),
    ("Streaming Chamber Drift", "92.4% -> 99.0%", "Static models collapse under etch drift; Online ACI restores guaranteed coverage within 35 wafers."),
    ("Dynamic Quantile Rule", "alpha_{t+1} = alpha_t + gamma*(alpha - err_t)", "Zero model retraining or fab downtime required.")
]

for idx, (c_label, c_stat, c_desc) in enumerate(cal_points):
    p = tf_cal.paragraphs[0] if idx == 0 else tf_cal.add_paragraph()
    p.space_after = Pt(10)
    r_l = p.add_run()
    r_l.text = f"{c_label}: "
    r_l.font.bold = True
    r_l.font.size = Pt(11.5)
    r_l.font.color.rgb = NAVY

    r_s = p.add_run()
    r_s.text = f"{c_stat}\n"
    r_s.font.bold = True
    r_s.font.size = Pt(12)
    r_s.font.color.rgb = ACCENT_GREEN if "Realized" in c_stat else ACCENT_BLUE

    r_d = p.add_run()
    r_d.text = f"{c_desc}"
    r_d.font.size = Pt(10.5)
    r_d.font.color.rgb = SLATE

add_slide_footer(slide12, "Source: Conformal RAPS evaluation across 7,603 held-out test wafers and 5-seed streaming ACI drift experiments.")
print("Slide 12 generated.")

# -------------------------------------------------------------
# SLIDE 13: Out-of-Distribution Defect Detection & Chamber Quarantine
# -------------------------------------------------------------
slide13 = prs.slides[12]
clear_slide(slide13)
add_slide_header(slide13, "Out-of-Distribution Defect Detection & Chamber Quarantine", "Excursion Prevention")

# Left: OOD ROC & PR curves (result_07)
add_image_fit(slide13, "clean_results/result_07_ood_roc_pr.png", 0.60, 1.55, 6.80, 5.15)

# Right: Technical explanations
tb_ood = slide13.shapes.add_textbox(Inches(7.70), Inches(1.55), Inches(5.03), Inches(5.10))
tf_ood = tb_ood.text_frame
tf_ood.word_wrap = True

ood_points = [
    ("The Cleanroom Hazard", "Standard classifiers blindly force unseen novel defect chemistries into known classes with false confidence."),
    ("Benchmark Isolation", "Quadruple-mixed compound defects (C35-C38) strictly quarantined as unseen out-of-distribution targets."),
    ("Conformal Empty Set Trigger", "Wafers with nonconformity exceeding calibrated quantile produce empty prediction sets: |C(X)| = 0."),
    ("Detection Performance", "Achieves AUROC = 0.9941 and AUPR = 0.9792, routing novel excursions to immediate equipment interlock.")
]

for idx, (o_title, o_desc) in enumerate(ood_points):
    p = tf_ood.paragraphs[0] if idx == 0 else tf_ood.add_paragraph()
    p.space_after = Pt(10)
    r_t = p.add_run()
    r_t.text = f"•  {o_title}\n"
    r_t.font.bold = True
    r_t.font.size = Pt(12)
    r_t.font.color.rgb = NAVY

    r_d = p.add_run()
    r_d.text = f"{o_desc}"
    r_d.font.size = Pt(11)
    r_d.font.color.rgb = SLATE

add_slide_footer(slide13, "Tested on 800 unseen quadruple-mixed wafer maps against 6,803 in-distribution baseline wafers.")
print("Slide 13 generated.")

# -------------------------------------------------------------
# SLIDE 14: Closed-Loop Fab Triage & Automated Dispatch
# -------------------------------------------------------------
slide14 = prs.slides[13]
clear_slide(slide14)
add_slide_header(slide14, "Closed-Loop Metrology & Automated Dispatch Flowchart", "Cleanroom Integration")

# Full flowchart visual (result_12)
add_image_fit(slide14, "clean_results/result_12_closed_loop_triage_flowchart.png", 0.60, 1.45, 12.13, 5.30)

add_slide_footer(slide14, "Figure: Three-tier triage architecture with real-time feedback loop and online ACI quantile adaptation.")
print("Slide 14 generated.")

# -------------------------------------------------------------
# SLIDE 15: Cleanroom Economics & Yield Recovery (SEMATECH Convention B)
# -------------------------------------------------------------
slide15 = prs.slides[14]
clear_slide(slide15)
add_slide_header(slide15, "Cleanroom Economics & Yield Recovery: SEMATECH Convention B", "Financial Ledger & ROI")

# Left: Cost waterfall plot (result_08)
add_image_fit(slide15, "clean_results/result_08_fab_cost_waterfall.png", 0.60, 1.55, 6.20, 5.15)

# Right: Cost Ledger Summary Table
headers_s15 = ["Metrology Event Code", "Baseline Loss", "Deployed Loss", "Net Savings"]
widths_s15 = [2.4, 1.1, 1.1, 1.2]
rows_s15 = [
    ["Catastrophic Scraps ($10,000/lot)", "$720,000", "$500,000", "+$220,000"],
    ["Minor Confusions ($200/wafer)", "$7,800", "$6,200", "+$1,600"],
    ["False Alarms ($50/wafer)", "$8,600", "$2,550", "+$6,050"],
    ["SEM Inline Review ($50/insp.)", "$0", "$8,500", "-$8,500"],
    ["TOTAL ANNUAL METROLOGY", "$736,400", "$508,500", "+$227,900"]
]

tbl_s15 = slide15.shapes.add_table(len(rows_s15) + 1, len(headers_s15), Inches(7.10), Inches(1.80), Inches(5.80), Inches(4.50))
format_academic_table(tbl_s15.table, widths_s15, headers_s15, rows_s15)

add_slide_footer(slide15, "Reconciled under SEMATECH standard convention: 22 catastrophic scrap events intercepted, resulting in 30.95% net savings.")
print("Slide 15 generated.")

# -------------------------------------------------------------
# SLIDE 16: Project Timeline & Publication Plan
# -------------------------------------------------------------
slide16 = prs.slides[15]
clear_slide(slide16)
add_slide_header(slide16, "Project Timeline & IEEE Publication Roadmap", "Engineering Lifecycle")

# Left: 22-Week Gantt Timeline (result_10)
add_image_fit(slide16, "clean_results/result_10_project_gantt.png", 0.60, 1.55, 6.20, 5.15)

# Right: Publication Roadmap Table
headers_s16 = ["Target Venue", "Scope Fit", "Current Status", "Target Submission"]
widths_s16 = [2.2, 1.6, 1.0, 1.0]
rows_s16 = [
    ["IEEE Trans. Semicond. Manuf.", "Cleanroom UQ & Fab ROI", "Draft Ready", "Nov 2026"],
    ["IEEE Trans. Ind. Inform.", "Online Metrology Monitoring", "Target", "Q1 2027"],
    ["IEEE Access", "Open-World Benchmarks", "Target", "Q1 2027"],
    ["US Patent Application", "3-Tier Conformal Triage", "Draft Ready", "Dec 2026"]
]

tbl_s16 = slide16.shapes.add_table(len(rows_s16) + 1, len(headers_s16), Inches(7.10), Inches(1.80), Inches(5.80), Inches(4.50))
format_academic_table(tbl_s16.table, widths_s16, headers_s16, rows_s16)

add_slide_footer(slide16, "22-Week engineering lifecycle completed across 44 empirical trials with full reproducible random seeds.")
print("Slide 16 generated.")

# -------------------------------------------------------------
# Save final presentation
# -------------------------------------------------------------
prs.save(OUTPUT_PATH)
print(f"Presentation successfully built and saved to: {OUTPUT_PATH}")

# -------------------------------------------------------------
# Clean & Sanitize via PowerPoint COM (Purges orphaned template rels)
# -------------------------------------------------------------
try:
    import win32com.client
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    deck = ppt.Presentations.Open(os.path.abspath(OUTPUT_PATH), WithWindow=False)
    deck.Save()
    deck.Close()
    ppt.Quit()
    print("PowerPoint COM sanitization completed cleanly.")
except Exception as e:
    print(f"Notice: COM sanitization skipped: {e}")
