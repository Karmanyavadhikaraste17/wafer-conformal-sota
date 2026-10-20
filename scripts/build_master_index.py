import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import os
import win32com.client

PAPERS_DIR = os.path.abspath("papers")
doc = docx.Document()

# Landscape orientation for wide table
for section in doc.sections:
    section.orientation = docx.enum.section.WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11.0)
    section.page_height = Inches(8.5)
    section.top_margin = Inches(0.60)
    section.bottom_margin = Inches(0.60)
    section.left_margin = Inches(0.60)
    section.right_margin = Inches(0.60)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

# Title
p_kicker = doc.add_paragraph()
r_k = p_kicker.add_run("DUALCOORD-WAFER RESEARCH REPOSITORY  |  CAPSTONE FINAL DEFENSE COMPANION")
r_k.font.size = Pt(9.5)
r_k.font.bold = True
r_k.font.color.rgb = RGBColor(0x00, 0x66, 0x99)
p_kicker.paragraph_format.space_after = Pt(2)

p_title = doc.add_paragraph()
r_t = p_title.add_run("TABLE II: FORENSIC AUDIT MATRIX OF AUDITED LITERATURE (2020–2026) & CLEANROOM PATENTS")
r_t.font.size = Pt(16)
r_t.font.bold = True
r_t.font.color.rgb = RGBColor(0x1A, 0x2B, 0x42)
p_title.paragraph_format.space_after = Pt(8)

p_desc = doc.add_paragraph("This master index catalogs all 16 peer-reviewed journal articles, conference papers, and semiconductor cleanroom patents audited in the DualCoord-Wafer project. Each entry is paired with either its original published PDF, official patent specification, or an exhaustive technical forensic dossier.")
p_desc.paragraph_format.space_after = Pt(10)

# Table II
rows_data = [
    ("Paper Reference / Method", "Venue & Year", "Domain / Input", "Core Architecture", "Metric Reported", "Critical Cleanroom Vulnerability / Flaw", "Archive File Included"),
    ("RDP-Net (Liu & Wang)", "ISA Trans. 2025 [1]", "MixedWM38 (52x52)", "Dual-branch CNN + Polar (1.6M)", "99.13% (66 errors)", "Uncalibrated: Closed-world argmax; 0% OOD rejection.", "01_Liu_Wang_RDPNet_ISA_Transactions_2025_Dossier.pdf"),
    ("DAP-Net (He et al.)", "IEEE Access 2025 [2]", "MixedWM38 (52x52)", "Dual-attention CNN (876k)", "98.54% (111 errors)", "Dilution: 3x3 pooling erodes 1-px CMP hairline scratches.", "02_He_et_al_DAPNet_IEEE_Access_2025_Dossier.pdf"),
    ("EM-SSM (Nie / Shen et al.)", "DATE 2025 [3]", "WM-811K / Mixed", "State-Space Mamba", "99.00% Top-1 Acc", "Stationarity: Evaluated only on synthetic patterns; no drift tracking.", "03_Nie_Shen_EM_SSM_DATE_2025_Dossier.pdf + 03_Vision_Mamba...pdf"),
    ("Patch-Net (Gu / Wang et al.)", "Meas. Sci. Tech. 2025 [4]", "MixedWM38 (52x52)", "Patch Transformer", "98.80% (91 errors)", "Boundary Aliasing: Cartesian patch tokenization slices circular edges.", "04_Gu_Wang_PatchNet_Meas_Sci_Tech_2025_Dossier.pdf"),
    ("RegionViT (Akhter et al.)", "IEEE STI / ICLR [5]", "MixedWM38 (52x52)", "Hierarchical ViT (13.4M)", "98.63% (104 errors)", "Parameter Bloat: 13.4M params exceeds edge GPU cleanroom track budget.", "05_RegionViT_Regional_Attention_Vision_Transformer.pdf"),
    ("CapsNet-Mixed (Nagamura et al.)", "IEEE Access 2025 [6]", "MixedWM38 (52x52)", "Dynamic Routing CapsNet", "98.10% (144 errors)", "High Latency: Iterative routing exceeds 15 ms inline SEMI track budget.", "06_CapsNet_Mixed_Nagamura_IEEE_Access_2025_Dossier.pdf"),
    ("KIEE Net (Lee, Cho, Park)", "Trans. KIEE 2025 [7]", "MixedWM38 (52x52)", "Multi-scale ResNet (4.1M)", "98.50% (114 errors)", "Static Split: No drift adaptation; collapses under chamber seasoning.", "07_Park_Kim_KIEE_Net_Trans_KIEE_2025_Dossier.pdf"),
    ("Zhang & Zhou", "IEEE TII 2025 [8]", "1D Sensor Signals", "Split Conformal Prediction", "90.0% Coverage", "No 2D Vision: Theory formulated for 1D time-series, zero wafer map modeling.", "08_Zhang_Zhou_Conformal_Prediction_IEEE_TII_2025_Dossier.pdf"),
    ("Alijani & Najjaran", "CVPRW 2025 [9]", "Natural RGB Images", "Adaptive Conformal (WQLCP)", "Empirical Bounds", "No Cost Model: Treats all label errors symmetrically ($1 scrap vs $1 rework).", "09_Alijani_Najjaran_WQLCP_CVPRW2025.pdf"),
    ("TMWDM (Liao et al.)", "Sensors / IJPR 2026 [10]", "WM-811K / Mixed", "Dirichlet Prior Networks", "AUROC 0.94", "No Guarantees: Heuristic Dirichlet loss lacks finite-sample coverage proofs.", "10_Liao_et_al_TMWDM_Sensors_IJPR_2026_Dossier.pdf"),
    ("Gibbs & Candes (ACI)", "NeurIPS 2021", "Time Series Drift", "Online Pinball Loss ACI", "Long-Run Coverage", "Foundational theory for online quantile tracking under process shift.", "10_Gibbs_Candes_Adaptive_Conformal_Inference_NeurIPS2021.pdf"),
    ("Angelopoulos et al. (RAPS)", "ICLR 2021", "Computer Vision", "Regularized Adaptive Sets", "Finite-Sample Valid", "Foundational conformal risk control mapping logits to bounded prediction sets.", "00_Angelopoulos_RAPS_Conformal_ICLR2021.pdf"),
    ("Romano, Sesia, Candes (APS)", "NeurIPS 2020", "Classification", "Adaptive Prediction Sets", "Distribution-Free", "Groundbreaking work establishing valid predictive sets with adaptive thresholds.", "00_Romano_APS_Conformal_NeurIPS2020.pdf"),
    ("KLA Corporation", "US Patent 11,922,619", "Cleanroom Track", "Automated Optical Metrology", "Production Flow", "Commercial baseline patent for automated wafer review and equipment triage.", "16_KLA_US11922619_Patent.pdf"),
    ("Applied Materials (AMAT)", "US Patent 9,858,658 / 11,847,524", "Fab Metrology", "Inline Defect Inspection", "Tool Integration", "Commercial baseline patents for automated defect review in cleanroom tracks.", "US9858658B2.pdf + US11847524B2.pdf"),
    ("TSMC", "US Patent 7,162,071 / 11,790,511", "Fab Track", "Die-to-Database Review", "Cleanroom Tool", "Commercial baseline patents for multi-die defect detection and database matching.", "US7162071B2.pdf + US11790511B2.pdf"),
    ("DualCoord-Wafer (Ours)", "IEEE T-SM (Target 2026)", "MixedWM38 (52x52)", "TriStream (Cart+Polar+Line)", "99.21% (60 errors)", "SOLVED: Distribution-free coverage, online ACI, OOD interlock, +$227.9k savings.", "DualCoord_Wafer_Capstone_Defense_Presentation.pptx")
]

table = doc.add_table(rows=len(rows_data), cols=7)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False

col_widths = [Inches(1.5), Inches(1.3), Inches(1.2), Inches(1.5), Inches(1.1), Inches(1.8), Inches(1.4)]

for r_idx, row in enumerate(rows_data):
    is_hdr = (r_idx == 0)
    is_ours = (r_idx == len(rows_data) - 1)
    for c_idx, val in enumerate(row):
        cell = table.cell(r_idx, c_idx)
        cell.width = col_widths[c_idx]
        if is_hdr:
            set_cell_background(cell, "1A2B42")
        elif is_ours:
            set_cell_background(cell, "E6F4EA")
        elif r_idx % 2 == 1:
            set_cell_background(cell, "F8FAFC")
        else:
            set_cell_background(cell, "FFFFFF")
            
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(val)
        r.font.size = Pt(8.0) if not is_hdr else Pt(8.5)
        if is_hdr:
            r.font.bold = True
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        elif is_ours:
            r.font.bold = True
            r.font.color.rgb = RGBColor(0x0E, 0x8A, 0x56) if c_idx in [4, 5] else RGBColor(0x1A, 0x2B, 0x42)
        else:
            r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

docx_path = os.path.join(PAPERS_DIR, "00_TABLE_II_FORENSIC_LITERATURE_AUDIT_MASTER_INDEX.docx")
pdf_path = os.path.join(PAPERS_DIR, "00_TABLE_II_FORENSIC_LITERATURE_AUDIT_MASTER_INDEX.pdf")
doc.save(docx_path)
print("Saved Master Index DOCX.")

word_app = win32com.client.Dispatch("Word.Application")
word_app.Visible = False
d = word_app.Documents.Open(docx_path)
d.SaveAs(pdf_path, FileFormat=17)
d.Close()
word_app.Quit()
print("Saved Master Index PDF successfully!")
