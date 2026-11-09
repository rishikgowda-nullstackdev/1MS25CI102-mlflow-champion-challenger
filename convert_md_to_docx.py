"""
convert_md_to_docx.py
=====================
Converts SUBMISSION_REPORT.md into a fully styled, editable Microsoft Word document (.docx).
Embeds tables, formatting, and evaluation plots directly into the Word document.
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

MD_PATH = "SUBMISSION_REPORT.md"
DOCX_PATH = "SUBMISSION_REPORT.docx"


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets internal padding for a cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)


def format_run_markdown(paragraph, text):
    """
    Parses basic bold (**text** or *text*) in a text snippet and adds runs.
    """
    pattern = re.compile(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)')
    tokens = pattern.split(text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith('*') and token.endswith('*'):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        elif token.startswith('`') and token.endswith('`'):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(199, 37, 78)
        else:
            paragraph.add_run(token)


def create_submission_docx():
    if not os.path.exists(MD_PATH):
        print(f"Error: {MD_PATH} not found.")
        return

    doc = docx.Document()

    # Page Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Base Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)

    with open(MD_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_table = False
    table_lines = []
    in_code_block = False
    code_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if not table_lines:
            return
        
        # Parse table rows
        rows_data = []
        for line in table_lines:
            parts = [c.strip() for c in line.strip().split('|')]
            if len(parts) >= 3:
                # Discard empty first and last elements if table was fenced with |
                row_cells = parts[1:-1] if parts[0] == '' and parts[-1] == '' else parts
                # Skip divider rows (e.g., :---|:---:)
                if any(re.match(r'^:?-+:?$', c) for c in row_cells if c):
                    continue
                rows_data.append(row_cells)

        if rows_data:
            num_cols = max(len(r) for r in rows_data)
            # Pad rows to uniform length
            for r in rows_data:
                while len(r) < num_cols:
                    r.append("")

            t = doc.add_table(rows=len(rows_data), cols=num_cols)
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = True

            # Style Table
            for row_idx, row_cells in enumerate(rows_data):
                is_header = (row_idx == 0)
                row_obj = t.rows[row_idx]
                for col_idx, cell_text in enumerate(row_cells):
                    cell = row_obj.cells[col_idx]
                    cell.text = ""
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    p.paragraph_format.line_spacing = 1.15
                    format_run_markdown(p, cell_text)

                    if is_header:
                        set_cell_background(cell, "2C3E50")
                        for r in p.runs:
                            r.font.bold = True
                            r.font.color.rgb = RGBColor(255, 255, 255)
                    else:
                        bg_hex = "F8F9FA" if row_idx % 2 == 1 else "FFFFFF"
                        set_cell_background(cell, bg_hex)

                    set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

            # Add small spacing after table
            doc.add_paragraph()

        in_table = False
        table_lines = []

    def flush_code_block():
        nonlocal in_code_block, code_lines
        if not code_lines:
            return
        
        block_text = "".join(code_lines)
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Inches(0.2)
        
        run = p.add_run(block_text)
        run.font.name = "Consolas"
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(30, 30, 30)

        in_code_block = False
        code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip('\r\n')

        # Code block handling
        if line.startswith("```"):
            if in_code_block:
                flush_code_block()
            else:
                if in_table:
                    flush_table()
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line + "\n")
            continue

        # Table detection
        if line.startswith("|") and line.endswith("|"):
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            flush_table()

        # Horizontal Rule
        if re.match(r'^-{3,}$', line.strip()):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            p_border = p.add_run("―" * 45)
            p_border.font.color.rgb = RGBColor(180, 180, 180)
            continue

        # Heading 1 (# ...)
        if line.startswith("# "):
            title_text = line[2:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(title_text)
            run.font.name = "Calibri"
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(26, 82, 118)
            continue

        # Heading 2 (## ...)
        if line.startswith("## "):
            h2_text = line[3:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(h2_text)
            run.font.name = "Calibri"
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(41, 128, 185)
            continue

        # Heading 3 (### ...)
        if line.startswith("### "):
            h3_text = line[4:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(h3_text)
            run.font.name = "Calibri"
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(52, 73, 94)
            continue

        # Heading 4 (#### ...)
        if line.startswith("#### "):
            h4_text = line[5:].strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(h4_text)
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = RGBColor(80, 80, 80)
            continue

        # Bullet List (- ... or * ...)
        if re.match(r'^\s*[-*]\s+', line):
            content = re.sub(r'^\s*[-*]\s+', '', line)
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            format_run_markdown(p, content)
            continue

        # Numbered List (1. ...)
        if re.match(r'^\s*\d+\.\s+', line):
            content = re.sub(r'^\s*\d+\.\s+', '', line)
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            format_run_markdown(p, content)
            continue

        # Regular Paragraph
        if line.strip():
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            format_run_markdown(p, line.strip())

    if in_table:
        flush_table()
    if in_code_block:
        flush_code_block()

    # Section 9: Append Evaluation Plots as Embedded Visual Figures
    doc.add_page_break()
    p_img_title = doc.add_paragraph()
    run_img = p_img_title.add_run("Appendix: Evaluation Visualizations & Dashboard Screenshots")
    run_img.font.name = "Calibri"
    run_img.font.size = Pt(16)
    run_img.font.bold = True
    run_img.font.color.rgb = RGBColor(26, 82, 118)

    images_to_embed = [
        ("outputs/confusion_matrix_comparison.png", "Figure 1: Confusion Matrices for Champion (DistilBERT) and Challenger (BERT SST-2)"),
        ("outputs/metrics_comparison_barchart.png", "Figure 2: Performance Metrics Comparison (Accuracy, Macro-F1, Precision, Recall)"),
        ("outputs/latency_comparison.png", "Figure 3: Average Inference Latency Comparison per sample (ms)"),
        ("screenshots/predict_terminal_output.png", "Figure 4: Terminal Execution of predict_from_mlflow.py (@champion URI)"),
        ("screenshots/registry_aliases.png", "Figure 5: MLflow Model Registry showing @champion and @challenger Aliases"),
        ("screenshots/mlflow_compare_view.png", "Figure 6: MLflow Experiment Tracking UI Runs Compare View")
    ]

    for img_path, caption in images_to_embed:
        if os.path.exists(img_path):
            p_cap = doc.add_paragraph()
            p_cap.paragraph_format.space_before = Pt(10)
            p_cap.paragraph_format.space_after = Pt(2)
            run_cap = p_cap.add_run(caption)
            run_cap.font.bold = True
            run_cap.font.size = Pt(11)
            run_cap.font.color.rgb = RGBColor(52, 73, 94)

            p_pic = doc.add_paragraph()
            p_pic.paragraph_format.space_before = Pt(2)
            p_pic.paragraph_format.space_after = Pt(8)
            p_pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_picture(img_path, width=Inches(6.2))

    doc.save(DOCX_PATH)
    print(f"Successfully generated editable Word document: {DOCX_PATH}")


if __name__ == "__main__":
    create_submission_docx()
