import os
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)
from reportlab.pdfgen import canvas

RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"
EQ_DIR = RESULTS_DIR / "equations"
RESULTS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
EQ_DIR.mkdir(parents=True, exist_ok=True)


def generate_equations():
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.family"] = "STIXGeneral"

    eqs = {
        "eq1": r"$\hat{y} = \beta_0 + \sum_{1 \leq |\alpha| \leq d} \beta_\alpha x^\alpha$",
        "eq2": r"$\binom{p+d}{d} - 1$",
        "eq3": r"$\sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^m \beta_j^2$",
        "eq4": r"$\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$",
        "eq5": r"$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$",
        "ridge_set": r"$\{0.001,\, 0.01,\, 0.1,\, 1,\, 10,\, 100,\, 1000\}$",
    }

    for name, tex in eqs.items():
        fig = plt.figure(figsize=(4.5, 0.5), dpi=300)
        fig.text(0.5, 0.5, tex, fontsize=11, ha="center", va="center")
        fig.savefig(EQ_DIR / f"{name}.png", dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.03)
        plt.close(fig)


def generate_plots():
    v1_deg = list(range(1, 11))
    v1_mse = [8.318903, 3.552212, 0.869064, 0.624337, 0.395527, 0.477003, 0.455818, 0.505881, 0.618216, 0.640541]
    v1_r2 = [0.161400, 0.641914, 0.912393, 0.937063, 0.960128, 0.951915, 0.954051, 0.949004, 0.937680, 0.935429]

    fig, ax1 = plt.subplots(figsize=(6.2, 3.2), dpi=300)
    c1 = "#1f77b4"
    ax1.set_xlabel("Polynomial degree", fontsize=9.5)
    ax1.set_ylabel("Validation MSE (log scale)", color=c1, fontsize=9.5)
    l1 = ax1.plot(v1_deg, v1_mse, color=c1, marker="o", linewidth=1.5, markersize=5, label="Validation MSE")
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.25, top=16)
    ax1.set_xticks(v1_deg)
    ax1.tick_params(axis="both", labelsize=8.5)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, linestyle="-", alpha=0.35, color="#e0e0e0")

    ax2 = ax1.twinx()
    c2 = "#ff7f0e"
    ax2.set_ylabel("Validation $R^2$", color=c2, fontsize=9.5)
    l2 = ax2.plot(v1_deg, v1_r2, color=c2, marker="s", linewidth=1.5, markersize=5, label="Validation $R^2$")
    ax2.set_ylim(bottom=0.05, top=1.12)
    ax2.tick_params(axis="y", labelcolor=c2, labelsize=8.5)

    l3 = ax1.axvline(x=5, color="#2ca02c", linestyle="--", linewidth=1.5, label="Selected degree = 5")

    lines = l1 + [l3] + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=1.0, facecolor="white", edgecolor="#cccccc")
    plt.title("var1: polynomial degree selection", fontsize=10.5, pad=6)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "degree_curve_var1.png", dpi=300)
    plt.close(fig)

    v2_deg = list(range(1, 21))
    v2_mse = [39.881477, 25.195123, 11.461392, 3.836090, 1.476417, 0.544465, 0.325143, 0.261128, 0.277104, 0.259689, 0.287239, 0.281402, 0.320029, 0.318016, 0.324019, 0.335610, 0.384211, 0.376426, 0.426255, 0.425356]
    v2_r2 = [0.249691, 0.525992, 0.784371, 0.927830, 0.972223, 0.989757, 0.993883, 0.995087, 0.994787, 0.995114, 0.994596, 0.994706, 0.993979, 0.994017, 0.993904, 0.993686, 0.992772, 0.992918, 0.991981, 0.991998]

    fig, ax1 = plt.subplots(figsize=(6.2, 2.75), dpi=300)
    ax1.set_xlabel("Polynomial degree", fontsize=9)
    ax1.set_ylabel("Validation MSE (log scale)", color=c1, fontsize=9)
    l1 = ax1.plot(v2_deg, v2_mse, color=c1, marker="o", linewidth=1.5, markersize=4, label="Validation MSE")
    ax1.set_yscale("log")
    ax1.set_ylim(bottom=0.15, top=65)
    ax1.set_xticks(v2_deg)
    ax1.tick_params(axis="both", labelsize=8)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(True, linestyle="-", alpha=0.35, color="#e0e0e0")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation $R^2$", color=c2, fontsize=9)
    l2 = ax2.plot(v2_deg, v2_r2, color=c2, marker="s", linewidth=1.5, markersize=4, label="Validation $R^2$")
    ax2.set_ylim(bottom=0.15, top=1.12)
    ax2.tick_params(axis="y", labelcolor=c2, labelsize=8)

    l3 = ax1.axvline(x=10, color="#2ca02c", linestyle="--", linewidth=1.5, label="Selected degree = 10")

    lines = l1 + [l3] + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8, framealpha=1.0, facecolor="white", edgecolor="#cccccc")
    plt.title("var2: polynomial degree selection", fontsize=10, pad=5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "degree_curve_var2.png", dpi=300)
    plt.close(fig)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 10)
        self.drawCentredString(A4[0] / 2.0, 36, str(self._pageNumber))
        self.restoreState()


def make_equation_block(img_path, eq_num_str, target_height_pt=24):
    with Image.open(img_path) as im:
        w, h = im.size
    width_pt = target_height_pt * (w / h)
    img = RLImage(str(img_path), width=width_pt, height=target_height_pt)
    num_p = Paragraph(f"({eq_num_str})", ParagraphStyle("EqNum", fontName="Times-Roman", fontSize=10, alignment=2))
    t = Table([[ "", img, num_p ]], colWidths=[40, 400, 47])
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def build_pdf():
    generate_equations()
    generate_plots()

    pdf_path = REPORT_DIR / "BT2024062_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=48,
        bottomMargin=42,
    )

    style_title = ParagraphStyle("DocTitle", fontName="Times-Roman", fontSize=17, leading=22, alignment=1, spaceAfter=8)
    style_author = ParagraphStyle("DocAuthor", fontName="Times-Roman", fontSize=12, leading=16, alignment=1)
    style_roll = ParagraphStyle("DocRoll", fontName="Times-Roman", fontSize=11, leading=15, alignment=1)
    style_date = ParagraphStyle("DocDate", fontName="Times-Roman", fontSize=11, leading=15, alignment=1, spaceAfter=12)
    style_repo = ParagraphStyle("DocRepo", fontName="Times-Roman", fontSize=9.5, leading=13, alignment=1, spaceAfter=18)

    style_h1 = ParagraphStyle("SecH1", fontName="Times-Bold", fontSize=13, leading=17, spaceBefore=12, spaceAfter=6, keepWithNext=True)
    style_h2 = ParagraphStyle("SecH2", fontName="Times-Bold", fontSize=11, leading=15, spaceBefore=9, spaceAfter=4, keepWithNext=True)
    style_body = ParagraphStyle("Body", fontName="Times-Roman", fontSize=9.5, leading=13.2, alignment=4, spaceAfter=5)
    style_body_first = ParagraphStyle("BodyFirst", fontName="Times-Roman", fontSize=9.5, leading=13.2, alignment=4, spaceAfter=5, firstLineIndent=16)
    style_caption = ParagraphStyle("Caption", fontName="Times-Roman", fontSize=9, leading=12, alignment=1, spaceBefore=4, spaceAfter=6)

    story = []

    # ================= PAGE 1 =================
    story.append(Paragraph("Assignment 1: Polynomial Regression", style_title))
    story.append(Paragraph("Vraj Vashi", style_author))
    story.append(Paragraph("Roll No: BT2024062", style_roll))
    story.append(Paragraph("October 2026", style_date))
    story.append(Paragraph('<b>GitHub Repository:</b> <font name="Courier" size="8.5">https://github.com/VrajVashi/BT2024062-ML-Assignment</font>', style_repo))

    story.append(Paragraph("1 &nbsp; Introduction", style_h1))
    story.append(Paragraph(
        "The objective of this assignment is to construct polynomial regression models for two personalized regression "
        "datasets, called <i>var1</i> and <i>var2</i>, and use the fitted models to predict the unknown target variable <i>y</i> in their "
        "corresponding test sets. The two problems have different numbers of inputs and different permitted maximum "
        "polynomial degrees. The var1 dataset has six input variables and permits polynomial degrees up to 10, while "
        "var2 has three input variables and permits degrees up to 20.",
        style_body
    ))
    story.append(Paragraph(
        "The main modelling decision is the polynomial degree. A degree that is too small may not represent the "
        "relationship between the inputs and target, whereas an unnecessarily high degree creates many correlated terms "
        "and can fit noise in the training data. Therefore, all model choices in this work were made using a held-out "
        "portion of the labelled training data. The test data was used only after the model choices had been finalized. Its "
        "target values were not available and it was not used for tuning.",
        style_body_first
    ))

    story.append(Paragraph("2 &nbsp; Dataset and Problem Description", style_h1))
    story.append(Paragraph(
        "The datasets were loaded and inspected using pandas. Both training sets contain 1,000 observations and have "
        "a numeric continuous target named <i>y</i>. Both test sets contain 1,000 observations with the same input columns "
        "as their corresponding training data, but without the target column. The dataset structure is summarized in "
        "Table 1.",
        style_body
    ))

    # Table 1
    story.append(Paragraph("Table 1: Dataset structure and missing-value summary.", style_caption))
    t1_data = [
        ["Dataset", "Rows", "Inputs", "Target", "Missing values"],
        ["var1 training", "1000", "6", "y", "0"],
        ["var1 test", "1000", "6", "Hidden", "0"],
        ["var2 training", "1000", "3", "y", "0"],
        ["var2 test", "1000", "3", "Hidden", "0"],
    ]
    t1 = Table(t1_data, colWidths=[120, 65, 65, 85, 95])
    t1.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "The inputs in both training datasets range from &minus;1 to 1. In the var1 training data, the target has mean "
        "0.7943, standard deviation 3.2082, minimum &minus;10.1054, and maximum 10.6104. In the var2 training data, the "
        "target has mean 2.2044, standard deviation 6.6875, minimum &minus;29.9989, and maximum 39.1383. Therefore, var2 "
        "has a wider target range and larger target variance.",
        style_body
    ))
    story.append(Paragraph(
        "There were no missing or non-finite values in any file. No duplicate complete rows were found in either "
        "training set. A small number of repeated input rows appeared in the test sets, but they were retained because "
        "they are valid test cases and each row requires a prediction.",
        style_body_first
    ))

    story.append(Paragraph("3 &nbsp; Methodology", style_h1))
    story.append(Paragraph("3.1 &nbsp; Polynomial Regression", style_h2))
    story.append(Paragraph(
        "Polynomial regression models nonlinear relationships by expanding the original variables into powers and inter"
        "action terms, followed by a linear regression model. For example, a degree-2 expansion of two variables includes "
        "<i>x</i><sub>1</sub>, <i>x</i><sub>2</sub>, <i>x</i><sub>1</sub><sup>2</sup>, <i>x</i><sub>1</sub><i>x</i><sub>2</sub>, and <i>x</i><sub>2</sub><sup>2</sup>. A general polynomial model of degree <i>d</i> can be written as",
        style_body
    ))
    story.append(make_equation_block(EQ_DIR / "eq1.png", "1", target_height_pt=25))

    story.append(PageBreak())

    # ================= PAGE 2 =================
    story.append(Paragraph(
        "where &alpha; is a multi-index and <i>x</i><sup>&alpha;</sup> represents powers and interactions whose total degree is no greater than <i>d</i>. "
        "The model is nonlinear in the original inputs but remains linear in its coefficients.<br/>"
        "For <i>p</i> original input variables and degree <i>d</i>, the number of generated polynomial terms, excluding the constant "
        "term, is",
        style_body
    ))
    story.append(make_equation_block(EQ_DIR / "eq2.png", "2", target_height_pt=24))

    story.append(Paragraph(
        "The number of terms grows rapidly with the degree. For example, var1 produces 461 terms at degree 5 and "
        "8,007 terms at degree 10. Var2 produces 285 terms at degree 10 and 1,770 terms at degree 20. This increase in "
        "dimensionality can cause overfitting and numerical instability.",
        style_body
    ))

    story.append(Paragraph("3.2 &nbsp; Preprocessing and Ridge Regression", style_h2))
    story.append(Paragraph(
        "Polynomial features were generated using <font name=\"Courier\">PolynomialFeatures</font> from scikit-learn with <font name=\"Courier\">include_bias=False</font>. The "
        "expanded features were standardized using <font name=\"Courier\">StandardScaler</font>. During validation, the scaler was fitted only on "
        "the modelling training portion, preventing information from the validation data from entering the preprocessing "
        "stage. Scaling is useful because different powers and interaction terms can have considerably different variances.<br/>"
        "Ridge regression was used after the polynomial expansion. Ridge minimizes the following objective:",
        style_body
    ))
    story.append(make_equation_block(EQ_DIR / "eq3.png", "3", target_height_pt=25))

    story.append(Paragraph(
        "where &lambda;, called <font name=\"Courier\">alpha</font> in scikit-learn, controls the amount of coefficient shrinkage. Polynomial terms can "
        "be highly correlated, especially at high degrees. Ridge regularization improves numerical stability and reduces "
        "overfitting while keeping the modelling approach straightforward.",
        style_body
    ))

    story.append(Paragraph("3.3 &nbsp; Validation and Model Selection", style_h2))
    story.append(Paragraph(
        "Each labelled dataset was divided into 80% modelling training data (800 rows) and 20% validation data (200 "
        "rows). A fixed <font name=\"Courier\">random_state=42</font> was used so that every candidate was evaluated on the same observations and "
        "the experiment could be reproduced.<br/>"
        "Every permitted integer degree was evaluated: degrees 1&ndash;10 for var1 and degrees 1&ndash;20 for var2. For each "
        "degree, the Ridge values",
        style_body
    ))

    # ridge set
    with Image.open(EQ_DIR / "ridge_set.png") as rim:
        rw, rh = rim.size
    r_target_h = 13.5
    r_target_w = r_target_h * (rw / rh)
    r_img = RLImage(str(EQ_DIR / "ridge_set.png"), width=r_target_w, height=r_target_h)
    r_table = Table([[r_img]], colWidths=[487])
    r_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(r_table)

    story.append(Paragraph(
        "were tested. The alpha with the lowest validation mean squared error (MSE) was retained for that degree. "
        "The final degree-alpha combination was selected using the lowest overall validation MSE. Validation <i>R</i><sup>2</sup> was also "
        "recorded as a secondary performance measure:",
        style_body
    ))
    story.append(make_equation_block(EQ_DIR / "eq4.png", "4", target_height_pt=25))
    story.append(make_equation_block(EQ_DIR / "eq5.png", "5", target_height_pt=26))

    story.append(Paragraph(
        "After model selection, each chosen pipeline was fitted again using all 1,000 labelled training observations. "
        "The refitted model was then used to generate the corresponding test predictions.",
        style_body
    ))

    story.append(Paragraph("4 &nbsp; Validation Results", style_h1))
    story.append(Paragraph("4.1 &nbsp; Results for var1", style_h2))
    story.append(Paragraph(
        "Table 2 shows the best alpha and validation scores obtained for every tested var1 degree.",
        style_body
    ))

    story.append(PageBreak())

    # ================= PAGE 3 =================
    story.append(Paragraph("Table 2: Validation results for var1.", style_caption))
    t2_data = [
        ["Degree", "Terms", "Best alpha", "Validation MSE", "Validation R2"],
        ["1", "6", "0.001", "8.318903", "0.161400"],
        ["2", "27", "0.001", "3.552212", "0.641914"],
        ["3", "83", "10", "0.869064", "0.912393"],
        ["4", "209", "10", "0.624337", "0.937063"],
        ["5", "461", "10", "0.395527", "0.960128"],
        ["6", "923", "100", "0.477003", "0.951915"],
        ["7", "1715", "100", "0.455818", "0.954051"],
        ["8", "3002", "100", "0.505881", "0.949004"],
        ["9", "5004", "100", "0.618216", "0.937680"],
        ["10", "8007", "100", "0.640541", "0.935429"],
    ]
    t2 = Table(t2_data, colWidths=[65, 65, 80, 130, 110])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("FONTNAME", (0, 5), (-1, 5), "Times-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Degree 5 with Ridge alpha 10 was selected for var1. It produced the lowest validation MSE of 0.395527 "
        "and the highest validation <i>R</i><sup>2</sup> of 0.960128. The large improvement from degree 1 to degree 5 shows that linear "
        "and low-degree models underfit the data. From degree 6 onward, the validation error increases even though the "
        "number of polynomial terms grows. This is evidence of increasing variance and overfitting. The complete degree "
        "comparison is shown in Figure 1.",
        style_body
    ))
    story.append(Spacer(1, 6))

    # Figure 1
    story.append(RLImage(str(RESULTS_DIR / "degree_curve_var1.png"), width=440, height=227))
    story.append(Paragraph("Figure 1: Polynomial degree selection for var1. The MSE axis uses a logarithmic scale.", style_caption))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.2 &nbsp; Results for var2", style_h2))
    story.append(Paragraph(
        "Table 3 presents the best validation result for each tested var2 degree.",
        style_body
    ))

    story.append(PageBreak())

    # ================= PAGE 4 =================
    story.append(Paragraph("Table 3: Validation results for var2.", style_caption))
    t3_data = [
        ["Degree", "Terms", "Best alpha", "Validation MSE", "Validation R2"],
        ["1", "3", "10", "39.881477", "0.249691"],
        ["2", "9", "0.001", "25.195123", "0.525992"],
        ["3", "19", "0.001", "11.461392", "0.784371"],
        ["4", "34", "0.001", "3.836090", "0.927830"],
        ["5", "55", "0.01", "1.476417", "0.972223"],
        ["6", "83", "0.001", "0.544465", "0.989757"],
        ["7", "119", "0.1", "0.325143", "0.993883"],
        ["8", "164", "0.1", "0.261128", "0.995087"],
        ["9", "219", "1", "0.277104", "0.994787"],
        ["10", "285", "1", "0.259689", "0.995114"],
        ["11", "363", "1", "0.287239", "0.994596"],
        ["12", "454", "1", "0.281402", "0.994706"],
        ["13", "559", "0.1", "0.320029", "0.993979"],
        ["14", "679", "10", "0.318016", "0.994017"],
        ["15", "815", "0.1", "0.324019", "0.993904"],
        ["16", "968", "0.1", "0.335610", "0.993686"],
        ["17", "1139", "10", "0.384211", "0.992772"],
        ["18", "1329", "10", "0.376426", "0.992918"],
        ["19", "1539", "10", "0.426255", "0.991981"],
        ["20", "1770", "10", "0.425356", "0.991998"],
    ]
    t3 = Table(t3_data, colWidths=[65, 65, 80, 130, 110], rowHeights=[9.2] * 21)
    t3.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("FONTNAME", (0, 10), (-1, 10), "Times-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "Degree 10 with Ridge alpha 1 was selected for var2. Its validation MSE is 0.259689 and its validation <i>R</i><sup>2</sup> is "
        "0.995114. The error decreases sharply from degree 1 through degree 8, indicating substantial nonlinearity and "
        "clear underfitting at low degrees. Degrees 8&ndash;12 have closer results (with degree 8 achieving 0.261128), but degree 10 gives the smallest measured "
        "MSE. Degrees above 12 do not improve validation performance and instead show a gradual increase in error. "
        "Therefore, degree 10 is preferred to the more complex degrees 11&ndash;20. This behaviour is illustrated in Figure 2.",
        style_body
    ))

    # Figure 2
    story.append(RLImage(str(RESULTS_DIR / "degree_curve_var2.png"), width=390, height=155))
    story.append(Paragraph("Figure 2: Polynomial degree selection for var2. The MSE axis uses a logarithmic scale.", style_caption))
    story.append(Spacer(1, 2))

    story.append(Paragraph("5 &nbsp; Final Models and Verification", style_h1))
    story.append(Paragraph(
        "The final var1 pipeline consists of a degree-5 polynomial expansion, feature standardization, and Ridge regression "
        "with alpha 10. The final var2 pipeline uses a degree-10 polynomial expansion, feature standardization, and Ridge "
        "regression with alpha 1. Both selected pipelines were refitted using their complete training datasets.",
        style_body
    ))
    story.append(Paragraph(
        "The generated prediction files are <font name=\"Courier\">BT2024062_pred_var1.csv</font> and <font name=\"Courier\">BT2024062_pred_var2.csv</font>. Each contains "
        "exactly 1,000 predictions in a single column named <i>y</i>, matching the sample submission format.",
        style_body
    ))
    story.append(Paragraph(
        "The complete program was run from start to finish. It verified that the feature columns and their order "
        "match between training and test data, that the test sets do not contain the target, and that all input values "
        "and predictions are finite. It also checked the degree restrictions, prediction row counts, and saved CSV column "
        "names. The saved files were read back after writing to verify the actual deliverables. All checks passed, and "
        "repeated executions produced identical prediction files. No hidden test target or test-derived score was used "
        "during model selection.",
        style_body
    ))

    story.append(PageBreak())

    # ================= PAGE 5 =================
    story.append(Paragraph("6 &nbsp; Conclusion", style_h1))
    story.append(Paragraph(
        "Polynomial regression successfully represented the nonlinear relationships in both datasets. Training-only valida"
        "tion selected degree 5 with Ridge alpha 10 for var1 and degree 10 with Ridge alpha 1 for var2. Their validation "
        "results were MSE 0.395527 with <i>R</i><sup>2</sup> = 0.960128, and MSE 0.259689 with <i>R</i><sup>2</sup> = 0.995114, respectively.",
        style_body
    ))
    story.append(Paragraph(
        "The experiments demonstrate the bias&ndash;variance trade-off. Low-degree models underfit both datasets, while "
        "unnecessarily high degrees generate many polynomial terms and reduce validation performance. Standardization "
        "and Ridge regularization made the polynomial models more stable, while validation-based model selection ensured "
        "that the hidden test set did not influence the modelling decisions.",
        style_body_first
    ))
    story.append(Spacer(1, 14))

    story.append(Paragraph("References", style_h1))
    ref_style = ParagraphStyle("Ref", fontName="Times-Roman", fontSize=9, leading=13, spaceAfter=8)
    story.append(Paragraph('[1] Scikit-learn Developers, &ldquo;PolynomialFeatures,&rdquo; <font name="Courier" size="8">https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html</font>.', ref_style))
    story.append(Paragraph('[2] Scikit-learn Developers, &ldquo;Ridge Regression,&rdquo; <font name="Courier" size="8">https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html</font>.', ref_style))
    story.append(Paragraph('[3] G. James, D. Witten, T. Hastie, and R. Tibshirani, <i>An Introduction to Statistical Learning</i>, Springer, 2021.', ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {pdf_path}")


if __name__ == "__main__":
    build_pdf()
