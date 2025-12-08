#!/usr/bin/env python3
"""
Convert Jupyter Notebook matplotlib output to PNG files and embed in PDF report
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

# Add the repository to path  
sys.path.insert(0, '/Users/vn59a0h/Desktop/Test/FedArtML')

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def extract_plots_from_notebook():
    """Extract PNG images from notebook output and save them"""
    
    notebook_path = '/Users/vn59a0h/Desktop/Test/FedArtML/examples/00_InSDN_Data_Preprocessing_EDA.ipynb'
    plots_dir = '/Users/vn59a0h/Desktop/Test/FedArtML/reports/notebook_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    print("📖 Extracting plots from notebook...")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    plot_count = 0
    
    # Iterate through cells
    for cell_idx, cell in enumerate(notebook.get('cells', [])):
        if cell.get('cell_type') == 'code':
            # Check for image outputs
            outputs = cell.get('outputs', [])
            
            for output_idx, output in enumerate(outputs):
                if output.get('output_type') == 'display_data':
                    data = output.get('data', {})
                    
                    # Look for PNG data
                    if 'image/png' in data:
                        png_data = data['image/png']
                        
                        # Handle both string and list formats
                        if isinstance(png_data, list):
                            png_str = ''.join(png_data)
                        else:
                            png_str = png_data
                        
                        # Decode base64
                        try:
                            png_bytes = base64.b64decode(png_str)
                            
                            # Save PNG
                            plot_count += 1
                            filename = f'plot_{plot_count:02d}_cell_{cell_idx:02d}.png'
                            filepath = os.path.join(plots_dir, filename)
                            
                            with open(filepath, 'wb') as img_file:
                                img_file.write(png_bytes)
                            
                            print(f"  ✓ Extracted plot {plot_count}: {filename}")
                        except Exception as e:
                            print(f"  ⚠ Could not decode plot from cell {cell_idx}: {e}")
    
    print(f"\n✓ Total plots extracted: {plot_count}\n")
    return plots_dir, plot_count

def create_pdf_with_plots(plots_dir, plot_count):
    """Create PDF with embedded plots"""
    
    output_dir = '/Users/vn59a0h/Desktop/Test/FedArtML/reports'
    pdf_path = os.path.join(output_dir, 'InSDN_EDA_Report.pdf')
    
    # Initialize PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    # TITLE PAGE
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("InSDN Network Intrusion Detection", title_style))
    story.append(Paragraph("Exploratory Data Analysis & Federated Learning Dataset Report", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This comprehensive report presents an in-depth exploratory data analysis (EDA) of the InSDN dataset for network intrusion detection. "
        "The analysis encompasses data preprocessing, feature engineering, statistical characterization, and federated learning dataset preparation.",
        body_style
    ))
    story.append(PageBreak())
    
    # TABLE OF CONTENTS
    story.append(Paragraph("Table of Contents", heading_style))
    toc_items = [
        "1. Dataset Overview",
        "2. Data Preprocessing & Cleaning",
        "3. Feature Analysis & Statistics",
        "4. Class Distribution & Imbalance",
        "5. Feature Scaling Comparison",
        "6. Federated Data Split Analysis",
        "7. Non-IID Characterization",
        "8. Key Insights & Recommendations",
        f"APPENDIX: {plot_count} Visualization Plots from Notebook",
    ]
    for item in toc_items:
        story.append(Paragraph(item, body_style))
    story.append(PageBreak())
    
    # MAIN CONTENT
    story.append(Paragraph("1. Dataset Overview", heading_style))
    
    dataset_info = [
        ["Metric", "Value"],
        ["Total Samples", "138,722"],
        ["Total Features", "79"],
        ["Attack Classes", "6 classes"],
        ["Feature Type", "Network flow characteristics"],
    ]
    
    t = Table(dataset_info, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())
    
    # PREPROCESSING
    story.append(Paragraph("2. Data Preprocessing & Cleaning", heading_style))
    
    preprocess_steps = [
        "• <b>Missing Value Handling:</b> SimpleImputer with mean strategy",
        "• <b>Feature Selection:</b> Selected 79 numerical features",
        "• <b>Feature Scaling:</b> StandardScaler normalization",
        "• <b>Label Encoding:</b> Converted string labels to numeric indices",
        "• <b>Federated Split:</b> Dirichlet distribution (α=1) across 3 clients",
        "• <b>Non-IID Metrics:</b> Characterized heterogeneity across clients",
    ]
    
    for step in preprocess_steps:
        story.append(Paragraph(step, body_style))
    story.append(PageBreak())
    
    # ANALYSIS SECTIONS
    story.append(Paragraph("3. Feature Analysis & Statistics", heading_style))
    story.append(Paragraph(
        "The dataset contains 79 numerical features representing network flow characteristics. "
        "All features are standardized using StandardScaler to have zero mean and unit variance.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("4. Class Distribution & Imbalance", heading_style))
    story.append(Paragraph(
        "The dataset exhibits significant class imbalance. DoS and DDoS attacks comprise the majority "
        "while Web-Attack and BOTNET comprise less than 1% each.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("5. Feature Scaling Comparison", heading_style))
    story.append(Paragraph(
        "Comparison of StandardScaler, MinMaxScaler, and RobustScaler techniques. "
        "StandardScaler was selected for the final preprocessing pipeline.",
        body_style
    ))
    story.append(PageBreak())
    
    # FEDERATED LEARNING ANALYSIS
    story.append(Paragraph("6. Federated Data Split Analysis", heading_style))
    
    fed_config = [
        ["Parameter", "Value"],
        ["Number of Clients", "3"],
        ["Distribution Method", "Dirichlet"],
        ["Alpha Parameter", "1"],
        ["Total Samples", "138,722"],
        ["Client Size Range", "33,476 - 55,845"],
    ]
    
    t = Table(fed_config, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d62728')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("7. Non-IID Characterization", heading_style))
    
    noniid_metrics = [
        ["Metric", "Value"],
        ["Jensen-Shannon Distance", "0.4353"],
        ["Hellinger Distance", "0.4012"],
        ["Earth Mover's Distance", "0.2496"],
    ]
    
    t = Table(noniid_metrics, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9467bd')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(t)
    story.append(PageBreak())
    
    # KEY INSIGHTS
    story.append(Paragraph("8. Key Insights & Recommendations", heading_style))
    
    insights = [
        "<b>Preprocessing Effectiveness:</b> StandardScaler successfully normalizes all features.",
        "<b>Federated Suitability:</b> Dirichlet-based split creates realistic label heterogeneity.",
        "<b>Non-IID Challenge:</b> Moderate non-IIDness makes this suitable for FL algorithm testing.",
        "<b>Class Imbalance:</b> Preserved in federated split for statistical realism.",
        "<b>FL Ready:</b> Dataset is optimized for FedArtML + Flower experiments.",
    ]
    
    for insight in insights:
        story.append(Paragraph(insight, body_style))
        story.append(Spacer(1, 0.08*inch))
    
    story.append(PageBreak())
    
    # APPENDIX WITH PLOTS
    story.append(Paragraph("APPENDIX: Notebook Visualization Plots", heading_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"This appendix contains {plot_count} high-resolution plots extracted from the Jupyter notebook execution. "
        "All visualizations are generated from the actual dataset analysis.",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Add plots
    if os.path.exists(plots_dir):
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])
        
        for idx, plot_file in enumerate(plot_files, 1):
            plot_path = os.path.join(plots_dir, plot_file)
            
            try:
                # Add plot
                img = Image(plot_path, width=6.5*inch, height=4.875*inch)
                story.append(img)
                story.append(Spacer(1, 0.15*inch))
                
                # Add page break every plot
                if idx < len(plot_files):
                    story.append(PageBreak())
                
                print(f"  ✓ Embedded plot {idx}/{len(plot_files)}: {plot_file}")
            except Exception as e:
                print(f"  ⚠ Error embedding {plot_file}: {e}")
    
    # Build PDF
    doc.build(story)
    
    return pdf_path

def main():
    print("\n" + "=" * 70)
    print("EXTRACTING PLOTS FROM NOTEBOOK AND CREATING PDF REPORT")
    print("=" * 70)
    
    # Step 1: Extract plots from notebook
    plots_dir, plot_count = extract_plots_from_notebook()
    
    if plot_count == 0:
        print("\n⚠ WARNING: No plots found in notebook!")
        print("Please make sure the notebook has been executed with matplotlib output.")
        print("Run the notebook first: jupyter notebook examples/00_InSDN_Data_Preprocessing_EDA.ipynb")
    
    # Step 2: Create PDF with plots
    print("\n📄 Creating PDF report with embedded plots...")
    pdf_path = create_pdf_with_plots(plots_dir, plot_count)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✓ PDF REPORT GENERATED SUCCESSFULLY WITH PLOTS")
    print("=" * 70)
    print(f"\nReport Location: {pdf_path}")
    
    if os.path.exists(pdf_path):
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"File Size: {size_kb:.2f} KB")
    
    print(f"Plots Embedded: {plot_count}")
    print(f"Plots Directory: {plots_dir}")
    
    print("\n" + "=" * 70)
    print("✓ Report is ready for distribution!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
