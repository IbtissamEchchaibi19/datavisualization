import random
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import inch
import os

def generate_random_date():
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
    return random_date.strftime("%d/%m/%y")

def generate_batch_number():
    prefixes = ["BTH", "HNY", "API", "SDR", "MNH", "PRD"]
    return f"{random.choice(prefixes)}{random.randint(1000, 9999)}"

def generate_apiary_data():
    locations = ["Fujairah", "Dubai", "Sharjah", "Abu Dhabi", "Ajman", "Ras Al Khaimah", "Umm Al Quwain",
                 "Masafi", "Dibba", "Taweeh", "Al Hajar", "Khor Fakkan", "Kalba", "Hatta", "Al farfar"]
    
    apiary_number = f"{random.randint(1, 99):02d}"
    location = random.choice(locations)
    gross_weight = round(random.uniform(80, 650), 1)
    drum_weight = round(random.uniform(5, 35), 1)
    net_weight = round(gross_weight - drum_weight, 1)
    beshara = random.choice([2.7, 2.5, 2.8, 3.0, 2.9, 2.6, 3.1, 2.4])
    production = round(net_weight - beshara, 1)
    num_hives = random.randint(15, 120)
    production_per_hive = round(production / num_hives, 2)
    num_supers = random.randint(15, 85)
    harvest_date = generate_random_date()

    return [
        f"{apiary_number} - {location}",
        gross_weight,
        drum_weight,
        net_weight,
        beshara,
        production,
        num_hives,
        production_per_hive,
        num_supers,
        harvest_date
    ]

def create_honey_production_pdf(filename, batch_number):
    doc = SimpleDocTemplate(filename, pagesize=A4, topMargin=0.3*inch,
                            leftMargin=0.15*inch, rightMargin=0.15*inch, bottomMargin=0.3*inch)

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title_style.fontSize = 12
    title_style.alignment = 1

    story = []

    title_text = f"SIDR 2024 - APIARIES HONEY PRODUCTION MAP - MANAHIL LLC.<br/>Batch: {batch_number}"
    title = Paragraph(title_text, title_style)
    story.append(title)
    story.append(Spacer(1, 0.1*inch))

    num_apiaries = random.randint(12, 15)
    data = []

    headers = [
        "Apiary\nNumber",
        "Gross\nweight (KG)",
        "Drum\nweight\n(KG)",
        "Net\nweight\n(KG)",
        "Beshara\n(KG)",
        "Production\n(KG)",
        "No. of\nProduction\nHives",
        "Production\nper hive\n(KG)",
        "No. of\nHive\nSupers",
        "Harvest\nDate"
    ]
    data.append(headers)

    total_net_weight = 0
    total_beshara = 0
    total_production = 0
    total_hives = 0
    production_per_hive_values = []

    for i in range(1, num_apiaries + 1):
        row_data = generate_apiary_data()
        row = row_data
        data.append(row)
        total_net_weight += row_data[3]
        total_beshara += row_data[4]
        total_production += row_data[5]
        total_hives += row_data[6]
        production_per_hive_values.append(row_data[7])

    cleaning_gross = round(random.uniform(150, 250), 1)
    cleaning_drum = 15.7
    cleaning_net = round(cleaning_gross - cleaning_drum, 1)
    cleaning_production = cleaning_net

    cleaning_row = [
        "Cleaning\nHarvest",
        cleaning_gross,
        cleaning_drum,
        cleaning_net,
        "-",
        cleaning_production,
        "-",
        "-",
        "-",
        generate_random_date()
    ]
    data.append(cleaning_row)

    total_net_weight += cleaning_net
    total_production += cleaning_production
    net_production = round(total_production - total_beshara, 1)
    avg_production_per_hive = round(sum(production_per_hive_values) / len(production_per_hive_values), 2) if production_per_hive_values else 0

    summary_row = [
        "",
        "",
        "",
        f"Net\nStock\n{round(total_net_weight, 1)}\nKg",
        f"Net\nBeshara\n{round(total_beshara, 1)}\nKg",
        f"Net\nProduction\n{net_production}KG",
        f"Total No.\nproduction\nhives\n{total_hives}",
        f"Average\nproduction\nper hive\n{avg_production_per_hive}KG",
        "",
        ""
    ]
    data.append(summary_row)

    # Adjusted column widths: Increased "Harvest Date" to 1.0*inch
    col_widths = [1.6*inch, 0.75*inch, 0.65*inch, 0.75*inch, 0.65*inch,
                  0.75*inch, 0.85*inch, 0.85*inch, 0.75*inch, 1.0*inch]

    table = Table(data, colWidths=col_widths, repeatRows=1)

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -3), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -3), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -3), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
        ('BACKGROUND', (0, -2), (-1, -2), colors.Color(1, 1, 0.8)),
        ('FONTNAME', (0, -2), (-1, -2), 'Helvetica-Bold'),
        ('FONTSIZE', (0, -2), (-1, -2), 6),
        ('BACKGROUND', (0, -1), (-1, -1), colors.Color(1, 1, 0.6)),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, -1), (-1, -1), 6),
        ('BACKGROUND', (5, -1), (5, -1), colors.yellow),
        ('LEFTPADDING', (0, 0), (-1, -1), 2),
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ])
    
    table.setStyle(table_style)
    story.append(table)
    doc.build(story)
    print(f"Generated: {filename} (Batch: {batch_number})")

def generate_multiple_pdfs(num_files=100):
    output_dir = "honey_production_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating {num_files} honey production PDF reports...")

    for i in range(1, num_files + 1):
        batch_num = generate_batch_number()
        filename = os.path.join(output_dir, f"honey_production_report_{i:03d}_{batch_num}.pdf")
        create_honey_production_pdf(filename, batch_num)

    print(f"\nSuccessfully generated {num_files} PDF files in '{output_dir}'!")

if __name__ == "__main__":
    try:
        batch_num = generate_batch_number()
        test_filename = f"test_honey_report_{batch_num}.pdf"
        create_honey_production_pdf(test_filename, batch_num)
        generate_multiple_pdfs(100)
        
    except ImportError:
        print("Please install reportlab: pip install reportlab")
    except Exception as e:
        print(f"An error occurred: {e}")