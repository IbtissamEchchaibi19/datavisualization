from faker import Faker
import random
import os
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Setup Faker and paths
fake = Faker()
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "newinvoices")
os.makedirs(output_dir, exist_ok=True)

# ReportLab setup
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'Title',
    parent=styles['Heading1'],
    alignment=1,  # Center
    spaceAfter=12
)
normal_style = styles['Normal']
heading_style = ParagraphStyle(
    'Heading',
    parent=styles['Heading2'],
    fontSize=12,
    leading=14,
    spaceAfter=6
)
small_style = ParagraphStyle(
    'Small',
    parent=styles['Normal'],
    fontSize=8,
    leading=10
)
table_heading = ParagraphStyle(
    'TableHeading',
    parent=styles['Normal'],
    fontSize=8,
    leading=10,
    alignment=1  # Center
)

# Generate invoice data
def generate_invoice_data():
    # Product variations
    products = [
        'Sidr Honey 50g', 'Acacia Honey 50g', 'Wildflower Honey 50g',
        'Mountain Honey 550g', 'Forest Honey 50g', 'Royal Jelly Honey 50g',
        'Manuka Honey 50g', 'Clover Honey 50g'
    ]

    # Customer locations (Emirates)
    locations = [
        'Fujairah', 'Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman',
        'Ras Al Khaimah', 'Umm Al Quwain'
    ]

    # Customer types
    customer_types = ['Retail Store', 'Supermarket', 'Hotel', 'Restaurant', 'Direct Consumer', 'Distributor']

    # Generate date
    start_date = datetime.now() - timedelta(days=730)
    end_date = datetime.now()
    invoice_date = start_date + timedelta(days=random.randint(0, 730))
    date_str = invoice_date.strftime('%d %b %Y')
    
    # Generate customer data
    customer_id = f"U{random.randint(301, 350)}"
    customer_name = f"Company {customer_id}"
    customer_location = random.choice(locations)
    customer_type = random.choice(customer_types)
    address = fake.street_address()
    city = customer_location
    po_box = f"P.O. Box {random.randint(1000, 9999)}"
    country = "United Arab Emirates"
    phone = f"+{random.randint(1, 999)}({random.randint(1, 999)}){random.randint(1000000, 9999999)}"
    trn = f"{random.randint(100000000, 999999999)}"
    
    # Determine payment status
    days_to_payment = random.choices(
        [random.randint(0, 10), random.randint(11, 30), random.randint(31, 60), random.randint(61, 90), -1],
        weights=[0.6, 0.2, 0.1, 0.05, 0.05],  # 60% pay quickly, 5% don't pay
        k=1
    )[0]
    
    payment_date = None if days_to_payment == -1 else invoice_date + timedelta(days=days_to_payment)
    payment_status = "Unpaid" if payment_date is None or payment_date > datetime.now() else "Paid"
    payment_term = "Immediate" if payment_status == "Paid" else f"Net {random.choice([15, 30, 45, 60])}"
    
    # Product data
    product = random.choice(products)
    qty = random.randint(1, 20)
    unit_price = round(random.uniform(200, 400), 2)
    gross = round(qty * unit_price, 2)
    discount_percentage = random.choices([0, 0.05, 0.1], weights=[0.8, 0.15, 0.05], k=1)[0]
    discount = round(gross * discount_percentage, 2)
    amount_excl_vat = gross - discount
    vat = round(amount_excl_vat * 0.05, 2)
    total = round(amount_excl_vat + vat, 2)
    
    # Calculate cost and profit
    cost_price = round(unit_price * 0.6, 2)  # Assume 60% of sale price is cost
    profit = round(amount_excl_vat - (cost_price * qty), 2)
    profit_margin = round((profit / amount_excl_vat) * 100, 2)
    
    # Invoice numbers
    invoice_no = f"SINU1301-{random.randint(2000000, 9999999)}"
    ref = f"FGDN-{random.randint(10, 99)} {fake.word().upper()}"
    do_no = f"SDHU1301-{random.randint(20000000, 99999999)}"
    customer_po = ref
    
    return {
        "invoice_date": date_str,
        "invoice_no": invoice_no,
        "ref": ref,
        "do_no": do_no,
        "customer_po": customer_po,
        "customer_name": customer_name,
        "customer_id": customer_id,
        "customer_location": customer_location,
        "customer_type": customer_type,
        "address_line1": address,
        "address_line2": f"{city}, {po_box}",
        "address_line3": country,
        "phone": phone,
        "trn": trn,
        "due_date": date_str,
        "payment_term": payment_term,
        "payment_status": payment_status,
        "payment_date": payment_date,
        "days_to_payment": days_to_payment if days_to_payment != -1 else None,
        "product": product,
        "uom": "UN",
        "qty": qty,
        "unit_price": unit_price,
        "gross": gross,
        "discount": discount,
        "amount_excl_vat": amount_excl_vat,
        "vat": vat,
        "total": total,
        "cost_price": cost_price,
        "profit": profit,
        "profit_margin": profit_margin,
        "words_amount": num_to_words(gross) + " Fils",
        "words_discount": num_to_words(discount) + " Fils" if discount > 0 else "Zero",
        "words_amount_excl_vat": num_to_words(amount_excl_vat) + " Fils",
        "words_vat": num_to_words(vat) + " Fils",
        "words_total": num_to_words(total) + " Fils",
        "bank_name": fake.company() + " Bank",
        "account_name": fake.company().upper() + " LLC SPC",
        "account_number": str(random.randint(10000000000, 99999999999)),
        "iban": f"AE{random.randint(100000000000000000000, 999999999999999999999)}",
        "swift": fake.lexify(text="????").upper() + "AEAE"
    }

# Convert number to words
def num_to_words(num):
    whole = int(num)
    frac = int(round((num - whole) * 100))
    
    units = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    
    if whole == 0:
        result = "Zero"
    elif whole < 10:
        result = units[whole]
    elif whole < 20:
        result = teens[whole - 10]
    elif whole < 100:
        result = tens[whole // 10]
        if whole % 10 > 0:
            result += "-" + units[whole % 10]
    elif whole < 1000:
        result = units[whole // 100] + " Hundred"
        if whole % 100 > 0:
            if whole % 100 < 10:
                result += " And " + units[whole % 100]
            elif whole % 100 < 20:
                result += " And " + teens[whole % 100 - 10]
            else:
                result += " And " + tens[(whole % 100) // 10]
                if (whole % 100) % 10 > 0:
                    result += "-" + units[(whole % 100) % 10]
    else:
        result = num_to_words(whole // 1000) + " Thousand"
        if whole % 1000 > 0:
            if whole % 1000 < 100:
                result += " And " + num_to_words(whole % 1000)
            else:
                result += " " + num_to_words(whole % 1000)
    
    # Add fractional part if exists
    if frac > 0:
        result += " And " + num_to_words(frac)
    
    return result

def create_invoice_pdf(data, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=15*mm, bottomMargin=15*mm, leftMargin=15*mm, rightMargin=15*mm)
    elements = []
    
    # Header section
    header_data = [
        [
            Paragraph("<b>ORGANIC HONEY FARM</b>", heading_style),
            Paragraph("<b>Tax Invoice</b>", heading_style),
            Paragraph("ORGANIC FARMS (L.L.C)<br/>22 Floor,Business Tower<br/>P.O. Box 7777, CITY<br/>Tel: 922222808 Fax: 922222808<br/>TRN: 100610585000003", small_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[90*mm, 50*mm, 55*mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),
        ('ALIGN', (2, 0), (2, 0), 'RIGHT'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 5*mm))
    
    # Customer Details box
    customer_data = [
        [Paragraph("<b>Customer Details</b>", normal_style), "", "", "", ""],
        [
            Paragraph(f"<b>Customer Name:</b> {data['customer_name']}", small_style),
            "",
            "",
            Paragraph(f"<b>Date:</b>", small_style),
            Paragraph(f"{data['invoice_date']}", small_style)
        ],
        [
            Paragraph(f"<b>Customer ID:</b> {data['customer_id']}", small_style),
            "",
            Paragraph(f"<b>TRN:</b> {data['trn']}", small_style),
            Paragraph(f"<b>Tax Invoice No:</b>", small_style),
            Paragraph(f"{data['invoice_no']}", small_style)
        ],
        [
            Paragraph(f"<b>Address:</b> {data['address_line1']}<br/>{data['address_line2']}<br/>{data['address_line3']}", small_style),
            "",
            "",
            Paragraph(f"<b>Ref:</b>", small_style),
            Paragraph(f"{data['ref']}", small_style)
        ],
        [
            Paragraph(f"<b>Customer Type:</b> {data['customer_type']}", small_style),
            "",
            "",
            Paragraph(f"<b>DO No:</b>", small_style),
            Paragraph(f"{data['do_no']}", small_style)
        ],
        [
            Paragraph(f"<b>Tel:</b> {data['phone']}", small_style),
            Paragraph(f"<b>Fax:</b>", small_style),
            "",
            Paragraph(f"<b>Customer PO No:</b>", small_style),
            Paragraph(f"{data['customer_po']}", small_style)
        ],
        [
            Paragraph(f"<b>Payment Status:</b> {data['payment_status']}", small_style),
            "",
            "",
            Paragraph(f"<b>Due Date:</b>", small_style),
            Paragraph(f"{data['due_date']}", small_style)
        ],
        [
            "",
            "",
            "",
            Paragraph(f"<b>Payment term:</b>", small_style),
            Paragraph(f"{data['payment_term']}", small_style)
        ]
    ]
    
    customer_table = Table(customer_data, colWidths=[40*mm, 30*mm, 40*mm, 30*mm, 45*mm])
    customer_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
        ('SPAN', (0, 0), (-1, 0)),  # Title spans all columns
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('SPAN', (0, 1), (2, 1)),  # Customer name spans 3 columns
        ('SPAN', (0, 3), (2, 3)),  # Address spans 3 columns
        ('SPAN', (0, 4), (2, 4)),  # Customer type
        ('SPAN', (0, 6), (2, 6)),  # Payment status
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(customer_table)
    elements.append(Spacer(1, 5*mm))
    
    # Invoice items table
    table_headers = [
        Paragraph("<b>Sr. No</b>", table_heading),
        Paragraph("<b>Item Description</b>", table_heading),
        Paragraph("<b>UOM</b>", table_heading),
        Paragraph("<b>QTY</b>", table_heading),
        Paragraph("<b>Unit Rate</b>", table_heading),
        Paragraph("<b>Gross Amount</b>", table_heading),
        Paragraph("<b>Discount</b>", table_heading),
        Paragraph("<b>Amount Excl. VAT</b>", table_heading),
        Paragraph("<b>VAT 5%</b>", table_heading),
        Paragraph("<b>Amount Incl. VAT</b>", table_heading)
    ]
    
    item_data = [
        table_headers,
        [
            "1",
            data['product'],
            data['uom'],
            data['qty'],
            f"{data['unit_price']:.2f}",
            f"{data['gross']:.2f}",
            f"{data['discount']:.2f}",
            f"{data['amount_excl_vat']:.2f}",
            f"{data['vat']:.2f}",
            f"{data['total']:.2f}"
        ],
        [
            "",
            "Total",
            "",
            "",
            "",
            f"{data['gross']:.2f}",
            f"{data['discount']:.2f}",
            f"{data['amount_excl_vat']:.2f}",
            f"{data['vat']:.2f}",
            f"{data['total']:.2f}"
        ]
    ]
    
    items_table = Table(item_data, colWidths=[10*mm, 40*mm, 10*mm, 10*mm, 20*mm, 20*mm, 20*mm, 20*mm, 15*mm, 20*mm])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(items_table)
    elements.append(Spacer(1, 5*mm))
    
    # Profitability Info - Only for internal use
    profit_data = [
        [Paragraph("<b>Profitability Information (Internal Use Only)</b>", small_style)],
        [Paragraph(f"Cost Price: AED {data['cost_price']:.2f} | Profit: AED {data['profit']:.2f} | Profit Margin: {data['profit_margin']}%", small_style)]
    ]
    
    profit_table = Table(profit_data, colWidths=[185*mm])
    profit_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (0, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(profit_table)
    elements.append(Spacer(1, 5*mm))
    
    # Totals Section
    total_data = [
        [
            Paragraph(f"Total Gross Amount (Dirham {data['words_amount']})", small_style),
            "AED",
            Paragraph(f"{data['gross']:.2f}", normal_style)
        ],
        [
            Paragraph(f"Discount (Dirham {data['words_discount']})", small_style),
            "AED",
            Paragraph(f"{data['discount']:.2f}", normal_style)
        ],
        [
            Paragraph(f"Total Excluding VAT (Dirham {data['words_amount_excl_vat']})", small_style),
            "AED",
            Paragraph(f"{data['amount_excl_vat']:.2f}", normal_style)
        ],
        [
            Paragraph(f"5% Total VAT (Dirham {data['words_vat']})", small_style),
            "AED",
            Paragraph(f"{data['vat']:.2f}", normal_style)
        ],
        [
            Paragraph(f"Total with VAT (Dirham {data['words_total']})", small_style),
            "AED",
            Paragraph(f"{data['total']:.2f}", normal_style)
        ]
    ]
    
    totals_table = Table(total_data, colWidths=[130*mm, 20*mm, 35*mm])
    totals_table.setStyle(TableStyle([
        ('ALIGN', (1, 0), (2, -1), 'RIGHT'),
        ('LINEABOVE', (0, -1), (-1, -1), 1, colors.black),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 5*mm))
    
    # Bank Details
    bank_details = f"""<b>Bank Details:</b>
A/C Payee: {data['account_name']}
Account Number: {data['account_number']}
IBAN: {data['iban']}
Bank: {data['bank_name']}
SWIFT: {data['swift']}"""
    
    elements.append(Paragraph(bank_details, small_style))
    elements.append(Spacer(1, 5*mm))
    
    # Terms and Conditions
    terms = """01. Customers are requested to check the goods at time of delivery, no claim for any shortage or damage will be accepted after delivery.
02. Once sold, goods will not be taken back.
03. Goods may be checked for expiry date at the time of taking delivery, our company will not be liable for unsold goods whether before or after the expiry date.
04. Payments have to be made from the same account name mentioned on the invoice under bill to section."""
    elements.append(Paragraph(terms, small_style))
    elements.append(Spacer(1, 5*mm))
    
    # Confirmation
    elements.append(Paragraph("We hereby confirm that we have received the goods in good condition as per this invoice.", small_style))
    elements.append(Spacer(1, 10*mm))
    
    # Signature section
    sig_data = [
        [
            Paragraph("Stamp ____________________", small_style),
            Paragraph("Signature ____________________", small_style)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[95*mm, 95*mm])
    elements.append(sig_table)
    elements.append(Spacer(1, 20*mm))
    
    # Footer
    elements.append(Paragraph("This is system generated Invoice/Credit note. Signature is not required.", small_style))
    elements.append(Spacer(1, 5*mm))
    elements.append(Paragraph("Page 1 of 1", small_style))
    
    # Build the PDF
    doc.build(elements)

# Function to generate multiple invoices and export for dashboard analysis
def generate_invoices_for_dashboard(num_invoices=100):
    """Generate sample invoice data for dashboard visualization and export to PDFs"""
    print(f"Generating {num_invoices} invoices in: {output_dir}")
    
    # Create a list to store all invoice data for potential CSV export
    all_invoice_data = []
    
    for i in range(num_invoices):
        data = generate_invoice_data()
        pdf_path = os.path.join(output_dir, f"invoices-{i+1}.pdf")
        create_invoice_pdf(data, pdf_path)
        
        # Store data for analysis
        all_invoice_data.append(data)
        
        print(f"Generated invoices_{i+1}.pdf")
    
    print(f"All invoices generated successfully in: {output_dir}")
    
    # Optional: Export invoice data to CSV for dashboard analysis
    # import pandas as pd
    # df = pd.DataFrame(all_invoice_data)
    # df.to_csv(os.path.join(output_dir, 'invoice_data.csv'), index=False)
    
    return all_invoice_data

# Generate invoices
invoice_data = generate_invoices_for_dashboard(10)