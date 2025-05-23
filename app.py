from flask import Flask, render_template, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
from handwriting_model import recognize_text
from PIL import Image, ImageDraw
import pdf2image
from fpdf import FPDF
from docx import Document

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    recognized_text = ""
    filename = ""
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(file_path)

            text_output = ""
            if filename.endswith(".pdf"):
                images = pdf2image.convert_from_path(file_path)
                for i, img in enumerate(images):
                    temp_path = os.path.join(UPLOAD_FOLDER, f"page_{i}.jpg")
                    img.save(temp_path)
                    text_output += recognize_text(temp_path) + "\n"
            else:
                text_output = recognize_text(file_path)

            recognized_text = text_output

            # Save as text for fallback
            with open(os.path.join(OUTPUT_FOLDER, "output.txt"), "w") as f:
                f.write(text_output)

            # Save as PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in text_output.split('\n'):
                pdf.multi_cell(0, 10, line)
            pdf_path = os.path.join(OUTPUT_FOLDER, "output.pdf")
            pdf.output(pdf_path)

            # Save as DOCX
            doc = Document()
            doc.add_paragraph(text_output)
            doc_path = os.path.join(OUTPUT_FOLDER, "output.docx")
            doc.save(doc_path)

            # Save as image
            img = Image.new('RGB', (1000, 800), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 10), text_output, fill=(0, 0, 0))
            img_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
            img.save(img_path)

    return render_template('index.html', recognized_text=recognized_text, filename=filename)

@app.route('/outputs/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)