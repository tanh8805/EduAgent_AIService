from pypdf import PdfReader

def read_pdf(file):
    try:
        reader = PdfReader(file)
        raw_text = ""
        for page_num,page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text + '\n'
        return raw_text.strip()
    except Exception as e:
        print(f"Error in reading pdf file: {e}")
        return ""