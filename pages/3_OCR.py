import os
import tempfile
import streamlit as st
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import ocrmypdf

# --- Streamlit UI ---
st.set_page_config(page_title="PDF OCR Converter", layout="centered")
st.title("📄 PDF OCR Tool with docTR & OCRmyPDF")
st.markdown("Upload a scanned PDF and get back a searchable PDF and Markdown text.")

uploaded_file = st.file_uploader("Upload your scanned PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_pdf_path = os.path.join(tmpdir, "input.pdf")
        output_pdf_path = os.path.join(tmpdir, "searchable_output.pdf")
        markdown_path = os.path.join(tmpdir, "output.md")

        # Save uploaded PDF to temporary path
        with open(input_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info("🔍 Performing OCR using docTR...")
        try:
            doc = DocumentFile.from_pdf(input_pdf_path)
            predictor = ocr_predictor(pretrained=True)
            result = predictor(doc)

            extracted_text = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join([word.value for word in line.words])
                        extracted_text += line_text + "\n"
                    extracted_text += "\n"
                extracted_text += "\n---\n\n"
        except Exception as e:
            st.error(f"docTR OCR failed: {e}")
            st.stop()

        st.success("✅ Text extraction complete!")

        st.info("📄 Creating searchable PDF using OCRmyPDF...")
        try:
            ocrmypdf.ocr(input_pdf_path, output_pdf_path, lang="eng", deskew=True,
                         output_type="pdf", optimize=0, force_ocr=True)
        except Exception as e:
            st.error(f"OCRmyPDF failed: {e}")
            st.stop()

        st.success("✅ Searchable PDF created!")

        markdown_content = f"# Extracted Content from Uploaded PDF\n\n{extracted_text}\n\n*Generated using docTR and OCRmyPDF.*"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Display download buttons
        with open(output_pdf_path, "rb") as f:
            st.download_button("📥 Download Searchable PDF", f, file_name="searchable_output.pdf", mime="application/pdf")

        with open(markdown_path, "rb") as f:
            st.download_button("📥 Download Markdown File", f, file_name="output.md", mime="text/markdown")
