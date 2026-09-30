from pdf_loader import extract_text_from_pdf


pages = extract_text_from_pdf(
    "sample.pdf"
)


for page in pages:

    print(
        f"\n--- Page {page['page']} ---\n"
    )

    print(
        page["text"][:500]
    )