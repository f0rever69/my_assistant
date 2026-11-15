import io
from zipfile import ZipFile

import pytest
from my_assistant.config.workspace import Workspace
from my_assistant.rag.documents import DocumentError, extract_passages, read_source


def test_text_locations_and_unmodified_chinese(tmp_path):
    path = tmp_path / "制度.md"
    path.write_bytes("# 报销\r\n\r\n全角金额：１００元\r\n下一行\r\n".encode())
    workspace = Workspace.from_path(tmp_path)
    raw, version = read_source(workspace, "制度.md")
    assert len(version) == 64
    passage = extract_passages(".md", raw)[1]
    assert passage.location == "行 3–4"
    assert passage.text == "全角金额：１００元\n下一行"
    (tmp_path / ".env.txt").write_text("private")
    with pytest.raises(PermissionError):
        read_source(workspace, ".env.txt")
    with pytest.raises(PermissionError):
        read_source(workspace, "../outside.txt")


def test_docx_locations_and_missing_text():
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p/><w:p><w:r><w:t>提交日期</w:t></w:r></w:p></w:body></w:document>",
        )
    [passage] = extract_passages(".docx", data.getvalue())
    assert (passage.location, passage.text) == ("第 2 段", "提交日期")
    with pytest.raises(DocumentError, match="no_extractable_text"):
        extract_passages(".txt", b"\n ")


def test_pdf_text_after_blank_page_has_correct_page_number():
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources "
            b"<< /Font << /F1 5 0 R >> >> /Contents 6 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length 37 >>\nstream\nBT /F1 12 Tf 10 100 Td (Policy) Tj ET\nendstream",
    ]
    pdf = b"%PDF-1.4\n"
    for i, obj in enumerate(objects, 1):
        pdf += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    pdf += b"trailer\n<< /Root 1 0 R /Size 7 >>\n%%EOF\n"
    [passage] = extract_passages(".pdf", pdf)
    assert passage.location == "第 2 页"
    assert passage.text.strip() == "Policy"
