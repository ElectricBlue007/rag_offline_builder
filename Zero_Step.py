import os
import fitz  # PyMuPDF
from pathlib import Path

def crop_pdf_remove_header_footer_and_text(
    input_pdf,
    output_pdf,
    top_margin_percent=0.08,
    bottom_margin_percent=0.08,
    remove_texts=None
):
    if remove_texts is None:
        remove_texts = []
    
    doc = fitz.open(input_pdf)

    for page in doc:
        # ۱) تنظیم CropBox جدید برای حذف هدر و فوتر
        rect = page.rect
        page_height = rect.height

        top_crop = page_height * top_margin_percent
        bottom_crop = page_height * bottom_margin_percent

        new_rect = fitz.Rect(
            rect.x0,
            rect.y0 + top_crop,
            rect.x1,
            rect.y1 - bottom_crop
        )
        page.set_cropbox(new_rect)

        # ۲) پیدا کردن و حذف عبارت‌ها از متن اصلی (با استفاده از quads برای دقت بیشتر)
        for text in remove_texts:
            # quads=True کمک می‌کند تا متن‌های زاویه‌دار یا کمی بهم‌ریخته بهتر پیدا شوند
            areas = page.search_for(text, quads=True)
            for area in areas:
                page.add_redact_annot(area, fill=(1, 1, 1))  # رنگ سفید

        # اعمال رداکشن‌های متن اصلی
        page.apply_redactions()

        # ۳) پیدا کردن و حذف عبارت‌ها از درون Textboxها (Annotations)
        for annot in page.annots():
            # بررسی محتوای متنی یادداشت/تکست‌باکس
            annot_content = annot.info.get("content", "")
            if annot_content:
                # اگر یکی از عبارت‌ها داخل این تکست‌باکس بود، کل آن را حذف کن
                if any(text.lower() in annot_content.lower() for text in remove_texts):
                    page.delete_annot(annot)

        # ۴) پیدا کردن و حذف عبارت‌ها از درون فیلدهای فرم (Widgets)
        for widget in page.widgets():
            widget_content = widget.field_value
            if widget_content and isinstance(widget_content, str):
                if any(text.lower() in widget_content.lower() for text in remove_texts):
                    page.delete_widget(widget)

    # ۵) ذخیره فایل با پارامترهای بهینه‌سازی
    # garbage=4 باعث می‌شود تمام اطلاعات رداکت شده و غیرقابل دسترس، برای همیشه از حجم فایل پاک شوند
    doc.save(output_pdf, garbage=4, deflate=True)
    doc.close()


def process_pdfs_recursive(input_folder, output_folder, remove_texts=None):
    """
    جستجوی بازگشتی (recursive) در تمام زیرپوشه‌ها برای یافتن فایل‌های PDF،
    پردازش آن‌ها و ذخیره با حفظ ساختار دقیق پوشه‌ها در پوشه خروجی.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # اطمینان از وجود پوشه اصلی خروجی
    output_path.mkdir(parents=True, exist_ok=True)

    # rglob("*.pdf") تمام فایل‌های PDF را در پوشه اصلی و تمام زیرپوشه‌ها پیدا می‌کند
    # (case-insensitive در ویندوز، و با استفاده از rglob در لینوکس/مک هم به خوبی کار می‌کند)
    pdf_files = list(input_path.rglob("*.pdf")) + list(input_path.rglob("*.PDF"))
    
    # حذف تکراری‌ها (در صورت وجود)
    pdf_files = list(set(pdf_files))

    for pdf_file in pdf_files:
        # محاسبه مسیر نسبی فایل نسبت به پوشه ورودی
        # مثال: اگر input_folder="PDF-Spanish" و فایل="PDF-Spanish/SubFolder/doc.pdf" باشد
        # relative_path خواهد بود: "SubFolder/doc.pdf"
        relative_path = pdf_file.relative_to(input_path)
        
        # ساخت مسیر کامل فایل در پوشه خروجی
        out_file = output_path / relative_path
        
        # ساختن تمام زیرپوشه‌های مورد نیاز در پوشه خروجی (اگر وجود نداشته باشند)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            crop_pdf_remove_header_footer_and_text(
                input_pdf=str(pdf_file),
                output_pdf=str(out_file),
                top_margin_percent=0.08,
                bottom_margin_percent=0.08,
                remove_texts=remove_texts
            )
            print(f"✅ پردازش شد: {out_file}")
        except Exception as e:
            print(f"❌ خطا در فایل {pdf_file}: {e}")


if __name__ == "__main__":
    input_folder = "./input"
    output_folder = "Output-pdf"
    
    # عبارت‌هایی که می‌خواهید حذف شوند (به حروف کوچک و بزرگ حساسیت نشان دادیم)
    remove_texts = [
        "© THALES NEDERLAND B.V. y/o sus proveedores Sujeto a las limitaciones de la leyenda de la página de título"
    ]

    print(f"شروع جستجو و پردازش فایل‌های PDF در '{input_folder}'...")
    process_pdfs_recursive(input_folder, output_folder, remove_texts=remove_texts)
    print("✅ عملیات با موفقیت به پایان رسید!")