#!/usr/bin/env python3
"""
Generate Sample Images for Testing
===================================
Creates synthetic land record images in multiple languages.
"""

import os
import random
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.image_utils import cv_to_pil, pil_to_cv


# Sample data for each language
SAMPLES = {
    "hi": {
        "title": "भूमि अभिलेख",
        "fields": [
            ("खसरा नंबर", "142/2A"),
            ("खतौनी नंबर", "00345"),
            ("मौजा", "रामपुर"),
            ("तहसील", "सदर"),
            ("जिला", "लखनऊ"),
            ("राज्य", "उत्तर प्रदेश"),
            ("रकबा", "0.2520 हेक्टेयर"),
            ("भूमि प्रकार", "कृषि भूमि"),
            ("नामांतरण नंबर", "MUT/2024/001234"),
            ("पंजीकरण दिनांक", "15/03/2024"),
            ("विक्रेता", "राम कुमार शर्मा"),
            ("क्रेता", "श्याम लाल गुप्ता"),
            ("स्टाम्प शुल्क", "₹ 50,000"),
        ]
    },
    "bn": {
        "title": "ভূমি রেকর্ড",
        "fields": [
            ("দাগ নম্বর", "১৪২/২A"),
            ("খতিয়ান নম্বর", "০০৩৪৫"),
            ("মৌজা", "রামপুর"),
            ("উপজেলা", "সদর"),
            ("জেলা", "লکھনৌ"),
            ("রাজ্য", "উত্তর প্রদেশ"),
            ("রকবা", "০.২৫২০ হেক্টেয়ার"),
            ("জমির ধরন", "কৃষি জমি"),
            ("নামজারি নম্বর", "MUT/2024/001234"),
            ("নিবন্ধন তারিখ", "১৫/০৩/২০২৪"),
            ("বিক্রেতা", "রাম কুমার শর্মা"),
            ("ক্রেতা", "শ্যাম লাল গুপ্তা"),
            ("স্ট্যাম্প ডিউটি", "৫০,০০০ টাকা"),
        ]
    },
    "ta": {
        "title": "நில பதிவு",
        "fields": [
            ("தொடர்பு எண்", "142/2A"),
            ("பட்டா எண்", "00345"),
            ("கிராமம்", "ராமபூர்"),
            ("தாலுகா", "சதர்"),
            ("மாவட்டம்", "லக்னவ்"),
            ("மாநிலம்", "உத்தரப் பிரதேசம்"),
            ("விரிவ்", "0.2520 ஹெக்டேர்"),
            ("நில வகை", "விவசாய நிலம்"),
            ("நாமமாற்று எண்", "MUT/2024/001234"),
            ("பதிவு நாள்", "15/03/2024"),
            ("விற்பனையாளர்", "ராம குமார் சர்மா"),
            ("வாங்குபவர்", "சியாம் லால் குப்தா"),
            ("ஸ்டாம்ப் கட்டணம்", "₹ 50,000"),
        ]
    },
    "te": {
        "title": "భూమి రికార్డు",
        "fields": [
            ("సర్వే నంబర్", "142/2A"),
            ("ఖతా నంబర్", "00345"),
            ("గ్రామం", "రామ్పుర్"),
            ("మండలం", "సదర్"),
            ("జిల్లా", "లక్నౌ"),
            ("రాష్ట్రం", "ఉత్తర్ ప్రదేశ్"),
            ("వ్యాసం", "0.2520 హెక్టేర్"),
            ("భూమి రకం", "వ్యవసాయ భూమి"),
            ("నామ మార్పు నంబర్", "MUT/2024/001234"),
            ("నివంధన తేదీ", "15/03/2024"),
            ("విక్రేత", "రామ్ కుమార్ شر్మా"),
            ("క్రేత", "శ్యామ్ లాల్ గుప్తా"),
            ("స్టాంప్ డ్యూటీ", "₹ 50,000"),
        ]
    },
    "kn": {
        "title": "ಭೂಮಿ ಲೇಖ",
        "fields": [
            ("ಸರ್ವೇ ನಂಬರ್", "142/2A"),
            ("ಖಾತಾ ನಂಬರ್", "00345"),
            ("ಗ್ರಾಮ", "ರಾಮಪುರ"),
            ("ತಾಲೂಕು", "ಸದರ್"),
            ("ಜಿಲ್ಲೆ", "ಲಕ್ನೌ"),
            ("ರಾಜ್ಯ", "ಉತ್ತರ ಪ್ರದೇಶ"),
            ("ವಿಸ್ತೀರ್ಣ", "0.2520 ಹೆಕ್ಟೇರ್"),
            ("ಭೂಮಿ ವಿಧ", "ಕೃಷಿ ಭೂಮಿ"),
            ("ನಾಮ ಬದಲಾವಣೆ ನಂಬರ್", "MUT/2024/001234"),
            ("ನೊಂದಣಿ ದಿನಾಂಕ", "15/03/2024"),
            ("ವಿಕ್ರೇತರ", "ರಾಮ್ ಕುಮಾರ್ ಶರ್ಮಾ"),
            ("ಕ್ರೇತರ", "ಶ್ಯಾಮ್ ಲಾಲ್ ಗುಪ್ತಾ"),
            ("ಸ್ಟಾಂಪ್ ಶುಲ್ಕ", "₹ 50,000"),
        ]
    },
    "ml": {
        "title": "ഭൂമി രേഖ",
        "fields": [
            ("സർവേ നമ്പർ", "142/2A"),
            ("പട്ടയം നമ്പർ", "00345"),
            ("ഗ്രാമം", "രാമ്പൂർ"),
            ("താലൂക്ക്", "സദർ"),
            ("ജില്ല", "ലക്നൗ"),
            ("രാജ്യം", "ഉത്തർ പ്രദേശ്"),
            ("വിസ്തീർണ്ണം", "0.2520 ഹെക്ടെയർ"),
            ("നില തരം", "കൃഷി ഭൂമി"),
            ("നാമ മാറ്റം നമ്പർ", "MUT/2024/001234"),
            ("നിബന്ധന തീയതി", "15/03/2024"),
            ("വിക്രേതാവ്", "രാം കുമാർ ശർമ്മ"),
            ("വാങ്ങുന്നവർ", "ശ്യാം ലാൽ ഗുപ്ത"),
            ("സ്റ്റാമ്പ് ഡ്യൂട്ടി", "₹ 50,000"),
        ]
    },
    "mr": {
        "title": "भूमी अभिलेख",
        "fields": [
            ("सर्वे क्रमांक", "142/2A"),
            ("सातबारा क्रमांक", "00345"),
            ("गाव", "रामपूर"),
            ("तालुका", "सदर"),
            ("जिल्हा", "लखनऊ"),
            ("राज्य", "उत्तर प्रदेश"),
            ("क्षेत्रफळ", "0.2520 हेक्टेयर"),
            ("भूमी प्रकार", "कृषी भूमि"),
            ("नामांतर क्रमांक", "MUT/2024/001234"),
            ("नोंदणी दिनांक", "15/03/2024"),
            ("विक्रेता", "राम कुमार शर्मा"),
            ("खरेदीदार", "श्याम लाल गुप्ता"),
            ("स्टाम्प शुल्क", "₹ 50,000"),
        ]
    },
    "gu": {
        "title": "જમીન રેકોર્ડ",
        "fields": [
            ("સર્વે નંબર", "142/2A"),
            ("સાતબારા નંબર", "00345"),
            ("ગામ", "રામપુર"),
            ("તાલુકા", "સદર"),
            ("જિલ્લો", "લખનઊ"),
            ("રાજ્ય", "ઉત્તર પ્રદેશ"),
            ("વિસ્તાર", "0.2520 હેક્ટર"),
            ("જમીન પ્રકાર", "કૃષિ જમીન"),
            ("નામ ફેરફાર નંબર", "MUT/2024/001234"),
            ("નોંધણી તારીખ", "15/03/2024"),
            ("વિક્રેતા", "રામ કુમાર શર્મા"),
            ("ખરીદાર", "શ્યામ લાલ ગુપ્તા"),
            ("સ્ટામ્પ ડ્યુટી", "₹ 50,000"),
        ]
    },
    "pa": {
        "title": "ਜ਼ਮੀਨ ਰਿਕਾਰਡ",
        "fields": [
            ("ਸਰਵੇ ਨੰਬਰ", "142/2A"),
            ("ਜਮਾਬੰਦੀ ਨੰਬਰ", "00345"),
            ("ਗਾਂਵ", "ਰਾਮਪੁਰ"),
            ("ਤਹਿਸੀਲ", "ਸਦਰ"),
            ("ਜ਼ਿਲ੍ਹਾ", "ਲਖਨਊ"),
            ("ਰਾਜ್ಯ", "ਉੱਤਰ ਪ੍ਰਦੇਸ਼"),
            ("ਰਕਬਾ", "0.2520 ਹੈਕਟੇਅਰ"),
            ("ਜ਼ਮੀਨ ਦੀ ਕਿਸਮ", "ਖੇਤੀਬਾੜੀ"),
            ("ਨਾਮ ਬਦਲੀ ਨੰਬਰ", "MUT/2024/001234"),
            ("ਰਜਿਸਟ੍ਰੀ ਤਾਰੀਖ", "15/03/2024"),
            ("ਵਿਕਰੇਤਾ", "ਰਾਮ ਕੁਮਾਰ ਸ਼ਰਮਾ"),
            ("ਖਰੀਦਦਾਰ", "ਸ਼ਿਆਮ ਲਾਲ ਗੁਪਤਾ"),
            ("ਸਟਾਮਪ ਡਿਊਟੀ", "₹ 50,000"),
        ]
    },
    "or": {
        "title": "ଭୂମି ରେକର୍ଡ",
        "fields": [
            ("ଖସରା ନମ୍ବର", "142/2A"),
            ("ଖତିୟାନ ନମ୍ବର", "00345"),
            ("ଗ୍ରାମ", "ରାମପୁର"),
            ("ତହସିଲ", "ସଦର"),
            ("ଜିଲ୍ଲା", "ଲଖନଉ"),
            ("ରାଜ୍ୟ", "ଉତ୍ତର ପ୍ରଦେଶ"),
            ("ରକବା", "0.2520 ହେକ୍ଟେର"),
            ("ଭୂମି ପ୍ରକାର", "କୃଷି ଭୂମି"),
            ("ନାମାନ୍ତରଣ ନମ୍ବର", "MUT/2024/001234"),
            ("ନିବନ୍ଧନ ତାରିଖ", "15/03/2024"),
            ("ବିକ୍ରେତା", "ରାମ କୁମାର ଶର୍ମା"),
            ("କ୍ରେତା", "ଶ୍ୟାମ ଲାଲ ଗୁପ୍ତା"),
            ("ଷ୍ଟାମ୍ପ ଶୁଳ୍କ", "ରୂ 50,000"),
        ]
    },
    "as": {
        "title": "ভূমি ৰেকର୍ଡ",
        "fields": [
            ("দাগ নম্বৰ", "142/2A"),
            ("খতিয়ান নম্বৰ", "00345"),
            ("গাঁও", "ৰামপুৰ"),
            ("মহকুমা", "সদৰ"),
            ("জিলা", "লখনؤ"),
            ("ৰাজ্য", "উত্তৰ প্ৰদেশ"),
            ("কښেত্ৰফল", "0.2520 হেক্টেৰ"),
            ("ভূমি প্ৰকাৰ", "কৃষি ভূমি"),
            ("নামজাৰি নম্বৰ", "MUT/2024/001234"),
            ("নিবন্ধন তাৰিখ", "15/03/2024"),
            ("বিক্ৰেতা", "ৰাম কুমাৰ শৰ্মা"),
            ("ক্ৰেতা", "শ্যাম লাল গুপ্ত"),
            ("ষ্টাম্প ডিউটি", "₹ 50,000"),
        ]
    },
    "ur": {
        "title": "زمین ریکارڈ",
        "fields": [
            ("سرvey نمبر", "142/2A"),
            ("ختنون نمبر", "00345"),
            ("موعزہ", "رام پور"),
            ("تحصیل", "صدر"),
            ("ضلع", "لکھنؤ"),
            ("صوبہ", "شمالی صوبہ"),
            ("رقبہ", "0.2520 ہیکٹر"),
            ("اراضی کی قسم", "زرعی زمین"),
            ("انتقال نمبر", "MUT/2024/001234"),
            ("رجسٹری تاریخ", "15/03/2024"),
            ("فروخت کنندہ", "رام کمار شرما"),
            ("خریدار", "شیام لال گپتا"),
            ("سٹامپ ڈیوٹی", "₹ 50,000"),
        ]
    },
    "en": {
        "title": "Land Record",
        "fields": [
            ("Survey Number", "142/2A"),
            ("Khata Number", "00345"),
            ("Village", "Rampur"),
            ("Tehsil", "Sadar"),
            ("District", "Lucknow"),
            ("State", "Uttar Pradesh"),
            ("Area", "0.2520 hectare"),
            ("Land Type", "Agricultural"),
            ("Mutation Number", "MUT/2024/001234"),
            ("Registration Date", "15/03/2024"),
            ("Seller", "Ram Kumar Sharma"),
            ("Buyer", "Shyam Lal Gupta"),
            ("Stamp Duty", "₹ 50,000"),
        ]
    }
}


def get_font(size=24):
    """Get a suitable font"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMalayalam-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansGujarati-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansGurmukhi-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/mangal.ttf",
    ]
    
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                pass
    
    return ImageFont.load_default()


def create_land_record_image(lang: str, size=(1000, 1400), add_noise=True) -> np.ndarray:
    """Create a synthetic land record image"""
    sample = SAMPLES.get(lang, SAMPLES["en"])
    
    # Create blank image with slight texture
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 245
    
    # Add paper texture
    if add_noise:
        texture = np.random.normal(0, 3, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    
    pil_img = cv_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    
    font_title = get_font(36)
    font_field = get_font(28)
    font_small = get_font(20)
    
    # Draw title
    y = 50
    draw.text((60, y), sample["title"], fill=(20, 20, 20), font=font_title)
    y += 60
    
    # Draw separator line
    draw.line([(60, y), (size[0] - 60, y)], fill=(100, 100, 100), width=2)
    y += 30
    
    # Draw fields
    for label, value in sample["fields"]:
        # Label
        draw.text((80, y), label + ":", fill=(40, 40, 40), font=font_field)
        # Value
        draw.text((400, y), value, fill=(0, 0, 0), font=font_field)
        y += 45
    
    # Add footer
    y += 20
    draw.line([(60, y), (size[0] - 60, y)], fill=(100, 100, 100), width=1)
    y += 20
    draw.text((80, y), "Government of India - Digital Land Records", fill=(100, 100, 100), font=font_small)
    
    # Add noise and blur for realism
    if add_noise:
        img_arr = pil_to_cv(pil_img)
        
        # Slight rotation
        angle = random.uniform(-1, 1)
        h, w = img_arr.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img_arr = cv2.warpAffine(img_arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Perspective transform (simulate photo angle)
        if random.random() < 0.3:
            pts1 = np.float32([[0,0], [w,0], [w,h], [0,h]])
            offset = 10
            pts2 = np.float32([
                [random.randint(-offset, offset), random.randint(-offset, offset)],
                [w + random.randint(-offset, offset), random.randint(-offset, offset)],
                [w + random.randint(-offset, offset), h + random.randint(-offset, offset)],
                [random.randint(-offset, offset), h + random.randint(-offset, offset)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img_arr = cv2.warpPerspective(img_arr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Gaussian noise
        noise = np.random.normal(0, 5, img_arr.shape).astype(np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Slight blur
        if random.random() < 0.5:
            img_arr = cv2.GaussianBlur(img_arr, (3, 3), 0.5)
        
        return img_arr
    
    return pil_to_cv(pil_img)


def main():
    parser = argparse.ArgumentParser(description="Generate sample land record images")
    parser.add_argument("--output-dir", default="data/samples", help="Output directory")
    parser.add_argument("--languages", nargs="+", default=list(SAMPLES.keys()))
    parser.add_argument("--per-language", type=int, default=5, help="Images per language")
    parser.add_argument("--clean", action="store_true", help="Generate clean images without noise")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 Generating {args.per_language} samples per language...")
    print(f"   Languages: {args.languages}")
    print(f"   Output: {output_dir}")
    
    total = 0
    for lang in args.languages:
        lang_dir = output_dir / lang
        lang_dir.mkdir(exist_ok=True)
        
        for i in range(args.per_language):
            img = create_land_record_image(lang, add_noise=not args.clean)
            
            filename = f"{lang}_sample_{i+1:03d}.jpg"
            filepath = lang_dir / filename
            cv2.imwrite(str(filepath), img)
            
            # Also save ground truth text
            sample = SAMPLES[lang]
            text_lines = [sample["title"]] + [f"{k}: {v}" for k, v in sample["fields"]]
            gt_text = "\n".join(text_lines)
            
            txt_path = lang_dir / f"{lang}_sample_{i+1:03d}.txt"
            txt_path.write_text(gt_text, encoding="utf-8")
            
            # Language tag
            lang_path = lang_dir / f"{lang}_sample_{i+1:03d}.lang"
            lang_path.write_text(lang)
            
            total += 1
        
        print(f"   ✅ {lang}: {args.per_language} images")
    
    print(f"\n🎉 Generated {total} total images in {output_dir}")
    print("   Each image has corresponding .txt (ground truth) and .lang files")


if __name__ == "__main__":
    import argparse
    main()