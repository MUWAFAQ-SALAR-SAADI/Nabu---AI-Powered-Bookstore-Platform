import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

import gradio as gr

# ------------------------------
# 1. إعداد مسارات الملفات وتجهيز البيانات
project_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

books_file = os.path.join(project_dir, "books_with_emotions.csv")
tagged_file = os.path.join(project_dir, "tagged_description.txt")

if not os.path.exists(books_file) or not os.path.exists(tagged_file):
    raise FileNotFoundError("❌ تأكد من وجود الملفات books_with_emotions.csv و tagged_description.txt في نفس مجلد المشروع!")

books = pd.read_csv(books_file)
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "https://via.placeholder.com/300x400?text=No+Cover",
    books["large_thumbnail"],
)

# ------------------------------
# 2. تهيئة نموذج البحث والـ Vectorstore
raw_documents = TextLoader(tagged_file, encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, hf_embeddings)

# ------------------------------
# 3. دالة استرجاع التوصيات وبناء البطاقات
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 8) -> pd.DataFrame:
    if not query.strip():
        query = "best books"
        
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]
        
    book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    cards_html = '<div class="cards-grid">'
    
    for idx, row in recommendations.iterrows():
        match_pct = np.random.randint(85, 98)
        price = np.random.choice([39975, 42475, 47475, 49975])
        rating = round(np.random.uniform(4.5, 4.9), 1)
        stores = np.random.randint(10, 18)
        
        authors_split = str(row["authors"]).split(";")
        author_name = authors_split[0] if len(authors_split) > 0 else "Unknown"
        
        # استخدام الكلاسات بدلاً من الستايلات المباشرة لتسهيل إضافة الحركات (Animations)
        cards_html += f'''
        <div class="book-card">
            <div class="card-image-wrapper">
                <img src="{row['large_thumbnail']}" class="card-image" />
                <span class="match-badge">✨ {match_pct}% Match</span>
            </div>
            <div class="card-content">
                <div>
                    <h4 class="card-title">{row['title']}</h4>
                    <p class="card-author">{author_name}</p>
                </div>
                <div>
                    <div class="card-rating">
                        <span>★ {rating} <span class="stores-text">({stores} stores)</span></span>
                    </div>
                    <div class="card-footer">
                        <span class="card-price">{price:,} IQD</span>
                        <button class="view-btn">View</button>
                    </div>
                </div>
            </div>
        </div>
        '''
    cards_html += '</div>'
    return cards_html

# ------------------------------
# 4. التصنيفات والشعور
categories = ["All"] + sorted([c for c in books["simple_categories"].dropna().unique()])
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# ------------------------------
# 5. التنسيق البرمجي (CSS) المحدث بحركات ولمسات تشبه React
custom_css = """
/* الخلفية العامة */
body, .gradio-container {
    background: radial-gradient(circle at top, #111827, #030712) !important;
    color: #F9FAFB !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    min-height: 100vh;
}

/* إخفاء الحاويات البيضاء التي تفرضها Gradio */
.gradio-container .form, .gradio-container .block, .gradio-container .wrap {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* تنسيق حقول الإدخال */
textarea, input, select, .gr-box, .gr-input, .gr-dropdown {
    background-color: rgba(17, 24, 39, 0.7) !important;
    border: 1px solid rgba(55, 65, 81, 0.5) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.3s ease !important;
}
textarea:focus, input:focus, select:focus {
    border-color: #10B981 !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
}

/* تنسيق زر البحث */
button.primary-btn {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}
button.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
}

/* شبكة البطاقات */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 24px;
    margin-top: 20px;
    padding-bottom: 40px;
}

/* البطاقة وحركات التحويم */
.book-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}
.book-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 30px rgba(16, 185, 129, 0.15), 0 0 15px rgba(16, 185, 129, 0.1);
    border-color: rgba(16, 185, 129, 0.3);
}

/* صورة البطاقة وشارة المطابقة */
.card-image-wrapper {
    position: relative;
    width: 100%;
    padding-top: 140%;
    background: #000;
    overflow: hidden;
}
.card-image {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.5s ease;
}
.book-card:hover .card-image {
    transform: scale(1.05);
}
.match-badge {
    position: absolute;
    top: 12px;
    right: 12px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.9), rgba(5, 150, 105, 0.9));
    color: #FFF;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: bold;
    backdrop-filter: blur(4px);
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

/* محتوى البطاقة الداخلي */
.card-content {
    padding: 16px;
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.card-title {
    margin: 0 0 6px 0;
    color: #F9FAFB;
    font-size: 15px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-author {
    margin: 0 0 12px 0;
    color: #9CA3AF;
    font-size: 12px;
}
.card-rating {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    color: #F3F4F6;
    font-size: 13px;
    font-weight: bold;
}
.stores-text {
    color: #6B7280;
    font-weight: normal;
    margin-left: 4px;
}
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 5px;
}
.card-price {
    color: #10B981;
    font-weight: 800;
    font-size: 14px;
}
.view-btn {
    background: rgba(31, 41, 55, 0.8);
    color: #F3F4F6;
    border: 1px solid rgba(75, 85, 99, 0.4);
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}
.view-btn:hover {
    background: #10B981;
    border-color: #10B981;
    color: #FFF;
}

/* إخفاء أزرار الأسهم الجانبية التي ظهرت في حقل النص */
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

/* تنسيقات الهيدر */
.header-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 40px;
}
.logo-container {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 26px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}
.hero-section {
    text-align: center;
    margin: 30px 0 40px 0;
}
.hero-title {
    font-size: 38px;
    font-weight: 800;
    background: linear-gradient(to right, #FFFFFF, #9CA3AF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.hero-sub {
    color: #9CA3AF;
    font-size: 15px;
    max-width: 600px;
    margin: 0 auto 30px auto;
    line-height: 1.6;
}
"""

# الشعار المدمج مطابق 100% للرسم المعتمد
nabu_logo_html = """
<div style="display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #111827, #000); width: 46px; height: 46px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 10px rgba(0,0,0,0.5);">
    <svg width="28" height="28" viewBox="0 0 100 100" fill="#10B981" xmlns="http://www.w3.org/2000/svg">
        <rect x="30" y="15" width="40" height="24" rx="3" />
        <rect x="25" y="41" width="50" height="6" rx="1" />
        <path d="M 25 48 C 10 48 10 70 25 70 C 32 70 36 62 36 55 Z" />
        <path d="M 36 48 H 70 L 62 58 L 56 90 L 40 62 Z" />
        <rect x="50" y="51" width="8" height="3" fill="#000000" rx="1"/>
    </svg>
</div>
"""

# ------------------------------
# 6. بناء الواجهة 
with gr.Blocks(theme=gr.themes.Base(), css=custom_css, title="Nabu - Book Recommender") as dashboard:
    
    gr.HTML(f"""
        <div class="header-nav">
            <div class="logo-container">
                {nabu_logo_html}
                <span>Nabu</span>
            </div>
            <div style="display: flex; align-items: center; gap: 24px;">
                <span style="color: #9CA3AF; font-size: 14px; font-weight: 500; cursor: pointer; transition: color 0.2s;" onmouseover="this.style.color='#FFF'" onmouseout="this.style.color='#9CA3AF'">Browse</span>
                <span style="color: #9CA3AF; font-size: 14px; font-weight: 500; cursor: pointer; transition: color 0.2s;" onmouseover="this.style.color='#FFF'" onmouseout="this.style.color='#9CA3AF'">Bookstores</span>
                <span style="color: #9CA3AF; font-size: 14px; font-weight: 500; cursor: pointer; transition: color 0.2s;" onmouseover="this.style.color='#FFF'" onmouseout="this.style.color='#9CA3AF'">My Lists</span>
                <span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: 8px 18px; border-radius: 10px; color: #FFF; font-size: 14px; font-weight: 600; cursor: pointer; backdrop-filter: blur(4px);">Sign In</span>
            </div>
        </div>
        <div class="hero-section">
            <div class="hero-title">Discover Books from Every Bookstore</div>
            <div class="hero-sub">Search millions of books across all bookstores. Find the best prices, nearest locations, and personalized recommendations.</div>
        </div>
    """)

    with gr.Row():
        user_query = gr.Textbox(
            show_label=False,
            placeholder="🔍 Search for books, authors, or describe a plot...",
            container=False,
            max_lines=1,
            scale=3
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Category", 
            show_label=False, 
            value="All", 
            scale=1, 
            container=False
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Tone", 
            show_label=False, 
            value="All", 
            scale=1, 
            container=False
        )
        submit_button = gr.Button("Search", elem_classes=["primary-btn"], scale=1)

    gr.HTML("<h3 style='color: #FFFFFF; font-size: 22px; font-weight: 700; margin-top: 40px; margin-bottom: 5px;'>Recommended For You</h3><p style='color: #9CA3AF; font-size: 14px; margin-top: 0;'>Based on your semantic search and mood filters</p>")

    output_html = gr.HTML()

    # تحديث النتائج عند ضغط الزر
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_html
    )
    
    # تحميل الكتب تلقائياً عند فتح الصفحة
    dashboard.load(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output_html
    )

# ------------------------------
# 7. تشغيل الواجهة
if __name__ == "__main__":
    dashboard.launch()