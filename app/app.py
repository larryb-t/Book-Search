from flask import Flask, render_template, request
from src.index import InvertedIndex
from src.ingest import find_pg_catalog, load_pg_catalog
from src.ingest.kaggle_books import find_books_csv, load_books_csv
from src.rank import search

dataset_root = '.'

pg_path = find_pg_catalog(dataset_root)
kaggle_path = find_books_csv(dataset_root)

records = []
if pg_path:
    records = load_pg_catalog(pg_path, filter_english=True)
    dataset_name = 'pg_catalog'
elif kaggle_path:
    records = load_books_csv(kaggle_path)
    dataset_name = 'kaggle_books'
else:
    dataset_name = None

index = InvertedIndex.build(records)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    if request.method == "POST":
        query = request.form["query"]
        results = search(index, query, top_k=10)
        # print([(r.book_id, r.display.get('title', ''), r.display.get('authors', ''), r.score) for r in results])

    return render_template("index.html", results=results)

app.run(debug=True)
