from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os, logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
app = Flask(__name__)

try:
    d = pd.read_csv('coursera_output.csv', encoding='utf-8')
    log.info(f"CSV loaded: {len(d)} records")
    cols = ['Course Name','University','Difficulty Level','Course Rating','Course URL','Course Description','Skills']
    for c in cols:
        if c not in d.columns: log.warning(f"'{c}' not found")
    txt_cols = ['Course Name','University','Course Description','Skills']
    for c in txt_cols:
        if c in d.columns:
            d[c] = d[c].astype(str).str.replace(r'\?', "'", regex=True).str.replace(r"\'", "'", regex=True).str.replace(r"\\", "", regex=True)
    
    if 'Keywords' not in d.columns:
        log.info("Creating 'Keywords'")
        d['Keywords'] = d['Course Name'].str.lower() + ' ' + d['Skills'].str.lower()
        d['Keywords'] = d['Keywords'].str.replace(r'\?', "'", regex=True).str.replace(r"\'", "'", regex=True).str.replace(r"\\", "", regex=True)
    
    if 'Course URL' in d.columns:
        d['Course URL'] = d['Course URL'].apply(lambda u: u if u.startswith(('http://','https://')) else f'https://{u}' if u else '')
    
    v = TfidfVectorizer(stop_words='english')
    X = v.fit_transform(d['Keywords'])
    log.info(f"TF-IDF done: {X.shape}")
except Exception as e:
    log.error(f"Data load error: {str(e)}")
    d = pd.DataFrame(columns=['Course Name','University','Difficulty Level','Course Rating','Course URL','Course Description','Skills','Keywords'])
    X = None
    v = None

@app.route('/')
def home(): return render_template('index.html')


@app.route('/api/recommend', methods=['POST'])
def api():
    try:
        if request.is_json:

            q = request.json.get('query','')
            diff = request.json.get('difficulty','all')
            r = float(request.json.get('rating',0))
            log.info(f"Query: '{q}', Difficulty: '{diff}', Rating: {r}")


            if X is None or v is None: return jsonify([]),500
            res = f(q,diff,r)
            for c in res.columns:
                if res[c].dtype == 'object': res[c] = res[c].fillna('')
            if 'Course Rating' in res.columns:
                res['Course Rating'] = res['Course Rating'].apply(lambda x: round(float(x),1))
            return jsonify(res.to_dict('records'))
        

        return jsonify({"error":"Request must be JSON"}),400
    except Exception as e:
        log.error(f"/api/recommend error: {str(e)}")
        return jsonify({"error":str(e)}),500

def f(q,diff='all',r=0,n=10):
    try:
        if not q.strip():
            log.info("Empty input, fallback to top rated")
            df = d.copy()
            if diff!='all': df = df[df['Difficulty Level']==diff]
            if r>0: df = df[df['Course Rating']>=r]
            if len(df)==0: df = d.copy()
            df = df.sort_values(by='Course Rating',ascending=False).head(n)
            df['similarity'] = 0.0
            return df
        
        log.info(f"Vectorizing: '{q}'")
        qv = v.transform([q])
        sims = cosine_similarity(qv,X).flatten()
        i = sims.argmax()
        top = d.iloc[i]
        log.info(f"Top match: '{top['Course Name']}' @ {sims[i]:.4f}")

        if 'Cluster' in d.columns:
            c = top['Cluster']
            df = d[d['Cluster']==c].copy()
            log.info(f"{len(df)} in cluster {c}")
        else:
            df = d.copy()
        if diff!='all':
            df = df[df['Difficulty Level']==diff]
            log.info(f"Filter diff='{diff}' → {len(df)}")
        if r>0:
            df = df[df['Course Rating']>=r]
            log.info(f"Filter rating>={r} → {len(df)}")
        if len(df)==0:
            log.warning("Empty after filters, reverting")
            df = d.copy()
            if r>0: df = df[df['Course Rating']>=r]
        idx = df.index
        df = df.copy()
        df['similarity'] = sims[idx]
        return df.sort_values(by='similarity',ascending=False).head(n)
    
    except Exception as e:
        log.error(f"Rec error: {str(e)}")
        r = pd.DataFrame(columns=d.columns)
        r['similarity'] = []
        return r

if __name__ == '__main__':
    app.run(debug=True)
