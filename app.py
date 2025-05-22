from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
import os, logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
app = Flask(__name__)

def safe_float(x):
    try:
        if pd.isna(x) or x == '' or str(x).lower() in ['not calibrated', 'n/a', 'none']:
            return 0.0
        return round(float(x), 1)
    except (ValueError, TypeError):
        return 0.0

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
    
    if 'Course Rating' in d.columns:
        d['Course Rating'] = d['Course Rating'].apply(safe_float)
        log.info(f"Cleaned Course Rating - min: {d['Course Rating'].min()}, max: {d['Course Rating'].max()}")
    
    if 'Keywords' not in d.columns:
        log.info("Creating 'Keywords'")
        d['Keywords'] = d['Course Name'].str.lower() + ' ' + d['Skills'].str.lower()
        d['Keywords'] = d['Keywords'].str.replace(r'\?', "'", regex=True).str.replace(r"\'", "'", regex=True).str.replace(r"\\", "", regex=True)
    
    if 'Course URL' in d.columns:
        d['Course URL'] = d['Course URL'].apply(lambda u: u if u.startswith(('http://','https://')) else f'https://{u}' if u else '')
    
    v = TfidfVectorizer(stop_words='english', max_features=5000)
    X = v.fit_transform(d['Keywords'])
    log.info(f"TF-IDF done: {X.shape}")
    
    knn = NearestNeighbors(n_neighbors=min(15, len(d)), metric='cosine', algorithm='brute')
    knn.fit(X)
    log.info("KNN model trained")
    
    if 'Cluster' in d.columns:
        unique_clusters = d['Cluster'].unique()
        log.info(f"Found {len(unique_clusters)} clusters")
    else:
        log.warning("No 'Cluster' column found")

except Exception as e:
    log.error(f"Data load error: {str(e)}")
    d = pd.DataFrame(columns=['Course Name','University','Difficulty Level','Course Rating','Course URL','Course Description','Skills','Keywords'])
    X = None
    v = None
    knn = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def api():
    try:
        if request.is_json:
            q = request.json.get('query','')
            diff = request.json.get('difficulty','all')
            r = float(request.json.get('rating',0))
            n = int(request.json.get('num_results', 10))
            log.info(f"Query: '{q}', Difficulty: '{diff}', Rating: {r}")

            if X is None or v is None or knn is None: 
                return jsonify([]),500
            
            res = recommend(q, diff, r, n)
            
            for c in res.columns:
                if res[c].dtype == 'object': 
                    res[c] = res[c].fillna('')
            if 'Course Rating' in res.columns:
                res['Course Rating'] = res['Course Rating'].apply(safe_float)
            
            return jsonify(res.to_dict('records'))
        
        return jsonify({"error":"Request must be JSON"}),400
    except Exception as e:
        log.error(f"/api/recommend error: {str(e)}")
        return jsonify({"error":str(e)}),500

def vote(qv, k=15):
    try:
        dists, inds = knn.kneighbors(qv, n_neighbors=k)
        neigh_courses = d.iloc[inds.flatten()]
        
        if 'Cluster' not in neigh_courses.columns:
            log.warning("No cluster column for voting")
            return None, {}
        
        neigh_clusters = neigh_courses['Cluster'].tolist()
        weights = 1 - dists.flatten()
        
        votes = {}
        for i, cluster in enumerate(neigh_clusters):
            if cluster in votes:
                votes[cluster] += weights[i]
            else:
                votes[cluster] = weights[i]
        
        winning_cluster = max(votes, key=votes.get)
        
        log.info(f"Votes: {votes}")
        log.info(f"Winner: {winning_cluster} with weight {votes[winning_cluster]:.4f}")
        
        return winning_cluster, votes
    
    except Exception as e:
        log.error(f"Voting error: {str(e)}")
        return None, {}

def recommend(q, diff='all', r=0, n=10):
    try:
        if not q.strip():
            log.info("Empty input, fallback to top rated")
            df = d.copy()
            if diff != 'all': 
                df = df[df['Difficulty Level'] == diff]
            if r > 0: 
                df = df[df['Course Rating'] >= r]
            if len(df) == 0: 
                df = d.copy()
            df = df.sort_values(by='Course Rating', ascending=False).head(n)
            df['similarity'] = 0.0
            df['predicted_cluster'] = 'N/A'
            return df
        
        log.info(f"Processing query: '{q}', diff: '{diff}', rating: {r}")
        
        qv = v.transform([q])
        predicted_cluster, vote_breakdown = vote(qv)
        
        if predicted_cluster is None:
            log.warning("No cluster prediction, using similarity fallback")
            sims = cosine_similarity(qv, X).flatten()
            df = d.copy()
            df['similarity'] = sims
            df['predicted_cluster'] = 'N/A'
        else:
            df = d[d['Cluster'] == predicted_cluster].copy()
            log.info(f"Cluster '{predicted_cluster}' - {len(df)} courses found")
            cluster_X = X[df.index]
            sims = cosine_similarity(qv, cluster_X).flatten()
            df['similarity'] = sims
            df['predicted_cluster'] = predicted_cluster
        
        log.info(f"Before filter: {len(df)} courses")
        
        if diff != 'all':
            df = df[df['Difficulty Level'] == diff]
            log.info(f"After difficulty filter: {len(df)}")
        
        if r > 0:
            df = df[df['Course Rating'] >= r]
            log.info(f"After rating filter: {len(df)}")
        
        if len(df) == 0:
            log.warning("No courses, expanding search")
            df = d.copy()
            if r > 0: df = df[df['Course Rating'] >= r]
            if diff != 'all': df = df[df['Difficulty Level'] == diff]
            if len(df) == 0: df = d.copy().sort_values(by='Course Rating', ascending=False).head(n * 2)
            
            sims = cosine_similarity(qv, X[df.index]).flatten()
            df['similarity'] = sims
            df['predicted_cluster'] = 'expanded'
        
        if len(df) > 0:
            result = df.sort_values(by='similarity', ascending=False).head(n)
            log.info(f"Final result: {len(result)}")
        else:
            result = pd.DataFrame(columns=list(d.columns) + ['similarity', 'predicted_cluster'])
        
        return result
    
    except Exception as e:
        log.error(f"Recommendation error: {str(e)}")
        return pd.DataFrame(columns=list(d.columns) + ['similarity', 'predicted_cluster'])

@app.route('/api/predict_cluster', methods=['POST'])
def predict_cluster_api():
    try:
        if request.is_json:
            q = request.json.get('query', '')
            k = int(request.json.get('k_neighbors', 15))
            
            if not q.strip():
                return jsonify({"error": "Query cannot be empty"}), 400
            
            if X is None or v is None or knn is None:
                return jsonify({"error": "Model not loaded"}), 500
            
            qv = v.transform([q])
            predicted_cluster, vote_breakdown = vote(qv, k)
            
            if predicted_cluster is None:
                return jsonify({"error": "Could not predict cluster"}), 500
            
            return jsonify({
                "query": q,
                "predicted_cluster": predicted_cluster,
                "vote_breakdown": vote_breakdown,
                "k_neighbors": k
            })
        
        return jsonify({"error": "Request must be JSON"}), 400
    
    except Exception as e:
        log.error(f"Cluster prediction API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug', methods=['POST'])
def debug_api():
    try:
        if request.is_json:
            q = request.json.get('query', '')
            diff = request.json.get('difficulty', 'all')
            
            if not q.strip():
                return jsonify({"error": "Query cannot be empty"}), 400
            
            if X is None or v is None or knn is None:
                return jsonify({"error": "Model not loaded"}), 500
            
            qv = v.transform([q])
            predicted_cluster, vote_breakdown = vote(qv)
            
            debug_info = {
                "query": q,
                "difficulty_filter": diff,
                "predicted_cluster": predicted_cluster,
                "vote_breakdown": vote_breakdown,
                "total_courses": len(d),
            }
            
            if predicted_cluster is not None:
                cluster_courses = d[d['Cluster'] == predicted_cluster]
                debug_info["cluster_size"] = len(cluster_courses)
                debug_info["cluster_difficulties"] = cluster_courses['Difficulty Level'].value_counts().to_dict()
                debug_info["cluster_ratings"] = {
                    "min": float(cluster_courses['Course Rating'].min()),
                    "max": float(cluster_courses['Course Rating'].max()),
                    "mean": float(cluster_courses['Course Rating'].mean())
                }
                
                if diff != 'all':
                    filtered = cluster_courses[cluster_courses['Difficulty Level'] == diff]
                    debug_info["filtered_size"] = len(filtered)
                else:
                    debug_info["filtered_size"] = len(cluster_courses)
            
            return jsonify(debug_info)
        
        return jsonify({"error": "Request must be JSON"}), 400
    
    except Exception as e:
        log.error(f"Debug API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
