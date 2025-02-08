import json
from compare_clustering_solutions import evaluate_clustering
from sentence_transformers import SentenceTransformer
import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
import spacy 
from sklearn.feature_extraction.text import TfidfVectorizer


class DynamicClustering:
    def __init__(self, similarity_threshold=0.7, min_cluster_size=10, max_iterations=10, convergence_threshold=0.98):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embeddings(self, requests):
        print("Generating embeddings for requests...")
        embeddings = self.model.encode(requests, convert_to_numpy=True)
        print(f"Embeddings generated: {embeddings.shape}")
        return embeddings

    def compute_centroid(self, cluster, embeddings):
        if not cluster:
            return None
        cluster_embeddings = np.array([embeddings[i] for i in cluster])
        similarity_matrix = cosine_similarity(cluster_embeddings)
        medoid_index = np.argmax(similarity_matrix.sum(axis=1))
        return embeddings[cluster[medoid_index]]

    def assign_to_clusters(self, embeddings):
        print("Assigning requests to initial clusters...")
        clusters = []
        centroids = []
        unclustered_requests = []
        
        for i, embedding in enumerate(embeddings):
            best_cluster = None
            best_similarity = 0
            
            for cluster_idx, centroid in enumerate(centroids):
                similarity = cosine_similarity([embedding], [centroid])[0][0]
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_idx
            
            if best_cluster is not None:
                clusters[best_cluster].append(i)
            else:
                clusters.append([i])
                centroids.append(embedding)
        
        print(f"Initial clustering completed: {len(clusters)} clusters formed.")
        return clusters, centroids, unclustered_requests

    def compute_cluster_shift(self, old_centroids, new_centroids):
        return np.mean([cosine_similarity([old], [new])[0][0] for old, new in zip(old_centroids, new_centroids)])

    def refine_clusters(self, embeddings, clusters, centroids):
        print("Refining clusters with iterative optimization...")
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}...")
            new_centroids = [self.compute_centroid(cluster, embeddings) for cluster in clusters]
            new_clusters = [[] for _ in range(len(new_centroids))]
            unclustered_requests = []
            
            for i, embedding in enumerate(embeddings):
                best_cluster = None
                best_similarity = 0

                for cluster_idx, centroid in enumerate(new_centroids):
                    similarity = cosine_similarity([embedding], [centroid])[0][0]
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster_idx

                if best_cluster is not None:
                    new_clusters[best_cluster].append(i)
                else:
                    unclustered_requests.append(i)
            
            shift = self.compute_cluster_shift(centroids, new_centroids)
            print(f"Cluster shift: {shift:.4f}")
            clusters, centroids = new_clusters, new_centroids

            if shift > self.convergence_threshold:
                print("Convergence threshold reached. Stopping refinement.")
                break
        
        print(f"Refinement complete. Final number of clusters: {len(clusters)}")
        return clusters, centroids, unclustered_requests

    def filter_clusters(self, clusters, unclustered_requests):
        print("Filtering clusters based on minimum size requirement...")
        valid_clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]
        final_unclustered = unclustered_requests + [i for cluster in clusters if len(cluster) < self.min_cluster_size for i in cluster]
        print(f"Valid clusters after filtering: {len(valid_clusters)}")
        print(f"Total unclustered requests: {len(final_unclustered)}")
        return valid_clusters, final_unclustered

    def extract_representatives(self, requests, embeddings, clusters, num_representatives=3):
        representatives = []
    
        for cluster in clusters:
            cur_reps = []

            # Extract embeddings of all requests in this cluster
            cluster_embeddings = np.array([embeddings[i] for i in cluster])

            # If the cluster has <= num_representatives, take all
            if len(cluster) <= num_representatives:
                cur_reps = [requests[i] for i in cluster]
            else:
                # Apply K-Means within the cluster
                kmeans = KMeans(n_clusters=num_representatives, random_state=42, n_init=10).fit(cluster_embeddings)

                # Find the closest requests to the centroids
                centroid_indices = []
                for centroid in kmeans.cluster_centers_:
                    closest_idx = np.argmin(np.linalg.norm(cluster_embeddings - centroid, axis=1))
                    centroid_indices.append(cluster[closest_idx])  # Map back to original request index

                cur_reps = [requests[i] for i in centroid_indices]  # ✅ Fix: List comprehension properly formatted

            representatives.append(cur_reps)

        return representatives
    
    def cluster_requests(self, requests, from_file=False):
        print("Starting clustering process...")
        if not from_file:
            embeddings = self.generate_embeddings(requests)
            clusters, centroids, unclustered_requests = self.assign_to_clusters(embeddings)
            clusters, centroids, unclustered_requests = self.refine_clusters(embeddings, clusters, centroids)
            valid_clusters, final_unclustered = self.filter_clusters(clusters, unclustered_requests)
            reps = self.extract_representatives(requests, embeddings, valid_clusters)
        else:
            valid_clusters, final_unclustered = load_clusters()
            embeddings = self.generate_embeddings(requests)
            reps = self.extract_representatives(requests, embeddings, valid_clusters)
        return valid_clusters, final_unclustered, reps

class ClusterNamer:
    def __init__(self, nlp_model="en_core_web_sm", top_n=3):
        self.nlp = spacy.load(nlp_model)
        self.top_n = top_n

    def preprocess_text(self, text):
        """Lowercase, remove stopwords, and lemmatize."""
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

    def extract_keywords(self, cluster_requests):
        """Extract top-n keywords using TF-IDF, with a fallback to noun phrase extraction."""
        if not cluster_requests:
            return "Unknown Cluster"

        processed_requests = [self.preprocess_text(req) for req in cluster_requests]
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1,3))
        X = vectorizer.fit_transform(processed_requests)

        # Rank words by their TF-IDF score
        feature_names = np.array(vectorizer.get_feature_names_out())
        avg_tfidf = X.mean(axis=0).A1
        top_keywords = feature_names[np.argsort(avg_tfidf)[-self.top_n:]]

        # If TF-IDF fails, extract noun phrases as a fallback
        if not top_keywords.any():
            top_keywords = self.extract_noun_phrases(cluster_requests)

        return " ".join(top_keywords)

    def extract_noun_phrases(self, cluster_requests):
        """Extract most common noun phrases from the requests as a backup."""
        all_noun_phrases = []
        for text in cluster_requests:
            doc = self.nlp(text.lower())
            noun_phrases = [" ".join(token.text for token in np) for np in doc.noun_chunks]
            all_noun_phrases.extend(noun_phrases)

        if not all_noun_phrases:
            return ["General Inquiry"]  # Fallback

        most_common_phrases = [phrase for phrase, _ in Counter(all_noun_phrases).most_common(self.top_n)]
        return most_common_phrases

    def clean_redundancies(self, phrase):
        words = phrase.split()
        cleaned_words = []
        seen = set()
    
        for word in words:
            if word not in seen:
                # Fix the "pende" word
                if word == "pende":
                    word = "Pending"
                cleaned_words.append(word)
                seen.add(word)

        return " ".join(cleaned_words).title()  # Convert to readable title format

    def extract_cluster_name(self, cluster_requests):
        """Generate a meaningful and readable cluster name."""
        raw_name = self.extract_keywords(cluster_requests)
        cleaned_name = self.clean_redundancies(raw_name)
        return cleaned_name if cleaned_name else "General Inquiry"


def get_requests(data_filename):
    df = pd.read_csv(data_filename)
    return df['text'].astype(str).apply(lambda x: x.lower().strip())

def save_clusters(valid_clusters, final_unclustered):
    with open("clustering_results60.json", "w") as f:
        json.dump({"valid_clusters": valid_clusters, "final_unclustered": final_unclustered}, f)

def load_clusters(filename="clustering_results60.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return data["valid_clusters"], data["final_unclustered"]

def save_clustered_requests(output_file, valid_clusters, final_unclustered, reps, cluster_names, requests):
    """
    Saves the clustered requests and unclustered requests into a structured JSON file.
    Ensures that requests match the original dataset exactly.
    """

    output_data = {
        "cluster_list": []
    }

    for i, cluster in enumerate(valid_clusters):
        cluster_data = {
            "cluster_name": cluster_names[i],
            "requests": [requests[idx] for idx in cluster],  # Exact original text
            "representatives": reps[i]  # List of representative requests
        }
        output_data["cluster_list"].append(cluster_data)

    # Store unclustered requests without modification
    output_data["unclustered"] = [requests[idx] for idx in final_unclustered]

    # Save JSON while ensuring correct encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Clustered data saved to {output_file}")

def analyze_unrecognized_requests(data_file, output_file, num_representatives, min_size):
    clustering_model = DynamicClustering(similarity_threshold=0.6, min_cluster_size=int(min_size), max_iterations=10)
    namer = ClusterNamer()
    
    requests = get_requests(data_file)
    
    valid_clusters, final_unclustered, reps = clustering_model.cluster_requests(requests.copy())

    cluster_names = {}
    for i, cluster in enumerate(valid_clusters):
        cluster_requests = [requests[idx] for idx in cluster]  # Get request texts
        cluster_names[i] = namer.extract_cluster_name(cluster_requests)
    
    save_clustered_requests('output/banking-clusters-min-size-10.json',valid_clusters, final_unclustered,reps, cluster_names, requests)


def load_clusters(filename="clustering_results60.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return data["valid_clusters"], data["final_unclustered"]


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)



    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])


    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
