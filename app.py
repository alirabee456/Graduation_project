from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import process
import neattext as ntx

app = Flask(__name__)

# Load the data
try:
    df = pd.read_csv("udemy_courses-cleaned.csv")
    print("CSV loaded successfully")
except Exception as e:
    print(f"Error loading CSV: {e}")
    raise

# Clean the course titles
try:
    df['course_title_cleaned'] = df['course_title'].apply(ntx.remove_stopwords).apply(ntx.remove_special_characters)
    print("Course titles cleaned successfully")
except Exception as e:
    print(f"Error cleaning course titles: {e}")
    raise

# Create the count vectorizer matrix
try:
    cv = CountVectorizer()
    title_matrix = cv.fit_transform(df['course_title_cleaned'].drop_duplicates()).toarray()
    print("Count vectorizer matrix created successfully")
except Exception as e:
    print(f"Error creating count vectorizer matrix: {e}")
    raise

# Compute the cosine similarity matrix
try:
    sim_matrix = cosine_similarity(title_matrix)
    print("Cosine similarity matrix computed successfully")
except Exception as e:
    print(f"Error computing cosine similarity matrix: {e}")
    raise

# Create a course index series
try:
    course_index = pd.Series(df.index, index=df['course_title']).drop_duplicates()
    print("Course index series created successfully")
except Exception as e:
    print(f"Error creating course index series: {e}")
    raise

def my_rec_sys(my_title):
    try:
        # Check if the input title is just one word
        if len(my_title.split()) == 1:
            # Filter courses that contain the input word in their titles
            matching_courses = df[df['course_title_cleaned'].str.contains(my_title, case=False)]
            if matching_courses.empty:
                return jsonify({"error": "No courses found. Please enter a valid course title."})
            return matching_courses.to_dict(orient='records')
        
        # If input title has more than one word, proceed with existing logic
        idx = course_index.get(my_title)
        if idx is None:
            closest_match, score = process.extractOne(my_title, course_index.index)
            if score >= 70:
                idx = course_index[closest_match]
            else:
                return jsonify({"error": "Course not found. Please enter a valid course title."})
        
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        course_indices = [i[0] for i in sim_scores]
        return df.iloc[course_indices].to_dict(orient='records')
    except Exception as e:
        print(f"Error in recommendation system: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    course_title = request.args.get('course_title')
    if not course_title:
        return jsonify({"error": "No course title entered"})
    recommendations = my_rec_sys(course_title)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
