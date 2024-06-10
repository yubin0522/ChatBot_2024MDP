from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# wellness_dataset.csv 파일을 읽어오고 embedding을 JSON에서 파싱합니다.
df = pd.read_csv('wellness_dataset.csv')
df['embedding'] = df['embedding'].apply(json.loads)

@app.route('/generate_response', methods=['POST'])
def generate_response():
    user_input = request.json['query']
    print(f"Received query: {user_input}")  # 요청 로그
    embedding = model.encode(user_input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    response = answer['챗봇']
    print(f"Sending response: {response}")  # 응답 로그
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
