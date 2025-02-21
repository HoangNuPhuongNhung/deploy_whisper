from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# Load Whisper model (có thể đổi "base" thành "small", "medium", "large" nếu muốn)
model = whisper.load_model("large")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = f"temp/{file.filename}"
    
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)

    # Chạy Whisper với word_timestamps=True
    result = model.transcribe(file_path, word_timestamps=True)
    
    # Xóa file tạm
    os.remove(file_path)

    # Lấy danh sách từ và timestamp
    words_with_timestamps = []
    for segment in result["segments"]:
        start = segment["start"]  # Thời gian bắt đầu
        end = segment["end"]  # Thời gian kết thúc
        text = segment["text"]  # Nội dung
        words_with_timestamps.append(f"[{start:.2f}s - {end:.2f}s] {text}")

    return jsonify({
    "formatted_segments": words_with_timestamps
})


if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8000))  # Railway cấp port động
    print(f"Server is running on port {port}")
    serve(app, host="0.0.0.0", port=port)