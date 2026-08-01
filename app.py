import os
import json
import base64
import subprocess
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import googleapiclient.http
from google.auth.transport.requests import Request
import pickle
import anthropic

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB max

# YouTube API scopes
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]

TOKEN_FILE = "token.pickle"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_client_config():
    """Build OAuth client config from environment variables."""
    return {
        "installed": {
            "client_id": os.environ.get("YT_CLIENT_ID"),
            "client_secret": os.environ.get("YT_CLIENT_SECRET"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback")]
        }
    }


def get_youtube_service():
    """Load saved credentials and return an authenticated YouTube API client."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "wb") as token:
            pickle.dump(creds, token)

    if not creds or not creds.valid:
        return None

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def extract_frames(video_path, num_frames=6):
    """Extract evenly-spaced frames from the video as base64 JPEGs."""
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True
        )
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            duration = 30.0

        interval = max(duration / (num_frames + 1), 1)
        for i in range(1, num_frames + 1):
            timestamp = interval * i
            frame_path = os.path.join(tmpdir, f"frame_{i}.jpg")
            subprocess.run(
                ["ffmpeg", "-ss", str(timestamp), "-i", video_path,
                 "-frames:v", "1", "-q:v", "3", "-y", frame_path],
                capture_output=True
            )
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frames.append(base64.b64encode(f.read()).decode("utf-8"))
    return frames


def extract_audio_transcript(video_path):
    """Extract audio and transcribe it using OpenAI Whisper (local, free)."""
    try:
        import whisper
    except ImportError:
        return ""

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "mp3", "-y", audio_path],
            capture_output=True
        )
        if not os.path.exists(audio_path):
            return ""
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result.get("text", "")


def generate_seo_content(video_path):
    """Use Claude to analyze video frames + audio transcript and generate SEO content."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"title": "Untitled Video", "description": "", "tags": []}

    client = anthropic.Anthropic(api_key=api_key)

    frames = extract_frames(video_path)
    transcript = extract_audio_transcript(video_path)

    content_blocks = []
    for frame_b64 in frames:
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame_b64}
        })

    prompt_text = (
        "Ye ek video ke frames hain (visual scenes) aur uska audio transcript neeche diya hai. "
        "Is video ke liye YouTube par upload karne ke liye SEO-optimized content banayein.\n\n"
        f"Audio Transcript:\n{transcript[:3000]}\n\n"
        "Respond ONLY in valid JSON format, no other text, no markdown fences:\n"
        '{"title": "...", "description": "...", "tags": ["tag1", "tag2", ...]}\n\n'
        "Title: catchy, SEO-friendly, under 100 characters.\n"
        "Description: detailed, engaging, keyword-rich, includes 5-8 relevant hashtags at the end.\n"
        "Tags: 10-15 relevant search tags as an array of strings."
    )
    content_blocks.append({"type": "text", "text": prompt_text})

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": content_blocks}]
    )

    raw_text = message.content[0].text.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        data = {"title": "Untitled Video", "description": raw_text, "tags": []}

    return data


@app.route("/")
def index():
    youtube = get_youtube_service()
    authorized = youtube is not None
    return render_template("index.html", authorized=authorized)


@app.route("/authorize")
def authorize():
    flow = google_auth_oauthlib.flow.Flow.from_client_config(
        get_client_config(), scopes=SCOPES
    )
    flow.redirect_uri = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback")
    authorization_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    session["state"] = state
    return redirect(authorization_url)


@app.route("/oauth2callback")
def oauth2callback():
    state = session.get("state")
    flow = google_auth_oauthlib.flow.Flow.from_client_config(
        get_client_config(), scopes=SCOPES, state=state
    )
    flow.redirect_uri = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback")
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    with open(TOKEN_FILE, "wb") as token:
        pickle.dump(creds, token)

    flash("YouTube account connected successfully!", "success")
    return redirect(url_for("index"))


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        flash("Koi video file nahi mili.", "error")
        return redirect(url_for("index"))

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        flash("Sahi video file select karein (mp4, mov, avi, mkv, webm).", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    seo_data = generate_seo_content(filepath)

    session["pending_filename"] = filename
    session["seo_title"] = seo_data.get("title", "Untitled Video")
    session["seo_description"] = seo_data.get("description", "")
    session["seo_tags"] = ", ".join(seo_data.get("tags", []))

    return redirect(url_for("preview"))


@app.route("/preview")
def preview():
    if "pending_filename" not in session:
        return redirect(url_for("index"))
    return render_template(
        "preview.html",
        title=session.get("seo_title", ""),
        description=session.get("seo_description", ""),
        tags=session.get("seo_tags", ""),
    )


@app.route("/upload", methods=["POST"])
def upload():
    youtube = get_youtube_service()
    if not youtube:
        flash("Pehle YouTube account connect karein.", "error")
        return redirect(url_for("index"))

    filename = session.get("pending_filename")
    if not filename:
        flash("Pehle video analyze karein.", "error")
        return redirect(url_for("index"))

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        flash("Video file nahi mili, dobara try karein.", "error")
        return redirect(url_for("index"))

    title = request.form.get("title", "").strip() or "Untitled Video"
    description = request.form.get("description", "").strip()
    tags = [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()]
    privacy = request.form.get("privacy", "private")
    category_id = request.form.get("category", "22")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    try:
        media = googleapiclient.http.MediaFileUpload(
            filepath, chunksize=-1, resumable=True
        )
        request_upload = youtube.videos().insert(
            part="snippet,status", body=body, media_body=media
        )
        response = None
        while response is None:
            status, response = request_upload.next_chunk()

        video_id = response.get("id")
        video_url = f"https://youtube.com/watch?v={video_id}"
        flash(f"Video upload ho gayi! Link: {video_url}", "success")
    except googleapiclient.errors.HttpError as e:
        flash(f"Upload mein error aayi: {e}", "error")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        session.pop("pending_filename", None)
        session.pop("seo_title", None)
        session.pop("seo_description", None)
        session.pop("seo_tags", None)

    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
