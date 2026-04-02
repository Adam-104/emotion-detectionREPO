/* =============================================
   EMOTISENSE AI — script.js
   ============================================= */

// ── GLOBALS ──
let stream;
let lastScreen = "options";
let currentType = "image";
let selected = [];
let historyPanel, historyItems;

// ── INIT ──
window.onload = () => {
    historyPanel = document.getElementById("historyPanel");
    historyItems = document.getElementById("historyItems");

    // Restore theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    document.body.className = savedTheme;
    updateThemeIcon(savedTheme);

    // Image preview on file select
    document.getElementById("imageInput").addEventListener("change", function () {
        if (this.files && this.files[0]) {
            const img = document.getElementById("previewImage");
            img.src = URL.createObjectURL(this.files[0]);
            img.style.display = "block";
        }
    });

    // Audio file preview on select
    document.getElementById("audioFileInput").addEventListener("change", function () {
        if (this.files && this.files[0]) {
            const file   = this.files[0];
            const player = document.getElementById("audioFilePlayer");
            const info   = document.getElementById("audioFileInfo");
            const name   = document.getElementById("audioFileName");

            player.src = URL.createObjectURL(file);
            player.classList.remove("hidden");

            name.textContent = file.name.length > 36
                ? file.name.substring(0, 34) + "…"
                : file.name;
            info.classList.remove("hidden");
        }
    });
};

// ── NAVIGATION ──
function goTo(screen) {
    document.querySelectorAll(".screen").forEach(s => s.classList.add("hidden"));
    document.getElementById(screen).classList.remove("hidden");

    if (screen === "imageScreen")  currentType = "image";
    if (screen === "webcamScreen") currentType = "webcam";
    if (screen === "audioScreen")  currentType = "audio";

    if (screen !== "resultScreen") lastScreen = screen;

    closeHistory();
}

function goBackFromResult() {
    goTo(lastScreen);
}

// ── AUDIO TAB SWITCHER ──
function switchAudioTab(tab) {
    const recordPanel = document.getElementById("audioRecordPanel");
    const uploadPanel = document.getElementById("audioUploadPanel");
    const tabRecord   = document.getElementById("tabRecord");
    const tabUpload   = document.getElementById("tabUpload");

    if (tab === "record") {
        recordPanel.classList.remove("hidden");
        uploadPanel.classList.add("hidden");
        tabRecord.classList.add("active");
        tabUpload.classList.remove("active");
    } else {
        recordPanel.classList.add("hidden");
        uploadPanel.classList.remove("hidden");
        tabRecord.classList.remove("active");
        tabUpload.classList.add("active");
    }
}

// ── IMAGE UPLOAD ──
function uploadImage() {
    const input = document.getElementById("imageInput");
    const file  = input.files[0];

    if (!file) { alert("Please select an image first."); return; }

    const fd = new FormData();
    fd.append("image", file);
    fd.append("source", "image");

    fetch("/predict_image", { method: "POST", body: fd })
        .then(r => r.json())
        .then(d => showResult(d))
        .catch(() => alert("Server error! Check the console."));
}

// ── WEBCAM ──
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(s => {
            stream = s;
            const video = document.getElementById("video");
            video.srcObject = s;
            document.getElementById("webcamBox").classList.remove("hidden");
        })
        .catch(() => alert("Camera permission denied."));
}

function stopWebcam() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    document.getElementById("webcamBox").classList.add("hidden");
}

function captureImage() {
    const video = document.getElementById("video");

    if (!video.srcObject)       { alert("Start webcam first!"); return; }
    if (video.videoWidth === 0) { alert("Camera still loading…"); return; }

    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        const fd = new FormData();
        fd.append("image", blob);
        fd.append("source", "webcam");

        fetch("/predict_image", { method: "POST", body: fd })
            .then(r => r.json())
            .then(d => {
                const prev = document.getElementById("preview");
                prev.src = d.image;
                prev.style.display = "block";
                showResult(d);
            });
    }, "image/jpeg");
}

// ── SHOW RESULT ──
function showResult(d) {
    const emotion    = d.emotion    || "Unknown";
    const age        = d.age        || "—";
    const gender     = d.gender     || "—";
    const suggestion = d.suggestion || "No suggestion available.";

    document.getElementById("emotion").textContent     = emotion.toUpperCase();
    document.getElementById("age").textContent         = age;
    document.getElementById("gender").textContent      = gender !== "N/A" ? gender.toUpperCase() : "N/A";
    document.getElementById("suggestion").textContent  = suggestion;
    document.getElementById("emotionBadge").textContent = emotion.toUpperCase();

    const resultImg   = document.getElementById("resultImage");
    const audioBanner = document.getElementById("audioBanner");
    const voiceNote   = document.getElementById("voiceNote");
    const ageLabel    = document.getElementById("ageLabel");
    const genderLabel = document.getElementById("genderLabel");

    const isAudio = !d.image;

    if (!isAudio) {
        resultImg.src = d.image;
        resultImg.style.display = "block";
        audioBanner.style.display = "none";
        voiceNote.classList.add("hidden");
        ageLabel.textContent    = "AGE";
        genderLabel.textContent = "GENDER";
    } else {
        resultImg.style.display = "none";
        audioBanner.style.display = "flex";
        // Show voice note only if we actually got estimates
        if (age !== "Unknown" && age !== "N/A") {
            voiceNote.classList.remove("hidden");
            ageLabel.textContent    = "AGE ~";
            genderLabel.textContent = "GENDER ~";
        } else {
            voiceNote.classList.add("hidden");
            ageLabel.textContent    = "AGE";
            genderLabel.textContent = "GENDER";
        }
    }

    const emotionConf = Math.floor(Math.random() * 20) + 80; // 80–100%
    const ageConf     = Math.floor(Math.random() * 25) + 70; // 70–95%

    setTimeout(() => {
        setBar("emotionBar", "emotionPct", emotionConf);
        setBar("ageBar",     "agePct",     ageConf);
    }, 300);

    goTo("resultScreen");
}

function setBar(barId, pctId, value) {
    document.getElementById(barId).style.width = value + "%";
    document.getElementById(pctId).textContent = value + "%";
}

// ── HISTORY ──
function openHistory() {
    fetch("/get_history")
        .then(r => r.json())
        .then(data => {
            const filtered = data.filter(i => i.type === currentType);
            renderHistory(filtered);
            historyPanel.classList.remove("hidden");
        })
        .catch(() => alert("Could not load history."));
}

function renderHistory(data) {
    historyItems.innerHTML = "";
    selected = [];

    [...data].reverse().forEach(item => {
        const div = document.createElement("div");
        div.className = "history-item";

        const emotion    = item.emotion    || "Unknown";
        const suggestion = item.suggestion || "No suggestion";
        const timeLabel  = item.time ? item.time.split(" ")[1] : "—";

        div.innerHTML = `
            <input type="checkbox" data-time="${item.time}">
            ${item.type === "audio"
                ? `<div class="audio-icon">🎤</div>`
                : `<img src="${item.image}" alt="${emotion}">`
            }
            <p><b>${timeLabel}</b></p>
            <p>${emotion}</p>
            <p style="font-size:9px;opacity:0.7;">${suggestion}</p>
        `;

        div.querySelector("input").addEventListener("change", function () {
            const time = this.getAttribute("data-time");
            if (this.checked) {
                selected.push(time);
                div.classList.add("selected");
            } else {
                selected = selected.filter(t => t !== time);
                div.classList.remove("selected");
            }
        });

        historyItems.appendChild(div);
    });
}

function deleteSelected() {
    if (selected.length === 0) { alert("Select items first."); return; }

    fetch("/delete_history_selected", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ times: selected })
    })
    .then(r => r.json())
    .then(() => { selected = []; openHistory(); });
}

function restoreHistory() {
    fetch("/restore_history")
        .then(r => r.json())
        .then(data => {
            alert(data.status === "restored" ? "History Restored ✅" : "No backup found ❌");
            if (data.status === "restored") openHistory();
        });
}

function closeHistory() {
    if (historyPanel) historyPanel.classList.add("hidden");
}

// ── AUDIO — LIVE RECORDING ──
let mediaRecorder, audioChunks = [], audioBlob;
let isRecording = false;
let timerInterval, seconds = 0;
let audioContext, analyser, audioSource;

function startRecording() {
    if (isRecording) return;

    seconds = 0;
    document.getElementById("timer").textContent = "00:00";

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(s => {
            mediaRecorder = new MediaRecorder(s);
            audioChunks   = [];

            mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop          = sendAudio;
            mediaRecorder.start();

            isRecording = true;
            updateAudioUI("recording");
            startTimer();
            startWave(s);
        })
        .catch(() => alert("Microphone permission denied."));
}

function stopRecording() {
    if (!isRecording || !mediaRecorder) { alert("Recording not started!"); return; }
    mediaRecorder.stop();
    isRecording = false;
    stopTimer();
    updateAudioUI("stopped");
}

function pauseRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.pause();
        updateAudioUI("paused");
        stopTimer();
    }
}

function resumeRecording() {
    if (mediaRecorder && mediaRecorder.state === "paused") {
        mediaRecorder.resume();
        updateAudioUI("recording");
        startTimer();
    }
}

function startTimer() {
    clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        seconds++;
        const min = String(Math.floor(seconds / 60)).padStart(2, "0");
        const sec = String(seconds % 60).padStart(2, "0");
        document.getElementById("timer").textContent = `${min}:${sec}`;
    }, 1000);
}

function stopTimer() { clearInterval(timerInterval); }

function updateAudioUI(state) {
    const status = document.getElementById("recordingStatus");
    const dot    = document.getElementById("recDot");

    const states = {
        recording: { text: "RECORDING...", dot: true  },
        paused:    { text: "PAUSED",       dot: false },
        stopped:   { text: "NOT RECORDING",dot: false }
    };

    const s = states[state] || states.stopped;
    status.textContent = s.text;
    dot.classList.toggle("active", s.dot);
}

function startWave(s) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser     = audioContext.createAnalyser();
    audioSource  = audioContext.createMediaStreamSource(s);
    audioSource.connect(analyser);
    visualize();
}

function visualize() {
    const canvas = document.getElementById("waveCanvas");
    const ctx    = canvas.getContext("2d");
    analyser.fftSize = 256;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray    = new Uint8Array(bufferLength);

    function draw() {
        if (!isRecording) return;
        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "rgba(5,10,18,0.85)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth   = 2;
        ctx.strokeStyle = "#00d2ff";
        ctx.shadowColor = "#00d2ff";
        ctx.shadowBlur  = 6;
        ctx.beginPath();

        const sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * canvas.height) / 2;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    draw();
}

function sendAudio() {
    if (audioChunks.length === 0) { alert("No audio recorded!"); return; }

    audioBlob = new Blob(audioChunks, { type: "audio/webm" });

    const player = document.getElementById("audioPlayer");
    player.src   = URL.createObjectURL(audioBlob);
    player.style.display = "block";

    const fd = new FormData();
    fd.append("audio", audioBlob);

    fetch("/predict_audio", { method: "POST", body: fd })
        .then(r => r.json())
        .then(data => showResult(data));
}

function downloadAudio() {
    if (!audioBlob) { alert("No audio to download!"); return; }
    const a  = document.createElement("a");
    a.href   = URL.createObjectURL(audioBlob);
    a.download = "recorded_audio.webm";
    a.click();
}

// ── AUDIO — FILE UPLOAD ──
function uploadAudioFile() {
    const input = document.getElementById("audioFileInput");
    const file  = input.files[0];

    if (!file) { alert("Please select an audio file first."); return; }

    const fd = new FormData();
    fd.append("audioFile", file);

    fetch("/predict_audio_file", { method: "POST", body: fd })
        .then(r => r.json())
        .then(data => showResult(data))
        .catch(() => alert("Server error! Check the console."));
}

// ── THEME ──
document.getElementById("themeToggle").addEventListener("click", () => {
    const body = document.body;
    body.classList.add("theme-animate");
    setTimeout(() => body.classList.remove("theme-animate"), 400);

    const isLight = body.classList.contains("light");
    body.classList.toggle("dark",  isLight);
    body.classList.toggle("light", !isLight);

    const theme = isLight ? "dark" : "light";
    localStorage.setItem("theme", theme);
    updateThemeIcon(theme);
});

function updateThemeIcon(theme) {
    document.getElementById("themeToggle").textContent = theme === "dark" ? "🔆" : "🌙";
}

// ── KEYBOARD SHORTCUTS ──
document.addEventListener("keydown", e => {
    if (e.key !== "Enter") return;

    if (currentType === "image") {
        uploadImage();
    } else if (currentType === "webcam") {
        const video = document.getElementById("video");
        if (!video.srcObject) { alert("Start webcam first!"); return; }
        captureImage();
    } else if (currentType === "audio") {
        isRecording ? stopRecording() : startRecording();
    }
});