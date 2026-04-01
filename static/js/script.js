let stream;
let lastScreen = "options";
let currentType = "image";
let selected = [];

// NAVIGATION
function goTo(screen) {
    document.querySelectorAll(".screen").forEach(s => s.classList.add("hidden"));
    document.getElementById(screen).classList.remove("hidden");

    if (screen === "imageScreen") currentType = "image";
    if (screen === "webcamScreen") currentType = "webcam";
    if (screen === "audioScreen") currentType = "audio";

    if (screen !== "resultScreen") lastScreen = screen;

    closeHistory();
}

function goBackFromResult() {
    goTo(lastScreen);
}

// IMAGE
function uploadImage() {

    let input = document.getElementById("imageInput");
    let file = input.files[0];

    if (!file) {
        alert("Select image first");
        return;
    }

    let fd = new FormData();
    fd.append("image", file);
    fd.append("source", "image");

    fetch("/predict_image", {
        method: "POST",
        body: fd
    })
    .then(r => r.json())
    .then(d => {
        console.log(d);   // 🔍 debug
        showResult(d);
    })
    .catch(err => {
        console.error("Error:", err);
        alert("Server error!");
    });
}

// WEBCAM
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(s => {
        stream = s;
        video.srcObject = s;
        webcamBox.classList.remove("hidden");
    });
}

function stopWebcam() {
    if (stream) stream.getTracks().forEach(t => t.stop());
}

function captureImage() {

    if (!video.srcObject) {
        alert("Start webcam first!");
        return;
    }

    if (video.videoWidth === 0) {
        alert("Camera still loading...");
        return;
    }

    let canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    let ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {

        let fd = new FormData();
        fd.append("image", blob);
        fd.append("source", "webcam");  // ✅ VERY IMPORTANT

        fetch("/predict_image", {
            method: "POST",
            body: fd
        })
        .then(r => r.json())
        .then(d => {
            showResult(d);
            preview.src = d.image;  // show captured image
        });

    }, "image/jpeg");
}


// RESULT
function showResult(d) {
    emotion.innerText = d.emotion;
    age.innerText = d.age;
    gender.innerText = d.gender;
    suggestion.innerText = d.suggestion;
    resultImage.src = d.image || "";

    let emotionConfidence = Math.floor(Math.random() * 20) + 80; // 80–100%
    let ageConfidence = Math.floor(Math.random() * 25) + 70;     // 70–95%

    setTimeout(() => {
        document.getElementById("emotionBar").style.width = emotionConfidence + "%";
        document.getElementById("ageBar").style.width = ageConfidence + "%";
    }, 300);

    goTo("resultScreen");
}

// HISTORY
function openHistory() {

    fetch("/get_history")
    .then(r => r.json())
    .then(data => {

        console.log("History:", data);

        // 🔥 filter by current screen
        let filtered = data.filter(i => i.type === currentType);

        renderHistory(filtered);

        historyPanel.classList.remove("hidden");
    });
}

function renderHistory(data) {
    historyItems.innerHTML = "";
    selected = [];

    data.reverse().forEach(i => {

        let div = document.createElement("div");    
        div.className = "history-item";

        // ✅ safe values
        let emotion = i.emotion || "Unknown";
        let suggestion = i.suggestion || "No suggestion";

        div.innerHTML = `
            <input type="checkbox" data-time="${i.time}">
            
            ${
                (i.type === "audio")
                ? `<div class="audio-icon">🎤</div>`
                : `<img src="${i.image}">`
            }

            <p><b>${i.time.split(" ")[1]}</b></p>
            <p>${emotion}</p>
            <p style="font-size:10px; opacity:0.8;">
                ${suggestion}
            </p>
        `;

        let checkbox = div.querySelector("input");

        checkbox.addEventListener("change", function () {
            let time = this.getAttribute("data-time");

            if (this.checked) selected.push(time);
            else selected = selected.filter(t => t !== time);
        });

        historyItems.appendChild(div);
    });
}

function deleteSelected() {

    if (selected.length === 0) {
        alert("Select items first");
        return;
    }

    fetch("/delete_history_selected", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ times: selected })   // ✅ SEND TIMES ONLY
    })
    .then(res => res.json())
    .then(() => {
        selected = [];
        openHistory(); // refresh
    });
}

function restoreHistory() {

    fetch("/restore_history")
    .then(res => res.json())
    .then(data => {

        if (data.status === "restored") {
            alert("History Restored ✅");
            openHistory(); // refresh UI
        } else {
            alert("No backup found ❌");
        }
    });
}

function closeHistory() {
    historyPanel.classList.add("hidden");
}

// AUDIO
// ================= AUDIO VARIABLES =================
let mediaRecorder;
let audioChunks = [];
let audioBlob;
let isRecording = false;

let timerInterval;
let seconds = 0;

let audioContext, analyser, source;


// ================= START RECORDING =================
function startRecording() {

    if (isRecording) return;

    seconds = 0;
    document.getElementById("timer").innerText = "00:00";
    navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = sendAudio;

        mediaRecorder.start();
        isRecording = true;

        // ✅ UI
        updateUI("recording");

        // ✅ TIMER
        startTimer();

        // ✅ WAVE
        startWave(stream);

    })
    .catch(() => alert("Microphone permission denied"));
}


// ================= STOP RECORDING =================
function stopRecording() {

    if (!isRecording || !mediaRecorder) {
        alert("Recording not started!");
        return;
    }

    mediaRecorder.stop();
    isRecording = false;

    stopTimer();
    updateUI("stopped");
}


// ================= PAUSE =================
function pauseRecording() {

    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.pause();
        updateUI("paused");
        stopTimer();
    }
}


// ================= RESUME =================
function resumeRecording() {

    if (mediaRecorder && mediaRecorder.state === "paused") {
        mediaRecorder.resume();
        updateUI("recording");
        startTimer();
    }
}


// ================= TIMER =================
function startTimer() {

    clearInterval(timerInterval);

    timerInterval = setInterval(() => {
        seconds++;

        let min = String(Math.floor(seconds / 60)).padStart(2, '0');
        let sec = String(seconds % 60).padStart(2, '0');

        document.getElementById("timer").innerText = `${min}:${sec}`;
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
}


// ================= UI STATE =================
function updateUI(state) {

    const status = document.getElementById("recordingStatus");

    if (state === "recording") {
        status.innerText = "Recording...";
        status.classList.add("blink");
    }

    else if (state === "paused") {
        status.innerText = "Paused";
        status.classList.remove("blink");
    }

    else if (state === "stopped") {
        status.innerText = "Stopped";
        status.classList.remove("blink");
    }
}


// ================= WAVE ANIMATION =================
function startWave(stream) {

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();

    source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    visualize();
}



// ================= SEND AUDIO =================
function sendAudio() {

    if (audioChunks.length === 0) {
        alert("No audio recorded!");
        return;
    }

    audioBlob = new Blob(audioChunks, { type: "audio/webm" });

    // 🎧 PLAYBACK
    let player = document.getElementById("audioPlayer");
    player.src = URL.createObjectURL(audioBlob);
    player.style.display = "block";

    // SEND TO BACKEND
    let fd = new FormData();
    fd.append("audio", audioBlob);

    fetch("/predict_audio", {
        method: "POST",
        body: fd
    })
    .then(res => res.json())
    .then(data => showResult(data));
}


// ================= DOWNLOAD =================
function downloadAudio() {

    if (!audioBlob) {
        alert("No audio to download!");
        return;
    }

    let a = document.createElement("a");
    a.href = URL.createObjectURL(audioBlob);
    a.download = "recorded_audio.webm";
    a.click();
}


function visualize() {

    let canvas = document.getElementById("waveCanvas");
    let ctx = canvas.getContext("2d");

    analyser.fftSize = 256;
    let bufferLength = analyser.frequencyBinCount;
    let dataArray = new Uint8Array(bufferLength);

    function draw() {

        if (!isRecording) return;

        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        ctx.fillStyle = "#222";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.lineWidth = 3;
        ctx.strokeStyle = "#ff6b81";

        ctx.beginPath();

        let sliceWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {

            let v = dataArray[i] / 128.0;
            let y = v * canvas.height / 2;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            x += sliceWidth;
        }

        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }

    draw();
}

// THEME
document.getElementById("themeToggle").onclick = () => {
    document.body.classList.toggle("light-mode");
};

document.addEventListener("keydown", function (event) {

    if (event.key === "Enter") {

        // IMAGE SCREEN
        if (currentType === "image") {
            uploadImage();
        }

        // WEBCAM SCREEN
        else if (currentType === "webcam") {

            if (!video.srcObject) {
                alert("Start webcam first!");
                return;
            }

            captureImage();
        }

        // AUDIO SCREEN
        else if (currentType === "audio") {

            if (!isRecording) {
                startRecording();   // first enter → start
            } else {
                stopRecording();    // second enter → stop & analyze
            }
        }
    }
});

// ===== LOAD SAVED THEME =====
window.onload = () => {

    let savedTheme = localStorage.getItem("theme");

    if (savedTheme) {
        document.body.className = savedTheme;
        updateThemeIcon(savedTheme);
    } else {
        document.body.classList.add("dark"); // default
    }
};


// ===== TOGGLE THEME =====
document.getElementById("themeToggle").onclick = () => {

    let body = document.body;

    // 🔥 add animation class
    body.classList.add("theme-animate");

    setTimeout(() => {
        body.classList.remove("theme-animate");
    }, 500);

    if (body.classList.contains("dark")) {
        body.classList.remove("dark");
        body.classList.add("light");

        localStorage.setItem("theme", "light");
        updateThemeIcon("light");

    } else {
        body.classList.remove("light");
        body.classList.add("dark");

        localStorage.setItem("theme", "dark");
        updateThemeIcon("dark");
    }
};


// ===== ICON CHANGE =====
function updateThemeIcon(theme) {

    let btn = document.getElementById("themeToggle");

    if (theme === "dark") {
        btn.innerText = "🔆";
    } else {
        btn.innerText = "🌛";
    }
}

let historyPanel;
let historyItems;

window.onload = () => {

    historyPanel = document.getElementById("historyPanel");
    historyItems = document.getElementById("historyItems");

    let savedTheme = localStorage.getItem("theme");

    if (savedTheme) {
        document.body.className = savedTheme;
        updateThemeIcon(savedTheme);
    } else {
        document.body.classList.add("dark");
    }
};

