from pathlib import Path

p = Path("templates/dashboard.html")
s = p.read_text()

marker = "  // Poll lightweight stats"

code = r'''
  // ================================
  // REAL-TIME ALERT API + SOUND
  // ================================

  const shownAlertIds = new Set();
  let alertSoundEnabled = false;
  let audioContext = null;

  function enableAlertSound() {
    try {
      audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
      audioContext.resume();
      alertSoundEnabled = true;

      const btn = document.getElementById("alert-sound-btn");
      if (btn) {
        btn.textContent = "?? SOUND ON";
        btn.style.opacity = "1";
      }

      playAlertSound("LOW");
    } catch (e) {
      console.log("Sound enable error:", e);
    }
  }

  function playAlertSound(severity = "LOW") {
    if (!alertSoundEnabled) return;

    try {
      audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();

      const now = audioContext.currentTime;
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();

      oscillator.connect(gain);
      gain.connect(audioContext.destination);

      if (severity === "CRITICAL") {
        oscillator.frequency.value = 1000;
      } else if (severity === "HIGH") {
        oscillator.frequency.value = 800;
      } else {
        oscillator.frequency.value = 600;
      }

      oscillator.type = "square";

      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.30);

      oscillator.start(now);
      oscillator.stop(now + 0.32);
    } catch (e) {
      console.log("Alert sound error:", e);
    }
  }

  // Add sound button to the alert panel
  (function addAlertSoundButton() {
    const panel = document.querySelector(".alerts-panel") ||
                  document.querySelector(".alert-panel") ||
                  document.querySelector("#alerts-panel");

    const container = panel || document.body;

    const btn = document.createElement("button");
    btn.id = "alert-sound-btn";
    btn.textContent = "?? ENABLE SOUND";
    btn.title = "Enable alert sound";

    btn.style.cssText = `
      position: fixed;
      right: 25px;
      top: 92px;
      z-index: 9999;
      padding: 7px 12px;
      border: 1px solid #25d99a;
      border-radius: 5px;
      background: #101a20;
      color: #25d99a;
      font-family: monospace;
      font-size: 11px;
      cursor: pointer;
    `;

    btn.onclick = enableAlertSound;
    container.appendChild(btn);
  })();

  async function loadRealtimeAlerts() {
    try {
      const response = await fetch("/api/alerts", {
        cache: "no-store"
      });

      if (!response.ok) return;

      const alerts = await response.json();

      alerts.forEach(alert => {
        if (!alert || !alert.id) return;

        if (!shownAlertIds.has(alert.id)) {
          shownAlertIds.add(alert.id);

          renderAlert(alert, true);

          if (alert.event_type === "person_detected") {
            playAlertSound(alert.severity || "LOW");
          } else {
            playAlertSound(alert.severity || "LOW");
          }
        }
      });

    } catch (error) {
      console.log("Realtime alert polling error:", error);
    }
  }

  // Load existing alerts immediately
  loadRealtimeAlerts();

  // Check for new alerts every second
  setInterval(loadRealtimeAlerts, 1000);

'''

if "loadRealtimeAlerts()" in s:
    print("REAL-TIME ALERT CODE ALREADY EXISTS")
elif marker not in s:
    print("ERROR: marker not found")
else:
    p.write_text(s.replace(marker, code + marker, 1))
    print("REAL-TIME ALERT + SOUND ADDED")
