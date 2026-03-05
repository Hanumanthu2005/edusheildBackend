document.addEventListener('DOMContentLoaded', () => {
    const warningLog = document.getElementById('warning-log');
    let tabSwitchCount = 0;

    // Function to log warnings on the UI and send to the Flask backend
    function triggerSystemViolation(violationType, message) {
        // 1. Update UI
        const time = new Date().toLocaleTimeString();
        const alertHtml = `<div class="warning-item">[${time}] <b>${violationType}:</b> ${message}</div>`;
        warningLog.insertAdjacentHTML('afterbegin', alertHtml);

        // 2. Send to Flask Backend
        fetch('/api/log_violation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ type: violationType })
        }).catch(err => console.error("Failed to log violation:", err));
    }

    // Monitor 1: Tab Switching (Visibility API)
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            tabSwitchCount++;
            triggerSystemViolation("Tab Switch", `User left the exam tab. (Total: ${tabSwitchCount})`);
        }
    });

    // Monitor 2: Window Focus (Did they click onto another monitor/application?)
    window.addEventListener("blur", () => {
        triggerSystemViolation("Window Unfocused", "Exam window lost focus.");
    });

    // Prevent copy/pasting
    document.addEventListener('copy', (e) => {
        e.preventDefault();
        triggerSystemViolation("Clipboard", "Copying content is disabled.");
    });
    
    document.addEventListener('paste', (e) => {
        e.preventDefault();
        triggerSystemViolation("Clipboard", "Pasting content is disabled.");
    });
});