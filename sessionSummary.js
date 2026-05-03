// --- SESSION SUMMARY MODULE ---
// Generates coaching tips and controls the post-video summary modal.

/**
 * Generate contextual coaching tips based on session metrics.
 * Returns an array of { icon, text, type } objects.
 */
function generateTips(summary) {
    const tips = [];
    const stability = parseFloat(summary.avgStability);
    const smoothness = parseFloat(summary.smoothnessScore);
    const peak = parseFloat(summary.peakVelocity);

    // --- Stability Tips ---
    if (stability < 50) {
        tips.push({
            icon: '⚠️',
            text: 'Your center of mass was often far from the optimal tension line. Focus on hip positioning — keep your weight over your feet and hips close to the wall.',
            type: 'warning'
        });
    } else if (stability < 70) {
        tips.push({
            icon: '💡',
            text: 'Good base awareness! Try flagging or drop-kneeing to fine-tune alignment on steeper sections.',
            type: 'tip'
        });
    } else {
        tips.push({
            icon: '✅',
            text: 'Excellent body positioning — you stayed well-aligned with the tension line throughout the climb.',
            type: 'success'
        });
    }

    // --- Smoothness Tips ---
    if (smoothness < 50) {
        tips.push({
            icon: '⚠️',
            text: 'Your movement had significant speed variation. Practice slow, controlled reaches — try "quiet feet" drills to build body tension.',
            type: 'warning'
        });
    } else if (smoothness < 75) {
        tips.push({
            icon: '💡',
            text: 'Decent flow! Minimize campus-style lunges by pre-reading the next two holds before you move.',
            type: 'tip'
        });
    } else {
        tips.push({
            icon: '✅',
            text: 'Very fluid movement — great climbing economy. Your body tension is dialed.',
            type: 'success'
        });
    }

    // --- Peak Velocity (Dyno Detection) ---
    if (peak > 0.25) {
        tips.push({
            icon: '⚡',
            text: 'High peak velocity detected (was it a dyno?).',
            type: 'tip'
        });
    }

    // --- General Encouragement ---
    if (stability >= 70 && smoothness >= 75) {
        tips.push({
            icon: '🏆',
            text: 'Good session! Both your positioning and movement quality are strong.',
            type: 'success'
        });
    }

    return tips;
}

/**
 * Builds or reveals the session-summary modal and populates it with data.
 * @param {object} summary - The object returned by getSessionSummary()
 * @param {object} callbacks - { onReplay, onDismiss }
 */ 
export function showSessionSummary(summary, callbacks = {}) {
    let modal = document.getElementById('session-summary-modal');
    if (!modal) return; // Bail if the HTML isn't in the page

    const tips = generateTips(summary);

    // --- Populate Metrics ---
    const setVal = (id, val) => {
        const el = modal.querySelector(`#${id}`);
        if (el) el.textContent = val;
    };

    setVal('summary-stability', `${summary.avgStability}%`);
    setVal('summary-smoothness', summary.smoothnessScore);
    setVal('summary-avg-velocity', `${(parseFloat(summary.avgVelocity) * 100).toFixed(1)}% bh/f`);
    setVal('summary-peak-velocity', `${(parseFloat(summary.peakVelocity) * 100).toFixed(1)}% bh/f`);
    setVal('summary-frames', summary.totalFramesAnalyzed.toLocaleString());

    // Color-code the headline metrics
    const stabilityEl = modal.querySelector('#summary-stability');
    const smoothnessEl = modal.querySelector('#summary-smoothness');
    if (stabilityEl) {
        stabilityEl.style.color = parseFloat(summary.avgStability) > 70 ? '#67f2c4' : parseFloat(summary.avgStability) > 50 ? '#ffd166' : '#ff6b6b';
    }
    if (smoothnessEl) {
        smoothnessEl.style.color = parseFloat(summary.smoothnessScore) > 75 ? '#67f2c4' : parseFloat(summary.smoothnessScore) > 50 ? '#ffd166' : '#ff6b6b';
    }

    // --- Populate Tips ---
    const tipsList = modal.querySelector('#summary-tips-list');
    if (tipsList) {
        tipsList.innerHTML = '';
        tips.forEach(tip => {
            const li = document.createElement('li');
            li.className = `tip-item tip-${tip.type}`;
            li.innerHTML = `<span class="tip-icon">${tip.icon}</span><span class="tip-text">${tip.text}</span>`;
            tipsList.appendChild(li);
        });
    }

    // --- Wire Buttons ---
    const replayBtn = modal.querySelector('#summary-replay-btn');
    const dismissBtn = modal.querySelector('#summary-dismiss-btn');

    // Clone & replace to remove old listeners
    if (replayBtn) {
        const newReplay = replayBtn.cloneNode(true);
        replayBtn.parentNode.replaceChild(newReplay, replayBtn);
        newReplay.addEventListener('click', () => {
            hideSessionSummary();
            if (callbacks.onReplay) callbacks.onReplay();
        });
    }
    if (dismissBtn) {
        const newDismiss = dismissBtn.cloneNode(true);
        dismissBtn.parentNode.replaceChild(newDismiss, dismissBtn);
        newDismiss.addEventListener('click', () => {
            hideSessionSummary();
            if (callbacks.onDismiss) callbacks.onDismiss();
        });
    }

    // --- Show ---
    modal.classList.add('visible');
}

/**
 * Hides the session-summary modal with a transition.
 */
export function hideSessionSummary() {
    const modal = document.getElementById('session-summary-modal');
    if (modal) modal.classList.remove('visible');
}
