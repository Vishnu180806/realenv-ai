let S = { sid: null, task: null, step: 0, cum: 0, done: false, actionsDone: [] };
let _loading = false;

document.addEventListener('DOMContentLoaded', () => {
    // Cursor Follower
    const cursor = document.querySelector('.cursor-follower');
    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    });

    // Reveal on Scroll
    const revealElements = document.querySelectorAll('.reveal, .reveal-text, .reveal-text-delayed, .reveal-btns');
    
    const revealOnScroll = () => {
        const windowHeight = window.innerHeight;
        revealElements.forEach(el => {
            const elementTop = el.getBoundingClientRect().top;
            const elementVisible = 150;
            if (elementTop < windowHeight - elementVisible) {
                el.classList.add('active');
            }
        });
    };

    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll(); // Initial check

    // Parallax Effect on Hero Sphere
    const sphere = document.querySelector('.glass-sphere');
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        if (sphere) {
            sphere.style.transform = `translateY(${scrolled * 0.3}px) rotate(${scrolled * 0.1}deg)`;
        }
    });

    // Smooth Scroll for Nav Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            if (this.getAttribute('href') === '#dashboard') return; // Handled by showDashboard
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
});

// ─── API Integration ────────────────────────────────────────────────────────

async function api(method, path, body) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const r = await fetch(path, opts);
    const d = await r.json();
    if (d.detail) throw new Error(Array.isArray(d.detail) ? d.detail.map(e=>e.msg).join(', ') : d.detail);
    return d;
}

function g(id) { return document.getElementById(id); }

async function startEpisode() {
    if (_loading) return;
    _loading = true;
    const tid = g('task-select').value;
    const startBtn = g('start-btn');
    startBtn.innerText = 'Initializing...';
    startBtn.disabled = true;

    try {
        const d = await api('POST', '/reset', { task_id: tid });
        S = { sid: d.session_id, task: tid, step: 0, cum: 0, done: false, actionsDone: [] };
        const obs = d.observation;

        // Update UI State
        g('metrics-card').style.opacity = '1';
        g('log-card').style.opacity = '1';
        g('ticket-card').style.opacity = '1';
        g('reset-btn').style.display = 'block';
        g('empty-ticket-msg').style.display = 'none';
        g('active-ticket').style.display = 'block';

        // Update Ticket
        g('t-subject').innerText = obs.ticket.subject;
        g('t-body').innerText = obs.ticket.body;
        g('t-meta').innerHTML = `
            <span class="btn-glass" style="padding: 5px 10px; font-size: 0.75rem">${obs.ticket.customer.name}</span>
            <span class="btn-glass" style="padding: 5px 10px; font-size: 0.75rem">${obs.ticket.customer.account_tier.toUpperCase()}</span>
        `;

        updateMetrics(0, 0, obs.max_steps);
        addLog('System', `Episode started: ${tid} (Session: ${S.sid.slice(0,8)})`);

        // Show original customer message in conversation thread
        g('convo-thread').style.display = 'block';
        g('convo-messages').innerHTML = '';
        addConvoMessage('customer', obs.ticket.customer.name, obs.ticket.body);
        
        // --- Automatically run bot sequences for the baseline ---
        autoRunBot(tid);

    } catch(e) { 
        addLog('Error', 'Start failed: ' + e.message); 
    } finally {
        startBtn.innerText = 'Start Episode';
        startBtn.disabled = false;
        _loading = false;
    }
}

const BOT_DATASET = {
    'billing_dispute_v1': [
        { action_type: 'classify', category: 'billing', priority: 'high', confidence: 1.0 },
        { action_type: 'draft_response', subject: 'Re: Duplicate Charge', body: 'Hello, I apologize for the duplicate charge. I have verified your account and issued a refund of $49.99 for the second charge immediately.', tone: 'professional' },
        { action_type: 'resolve', resolution_summary: 'Confirmed duplicate charge and issued a refund immediately.', satisfied: true }
    ],
    'technical_outage_v1': [
        { action_type: 'classify', category: 'technical', priority: 'urgent', confidence: 1.0 },
        { action_type: 'request_info', questions: ['Could you confirm the exact error message?', 'Are you using the API directly or a client?'], body: 'We are investigating the 500 errors. Could you provide a bit more logging information?' },
        { action_type: 'draft_response', subject: 'Re: 500 Outage', body: 'Thank you for the additional information. I have reviewed the logs and am escalating this immediately to our engineering team.', tone: 'professional' },
        { action_type: 'escalate', team: 'engineering', reason: 'High priority outage affecting API across STANDARD tier apps. 500 errors verified.', internal_notes: 'Logs attached.' }
    ],
    'enterprise_complaint_v1': [
        { action_type: 'classify', category: 'billing', priority: 'urgent', confidence: 1.0 },
        { action_type: 'request_info', questions: ["Which specific invoice ID is incorrect?", "What was the previous expected usage metric?"], body: 'I apologize for the API quota billing discrepancy. Could you specify the exact invoice ID so I can investigate?' },
        { action_type: 'draft_response', subject: 'Re: Enterprise Billing', body: 'Thank you for providing the invoice IDs. I am escalating this to the management team to correct the discrepancy for your enterprise account.', tone: 'professional' },
        { action_type: 'escalate', team: 'management', reason: 'Enterprise client facing billing error on high volume API usage.', internal_notes: 'Requires manager approval for invoice adjustment.' }
    ]
};

async function autoRunBot(tid) {
    const sequence = BOT_DATASET[tid] || [];
    for (let i = 0; i < sequence.length; i++) {
        if (S.done) break;
        await new Promise(r => setTimeout(r, 1500)); // Bot "typing" delay
        await botStep(sequence[i]);
    }
}

async function botStep(action) {
    if (!S.sid || S.done) return;

    try {
        const type = action.action_type;
        const d = await api('POST', '/step', { session_id: S.sid, action });
        
        S.step = d.observation.step;
        S.cum = d.observation.cumulative_reward;
        S.done = d.done;

        const rwd = d.reward !== undefined ? d.reward : 0;
        const fb = (d.info && d.info.feedback) || 'Action processed.';
        
        updateMetrics(S.cum, S.step, d.observation.max_steps);
        addLog(type.toUpperCase(), `Reward: ${rwd >= 0 ? '+' : ''}${rwd.toFixed(3)} | ${fb}`);
        addRewardDot(rwd);

        // ── Show bot reply in conversation thread ──
        if (type === 'draft_response') {
            addConvoMessage('bot', 'AI Assistant', action.body);
        } else if (type === 'request_info') {
            const qs = action.questions.map((q, i) => `${i+1}. ${q}`).join('\n');
            addConvoMessage('bot', 'AI Assistant', `${action.body}\n\n${qs}`);
        } else if (type === 'escalate') {
            addConvoMessage('system', '', `🔀 Ticket escalated to <strong>${action.team}</strong> — ${action.reason}`);
        } else if (type === 'resolve') {
            addConvoMessage('bot', 'AI Assistant', action.resolution_summary);
            addConvoMessage('system', '', '✅ Ticket resolved');
        } else if (type === 'classify') {
            addConvoMessage('system', '', `🏷️ Classified as <strong>${action.category}</strong> · Priority: <strong>${action.priority}</strong>`);
        }

        // Show Score feedback in conversation
        const scoreStyle = rwd > 0 ? "color:#00ff88;" : (rwd < 0 ? "color:#ff4444;" : "color:#ffbb00;");
        const scoreSign = rwd >= 0 ? '+' : '';
        addConvoMessage('system', '', `<span style="${scoreStyle}">Bot received ${scoreSign}${rwd.toFixed(2)} score</span> (${fb})`);

        if (S.done) {
            const sc = await api('GET', '/score/' + S.sid);
            addLog('FINAL', `Episode Complete. Final Score: ${sc.final_score.toFixed(4)}`);
        }

    } catch(e) { 
        addLog('Error', `Action failed: ${e.message}`); 
    }
}

function updateMetrics(score, step, maxSteps) {
    g('m-score').innerText = score.toFixed(4);
    g('m-steps').innerText = `${step}/${maxSteps}`;
}

function addLog(label, msg) {
    const container = g('log-container');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<strong>[${label}]</strong> ${msg}`;
    container.prepend(entry);
}

function addConvoMessage(role, sender, text) {
    const container = g('convo-messages');
    const bubble = document.createElement('div');
    bubble.className = `convo-msg ${role}`;
    // Convert newlines to <br> for display
    const htmlText = String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
    bubble.innerHTML = sender
        ? `<div class="convo-sender">${sender}</div><div>${htmlText}</div>`
        : `<div>${text}</div>`; // system msgs allow HTML
    container.appendChild(bubble);
    // Auto-scroll thread into view
    bubble.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function addRewardDot(r) {
    const container = g('reward-dots');
    const dot = document.createElement('div');
    dot.className = 'rdot ' + (r > 0.1 ? 'high' : r > 0 ? 'mid' : 'low');
    container.appendChild(dot);
}

function resetUI() {
    S = { sid: null, task: null, step: 0, cum: 0, done: false, actionsDone: [] };
    g('metrics-card').style.opacity = '0.5';
    g('log-card').style.opacity = '0.5';
    g('ticket-card').style.opacity = '0.5';
    g('reset-btn').style.display = 'none';
    g('empty-ticket-msg').style.display = 'block';
    g('active-ticket').style.display = 'none';
    g('log-container').innerHTML = '<div class="log-entry">Console reset. Ready for new episode.</div>';
    g('reward-dots').innerHTML = '';
    // Clear conversation thread
    g('convo-messages').innerHTML = '';
    g('convo-thread').style.display = 'none';
}
