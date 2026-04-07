import streamlit as st
import random
import time
from env.environment import make_env
from env.models import (
    ClassifyAction, DraftResponseAction, EscalateAction,
    RequestInfoAction, ResolveAction, IssueCategory, Priority,
    EscalationTeam, ActionType
)
from tasks.registry import ALL_TASK_IDS

st.set_page_config(page_title="Support AI v2.1", layout="wide", page_icon="💎")

# --- ADVANCED 3D PARTICLE BACKGROUND ---
st.components.v1.html("""
<div id="three-container" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -2;"></div>
<div id="gradient-overlay" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; background: radial-gradient(circle at 50% 50%, rgba(106, 17, 203, 0.2) 0%, rgba(37, 117, 252, 0.1) 100%); mix-blend-mode: overlay;"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('three-container').appendChild(renderer.domElement);

    // Particle System
    const particlesGeometry = new THREE.BufferGeometry();
    const posArray = new Float32Array(5000 * 3);
    for(let i=0; i < 5000 * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * 100;
    }
    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    
    const particlesMaterial = new THREE.PointsMaterial({
        size: 0.15,
        color: '#ffffff',
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending
    });
    
    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);
    
    // Abstract Flowing Shape
    const geometry = new THREE.IcosahedronGeometry(15, 2);
    const material = new THREE.MeshNormalMaterial({ wireframe: true, transparent: true, opacity: 0.3 });
    const sphere = new THREE.Mesh(geometry, material);
    scene.add(sphere);

    camera.position.z = 40;

    let mouseX = 0;
    let mouseY = 0;
    document.addEventListener('mousemove', (e) => {
        mouseX = (e.clientX - window.innerWidth / 2) * 0.05;
        mouseY = (e.clientY - window.innerHeight / 2) * 0.05;
    });

    function animate() {
        requestAnimationFrame(animate);
        particlesMesh.rotation.y += 0.002;
        sphere.rotation.y += 0.005;
        sphere.rotation.x += 0.003;
        
        particlesMesh.rotation.y += (mouseX * 0.001);
        particlesMesh.rotation.x += (mouseY * 0.001);
        
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
</script>
<style>
    body { margin: 0; overflow: hidden; background: #0b0e14; }
</style>
""", height=0)

# --- GLOWING GLASSMORPHISM CSS ---
st.markdown("""
    <style>
    /* Force Streamlit backgrounds to transparent */
    .stApp, .main, .stAppHeader, [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        color: white !important;
    }

    /* Cards with better contrast and glow */
    .ticket-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(20px) saturate(180%);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f1f5f9 !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .ticket-card::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(37, 117, 252, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }

    /* Global text color - less aggressive to allow default component colors */
    h1, h2, h3, h4, h5, h6, b {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Target markdown and normal text */
    .stMarkdown p, .stMarkdown span, label {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ensure warning/info boxes have dark text for visibility on light backgrounds */
    [data-testid="stNotification"] p, [data-testid="stNotification"] li {
        color: #1e293b !important;
    }

    /* Fix expander header text color */
    [data-testid="stExpander"] p, [data-testid="stExpander"] span, [data-testid="stExpander"] summary {
        color: #f8fafc !important;
    }

    /* JSON display visibility - force background and text color */
    [data-testid="stJson"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 10px;
    }
    
    [data-testid="stJson"] span {
        color: #94a3b8 !important; /* light gray for labels */
    }
    
    [data-testid="stJson"] .string {
        color: #38bdf8 !important; /* blue for strings */
    }

    h1 {
        text-shadow: 0 0 20px rgba(37, 117, 252, 0.5);
        font-weight: 800 !important;
    }


    .stButton>button {
        background: linear-gradient(135deg, rgba(37, 117, 252, 0.6) 0%, rgba(106, 17, 203, 0.6) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(37, 117, 252, 0.9) 0%, rgba(106, 17, 203, 0.9) 100%);
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 10px 20px rgba(37, 117, 252, 0.4);
    }

    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 16px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Session State
if "env" not in st.session_state:
    st.session_state.env = None
    st.session_state.obs = None
    st.session_state.history = []
    st.session_state.feedback = ""

def reset_env(task_id):
    st.session_state.env = make_env(task_id)
    st.session_state.obs = st.session_state.env.reset()
    st.session_state.history = []
    st.session_state.feedback = "New task started. Waiting for classification."

# Sidebar
with st.sidebar:
    st.title("⚙️ Control Panel")
    task_id = st.selectbox("Select Scenario", ALL_TASK_IDS, index=1)
    if st.button("🔄 Reset / Start New Episode"):
        reset_env(task_id)
    
    st.divider()
    if st.session_state.env:
        st.metric("Current Score", f"{st.session_state.env.final_score():.2f}")
        st.metric("Steps Used", f"{st.session_state.obs['step']} / {st.session_state.obs['max_steps']}")

# Main UI
st.title("🎫 CustomerSupport AI Simulation")

if not st.session_state.env:
    st.info("👈 Select a task and click Reset to begin!")
else:
    obs = st.session_state.obs
    ticket = obs["ticket"]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📬 Incoming Ticket")
        with st.container():
            st.markdown(f"""
            <div class="ticket-card">
                <h3>{ticket['subject']}</h3>
                <p><b>Customer:</b> {ticket['customer']['name']} (<span class="status-badge" style="background:#e3f2fd; color:#1565c0;">{ticket['customer']['account_tier'].upper()}</span>)</p>
                <hr>
                <p style="font-size: 1.1rem; line-height: 1.6;">{ticket['body']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        if st.session_state.feedback:
            st.warning(f"💡 **Latest Feedback**: {st.session_state.feedback}")

    with col2:
        st.subheader("🛠️ Agent Actions")
        
        if obs["resolved"]:
            st.success("✅ Episode Complete!")
            st.balloons()
        else:
            tabs = st.tabs(["Classify", "Response", "Info", "Escalate", "Resolve"])
            
            with tabs[0]:
                c_cat = st.selectbox("Category", list(IssueCategory))
                c_prio = st.selectbox("Priority", list(Priority))
                if st.button("Apply Classification"):
                    action = ClassifyAction(category=c_cat, priority=c_prio, confidence=1.0)
                    res = st.session_state.env.step(action)
                    st.session_state.obs = res.observation
                    st.session_state.feedback = res.info["feedback"]
                    st.rerun()

            with tabs[1]:
                r_body = st.text_area("Draft Message", placeholder="Hi there, I'm sorry to hear...")
                if st.button("Send Response"):
                    action = DraftResponseAction(subject=f"Re: {ticket['subject']}", body=r_body)
                    res = st.session_state.env.step(action)
                    st.session_state.obs = res.observation
                    st.session_state.feedback = res.info["feedback"]
                    st.rerun()
            
            with tabs[2]:
                q1 = st.text_input("Question 1")
                if st.button("Request Info"):
                    action = RequestInfoAction(questions=[q1], body=f"Please provide: {q1}")
                    res = st.session_state.env.step(action)
                    st.session_state.obs = res.observation
                    st.session_state.feedback = res.info["feedback"]
                    st.rerun()

            with tabs[3]:
                e_team = st.selectbox("Escalation Team", list(EscalationTeam))
                e_reason = st.text_area("Reason for Escalation")
                if st.button("Escalate Ticket"):
                    action = EscalateAction(team=e_team, reason=e_reason)
                    res = st.session_state.env.step(action)
                    st.session_state.obs = res.observation
                    st.session_state.feedback = res.info["feedback"]
                    st.rerun()

            with tabs[4]:
                res_sum = st.text_area("Resolution Summary")
                if st.button("Resolve Ticket", type="primary"):
                    action = ResolveAction(resolution_summary=res_sum, satisfied=True)
                    res = st.session_state.env.step(action)
                    st.session_state.obs = res.observation
                    st.session_state.feedback = res.info["feedback"]
                    st.rerun()

    st.divider()
    st.subheader("📜 Action History")
    for act in reversed(obs["actions_taken"]):
        with st.expander(f"{act['action_type'].upper()} Step"):
            st.json(act)
