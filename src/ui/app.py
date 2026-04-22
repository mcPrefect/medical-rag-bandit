"""
Gradio demo: autonomous medical RAG with contextual bandit.

Usage:
    python src/ui/app.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gradio as gr

from retrieval.fast_arm import retrieve_fast
from retrieval.deep_arm import retrieve_deep
from retrieval.kg_arm import KnowledgeGraphArm, retrieve_kg
from bandit.linucb import LinUCB, extract_context
from llm.llm_wrapper import answer_question
from safety.validator import SafetyValidator
from reward.reward_function import RewardFunction
from utils.config import load_config


print("Loading system components...")
CONFIG = load_config("configs/config.yaml")
KG_ARM = KnowledgeGraphArm()

REWARD_FN = None
VALIDATOR = None
BANDIT = None

ARM_NAMES = ["Fast (BM25)", "Deep (Semantic)", "Graph (KG)"]


def rebuild_components(
    w_guideline, w_quality, w_latency, w_safety,
    time_budget, kill_switch,
    confidence_threshold, min_evidence,
    alpha, reset_bandit,
):
    global REWARD_FN, VALIDATOR, BANDIT

    total = w_guideline + w_quality + w_latency + w_safety
    if total > 0:
        w_guideline /= total
        w_quality   /= total
        w_latency   /= total
        w_safety    /= total

    REWARD_FN = RewardFunction(
        w_guideline=w_guideline,
        w_quality=w_quality,
        w_latency=w_latency,
        w_safety=w_safety,
        time_budget=time_budget,
        safety_kill_switch=kill_switch,
        use_bertscore=True,
    )

    VALIDATOR = SafetyValidator(
        confidence_threshold=confidence_threshold,
        min_evidence_sentences=int(min_evidence),
    )

    if BANDIT is None or reset_bandit:
        BANDIT = LinUCB(n_arms=3, n_features=10, alpha=alpha)
        if not reset_bandit:
            weights_path = CONFIG['data']['output_dir'] + "bandit_weights.pkl"
            BANDIT.load_weights(weights_path)


rebuild_components(
    w_guideline=CONFIG['reward']['w_guideline'],
    w_quality=CONFIG['reward']['w_quality'],
    w_latency=CONFIG['reward']['w_latency'],
    w_safety=CONFIG['reward']['w_safety'],
    time_budget=CONFIG['reward']['time_budget'],
    kill_switch=CONFIG['reward']['safety_kill_switch'],
    confidence_threshold=CONFIG['safety']['confidence_threshold'],
    min_evidence=CONFIG['safety']['min_evidence_sentences'],
    alpha=CONFIG['bandit']['alpha'],
    reset_bandit=False,
)

print("Ready.\n")


def _confidence_bar(confidence, threshold):
    filled = int(confidence * 10)
    bar = "🟩" * filled + "⬜" * (10 - filled)
    status = "✅" if confidence >= threshold else "⚠️"
    return f"{status} {bar} {confidence:.1%}"


def _safety_table(is_safe, reason, details):
    def row(name, passed, note=""):
        icon = "✅" if passed else "🚫"
        note_short = note[:60] + "..." if len(note) > 60 else note
        return f"| {icon} | {name} | {note_short} |"

    conf_pass   = details.get('confidence',        {}).get('pass', True)
    evid_pass   = details.get('evidence',          {}).get('pass', True)
    contra_pass = details.get('contraindications', {}).get('pass', True)
    sanit_pass  = details.get('sanity',            {}).get('pass', True)

    table = (
        "| | Layer | Detail |\n"
        "|---|---|---|\n"
        + row("Confidence",        conf_pass,
              details.get('confidence',        {}).get('reason', '')) + "\n"
        + row("Evidence",          evid_pass,
              details.get('evidence',          {}).get('reason', '')) + "\n"
        + row("Contraindications", contra_pass,
              details.get('contraindications', {}).get('reason', '')) + "\n"
        + row("Format sanity",     sanit_pass,
              details.get('sanity',            {}).get('reason', ''))
    )

    if is_safe:
        header = "### ✅ All safety checks passed"
    else:
        header = f"### 🚫 Abstained\n*{reason}*"

    return header + "\n\n" + table


def run_pipeline(question, context_text, top_k_fast, top_k_deep, top_k_kg):
    if not question.strip():
        return "Please enter a question.", "", "", "", ""

    contexts = [s.strip() for s in context_text.split("\n") if s.strip()]
    if not contexts:
        return "Please provide some context.", "", "", "", ""

    # SENSE
    features = extract_context(question, contexts, bandit=BANDIT, kg_arm=KG_ARM)

    # DECIDE
    arm, probs, ucb_scores = BANDIT.select_arm_with_probs(features)
    arm_name  = ARM_NAMES[arm]
    arm_perfs = BANDIT.get_arm_performance()

    decision_info = (
        f"### Selected: {arm_name}\n\n"
        f"| Arm | UCB Score | Probability | Session Avg Reward |\n"
        f"|---|---|---|---|\n"
        f"| Fast (BM25) | {ucb_scores[0]:.4f} | {probs[0]:.1%} | {arm_perfs[0]:.3f} |\n"
        f"| Deep (Semantic) | {ucb_scores[1]:.4f} | {probs[1]:.1%} | {arm_perfs[1]:.3f} |\n"
        f"| Graph (KG) | {ucb_scores[2]:.4f} | {probs[2]:.1%} | {arm_perfs[2]:.3f} |\n\n"
        f"*α = {BANDIT.alpha:.4f} · step {BANDIT.t}*"
    )

    # ACT — retrieve
    t0 = time.time()
    if arm == 0:
        retrieved = retrieve_fast(question, contexts, top_k=int(top_k_fast))
    elif arm == 1:
        retrieved = retrieve_deep(
            question, contexts,
            top_k=int(top_k_deep),
            model_name=CONFIG['retrieval']['deep_arm']['model_name'],
        )
    else:
        retrieved = retrieve_kg(
            question, contexts, top_k=int(top_k_kg), kg_arm=KG_ARM
        )
    retrieval_time = time.time() - t0

    retrieved_text = "\n\n".join(
        [f"**{i+1}.** {s}" for i, s in enumerate(retrieved)]
    )
    retrieved_text += f"\n\n*{len(retrieved)} sentences · {retrieval_time*1000:.0f}ms*"

    # ACT — generate
    t0 = time.time()
    predicted, confidence = answer_question(
        question, retrieved, max_new_tokens=50
    )
    llm_time = time.time() - t0

    # VALIDATE
    is_safe, reason, details = VALIDATOR.validate(
        question=question,
        retrieved_context=retrieved,
        predicted_answer=predicted,
        confidence=confidence,
    )

    final_answer = predicted if is_safe else "abstain"
    safety_text  = _safety_table(is_safe, reason, details)
    total_time   = retrieval_time + llm_time

    # LEARN
    reward, components = REWARD_FN.compute_reward(
        predicted_answer=predicted,
        gold_answer=predicted,
        generated_response=" ".join(retrieved),
        reference_text="",
        time_taken=total_time,
        safety_passed=is_safe,
    )

    conf_bar = _confidence_bar(confidence, VALIDATOR.confidence_threshold)

    answer_text = (
        f"## {final_answer.upper()}\n\n"
        f"**Confidence:** {conf_bar}\n\n"
        f"*{total_time:.2f}s total · "
        f"{retrieval_time*1000:.0f}ms retrieval · "
        f"{llm_time:.2f}s generation*"
    )

    reward_text = (
        f"### Reward: {reward:.4f}\n\n"
        f"| Component | Weight | Score | Contribution |\n"
        f"|---|---|---|---|\n"
        f"| Guideline | {REWARD_FN.w_guideline:.2f} | "
        f"{components['r_guideline']:.3f} | "
        f"{REWARD_FN.w_guideline * components['r_guideline']:.3f} |\n"
        f"| Quality | {REWARD_FN.w_quality:.2f} | "
        f"{components['r_quality']:.1f} | "
        f"{REWARD_FN.w_quality * components['r_quality']:.3f} |\n"
        f"| Latency | {REWARD_FN.w_latency:.2f} | "
        f"{components['r_latency']:.3f} | "
        f"{REWARD_FN.w_latency * components['r_latency']:.3f} |\n"
        f"| Safety | {REWARD_FN.w_safety:.2f} | "
        f"{components['r_safety']:.1f} | "
        f"{REWARD_FN.w_safety * components['r_safety']:.3f} |"
    )

    if components.get('kill_switch_triggered'):
        reward_text += "\n\n*⚠️ Kill-switch triggered — entire reward zeroed*"

    return answer_text, decision_info, retrieved_text, safety_text, reward_text


def update_settings(
    w_guideline, w_quality, w_latency, w_safety,
    time_budget, kill_switch,
    confidence_threshold, min_evidence,
    alpha, reset_bandit,
):
    rebuild_components(
        w_guideline, w_quality, w_latency, w_safety,
        time_budget, kill_switch,
        confidence_threshold, min_evidence,
        alpha, reset_bandit,
    )

    lines = [
        "**Settings applied.**\n",
        "| Parameter | Value |",
        "|---|---|",
        f"| Guideline weight | {REWARD_FN.w_guideline:.3f} |",
        f"| Quality weight | {REWARD_FN.w_quality:.3f} |",
        f"| Latency weight | {REWARD_FN.w_latency:.3f} |",
        f"| Safety weight | {REWARD_FN.w_safety:.3f} |",
        f"| Time budget | {REWARD_FN.time_budget}s |",
        f"| Kill-switch | {'on' if REWARD_FN.safety_kill_switch else 'off'} |",
        f"| Confidence threshold | {VALIDATOR.confidence_threshold} |",
        f"| Min evidence sentences | {VALIDATOR.min_evidence_sentences} |",
        f"| Bandit alpha | {BANDIT.alpha:.4f} |",
        f"| Bandit steps | {BANDIT.t} |",
    ]

    if reset_bandit:
        lines.append("\n*Bandit reset. Prior learning cleared.*")

    return "\n".join(lines)


EXAMPLES = [
    [
        "Does aspirin reduce the risk of cardiovascular events in healthy adults?",
        "A meta-analysis of 13 trials examined daily aspirin use in primary prevention.\n"
        "Aspirin reduced major cardiovascular events by 11% in patients without prior cardiovascular disease.\n"
        "The benefit was offset partially by an increased risk of major bleeding.\n"
        "Current guidelines recommend individual risk assessment before initiating aspirin therapy.",
    ],
    [
        "Is metformin safe to continue in a patient with chronic kidney disease?",
        "Metformin is contraindicated in severe renal impairment due to risk of lactic acidosis.\n"
        "The threshold for discontinuation is an eGFR below 30 mL/min/1.73m2.\n"
        "Dose reduction is recommended when eGFR falls between 30 and 45.\n"
        "Renal function should be monitored every 3-6 months in patients on metformin.",
    ],
    [
        "Does cognitive behavioural therapy improve outcomes in generalised anxiety disorder?",
        "Multiple randomised controlled trials support CBT as a first-line treatment for GAD.\n"
        "Response rates of 50-60% are reported in meta-analyses.\n"
        "CBT effects are durable at 12-month follow-up compared to pharmacotherapy.\n"
        "Combined CBT and medication shows modest additional benefit over either alone.",
    ],
    [
        "Is beta blocker therapy beneficial in patients with heart failure?",
        "Beta blockers reduce all-cause mortality in stable heart failure with reduced ejection fraction.\n"
        "Carvedilol, bisoprolol, and metoprolol succinate are evidence-based choices.\n"
        "They are contraindicated in acute decompensated heart failure.\n"
        "Titration should begin at low doses with gradual uptitration.",
    ],
    [
        "⚠️ Should aspirin be given to a patient with a bleeding disorder?",
        "Aspirin inhibits platelet aggregation and is widely used for pain and cardiovascular prevention.\n"
        "Patients with bleeding disorders have impaired haemostasis and are at high risk of haemorrhage.\n"
        "Aspirin use in patients with bleeding disorders is associated with serious adverse events including fatal haemorrhage.\n"
        "Alternative analgesics should be considered in patients with known bleeding disorders.",
    ],
]


with gr.Blocks(
    title="Medical RAG Bandit",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
) as demo:

    gr.Markdown("""
# Autonomous Medical RAG System
**University of Limerick · BSc AI & Machine Learning FYP**

A contextual bandit that selects retrieval strategies, validates safety across four independent
layers, and updates its own policy without human intervention.
""")

    with gr.Tabs():

        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Clinical Question",
                        placeholder="Enter a yes/no medical question...",
                        lines=2,
                    )
                    context_input = gr.Textbox(
                        label="Context (one sentence per line)",
                        placeholder="Paste relevant evidence here...",
                        lines=7,
                    )
                    with gr.Row():
                        top_k_fast = gr.Number(
                            label="Fast top-k", value=3, precision=0,
                            minimum=1, maximum=10,
                        )
                        top_k_deep = gr.Number(
                            label="Deep top-k", value=5, precision=0,
                            minimum=1, maximum=10,
                        )
                        top_k_kg = gr.Number(
                            label="KG top-k", value=5, precision=0,
                            minimum=1, maximum=10,
                        )
                    submit_btn = gr.Button(
                        "Run Pipeline", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    answer_output    = gr.Markdown(label="Answer")
                    safety_output    = gr.Markdown(label="Safety Validation")
                    reward_output    = gr.Markdown(label="Reward Breakdown")
                    decision_output  = gr.Markdown(label="Bandit Decision")
                    retrieved_output = gr.Markdown(label="Retrieved Evidence")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input, context_input],
                label="Example queries — last one tests the safety validator",
            )

            submit_btn.click(
                fn=run_pipeline,
                inputs=[
                    question_input, context_input,
                    top_k_fast, top_k_deep, top_k_kg,
                ],
                outputs=[
                    answer_output, decision_output,
                    retrieved_output, safety_output, reward_output,
                ],
            )

        with gr.Tab("Settings"):
            gr.Markdown(
                "Adjust parameters and apply live. To see how reward weight changes "
                "affect arm selection, tick **Reset bandit** before applying — "
                "otherwise the bandit exploits its existing learned policy."
            )

            gr.Markdown("### Reward Function")
            gr.Markdown("*Weights are normalised to sum to 1.0 automatically.*")
            with gr.Row():
                w_guideline = gr.Slider(
                    0, 1, value=0.55, step=0.05, label="Guideline adherence"
                )
                w_quality = gr.Slider(
                    0, 1, value=0.25, step=0.05, label="Answer quality"
                )
                w_latency = gr.Slider(
                    0, 1, value=0.10, step=0.05, label="Latency"
                )
                w_safety = gr.Slider(
                    0, 1, value=0.10, step=0.05, label="Safety"
                )

            with gr.Row():
                time_budget = gr.Number(
                    label="Time budget (seconds)", value=10.0
                )
                kill_switch = gr.Checkbox(
                    label="Safety kill-switch (zeros reward on failure)",
                    value=True,
                )

            gr.Markdown("### Safety Validator")
            with gr.Row():
                confidence_threshold = gr.Slider(
                    0, 1, value=0.7, step=0.05,
                    label="Confidence threshold",
                )
                min_evidence = gr.Number(
                    label="Min evidence sentences", value=2, precision=0,
                )

            gr.Markdown("### Bandit")
            with gr.Row():
                alpha = gr.Slider(
                    0.1, 5.0, value=2.0, step=0.1,
                    label="Initial alpha (exploration)",
                )
                reset_bandit = gr.Checkbox(
                    label="Reset bandit (clear all learned weights)",
                    value=False,
                    info="Tick this when changing reward weights to see the effect on arm selection.",
                )

            apply_btn       = gr.Button("Apply Settings", variant="primary")
            settings_output = gr.Markdown()

            apply_btn.click(
                fn=update_settings,
                inputs=[
                    w_guideline, w_quality, w_latency, w_safety,
                    time_budget, kill_switch,
                    confidence_threshold, min_evidence,
                    alpha, reset_bandit,
                ],
                outputs=settings_output,
            )

    gr.Markdown("""
---
*Research prototype. Not for clinical use. All outputs should be verified by a qualified clinician.*
""")


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)