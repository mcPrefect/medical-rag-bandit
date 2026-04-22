"""
Gradio demo: autonomous medical RAG with contextual bandit.
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
        w_quality /= total
        w_latency /= total
        w_safety /= total

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


def run_pipeline(question, context_text, top_k_fast, top_k_deep, top_k_kg):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    contexts = [s.strip() for s in context_text.split("\n") if s.strip()]
    if not contexts:
        return "Please provide some context.", "", "", ""

    features = extract_context(question, contexts, bandit=BANDIT, kg_arm=KG_ARM)

    arm, probs, ucb_scores = BANDIT.select_arm_with_probs(features)
    arm_name = ARM_NAMES[arm]

    decision_info = (
        f"### Selected: {arm_name}\n\n"
        f"| Arm | UCB Score | Probability |\n"
        f"|---|---|---|\n"
        f"| Fast (BM25) | {ucb_scores[0]:.4f} | {probs[0]:.1%} |\n"
        f"| Deep (Semantic) | {ucb_scores[1]:.4f} | {probs[1]:.1%} |\n"
        f"| Graph (KG) | {ucb_scores[2]:.4f} | {probs[2]:.1%} |\n\n"
        f"*alpha = {BANDIT.alpha:.4f}, step {BANDIT.t}*"
    )

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
        retrieved = retrieve_kg(question, contexts, top_k=int(top_k_kg), kg_arm=KG_ARM)
    retrieval_time = time.time() - t0

    retrieved_text = "\n\n".join([f"**{i+1}.** {s}" for i, s in enumerate(retrieved)])
    retrieved_text += f"\n\n*Retrieved {len(retrieved)} sentences in {retrieval_time*1000:.0f}ms*"

    t0 = time.time()
    predicted, confidence = answer_question(question, retrieved, max_new_tokens=50)
    llm_time = time.time() - t0

    is_safe, reason, details = VALIDATOR.validate(
        question=question,
        retrieved_context=retrieved,
        predicted_answer=predicted,
        confidence=confidence,
    )

    if is_safe:
        safety_text = f"**Passed** - all safety checks clear\n\n*{reason}*"
        final_answer = predicted
    else:
        safety_text = f"**Abstained**\n\n*{reason}*"
        final_answer = "abstain"

    total_time = retrieval_time + llm_time

    answer_text = (
        f"## {final_answer.upper()}\n\n"
        f"*Confidence: {confidence:.3f} (threshold {VALIDATOR.confidence_threshold})*\n\n"
        f"*{total_time:.2f}s total, {retrieval_time*1000:.0f}ms retrieval, "
        f"{llm_time:.2f}s generation*"
    )

    return answer_text, decision_info, retrieved_text, safety_text


EXAMPLES = [
    [
        "Does aspirin reduce the risk of cardiovascular events in healthy adults?",
        "A meta-analysis of 13 trials examined daily aspirin use in primary prevention.\nAspirin reduced major cardiovascular events by 11% in patients without prior cardiovascular disease.\nThe benefit was offset partially by an increased risk of major bleeding.\nCurrent guidelines recommend individual risk assessment before initiating aspirin therapy.",
    ],
    [
        "Is metformin safe to continue in a patient with chronic kidney disease?",
        "Metformin is contraindicated in severe renal impairment due to risk of lactic acidosis.\nThe threshold for discontinuation is an eGFR below 30 mL/min/1.73m2.\nDose reduction is recommended when eGFR falls between 30 and 45.\nRenal function should be monitored every 3-6 months in patients on metformin.",
    ],
    [
        "Does cognitive behavioural therapy improve outcomes in generalised anxiety disorder?",
        "Multiple randomised controlled trials support CBT as a first-line treatment for GAD.\nResponse rates of 50-60% are reported in meta-analyses.\nCBT effects are durable at 12-month follow-up compared to pharmacotherapy.\nCombined CBT and medication shows modest additional benefit over either alone.",
    ],
    [
        "Is beta blocker therapy beneficial in patients with heart failure?",
        "Beta blockers reduce all-cause mortality in stable heart failure with reduced ejection fraction.\nCarvedilol, bisoprolol, and metoprolol succinate are evidence-based choices.\nThey are contraindicated in acute decompensated heart failure.\nTitration should begin at low doses with gradual uptitration.",
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
**University of Limerick - BSc AI & Machine Learning FYP**

A contextual bandit that automatically selects retrieval strategies, validates safety,
and updates its own policy without human intervention.
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
                        lines=8,
                    )
                    with gr.Row():
                        top_k_fast = gr.Number(label="Fast top-k", value=3, precision=0, minimum=1, maximum=10)
                        top_k_deep = gr.Number(label="Deep top-k", value=5, precision=0, minimum=1, maximum=10)
                        top_k_kg   = gr.Number(label="KG top-k",   value=5, precision=0, minimum=1, maximum=10)
                    submit_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

                with gr.Column(scale=1):
                    answer_output    = gr.Markdown(label="Answer")
                    decision_output  = gr.Markdown(label="Bandit Decision")
                    safety_output    = gr.Markdown(label="Safety Validation")
                    retrieved_output = gr.Markdown(label="Retrieved Evidence")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input, context_input],
                label="Example queries (click to load)",
            )

            submit_btn.click(
                fn=run_pipeline,
                inputs=[question_input, context_input,
                        top_k_fast, top_k_deep, top_k_kg],
                outputs=[answer_output, decision_output,
                         retrieved_output, safety_output],
            )

        with gr.Tab("Settings"):
            gr.Markdown("Adjust the system's parameters and apply them live. Changes take effect on the next query.")

            gr.Markdown("### Reward Function")
            gr.Markdown("*Weights are automatically normalised to sum to 1.0.*")
            with gr.Row():
                w_guideline = gr.Slider(0, 1, value=0.55, step=0.05, label="Guideline adherence")
                w_quality   = gr.Slider(0, 1, value=0.25, step=0.05, label="Answer quality")
                w_latency   = gr.Slider(0, 1, value=0.10, step=0.05, label="Latency")
                w_safety    = gr.Slider(0, 1, value=0.10, step=0.05, label="Safety")

            with gr.Row():
                time_budget = gr.Number(label="Time budget (seconds)", value=10.0)
                kill_switch = gr.Checkbox(label="Safety kill-switch (zeros reward on failure)", value=True)

            gr.Markdown("### Safety Validator")
            with gr.Row():
                confidence_threshold = gr.Slider(0, 1, value=0.7, step=0.05, label="Confidence threshold")
                min_evidence         = gr.Number(label="Min evidence sentences", value=2, precision=0)

            gr.Markdown("### Bandit")
            with gr.Row():
                alpha        = gr.Slider(0.1, 5.0, value=2.0, step=0.1, label="Initial alpha (exploration)")
                reset_bandit = gr.Checkbox(label="Reset bandit (clear all learned weights)", value=False)

            apply_btn       = gr.Button("Apply Settings", variant="primary")
            settings_output = gr.Markdown()

            apply_btn.click(
                fn=update_settings,
                inputs=[w_guideline, w_quality, w_latency, w_safety,
                        time_budget, kill_switch,
                        confidence_threshold, min_evidence,
                        alpha, reset_bandit],
                outputs=settings_output,
            )



if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
