import os
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from models.gan_module import PrecipGAN
from utils import model_classes


DEFAULT_MODEL_PATH = "lightning/precip_gan/GAN_epoch=3-val_loss=0.001675.ckpt"
DEFAULT_H5_PATH = "data/CIKM/cikm_oversampled_v2.h5"
NUM_INPUT_FRAMES = 12
BATCH_THRESHOLD = 0.3


st.set_page_config(page_title="Radar Nowcasting Demo", layout="wide")

st.markdown(
    """
<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 2rem;}
    div.stButton > button:first-child {
        height: 3em;
        width: 100%;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def calculate_all_metrics(target: np.ndarray, pred: np.ndarray, threshold: float = 0.3) -> Tuple[float, float, float, float, float]:
    mse = float(np.mean((target - pred) ** 2))
    mae = float(np.mean(np.abs(target - pred)))

    t_mask = (target >= threshold).astype(np.int32)
    p_mask = (pred >= threshold).astype(np.int32)

    hits = int(np.sum(t_mask * p_mask))
    false_alarms = int(np.sum((1 - t_mask) * p_mask))
    misses = int(np.sum(t_mask * (1 - p_mask)))
    correct_negatives = int(np.sum((1 - t_mask) * (1 - p_mask)))

    csi_denom = hits + misses + false_alarms
    csi = hits / csi_denom if csi_denom > 0 else 0.0

    far_denom = hits + false_alarms
    far = false_alarms / far_denom if far_denom > 0 else 0.0

    hss_num = 2 * (hits * correct_negatives - misses * false_alarms)
    hss_den = (hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (false_alarms + correct_negatives)
    hss = hss_num / hss_den if hss_den > 0 else 0.0

    return mse, mae, csi, far, hss


def plot_img(img_data: np.ndarray, cmap: str = "jet"):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_data, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    return fig


def plot_csi_curve(csi_model: List[float], csi_base: List[float], model_label: str):
    x_axis = [6 * (i + 1) for i in range(len(csi_model))]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(x_axis, csi_model, "o-", color="#d1495b", linewidth=2, label=model_label)
    ax.plot(x_axis, csi_base, "s--", color="#7f8c8d", linewidth=1.5, label="Persistence")
    ax.set_xlabel("Lead Time (min)")
    ax.set_ylabel("CSI")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def get_available_model_paths() -> List[str]:
    model_paths: List[str] = []
    for root in [os.path.join("lightning", "precip_regression"), os.path.join("lightning", "precip_gan")]:
        if not os.path.exists(root):
            continue
        for current_root, _, files in os.walk(root):
            for name in files:
                if name.endswith(".ckpt"):
                    model_paths.append(os.path.join(current_root, name))

    model_paths = sorted(model_paths)
    if DEFAULT_MODEL_PATH in model_paths:
        model_paths.remove(DEFAULT_MODEL_PATH)
        model_paths.insert(0, DEFAULT_MODEL_PATH)
    return model_paths


@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None, None, f"Checkpoint not found: {model_path}"

    try:
        if "precip_gan" in model_path or os.path.basename(model_path).startswith("GAN_"):
            gan_model = PrecipGAN.load_from_checkpoint(model_path, map_location="cpu")
            model = gan_model.generator.cpu().eval()
            return model, "GAN Generator", None

        model_cls, model_label = model_classes.get_model_class(os.path.basename(model_path))
        loaded_model = model_cls.load_from_checkpoint(model_path, map_location="cpu")
        loaded_model = loaded_model.cpu().eval()
        return loaded_model, model_label, None
    except Exception as exc:
        return None, None, str(exc)


@st.cache_data
def get_h5_info(h5_path: str) -> Dict[str, int]:
    with h5py.File(h5_path, "r") as f:
        shape = f["test"]["images"].shape
    return {
        "num_samples": int(shape[0]),
        "num_frames": int(shape[1]),
        "height": int(shape[2]),
        "width": int(shape[3]),
    }


@st.cache_data
def load_sequence(h5_path: str, index: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        sample = np.array(f["test"]["images"][index], dtype=np.float32)
    return np.clip(sample, 0.0, 1.0)


def get_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def autoregressive_inference(model: torch.nn.Module, initial_seq: np.ndarray, steps: int) -> List[np.ndarray]:
    device = get_device(model)
    current_input = torch.from_numpy(initial_seq).float().unsqueeze(0).unsqueeze(2).to(device)
    predictions: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(steps):
            bsz, timesteps, channels, height, width = current_input.shape
            model_input = current_input.view(bsz, timesteps * channels, height, width)
            pred_next = model(model_input)
            if pred_next.ndim == 3:
                pred_next = pred_next.unsqueeze(1)
            pred_next = pred_next.float()
            pred_np = np.clip(pred_next.squeeze().detach().cpu().numpy(), 0.0, 1.0)
            predictions.append(pred_np)
            current_input = torch.cat([current_input[:, 1:, ...], pred_next.unsqueeze(1)], dim=1)

    return predictions


def evaluate_sequence(model: torch.nn.Module, sequence: np.ndarray) -> Dict[str, object]:
    input_seq = sequence[:NUM_INPUT_FRAMES]
    gt_frames = sequence[NUM_INPUT_FRAMES:]
    steps = int(gt_frames.shape[0])
    predictions = autoregressive_inference(model, input_seq, steps)

    metrics_model = []
    metrics_base = []
    persistence_img = np.clip(input_seq[-1], 0.0, 1.0)

    for step in range(steps):
        gt_img = np.clip(gt_frames[step], 0.0, 1.0)
        pred_img = predictions[step]
        metrics_model.append(calculate_all_metrics(gt_img, pred_img, threshold=BATCH_THRESHOLD))
        metrics_base.append(calculate_all_metrics(gt_img, persistence_img, threshold=BATCH_THRESHOLD))

    return {
        "input_seq": input_seq,
        "gt_frames": gt_frames,
        "predictions": predictions,
        "metrics_model": metrics_model,
        "metrics_base": metrics_base,
        "steps": steps,
        "persistence_img": persistence_img,
    }


def evaluate_batch(model: torch.nn.Module, h5_path: str, num_eval: int):
    metrics_pool = {
        "csi_ai": [],
        "csi_base": [],
        "mse_ai": [],
        "mse_base": [],
        "mae_ai": [],
        "mae_base": [],
        "far_ai": [],
        "far_base": [],
        "hss_ai": [],
        "hss_base": [],
    }

    progress_bar = st.progress(0)
    status_text = st.empty()
    debug_log = st.expander("Debug Log", expanded=False)

    with h5py.File(h5_path, "r") as f:
        dataset = f["test"]["images"]
        total_steps = int(dataset.shape[1] - NUM_INPUT_FRAMES)
        for key in metrics_pool:
            metrics_pool[key] = [[] for _ in range(total_steps)]

        valid_count = 0
        for index in range(num_eval):
            status_text.text(f"Evaluating sample {index + 1}/{num_eval}")
            progress_bar.progress((index + 1) / num_eval)

            try:
                sequence = np.array(dataset[index], dtype=np.float32)
                result = evaluate_sequence(model, np.clip(sequence, 0.0, 1.0))
                for step in range(result["steps"]):
                    model_metrics = result["metrics_model"][step]
                    base_metrics = result["metrics_base"][step]
                    metrics_pool["mse_ai"][step].append(model_metrics[0])
                    metrics_pool["mae_ai"][step].append(model_metrics[1])
                    metrics_pool["csi_ai"][step].append(model_metrics[2])
                    metrics_pool["far_ai"][step].append(model_metrics[3])
                    metrics_pool["hss_ai"][step].append(model_metrics[4])

                    metrics_pool["mse_base"][step].append(base_metrics[0])
                    metrics_pool["mae_base"][step].append(base_metrics[1])
                    metrics_pool["csi_base"][step].append(base_metrics[2])
                    metrics_pool["far_base"][step].append(base_metrics[3])
                    metrics_pool["hss_base"][step].append(base_metrics[4])
                valid_count += 1
            except Exception as exc:
                with debug_log:
                    st.write(f"Sample {index}: {exc}")

    progress_bar.empty()
    status_text.empty()
    return metrics_pool, valid_count


st.title("Weather Radar Nowcasting System")
st.caption("H5-backed demo using the CIKM test split and selectable regression or GAN checkpoints.")
st.divider()

available_model_paths = get_available_model_paths()
if not available_model_paths:
    st.error("No checkpoint files were found under lightning/precip_regression or lightning/precip_gan.")
    st.stop()

if not os.path.exists(DEFAULT_H5_PATH):
    st.error(f"Dataset file not found: {DEFAULT_H5_PATH}")
    st.stop()

dataset_info = get_h5_info(DEFAULT_H5_PATH)
num_future_frames = dataset_info["num_frames"] - NUM_INPUT_FRAMES
all_sequences = [f"sample_{idx:04d}" for idx in range(dataset_info["num_samples"])]

selected_model_path = st.sidebar.selectbox(
    "Model Checkpoint",
    available_model_paths,
    format_func=lambda path: path.replace("\\", "/"),
)
model, model_label, model_error = load_model(selected_model_path)
st.sidebar.caption(f"Current model: {model_label if model_label else 'Unavailable'}")
st.sidebar.caption(f"Dataset: {DEFAULT_H5_PATH}")
st.sidebar.caption(
    f"Test samples: {dataset_info['num_samples']} | Frames/sample: {dataset_info['num_frames']} | Future GT: {num_future_frames}"
)

if model is None:
    st.error(f"Failed to load model: {model_error}")
    st.stop()

if "seq_index" not in st.session_state:
    st.session_state.seq_index = 0
if "selected_seq_val" not in st.session_state:
    st.session_state.selected_seq_val = all_sequences[0]


def on_prev_click():
    if st.session_state.seq_index > 0:
        st.session_state.seq_index -= 1
        st.session_state.selected_seq_val = all_sequences[st.session_state.seq_index]


def on_next_click():
    if st.session_state.seq_index < len(all_sequences) - 1:
        st.session_state.seq_index += 1
        st.session_state.selected_seq_val = all_sequences[st.session_state.seq_index]


def on_selectbox_change():
    st.session_state.seq_index = all_sequences.index(st.session_state.selected_seq_val)


tab1, tab2 = st.tabs(["Single Sample", "Batch Evaluation"])

with tab1:
    st.caption(f"Sample progress: {st.session_state.seq_index + 1} / {len(all_sequences)}")

    col_prev, col_sel, col_next = st.columns([1, 4, 1])
    with col_prev:
        st.button("Previous", on_click=on_prev_click, disabled=st.session_state.seq_index == 0, use_container_width=True)
    with col_sel:
        st.selectbox(
            "Select sample",
            all_sequences,
            index=st.session_state.seq_index,
            key="selected_seq_val",
            on_change=on_selectbox_change,
            label_visibility="collapsed",
        )
    with col_next:
        st.button(
            "Next",
            on_click=on_next_click,
            disabled=st.session_state.seq_index == len(all_sequences) - 1,
            use_container_width=True,
        )

    current_idx = st.session_state.seq_index
    current_seq_name = all_sequences[current_idx]

    with st.spinner(f"Running inference for {current_seq_name}..."):
        try:
            sequence = load_sequence(DEFAULT_H5_PATH, current_idx)
            result = evaluate_sequence(model, sequence)
        except Exception as exc:
            st.error(f"Failed to process {current_seq_name}: {exc}")
            st.stop()

    with st.expander("Input Sequence", expanded=False):
        cols = st.columns(NUM_INPUT_FRAMES)
        for idx in range(NUM_INPUT_FRAMES):
            with cols[idx]:
                st.pyplot(plot_img(result["input_seq"][idx], cmap="gray"))
                st.caption(f"T-{NUM_INPUT_FRAMES - idx}")

    col_visual, col_analysis = st.columns([1.6, 1])
    with col_visual:
        st.subheader("Forecast Comparison")
        step_slider = st.slider(
            "Lead time",
            min_value=6,
            max_value=result["steps"] * 6,
            value=6,
            step=6,
            format="%d min",
            key=f"slider_{current_seq_name}",
        )
        pred_idx = (step_slider // 6) - 1

        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown("**Ground Truth**")
            st.pyplot(plot_img(result["gt_frames"][pred_idx]))
        with vc2:
            st.markdown("**Model Prediction**")
            st.pyplot(plot_img(result["predictions"][pred_idx]))
        with vc3:
            st.markdown("**Persistence**")
            st.pyplot(plot_img(result["persistence_img"]))

    with col_analysis:
        st.subheader("Skill Trend")
        csi_model_list = [metric[2] for metric in result["metrics_model"]]
        csi_base_list = [metric[2] for metric in result["metrics_base"]]
        st.pyplot(plot_csi_curve(csi_model_list, csi_base_list, model_label))
        st.divider()

        curr_m = result["metrics_model"][pred_idx]
        curr_b = result["metrics_base"][pred_idx]
        csi_improv = (curr_m[2] - curr_b[2]) / (curr_b[2] + 1e-6) * 100.0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("CSI", f"{curr_m[2]:.3f}", f"{csi_improv:+.1f}%")
        with m2:
            st.metric("MSE", f"{curr_m[0]:.4f}", f"{curr_m[0] - curr_b[0]:+.4f}", delta_color="inverse")
        with m3:
            st.metric("MAE", f"{curr_m[1]:.4f}", f"{curr_m[1] - curr_b[1]:+.4f}", delta_color="inverse")

        m4, m5 = st.columns(2)
        with m4:
            st.metric("FAR", f"{curr_m[3]:.3f}", f"{curr_m[3] - curr_b[3]:+.3f}", delta_color="inverse")
        with m5:
            st.metric("HSS", f"{curr_m[4]:.3f}", f"{curr_m[4] - curr_b[4]:+.3f}")

with tab2:
    st.subheader("Batch Evaluation on H5 Test Split")
    st.caption("This evaluates the selected checkpoint against the persistence baseline on the test split.")

    num_eval = st.slider("Number of test samples", 10, dataset_info["num_samples"], min(50, dataset_info["num_samples"]))
    start_batch = st.button("Run Batch Evaluation", key="run_batch")

    if start_batch:
        metrics_pool, valid_count = evaluate_batch(model, DEFAULT_H5_PATH, num_eval)

        if valid_count <= 0:
            st.error("No valid samples were evaluated.")
        else:
            avg_metrics = {}
            for key, values in metrics_pool.items():
                avg_metrics[key] = [float(np.mean(step_values)) if step_values else 0.0 for step_values in values]

            x_axis = [6 * (step + 1) for step in range(num_future_frames)]

            col_chart, col_data = st.columns([1.5, 1])
            with col_chart:
                st.subheader("Average CSI")
                st.pyplot(plot_csi_curve(avg_metrics["csi_ai"], avg_metrics["csi_base"], model_label))

            with col_data:
                st.subheader("Metrics Table")
                df_res = pd.DataFrame(
                    {
                        "Time (min)": x_axis,
                        f"{model_label} CSI": avg_metrics["csi_ai"],
                        "Base CSI": avg_metrics["csi_base"],
                        f"{model_label} MSE": avg_metrics["mse_ai"],
                        "Base MSE": avg_metrics["mse_base"],
                        f"{model_label} MAE": avg_metrics["mae_ai"],
                        "Base MAE": avg_metrics["mae_base"],
                        f"{model_label} FAR": avg_metrics["far_ai"],
                        "Base FAR": avg_metrics["far_base"],
                        f"{model_label} HSS": avg_metrics["hss_ai"],
                        "Base HSS": avg_metrics["hss_base"],
                    }
                )
                st.dataframe(df_res, use_container_width=True)

            mean_csi_ai = float(np.mean(avg_metrics["csi_ai"]))
            mean_csi_base = float(np.mean(avg_metrics["csi_base"]))
            mean_far_ai = float(np.mean(avg_metrics["far_ai"]))
            mean_far_base = float(np.mean(avg_metrics["far_base"]))
            mean_hss_ai = float(np.mean(avg_metrics["hss_ai"]))
            mean_hss_base = float(np.mean(avg_metrics["hss_base"]))

            st.success(f"Evaluated {valid_count} samples.")
            summary_df = pd.DataFrame(
                [
                    {"Metric": "CSI", model_label: mean_csi_ai, "Persistence": mean_csi_base},
                    {"Metric": "FAR", model_label: mean_far_ai, "Persistence": mean_far_base},
                    {"Metric": "HSS", model_label: mean_hss_ai, "Persistence": mean_hss_base},
                ]
            )
            st.dataframe(summary_df, use_container_width=True)
