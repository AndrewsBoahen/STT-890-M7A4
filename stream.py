import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris, fetch_california_housing
import warnings
warnings.filterwarnings("ignore")
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Andrews Conformal Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Conformal Prediction App")
st.header("Welcome to my first streamlit App! I built it to perform basic conformal prediction! Let's have fun!")
st.write("Upload your dataset, select your task and model, and get prediction intervals or sets!")


# ---- CUSTOM CSS ----
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9ff, #e8ecff);
        border-left: 4px solid #1a73e8;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .tutorial-step {
        background: #f0f7ff;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0;
        border-left: 4px solid #1a73e8;
    }
    .tutorial-step h4 {
        color: #1a73e8;
        margin-bottom: 0.3rem;
    }
    .highlight-box {
        background: #fff8e1;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---- PYTORCH MODELS ----
if TORCH_AVAILABLE:
    class TorchRegressorNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    class TorchClassifierNet(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    class PyTorchRegressor:
        def __init__(self, epochs=50, lr=0.01):
            self.epochs = epochs
            self.lr = lr
            self.model = None
            self.scaler = StandardScaler()
        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            X_t = torch.FloatTensor(X_scaled)
            y_t = torch.FloatTensor(y).reshape(-1, 1)
            self.model = TorchRegressorNet(X_t.shape[1])
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            self.model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = loss_fn(self.model(X_t), y_t)
                loss.backward()
                optimizer.step()
            return self
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            X_t = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                return self.model(X_t).numpy().flatten()

    class PyTorchClassifier:
        def __init__(self, epochs=50, lr=0.01):
            self.epochs = epochs
            self.lr = lr
            self.model = None
            self.scaler = StandardScaler()
            self.num_classes = None
        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            self.num_classes = len(np.unique(y))
            X_t = torch.FloatTensor(X_scaled)
            y_t = torch.LongTensor(y)
            self.model = TorchClassifierNet(X_t.shape[1], self.num_classes)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss()
            self.model.train()
            for _ in range(self.epochs):
                optimizer.zero_grad()
                loss = loss_fn(self.model(X_t), y_t)
                loss.backward()
                optimizer.step()
            return self
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            X_t = torch.FloatTensor(X_scaled)
            self.model.eval()
            with torch.no_grad():
                return torch.softmax(self.model(X_t), dim=1).numpy()

if TF_AVAILABLE:
    class KerasRegressor:
        def __init__(self, epochs=50):
            self.epochs = epochs
            self.model = None
            self.scaler = StandardScaler()
        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            self.model = keras.Sequential([
                keras.layers.Dense(64, activation="relu", input_shape=(X_scaled.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1)
            ])
            self.model.compile(optimizer="adam", loss="mse")
            self.model.fit(X_scaled, y, epochs=self.epochs, verbose=0, batch_size=32)
            return self
        def predict(self, X):
            return self.model.predict(self.scaler.transform(X), verbose=0).flatten()

    class KerasClassifier:
        def __init__(self, epochs=50):
            self.epochs = epochs
            self.model = None
            self.scaler = StandardScaler()
            self.num_classes = None
        def fit(self, X, y):
            X_scaled = self.scaler.fit_transform(X)
            self.num_classes = len(np.unique(y))
            self.model = keras.Sequential([
                keras.layers.Dense(64, activation="relu", input_shape=(X_scaled.shape[1],)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(self.num_classes, activation="softmax")
            ])
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            self.model.fit(X_scaled, y, epochs=self.epochs, verbose=0, batch_size=32)
            return self
        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)
        def predict_proba(self, X):
            return self.model.predict(self.scaler.transform(X), verbose=0)

# ---- CONFORMAL PREDICTION FUNCTIONS ----
def conformal_regression(model, X_train, y_train, X_test, y_test, alpha):
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model.fit(X_tr, y_tr)
    cal_preds = model.predict(X_cal)
    cal_scores = np.abs(y_cal - cal_preds)
    n = len(cal_scores)
    q_hat = np.quantile(cal_scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0))
    y_pred = model.predict(X_test)
    return y_pred, y_pred - q_hat, y_pred + q_hat, q_hat, cal_scores

def conformal_classification(model, X_train, y_train, X_test, alpha):
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    model.fit(X_tr, y_tr)
    cal_probs = model.predict_proba(X_cal)
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(cal_scores)
    q_hat = np.quantile(cal_scores, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0))
    test_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    prediction_sets = (1 - test_probs) <= q_hat
    return y_pred, prediction_sets, prediction_sets.sum(axis=1), q_hat, cal_scores

def generate_pdf_report(task, model_name, confidence, alpha, results_df,
                         coverage, q_hat, extra_metrics, cal_scores,
                         y_pred=None, lower=None, upper=None, y_test=None,
                         set_sizes=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("CustomTitle", parent=styles["Title"],
                                  fontSize=22, textColor=colors.HexColor("#1a73e8"),
                                  spaceAfter=6)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"],
                                    fontSize=13, textColor=colors.HexColor("#0d47a1"),
                                    spaceBefore=12, spaceAfter=4)
    body_style = ParagraphStyle("CustomBody", parent=styles["Normal"],
                                 fontSize=10, leading=14)
    small_style = ParagraphStyle("Small", parent=styles["Normal"],
                                  fontSize=8, textColor=colors.grey)

    story = []

    # ---- TITLE PAGE ----
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Conformal Prediction Analysis Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a73e8")))
    story.append(Spacer(1, 0.1*inch))

    from datetime import datetime
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", small_style))
    story.append(Spacer(1, 0.2*inch))

    # ---- SECTION 1: CONFIGURATION ----
    story.append(Paragraph("1. Analysis Configuration", heading_style))
    config_data = [
        ["Parameter", "Value"],
        ["Task Type", task],
        ["Model", model_name],
        ["Confidence Level", f"{confidence}%"],
        ["Significance Level (alpha)", str(alpha)],
        ["Quantile Threshold (q-hat)", f"{q_hat:.4f}"],
    ]
    config_table = Table(config_data, colWidths=[2.5*inch, 4*inch])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9ff")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9ff"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 0.2*inch))

    # ---- SECTION 2: PERFORMANCE METRICS ----
    story.append(Paragraph("2. Performance Metrics", heading_style))
    metrics_data = [["Metric", "Value"]]
    metrics_data.append(["Empirical Coverage", f"{coverage*100:.1f}%"])
    metrics_data.append(["Target Coverage", f"{confidence}%"])
    coverage_gap = coverage*100 - confidence
    metrics_data.append(["Coverage Gap", f"{coverage_gap:+.1f}%"])
    for k, v in extra_metrics.items():
        metrics_data.append([k, str(v)])

    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 4*inch])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d47a1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#e8f5e9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.2*inch))

    # ---- SECTION 3: PLOTS ----
    story.append(Paragraph("3. Visualizations", heading_style))

    # Plot 1: Prediction intervals or set sizes
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    if task == "Regression" and y_pred is not None:
        sorted_idx = np.argsort(y_pred)
        axes[0].plot(y_pred[sorted_idx], color="#1a73e8", label="Predicted", linewidth=1.5)
        axes[0].fill_between(range(len(sorted_idx)), lower[sorted_idx], upper[sorted_idx],
                             alpha=0.25, color="#1a73e8", label=f"{confidence}% Interval")
        axes[0].scatter(range(len(sorted_idx)), y_test[sorted_idx],
                        color="red", s=6, label="Actual", zorder=5)
        axes[0].set_title("Conformal Prediction Intervals", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel("Value")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].hist(set_sizes, bins=range(1, int(set_sizes.max()) + 2),
                     color="#1a73e8", edgecolor="white", align="left")
        axes[0].set_title("Prediction Set Size Distribution", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Set Size")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Calibration score distribution
    axes[1].hist(cal_scores, bins=30, color="#0d47a1", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=q_hat, color="red", linestyle="--", linewidth=1.5,
                    label=f"q-hat = {q_hat:.3f}")
    axes[1].set_title("Nonconformity Score Distribution", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Nonconformity Score")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    plt.close()

    from reportlab.platypus import Image as RLImage
    img = RLImage(img_buffer, width=6.5*inch, height=2.5*inch)
    story.append(img)
    story.append(Spacer(1, 0.2*inch))

    # ---- SECTION 4: RESULTS TABLE (first 20 rows) ----
    story.append(PageBreak())
    story.append(Paragraph("4. Prediction Results (first 20 rows)", heading_style))
    story.append(Paragraph("Full results available as CSV download in the app.", small_style))
    story.append(Spacer(1, 0.1*inch))

    preview_df = results_df.head(20)
    table_data = [list(preview_df.columns)]
    for _, row in preview_df.iterrows():
        table_data.append([str(v) for v in row.values])

    col_width = 6.5*inch / len(preview_df.columns)
    results_table = Table(table_data, colWidths=[col_width]*len(preview_df.columns))
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a73e8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9ff"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))

    # ---- SECTION 5: INTERPRETATION ----
    story.append(Paragraph("5. Interpretation", heading_style))
    if coverage * 100 >= confidence:
        interp = (f"The model achieved {coverage*100:.1f}% empirical coverage, which meets the "
                  f"target of {confidence}%. The conformal prediction {('intervals' if task == 'Regression' else 'sets')} "
                  f"are well-calibrated.")
    else:
        interp = (f"The model achieved {coverage*100:.1f}% empirical coverage, slightly below the "
                  f"target of {confidence}%. Consider increasing the dataset size or adjusting alpha.")
    story.append(Paragraph(interp, body_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        f"The quantile threshold q-hat = {q_hat:.4f} was computed from the calibration set "
        f"nonconformity scores. This value determines the width of prediction intervals (regression) "
        f"or the size of prediction sets (classification).",
        body_style
    ))

    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Generated by Conformal Prediction App", small_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ---- SIDEBAR ----
st.sidebar.markdown("## ⚙️ Settings")
task = st.sidebar.radio("Select Task Type", ["Regression", "Classification"])

if task == "Regression":
    model_options = ["Random Forest", "Linear Regression", "Decision Tree",
                     "Gradient Boosting", "Gaussian Process (RBF)", "Gaussian Process (Matern)"]
    if TF_AVAILABLE: model_options.append("Neural Network (Keras)")
    if TORCH_AVAILABLE: model_options.append("Neural Network (PyTorch)")
else:
    model_options = ["Random Forest", "Logistic Regression", "Decision Tree",
                     "Gradient Boosting", "Gaussian Process"]
    if TF_AVAILABLE: model_options.append("Neural Network (Keras)")
    if TORCH_AVAILABLE: model_options.append("Neural Network (PyTorch)")

model_name = st.sidebar.selectbox("Select Model", model_options)
if "Neural Network" in model_name:
    epochs = st.sidebar.slider("Training Epochs", 10, 200, 50, 10)
else:
    epochs = 50

alpha = st.sidebar.slider("Significance Level (alpha)", 0.01, 0.20, 0.10, 0.01,
                           help="Lower alpha = higher confidence")
confidence = int((1 - alpha) * 100)
st.sidebar.markdown(f"**Confidence Level: {confidence}%**")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

# ---- MAIN TITLE ----
st.markdown('<div class="main-title">Conformal Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Uncertainty quantification for machine learning models</div>', unsafe_allow_html=True)

# ---- TABS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Tutorial",
    "🧪 Try with Sample Data",
    "📂 Upload Your Data",
    "ℹ️ About Conformal Prediction"
])

# ====================
# TAB 1: TUTORIAL
# ====================
with tab1:
    st.header("How to Use This App")

    st.markdown("""
    <div class="tutorial-step">
        <h4>Step 1️⃣ — Choose your task type</h4>
        In the left sidebar, select either <b>Regression</b> (predicting numbers like house prices)
        or <b>Classification</b> (predicting categories like species type).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tutorial-step">
        <h4>Step 2️⃣ — Select a model</h4>
        Pick a model from the sidebar. Options include Random Forest, Gaussian Process,
        Neural Networks (if installed), and more. Not sure? Start with <b>Random Forest</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tutorial-step">
        <h4>Step 3️⃣ — Set the confidence level</h4>
        Use the <b>alpha slider</b> to set your significance level.
        Alpha = 0.10 means <b>90% confidence</b>. Lower alpha = wider intervals but more reliable coverage.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tutorial-step">
        <h4>Step 4️⃣ — Upload your data or use sample data</h4>
        Go to <b>Upload Your Data</b> tab to upload a CSV file, or go to
        <b>Try with Sample Data</b> to explore with built-in datasets first.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tutorial-step">
        <h4>Step 5️⃣ — Select your target and features</h4>
        Choose which column is the <b>target</b> (what you want to predict)
        and which columns are the <b>features</b> (predictors).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tutorial-step">
        <h4>Step 6️⃣ — Run and interpret results</h4>
        Click <b>Run Conformal Prediction</b> and explore the interactive charts,
        metrics, and downloadable results table.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight-box">
        💡 <b>Tip:</b> Try the <b>Sample Data tab</b> first to understand the output
        before uploading your own data!
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Understanding the Output")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**For Regression:**")
        st.markdown("""
        - **Predicted**: Model's point estimate
        - **Lower/Upper Bound**: The prediction interval
        - **Interval Width**: Smaller = more precise
        - **Empirical Coverage**: Should be ≥ confidence level
        - **RMSE**: Lower = better predictions
        """)
    with col2:
        st.markdown("**For Classification:**")
        st.markdown("""
        - **Predicted**: Most likely class
        - **Prediction Set**: All plausible classes
        - **Set Size**: Smaller = more confident
        - **Empirical Coverage**: Should be ≥ confidence level
        - **Accuracy**: Higher = better predictions
        """)

# ====================
# TAB 2: SAMPLE DATA
# ====================
with tab2:
    st.header("Try with Sample Data")
    st.write("Explore conformal prediction with built-in datasets — no upload needed!")

    sample_choice = st.selectbox(
        "Choose a sample dataset",
        ["California Housing (Regression)", "Iris Flowers (Classification)"]
    )

    if sample_choice == "California Housing (Regression)":
        data = fetch_california_housing(as_frame=True)
        df_sample = data.frame.sample(500, random_state=42).reset_index(drop=True)
        target_col = "MedHouseVal"
        st.markdown("""
        <div class="highlight-box">
        🏠 <b>California Housing Dataset</b>: Predict median house values based on 
        features like income, house age, and location. (500 random samples)
        </div>
        """, unsafe_allow_html=True)
    else:
        data = load_iris(as_frame=True)
        df_sample = data.frame
        df_sample["target"] = data.target_names[data.target]
        target_col = "target"
        st.markdown("""
        <div class="highlight-box">
        🌸 <b>Iris Dataset</b>: Classify iris flowers into 3 species based on 
        petal and sepal measurements.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Dataset Preview")
    st.dataframe(df_sample.head(10))
    st.write(f"Shape: {df_sample.shape[0]} rows × {df_sample.shape[1]} columns")

    # Interactive EDA plots
    st.subheader("Explore the Data")
    col1, col2 = st.columns(2)

    with col1:
        num_cols = df_sample.select_dtypes(include=np.number).columns.tolist()
        hist_col = st.selectbox("Distribution of:", num_cols)
        if sample_choice == "Iris Flowers (Classification)":
            fig = px.histogram(df_sample, x=hist_col, color="target",
                               barmode="overlay", template="plotly_white",
                               title=f"Distribution of {hist_col}")
        else:
            fig = px.histogram(df_sample, x=hist_col, template="plotly_white",
                               title=f"Distribution of {hist_col}", color_discrete_sequence=["#1a73e8"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        x_col = st.selectbox("Scatter X axis:", num_cols, index=0)
        y_col = st.selectbox("Scatter Y axis:", num_cols, index=1)
        if sample_choice == "Iris Flowers (Classification)":
            fig2 = px.scatter(df_sample, x=x_col, y=y_col, color="target",
                              template="plotly_white", title=f"{x_col} vs {y_col}")
        else:
            fig2 = px.scatter(df_sample, x=x_col, y=y_col, color=target_col,
                              template="plotly_white", title=f"{x_col} vs {y_col}",
                              color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df_sample[num_cols].corr()
    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     template="plotly_white", title="Feature Correlations")
    st.plotly_chart(fig3, use_container_width=True)

    # Run conformal prediction on sample data
    st.subheader("Run Conformal Prediction on Sample Data")
    sample_task = "Classification" if sample_choice == "Iris Flowers (Classification)" else "Regression"
    feature_cols = [col for col in df_sample.columns if col != target_col]

    if st.button("Run Conformal Prediction on Sample Data"):
        with st.spinner("Running conformal prediction..."):
            X = df_sample[feature_cols].copy()
            y = df_sample[target_col].copy()

            for col in X.select_dtypes(include="object").columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            le = None
            if sample_task == "Classification" and y.dtype == "object":
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            if sample_task == "Regression":
                if model_name == "Random Forest": base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_name == "Linear Regression": base_model = LinearRegression()
                elif model_name == "Gradient Boosting": base_model = GradientBoostingRegressor(random_state=42)
                elif model_name == "Gaussian Process (RBF)":
                    base_model = GaussianProcessRegressor(kernel=ConstantKernel()*RBF()+WhiteKernel(), random_state=42)
                elif model_name == "Gaussian Process (Matern)":
                    base_model = GaussianProcessRegressor(kernel=ConstantKernel()*Matern(nu=1.5)+WhiteKernel(), random_state=42)
                elif model_name == "Neural Network (Keras)" and TF_AVAILABLE: base_model = KerasRegressor(epochs=epochs)
                elif model_name == "Neural Network (PyTorch)" and TORCH_AVAILABLE: base_model = PyTorchRegressor(epochs=epochs)
                else: base_model = DecisionTreeRegressor(random_state=42)

                y_pred, lower, upper, q_hat, cal_scores = conformal_regression(base_model, X_train, y_train, X_test, y_test, alpha)
                interval_width = upper - lower
                coverage = np.mean((y_test >= lower) & (y_test <= upper))
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                st.markdown('<div class="success-box">✅ Conformal Prediction Complete!</div>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Confidence Level", f"{confidence}%")
                col2.metric("Empirical Coverage", f"{coverage*100:.1f}%")
                col3.metric("Avg Interval Width", f"{interval_width.mean():.3f}")
                col4.metric("RMSE", f"{rmse:.4f}")

                # Interactive prediction intervals plot
                st.subheader("Interactive Prediction Intervals")
                sorted_idx = np.argsort(y_pred)
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=list(range(len(sorted_idx))), y=upper[sorted_idx],
                    fill=None, mode="lines", line_color="rgba(26,115,232,0.3)", name="Upper Bound"
                ))
                fig_pred.add_trace(go.Scatter(
                    x=list(range(len(sorted_idx))), y=lower[sorted_idx],
                    fill="tonexty", mode="lines", line_color="rgba(26,115,232,0.3)",
                    fillcolor="rgba(26,115,232,0.15)", name="Prediction Interval"
                ))
                fig_pred.add_trace(go.Scatter(
                    x=list(range(len(sorted_idx))), y=y_pred[sorted_idx],
                    mode="lines", line=dict(color="#1a73e8", width=2), name="Predicted"
                ))
                fig_pred.add_trace(go.Scatter(
                    x=list(range(len(sorted_idx))), y=y_test[sorted_idx],
                    mode="markers", marker=dict(color="red", size=4), name="Actual"
                ))
                fig_pred.update_layout(template="plotly_white",
                                       title=f"{confidence}% Conformal Prediction Intervals ({model_name})",
                                       xaxis_title="Sample Index", yaxis_title="Value")
                st.plotly_chart(fig_pred, use_container_width=True)

                # Calibration scores distribution
                st.subheader("Calibration Score Distribution")
                fig_cal = px.histogram(x=cal_scores, nbins=30, template="plotly_white",
                                       title="Nonconformity Scores (Calibration Set)",
                                       color_discrete_sequence=["#1a73e8"],
                                       labels={"x": "Nonconformity Score", "y": "Count"})
                fig_cal.add_vline(x=q_hat, line_dash="dash", line_color="red",
                                  annotation_text=f"q̂ = {q_hat:.3f}", annotation_position="top right")
                st.plotly_chart(fig_cal, use_container_width=True)

                # Results table
                st.subheader("Detailed Results")
                results_df = pd.DataFrame({
                    "Actual": y_test.round(3),
                    "Predicted": y_pred.round(3),
                    "Lower Bound": lower.round(3),
                    "Upper Bound": upper.round(3),
                    "Interval Width": interval_width.round(3),
                    "Covered ✓": (y_test >= lower) & (y_test <= upper)
                }).reset_index(drop=True)
                st.dataframe(results_df)

            else:
                if model_name == "Random Forest": base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_name == "Logistic Regression": base_model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_name == "Gradient Boosting": base_model = GradientBoostingClassifier(random_state=42)
                elif model_name == "Gaussian Process":
                    base_model = GaussianProcessClassifier(kernel=ConstantKernel()*RBF(), random_state=42)
                elif model_name == "Neural Network (Keras)" and TF_AVAILABLE: base_model = KerasClassifier(epochs=epochs)
                elif model_name == "Neural Network (PyTorch)" and TORCH_AVAILABLE: base_model = PyTorchClassifier(epochs=epochs)
                else: base_model = DecisionTreeClassifier(random_state=42)

                y_pred, prediction_sets, set_sizes, q_hat, cal_scores = conformal_classification(base_model, X_train, y_train, X_test, alpha)
                coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])
                acc = accuracy_score(y_test, y_pred)

                st.markdown('<div class="success-box">✅ Conformal Prediction Complete!</div>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Confidence Level", f"{confidence}%")
                col2.metric("Empirical Coverage", f"{coverage*100:.1f}%")
                col3.metric("Avg Set Size", f"{set_sizes.mean():.2f}")
                col4.metric("Accuracy", f"{acc*100:.1f}%")

                # Set size distribution
                st.subheader("Prediction Set Size Distribution")
                fig_set = px.histogram(x=set_sizes, template="plotly_white",
                                       title="Distribution of Prediction Set Sizes",
                                       color_discrete_sequence=["#1a73e8"],
                                       labels={"x": "Set Size", "y": "Count"})
                st.plotly_chart(fig_set, use_container_width=True)

                # Calibration scores
                st.subheader("Calibration Score Distribution")
                fig_cal = px.histogram(x=cal_scores, nbins=30, template="plotly_white",
                                       title="Nonconformity Scores (Calibration Set)",
                                       color_discrete_sequence=["#1a73e8"],
                                       labels={"x": "Nonconformity Score", "y": "Count"})
                fig_cal.add_vline(x=q_hat, line_dash="dash", line_color="red",
                                  annotation_text=f"q̂ = {q_hat:.3f}", annotation_position="top right")
                st.plotly_chart(fig_cal, use_container_width=True)

                results_df = pd.DataFrame({
                    "Actual": le.inverse_transform(y_test) if le else y_test,
                    "Predicted": le.inverse_transform(y_pred) if le else y_pred,
                    "Prediction Set Size": set_sizes,
                    "Correct ✓": y_pred == y_test
                }).reset_index(drop=True)
                st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", data=csv,
                               file_name="conformal_predictions.csv", mime="text/csv")

# ====================
# TAB 3: UPLOAD DATA
# ====================
with tab3:
    st.header("Upload Your Own Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview:**")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # EDA
        st.subheader("Quick Data Exploration")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                hist_col = st.selectbox("Distribution of:", num_cols, key="upload_hist")
                fig = px.histogram(df, x=hist_col, template="plotly_white",
                                   color_discrete_sequence=["#1a73e8"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                corr = df[num_cols].corr()
                fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                 template="plotly_white", title="Correlation Heatmap")
                st.plotly_chart(fig2, use_container_width=True)

        target = st.selectbox("Select Target Column", df.columns)
        features = st.multiselect("Select Feature Columns",
                                  options=[col for col in df.columns if col != target],
                                  default=[col for col in df.columns if col != target])

        if features and target:
            X = df[features].copy()
            y = df[target].copy()

            for col in X.select_dtypes(include="object").columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

            le_target = None
            if task == "Classification" and y.dtype == "object":
                le_target = LabelEncoder()
                y = pd.Series(le_target.fit_transform(y))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            st.write(f"Training: **{len(X_train)}** samples | Test: **{len(X_test)}** samples")

            if st.button("Run Conformal Prediction"):
                with st.spinner(f"Training {model_name}..."):
                    if task == "Regression":
                        if model_name == "Random Forest": base_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        elif model_name == "Linear Regression": base_model = LinearRegression()
                        elif model_name == "Decision Tree": base_model = DecisionTreeRegressor(random_state=42)
                        elif model_name == "Gradient Boosting": base_model = GradientBoostingRegressor(random_state=42)
                        elif model_name == "Gaussian Process (RBF)":
                            base_model = GaussianProcessRegressor(kernel=ConstantKernel()*RBF()+WhiteKernel(), random_state=42)
                        elif model_name == "Gaussian Process (Matern)":
                            base_model = GaussianProcessRegressor(kernel=ConstantKernel()*Matern(nu=1.5)+WhiteKernel(), random_state=42)
                        elif model_name == "Neural Network (Keras)" and TF_AVAILABLE: base_model = KerasRegressor(epochs=epochs)
                        elif model_name == "Neural Network (PyTorch)" and TORCH_AVAILABLE: base_model = PyTorchRegressor(epochs=epochs)
                        else: base_model = DecisionTreeRegressor(random_state=42)

                        y_pred, lower, upper, q_hat, cal_scores = conformal_regression(base_model, X_train, y_train, X_test, y_test, alpha)
                        interval_width = upper - lower
                        coverage = np.mean((y_test >= lower) & (y_test <= upper))
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                        st.markdown('<div class="success-box">✅ Done!</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Confidence Level", f"{confidence}%")
                        col2.metric("Empirical Coverage", f"{coverage*100:.1f}%")
                        col3.metric("Avg Interval Width", f"{interval_width.mean():.3f}")
                        col4.metric("RMSE", f"{rmse:.4f}")

                        sorted_idx = np.argsort(y_pred)
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=list(range(len(sorted_idx))), y=upper[sorted_idx], fill=None, mode="lines", line_color="rgba(26,115,232,0.3)", name="Upper"))
                        fig_pred.add_trace(go.Scatter(x=list(range(len(sorted_idx))), y=lower[sorted_idx], fill="tonexty", mode="lines", line_color="rgba(26,115,232,0.3)", fillcolor="rgba(26,115,232,0.15)", name="Interval"))
                        fig_pred.add_trace(go.Scatter(x=list(range(len(sorted_idx))), y=y_pred[sorted_idx], mode="lines", line=dict(color="#1a73e8", width=2), name="Predicted"))
                        fig_pred.add_trace(go.Scatter(x=list(range(len(sorted_idx))), y=y_test[sorted_idx], mode="markers", marker=dict(color="red", size=4), name="Actual"))
                        fig_pred.update_layout(template="plotly_white", title="Conformal Prediction Intervals", xaxis_title="Sample", yaxis_title="Value")
                        st.plotly_chart(fig_pred, use_container_width=True)

                        fig_cal = px.histogram(x=cal_scores, nbins=30, template="plotly_white",
                                               title="Nonconformity Score Distribution",
                                               color_discrete_sequence=["#1a73e8"])
                        fig_cal.add_vline(x=q_hat, line_dash="dash", line_color="red",
                                          annotation_text=f"q̂={q_hat:.3f}")
                        st.plotly_chart(fig_cal, use_container_width=True)

                        results_df = pd.DataFrame({
                            "Actual": y_test.round(3), "Predicted": y_pred.round(3),
                            "Lower": lower.round(3), "Upper": upper.round(3),
                            "Width": interval_width.round(3),
                            "Covered ✓": (y_test >= lower) & (y_test <= upper)
                        }).reset_index(drop=True)
                        st.dataframe(results_df)

                    else:
                        if model_name == "Random Forest": base_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        elif model_name == "Logistic Regression": base_model = LogisticRegression(max_iter=1000, random_state=42)
                        elif model_name == "Decision Tree": base_model = DecisionTreeClassifier(random_state=42)
                        elif model_name == "Gradient Boosting": base_model = GradientBoostingClassifier(random_state=42)
                        elif model_name == "Gaussian Process":
                            base_model = GaussianProcessClassifier(kernel=ConstantKernel()*RBF(), random_state=42)
                        elif model_name == "Neural Network (Keras)" and TF_AVAILABLE: base_model = KerasClassifier(epochs=epochs)
                        elif model_name == "Neural Network (PyTorch)" and TORCH_AVAILABLE: base_model = PyTorchClassifier(epochs=epochs)
                        else: base_model = DecisionTreeClassifier(random_state=42)

                        y_pred, prediction_sets, set_sizes, q_hat, cal_scores = conformal_classification(base_model, X_train, y_train, X_test, alpha)
                        coverage = np.mean([prediction_sets[i, y_test[i]] for i in range(len(y_test))])
                        acc = accuracy_score(y_test, y_pred)

                        st.markdown('<div class="success-box">✅ Done!</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Confidence Level", f"{confidence}%")
                        col2.metric("Empirical Coverage", f"{coverage*100:.1f}%")
                        col3.metric("Avg Set Size", f"{set_sizes.mean():.2f}")
                        col4.metric("Accuracy", f"{acc*100:.1f}%")

                        fig_set = px.histogram(x=set_sizes, template="plotly_white",
                                               title="Prediction Set Sizes",
                                               color_discrete_sequence=["#1a73e8"])
                        st.plotly_chart(fig_set, use_container_width=True)

                        fig_cal = px.histogram(x=cal_scores, nbins=30, template="plotly_white",
                                               title="Nonconformity Score Distribution",
                                               color_discrete_sequence=["#1a73e8"])
                        fig_cal.add_vline(x=q_hat, line_dash="dash", line_color="red",
                                          annotation_text=f"q̂={q_hat:.3f}")
                        st.plotly_chart(fig_cal, use_container_width=True)

                        results_df = pd.DataFrame({
                            "Actual": le_target.inverse_transform(y_test) if le_target else y_test,
                            "Predicted": le_target.inverse_transform(y_pred) if le_target else y_pred,
                            "Set Size": set_sizes, "Correct ✓": y_pred == y_test
                        }).reset_index(drop=True)
                        st.dataframe(results_df)

                    csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Results as CSV", data=csv,
                                       file_name="conformal_results.csv", mime="text/csv")

                    # ---- OPTIONAL PDF REPORT ----
                    st.subheader("📄 Generate PDF Report (Optional)")
                    if st.checkbox("Generate a downloadable PDF report of this analysis"):
                        with st.spinner("Building PDF report..."):
                            if task == "Regression":
                                extra_metrics = {
                                    "RMSE": f"{rmse:.4f}",
                                    "Avg Interval Width": f"{interval_width.mean():.3f}",
                                    "Min Interval Width": f"{interval_width.min():.3f}",
                                    "Max Interval Width": f"{interval_width.max():.3f}",
                                }
                                pdf_buffer = generate_pdf_report(
                                    task, model_name, confidence, alpha,
                                    results_df, coverage, q_hat, extra_metrics,
                                    cal_scores, y_pred=y_pred, lower=lower,
                                    upper=upper, y_test=y_test
                                )
                            else:
                                extra_metrics = {
                                    "Accuracy": f"{acc*100:.1f}%",
                                    "Avg Prediction Set Size": f"{set_sizes.mean():.2f}",
                                    "Min Set Size": f"{int(set_sizes.min())}",
                                    "Max Set Size": f"{int(set_sizes.max())}",
                                }
                                pdf_buffer = generate_pdf_report(
                                    task, model_name, confidence, alpha,
                                    results_df, coverage, q_hat, extra_metrics,
                                    cal_scores, set_sizes=set_sizes
                                )

                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_buffer,
                            file_name="conformal_prediction_report.pdf",
                            mime="application/pdf"
                        )
    else:
        st.info("Upload a CSV file to get started.")

# ====================
# TAB 4: ABOUT
# ====================
with tab4:
    st.header("About Conformal Prediction")

    st.markdown("""
    ### What is Conformal Prediction?
    Conformal prediction is a framework for **uncertainty quantification** in machine learning.
    Unlike standard models that give only a single prediction, conformal prediction gives
    a **guaranteed coverage interval or set** — meaning it tells you not just *what* the
    model predicts, but *how confident* it is.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Key Properties
        - **Distribution-free**: Works with any model
        - **Finite sample guarantee**: Coverage is guaranteed for any sample size
        - **Model agnostic**: Random Forest, Neural Networks, GP — all work
        - **Calibrated**: Empirical coverage matches the confidence level
        """)
    with col2:
        st.markdown("""
        ### How It Works (Split Conformal)
        1. Split training data into *proper train* and *calibration* sets
        2. Train the model on the proper training set
        3. Compute **nonconformity scores** on the calibration set
        4. Use the (1-α) quantile of scores as the threshold **q̂**
        5. For new points, build intervals/sets using **q̂**
        """)

    st.markdown("""
    ### Models Available in This App
    | Model | Type | Best For |
    |-------|------|----------|
    | Random Forest | Both | General purpose, robust |
    | Linear/Logistic Regression | Both | Simple, interpretable |
    | Decision Tree | Both | Interpretable, fast |
    | Gradient Boosting | Both | High accuracy |
    | Gaussian Process (RBF) | Regression | Smooth functions |
    | Gaussian Process (Matern) | Regression | Rough/noisy data |
    | Gaussian Process | Classification | Small datasets |
    | Neural Network (Keras) | Both | Large, complex datasets |
    | Neural Network (PyTorch) | Both | Large, complex datasets |

    ### References
    - Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*
    - Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction*
    - Shafer, G., & Vovk, V. (2008). *A Tutorial on Conformal Prediction*
    """)