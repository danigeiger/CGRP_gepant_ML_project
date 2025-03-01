import streamlit as st
import pandas as pd
import joblib
import os
from padelpy import padeldescriptor

# Cached Loading Functions
@st.cache_data
def load_model():
    """Load trained Random Forest model."""
    return joblib.load("rf_reg.joblib") if os.path.exists("rf_reg.joblib") else None

@st.cache_data
def load_variance_selector():
    """Load VarianceThreshold selector."""
    return joblib.load("variance_selector.joblib") if os.path.exists("variance_selector.joblib") else None

@st.cache_data
def load_scaler():
    """Load StandardScaler for inverse transformation."""
    return joblib.load("target_scaler.joblib") if os.path.exists("target_scaler.joblib") else None

# Load model files
model = load_model()
variance_selector = load_variance_selector()
scaler = load_scaler()

# Define images for each tab
TAB_IMAGES = {
    "CGRP Receptor": "inhib_CGRP.gif",
    "Resume": "az_team.jpg"
}

# Define images for the Biography tab (side by side)
biography_pics = ["Domino.jpg", "Mom&Me.jpg", "papr.jpg"]

# Function to Detect ChEMBL ID and SMILES Columns
def detect_columns(df):
    """Detect which column is ChEMBL ID and which is SMILES."""
    if df.shape[1] != 2:
        st.error(f" Expected 2 columns, but found {df.shape[1]}.")
        return None

    col1, col2 = df.columns
    chembl_col = col1 if df[col1].astype(str).str.startswith("CHEMBL").sum() > df[col2].astype(str).str.startswith("CHEMBL").sum() else col2
    smiles_col = col1 if chembl_col == col2 else col2

    return chembl_col, smiles_col

# Function to Prepare .smi File for PaDEL
def prepare_padel_input(df, output_file="molecules.smi"):
    """Extract SMILES and molecule ID and save as .smi file for PaDEL."""
    detected_columns = detect_columns(df)
    if detected_columns is None:
        return None  

    chembl_col, smiles_col = detected_columns
    df_smiles = df[[smiles_col, chembl_col]]  # Order: SMILES first, ChEMBL ID second
    df_smiles.to_csv(output_file, sep='\t', index=False, header=False)
    
    return output_file

# Generate Molecular Fingerprints Using PaDEL
def generate_fingerprints(input_smi="molecules.smi", output_csv="fingerprints.csv"):
    """Generate molecular fingerprints with PaDEL."""
    padeldescriptor(mol_dir=input_smi, d_file=output_csv, fingerprints=True, retainorder=True)
    return output_csv

# Apply Variance Threshold Selection
def apply_variance_threshold(df_fingerprints):
    """Apply VarianceThreshold selector from training."""
    if variance_selector is None:
        st.error("⚠️ Variance selector is missing.")
        return None

    # Drop "Name" column if it exists
    if "Name" in df_fingerprints.columns:
        df_fingerprints = df_fingerprints.drop(columns=["Name"])

    # Ensure data is numeric
    df_fingerprints = df_fingerprints.apply(pd.to_numeric, errors='coerce')

    try:
        df_reduced = variance_selector.transform(df_fingerprints)
        return df_reduced
    except Exception as e:
        st.error(f" Error during feature selection: {e}")
        return None

# Undo Scaling & Convert Back to Nanomolar (nM)
def undo_scaling_and_convert(predictions):
    """Convert scaled IC50 predictions back to original nM values."""
    if scaler is None:
        return predictions  

    try:
        unscaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        ic50_nM = 10**(-unscaled_predictions) * 10**9  
        return ic50_nM
    except Exception as e:
        st.error(f"⚠️ Scaling error: {e}")
        return predictions

#  Make Predictions and Reverse Scaling
def predict_ic50(df_fingerprints):
    """Predict IC50 values and convert back to nanomolar (nM)."""
    if model is None:
        st.error("⚠️ No trained model found. Please upload `rf_reg.joblib`.")
        return None

    df_reduced = apply_variance_threshold(df_fingerprints)
    if df_reduced is None:
        st.error("Feature reduction failed. Check variance selector.")
        return None

    if model.n_features_in_ != df_reduced.shape[1]:
        st.error(f"Model expects {model.n_features_in_} features, but received {df_reduced.shape[1]}.")
        return None

    try:
        predictions = model.predict(df_reduced)
        ic50_predictions = undo_scaling_and_convert(predictions)
        return pd.DataFrame({"Predicted IC50 (nM)": ic50_predictions})

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        return None

# **Create Tabs**
tab1, tab2, tab3 = st.tabs(["CGRP Receptor", "Resume", "Biography"])

# **CGRP Receptor Tab**
with tab1:
    st.title("CGRP Receptor Antagonist IC50 Predictor")

    image_path = TAB_IMAGES["CGRP Receptor"]
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    # File uploader for TXT or CSV
    file_type = st.radio("Choose File Type:", ["CSV", "TXT"])
    uploaded_file = st.file_uploader(f"Upload a {file_type} file", type=["csv", "txt"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine="python", header=None)
            if df.shape[1] != 2:
                st.error(f"Expected 2 columns, but found {df.shape[1]}. Please check your file format.")
                st.stop()

            df.columns = ["ChEMBL_ID", "SMILES"]
            st.write("**Uploaded Data Sample**")
            st.write(df.head(10))

            smi_file = prepare_padel_input(df)
            if smi_file is None:
                st.stop()

            fingerprints_file = generate_fingerprints(smi_file)
            df_fingerprints = pd.read_csv(fingerprints_file)

            df_predictions = predict_ic50(df_fingerprints)
            if df_predictions is None:
                st.error("⚠️ No predictions were generated.")
                st.stop()

            st.write("**Predictions**")
            st.write(df_predictions.head(10))

            # Download Button
            pred_csv = df_predictions.to_csv(index=False).encode("utf-8")
            st.download_button("Download Prediction Results", pred_csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")

    # Example Files
    st.write("### Example Input Files")
    for file in ["sample_data.csv", "sample_data.txt"]:
        if os.path.exists(file):
            with open(file, "rb") as f:
                st.download_button(f" Download {file}", f.read(), file, "text/csv")


#  **Resume Tab**
with tab2:
    st.title("Resume")

    # Display Resume image
    image_path = TAB_IMAGES["Resume"]
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)

    st.write("""
    ## **Professional Summary**  
    I am passionate about **applying machine learning and AI to biomedical research**,  
    with a focus on **personalized medicine, biomarker analysis, and drug discovery**.  
    I aim to leverage computational tools to extract insights from **NCBI research**  
    and improve diagnostics, treatment optimization, and healthcare accessibility.
    """)

    # Education Section
    st.write("## **Education**")
    st.write("""
    - **Bachelor of Science**, University of Oregon (2017-2021)  
      *Focus: Bioinformatics, Computational Biology, Machine Learning in Healthcare*
    """)

    # Experience Section
    st.write("## **Experience**")

    st.write("""
    **Machine Learning & Bioinformatics Research**  
    - Developed and applied **machine learning models** for biomarker discovery, drug screening, and disease diagnostics.  
    - Conducted **biomarker analysis** using genomics, proteomics, and blood-based biomarkers.  
    - Utilized **NCBI datasets** and AI-driven methods to analyze disease mechanisms.  
    - Integrated **subjective and objective diagnostics** using computational tools.  

    **AstraZeneca – Inhalation Product Development (IPD)**  
    - Worked with **biologic drug formulations** for inhalation delivery.  
    - Set up and operated the **spray dryer** for mixing biologic drugs with leucine, isoleucine, and trehalose.  
    - Maintained **clean room conditions** and followed strict **GMP documentation**.  
    - Assisted in **document control** and regulatory compliance.  
    - Helped **decommission the facility** before the department's relocation to the East Coast.  
    """)

    # Technical Skills Section
    st.write("## **Technical Skills**")
    st.write("""
    - **Programming & Data Science:** Python, Pandas, Scikit-learn, TensorFlow, PyTorch  
    - **Bioinformatics & Computational Biology:** RDKit, PyMOL, RNA-seq, NCBI, ChEMBL, RCSB PBD  
    - **Machine Learning Applications:** Supervised/Unsupervised Learning, Feature Engineering, Predictive Modeling  
    - **Biomarker Analysis:** Genomics, Transcriptomics, Proteomics, Blood-Based Biomarkers  
    - **Diagnostics & Personalized Medicine:** AI-driven diagnostics, Clinical Data Analysis, Patient-Specific Treatment Models  
    - **Biologic Drug Development:** Inhalation Product Development, Spray Drying, GMP Documentation  
    """)


# **Biography Tab**
with tab3:
    st.title("Biography")

    # Display images side by side
    col1, col2, col3 = st.columns(3)
    with col1: st.image(biography_pics[0], use_container_width=True)
    with col2: st.image(biography_pics[1], use_container_width=True)
    with col3: st.image(biography_pics[2], use_container_width=True)

    st.write("""
    ### About Me  
    I am passionate about using **machine learning to improve personalized medicine, drug discovery, and diagnostics**.  
    I want to apply AI to **NCBI research** to find better ways to treat diseases, make healthcare more individualized, and improve diagnostics.  
    This includes both **subjective diagnostics**, like patient-reported symptoms, and **objective diagnostics** based on data collected using **biotechnology**.  
    With the right data and tools, I believe we can make **healthcare more proactive and accessible for everyone**.

    ### Why CGRP?  
    My interest in discovering alternative therapeutic agents, particularly **CGRP receptor antagonists**,  
    stems from a **permanent complication my mother (pictured above) experienced** after receiving a monoclonal antibody targeting the CGRP receptor.  
    Unfortunately, despite being a chronic migraine sufferer, she has yet to find a truly safe and effective treatment.

    ### **Research Interests**  
    - **AI-driven small-molecule drug discovery (e.g., CGRP Model)**  
    - **AI-driven biologic drug synthesis (protein-based therapeutics)**  
    - **DeepMind's AlphaFold for protein structure prediction**  
    - **AI applications in human genomics & precision medicine**  
    - **Molecular docking & fingerprint-based analysis**  
    - **Investigating biological aging, metabolism, and longevity interventions**  

    ### **Personal Interests**  
    - **Tutoring Math and Chemistry**  
    - **Working with Horses**  
    - **Watercoloring Botanicals**  
    - **Hiking with my dog **  

    **[GitHub](https://github.com/danigeiger)**
    """)

