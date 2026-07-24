import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from utils import inject_custom_css, ROOT_DIR, ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR

def main():
    inject_custom_css()
    
    st.markdown("""
    <div class="brand-container">
        <div style="font-size: 3rem; line-height: 1;">📊</div>
        <div>
            <div class="brand-title">Model Performance <span class="brand-accent">Analytics</span></div>
            <div class="brand-subtitle">Visualizing loss metrics, ROC curves, and classifier comparisons</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_curves, tab_logs = st.tabs(["Performance Curves", "Training & Model Comparison Logs"])
    
    with tab_curves:
        st.markdown("### **Diagnostic Performance Curves**")
        st.write(
            "StarSight evaluates training loops using cross-entropy losses alongside accuracy tracking. "
            "ROC-AUC curves highlight class separability for exoplanet candidates vs false positives."
        )
        
        # Row 1: Loss & Accuracy / ROC & Confusion Matrix
        c_loss, c_roc = st.columns(2)
        with c_loss:
            st.markdown('<div class="chart-wrap"><div class="chart-title">Loss & Accuracy Curves</div><div class="chart-subtitle">CNN validation loss progression across epochs</div>', unsafe_allow_html=True)
            img_loss = FIGURES_DIR / "loss_accuracy_curves.png"
            if img_loss.exists():
                st.image(Image.open(img_loss), use_container_width=True)
            else:
                st.info("Loss/accuracy curves not found.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_roc:
            st.markdown('<div class="chart-wrap"><div class="chart-title">ROC & Confusion Matrix</div><div class="chart-subtitle">Binary classification ROC curves and candidate distributions</div>', unsafe_allow_html=True)
            img_roc = FIGURES_DIR / "roc_confusion_matrix.png"
            if img_roc.exists():
                st.image(Image.open(img_roc), use_container_width=True)
            else:
                st.info("ROC/confusion matrix curves not found.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Row 2: CNN vs LightGBM
        st.markdown("### **Model Performance Comparison**")
        st.write(
            "Comparison curves illustrating the performance improvements obtained by wrapping the CNN encoder's "
            "128 penultimate feature representations with the downstream GBDT (LightGBM) classifier."
        )
        img_comp = FIGURES_DIR / "model_comparison_plots.png"
        if img_comp.exists():
            st.image(Image.open(img_comp), use_container_width=True)
        else:
            st.info("Model comparison curves not found.")
            
    with tab_logs:
        st.markdown("### **Classifier Comparison Metrics**")
        comp_csv = METRICS_DIR / "comparison.csv"
        if not comp_csv.exists():
            comp_csv = METRICS_DIR / "model_comparison_metrics.csv"
            
        if comp_csv.exists():
            try:
                df_comp = pd.read_csv(comp_csv)
                
                rows_html = ""
                for idx, row in df_comp.iterrows():
                    rows_html += f"""
                    <tr>
                        <td><strong>{row['model_name'].upper()}</strong></td>
                        <td style="text-align: right;">{row.get('accuracy', 0.0):.4f}</td>
                        <td style="text-align: right;">{row.get('precision', 0.0):.4f}</td>
                        <td style="text-align: right;">{row.get('recall', 0.0):.4f}</td>
                        <td style="text-align: right;">{row.get('f1_score', 0.0):.4f}</td>
                        <td style="text-align: right;">{row.get('roc_auc', 0.0):.4f}</td>
                        <td style="text-align: right;">{row.get('training_time', 0.0):.2f}s</td>
                        <td style="text-align: right;">{row.get('inference_time', 0.0)*1000.0:.2f}ms</td>
                    </tr>
                    """
                    
                table_html = f"""
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Model Classifier</th>
                            <th style="text-align: right;">Accuracy</th>
                            <th style="text-align: right;">Precision</th>
                            <th style="text-align: right;">Recall</th>
                            <th style="text-align: right;">F1 Score</th>
                            <th style="text-align: right;">ROC-AUC</th>
                            <th style="text-align: right;">Train Time</th>
                            <th style="text-align: right;">Inference Time</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading comparison logs: {e}")
        else:
            st.info("Comparison logs not found.")
            
        # Training Epoch history
        st.markdown("---")
        st.markdown("### **CNN Epoch Loss History**")
        history_csv = METRICS_DIR / "history.csv"
        if history_csv.exists():
            try:
                df_hist = pd.read_csv(history_csv)
                
                rows_html = ""
                # Show top 15 epochs or all
                for idx, row in df_hist.head(20).iterrows():
                    rows_html += f"""
                    <tr>
                        <td>Epoch {int(row['epoch'])}</td>
                        <td style="text-align: right;">{row['train_loss']:.5f}</td>
                        <td style="text-align: right;">{row['train_acc'] * 100.0:.2f}%</td>
                        <td style="text-align: right;">{row['val_loss']:.5f}</td>
                        <td style="text-align: right;">{row['val_acc'] * 100.0:.2f}%</td>
                    </tr>
                    """
                    
                table_html = f"""
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Epoch</th>
                            <th style="text-align: right;">Train Loss</th>
                            <th style="text-align: right;">Train Accuracy</th>
                            <th style="text-align: right;">Val Loss</th>
                            <th style="text-align: right;">Val Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
                """
                st.markdown(table_html, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading epoch history: {e}")
        else:
            st.info("history.csv epoch logs not found.")

if __name__ == "__main__":
    main()
