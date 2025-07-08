import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import sys
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def load_csv_with_error_handling(filepath):
    """
    Load CSV with multiple fallback strategies to handle parsing errors
    """
    try:
        # First attempt: Standard loading with error handling
        df = pd.read_csv(filepath, 
                        encoding='utf-8',
                        on_bad_lines='skip',  # Skip problematic lines
                        sep=',',
                        quotechar='"',
                        skipinitialspace=True)
        return df, "Loaded successfully with standard method"
        
    except Exception as e1:
        st.warning(f"First attempt failed: {str(e1)}")
        
        try:
            # Second attempt: Try with different encoding
            df = pd.read_csv(filepath, 
                            encoding='latin-1',
                            on_bad_lines='skip',
                            sep=',',
                            quotechar='"',
                            skipinitialspace=True)
            return df, "Loaded successfully with latin-1 encoding"
            
        except Exception as e2:
            st.warning(f"Second attempt failed: {str(e2)}")
            
            try:
                # Third attempt: More aggressive error handling
                df = pd.read_csv(filepath, 
                                encoding='utf-8',
                                on_bad_lines='skip',
                                sep=',',
                                quotechar='"',
                                skipinitialspace=True,
                                error_bad_lines=False,  # For older pandas versions
                                warn_bad_lines=False)   # For older pandas versions
                return df, "Loaded successfully with aggressive error handling"
                
            except Exception as e3:
                st.warning(f"Third attempt failed: {str(e3)}")
                
                try:
                    # Fourth attempt: Try reading line by line and fix manually
                    with open(filepath, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    
                    # Find the header
                    header = lines[0].strip().split(',')
                    expected_cols = len(header)
                    
                    # Clean the lines
                    clean_lines = [lines[0]]  # Keep header
                    for i, line in enumerate(lines[1:], 2):
                        fields = line.strip().split(',')
                        if len(fields) == expected_cols:
                            clean_lines.append(line)
                        elif len(fields) > expected_cols:
                            # Try to merge extra fields (common with text fields containing commas)
                            fixed_fields = fields[:expected_cols-1] + [','.join(fields[expected_cols-1:])]
                            clean_lines.append(','.join(fixed_fields) + '\n')
                            st.info(f"Fixed line {i}: merged {len(fields) - expected_cols + 1} fields into last column")
                        else:
                            st.warning(f"Skipped line {i}: too few fields ({len(fields)} vs {expected_cols})")
                    
                    # Create temporary cleaned file content
                    from io import StringIO
                    cleaned_content = ''.join(clean_lines)
                    df = pd.read_csv(StringIO(cleaned_content))
                    
                    return df, f"Loaded successfully after manual line fixing. Fixed {len(lines) - len(clean_lines)} problematic lines"
                    
                except Exception as e4:
                    st.error(f"All attempts failed. Final error: {str(e4)}")
                    return None, f"Failed to load CSV: {str(e4)}"

def show_dashboard():
    st.title("Participant Feedback Analysis")

    # Load CSV with error handling
    df, load_message = load_csv_with_error_handling('feedback.csv')
    
    if df is None:
        st.error("❌ Could not load the CSV file. Please check the file format and try again.")
        st.write("**Common solutions:**")
        st.write("1. Open the CSV in a text editor and check line 7 for extra commas")
        st.write("2. Ensure all text fields with commas are properly quoted")
        st.write("3. Check for missing or extra columns in the header")
        st.write("4. Try saving the file as UTF-8 encoded CSV")
        return
    
    st.success(f"✅ {load_message}")

    # Display basic info about the dataset
    st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show column names
    st.write("**Columns found:**")
    cols_display = st.columns(3)
    for i, col in enumerate(df.columns):
        with cols_display[i % 3]:
            st.write(f"• {col}")
    
    # Show first few rows
    st.subheader("📋 First 5 rows of data:")
    st.dataframe(df.head())

    # Define the score columns (adjust these based on your actual column names)
    score_columns = [
        'Clarity Score', 'Trust Score', 'UX Score', 'Satisfaction Score',
        'Ease of Use Score', 'Accuracy Score', 'Reliability Score',
        'Efficiency Score', 'Helpfulness Score', 'Innovation Score',
        'Design Score', 'Performance Score', 'Value Score',
        'Recommendation Score', 'Overall Score'
    ]
    
    # Filter to only include columns that exist in the dataframe
    existing_score_columns = [col for col in score_columns if col in df.columns]
    
    if not existing_score_columns:
        st.warning("⚠️ No predefined score columns found. Attempting to find numeric columns...")
        
        # Try to find numeric columns that might be scores
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        potential_score_columns = [col for col in numeric_columns if 'score' in col.lower() or 'rating' in col.lower()]
        
        if potential_score_columns:
            existing_score_columns = potential_score_columns
            st.info(f"Found potential score columns: {existing_score_columns}")
        else:
            st.write("**Available columns:**")
            for col in df.columns:
                st.write(f"• {col} (type: {df[col].dtype})")
            
            # Let user select columns
            selected_columns = st.multiselect(
                "Select columns to analyze as scores:",
                options=numeric_columns,
                default=numeric_columns[:5] if len(numeric_columns) > 0 else []
            )
            existing_score_columns = selected_columns

    if not existing_score_columns:
        st.warning("No score columns selected for analysis.")
        return

    # Show basic statistics
    st.header("📊 Basic Statistics")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Scores")
        for i, column in enumerate(existing_score_columns):
            if i < len(existing_score_columns) // 2:
                try:
                    # Convert to numeric and handle errors
                    numeric_data = pd.to_numeric(df[column], errors='coerce')
                    avg_score = numeric_data.mean()
                    valid_count = numeric_data.notna().sum()
                    st.write(f"**{column}:** {avg_score:.2f} ({valid_count} valid responses)")
                except:
                    st.write(f"**{column}:** Error calculating mean")
    
    with col2:
        st.subheader("Average Scores (continued)")
        for i, column in enumerate(existing_score_columns):
            if i >= len(existing_score_columns) // 2:
                try:
                    numeric_data = pd.to_numeric(df[column], errors='coerce')
                    avg_score = numeric_data.mean()
                    valid_count = numeric_data.notna().sum()
                    st.write(f"**{column}:** {avg_score:.2f} ({valid_count} valid responses)")
                except:
                    st.write(f"**{column}:** Error calculating mean")

    # Summary statistics table
    st.subheader("📈 Summary Statistics")
    try:
        # Convert columns to numeric
        numeric_df = df[existing_score_columns].apply(pd.to_numeric, errors='coerce')
        summary_stats = numeric_df.describe()
        st.dataframe(summary_stats)
    except Exception as e:
        st.error(f"Error creating summary statistics: {str(e)}")

    # Plotting histograms for each score column
    st.header("📊 Score Distributions")
    
    # Create a grid layout for better visualization
    cols_per_row = 3
    for i in range(0, len(existing_score_columns), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, column in enumerate(existing_score_columns[i:i+cols_per_row]):
            with cols[j]:
                st.subheader(f"{column}")
                try:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Remove any non-numeric values
                    numeric_data = pd.to_numeric(df[column], errors='coerce').dropna()
                    
                    if len(numeric_data) > 0:
                        # Determine appropriate bins based on data range
                        min_val = numeric_data.min()
                        max_val = numeric_data.max()
                        
                        if max_val <= 5:
                            bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                            xticks = [1, 2, 3, 4, 5]
                        else:
                            bins = 'auto'
                            xticks = None
                        
                        ax.hist(numeric_data, bins=bins, 
                               edgecolor='black', align='left', alpha=0.7, color='skyblue')
                        ax.set_xlabel('Score')
                        ax.set_ylabel('Number of Participants')
                        if xticks:
                            ax.set_xticks(xticks)
                        ax.set_title(f'{column} Distribution')
                        ax.grid(True, alpha=0.3)
                        
                        # Add statistics text
                        mean_val = numeric_data.mean()
                        std_val = numeric_data.std()
                        ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nN: {len(numeric_data)}',
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        st.pyplot(fig)
                    else:
                        st.write("❌ No valid numeric data found for this column")
                        
                except Exception as e:
                    st.write(f"❌ Error creating histogram: {str(e)}")
                
                plt.close()

    # Correlation matrix
    st.header("🔗 Correlation Analysis")
    try:
        # Select only numeric columns for correlation
        numeric_df = df[existing_score_columns].apply(pd.to_numeric, errors='coerce')
        numeric_cols = numeric_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            correlation_matrix = numeric_df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            
            # Add labels
            ax.set_xticks(range(len(numeric_cols)))
            ax.set_yticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax.set_yticklabels(numeric_cols)
            
            # Add correlation values
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")
            
            plt.colorbar(im, ax=ax)
            plt.title('Correlation Matrix of Scores')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.write("❌ Not enough numeric columns for correlation analysis")
            
    except Exception as e:
        st.write(f"❌ Error creating correlation matrix: {str(e)}")

    # Overall insights
    st.header("💡 Key Insights")
    try:
        numeric_df = df[existing_score_columns].apply(pd.to_numeric, errors='coerce')
        numeric_data = numeric_df.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            means = numeric_data.mean()
            stds = numeric_data.std()
            
            highest_avg = means.idxmax()
            lowest_avg = means.idxmin()
            
            st.write(f"🏆 **Highest average score:** {highest_avg} ({means[highest_avg]:.2f})")
            st.write(f"⚠️ **Lowest average score:** {lowest_avg} ({means[lowest_avg]:.2f})")
            
            # Standard deviation insights
            most_consistent = stds.idxmin()
            least_consistent = stds.idxmax()
            
            st.write(f"🎯 **Most consistent scores:** {most_consistent} (std: {stds[most_consistent]:.2f})")
            st.write(f"📈 **Most variable scores:** {least_consistent} (std: {stds[least_consistent]:.2f})")
            
            # Overall satisfaction
            overall_mean = means.mean()
            st.write(f"📊 **Overall average across all scores:** {overall_mean:.2f}")
            
    except Exception as e:
        st.write(f"❌ Error generating insights: {str(e)}")

    # Separator and dataset viewer
    st.markdown("---")
    st.subheader("📂 View Complete Dataset")
    
    if st.button("🔍 Launch Dataset Viewer"):
        try:
            dataset_script = "view_feedback_csv.py"
            dataset_port = "8507"

            subprocess.Popen(["streamlit", "run", dataset_script, "--server.port", str(dataset_port)])

            dev_ip = get_local_ip() 
            dataset_url = f"http://{dev_ip}:{dataset_port}"
            st.success(f"📊 Dataset Viewer started at: [Click to open it here]({dataset_url})")
        except Exception as e:
            st.error(f"Error launching dataset viewer: {str(e)}")

if __name__ == "__main__":
    show_dashboard()