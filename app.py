# app.py
import pandas as pd
import pdfplumber
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import numpy as np
import io
import base64
import requests
from PIL import Image
from urllib.parse import urlparse
import re
import os
import json
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Campaign Performance Benchmark Analysis",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for logo positioning and header styling
st.markdown("""
    <style>
        /* Remove default red from buttons and tabs */
        .stButton button {
            background-color: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1rem;
        }
        .stButton button:hover {
            background-color: #0052a3;
        }
        /* Style tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #0066cc;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0066cc !important;
            color: white !important;
        }
        /* Header styling */
        h1 {
            font-size: 1.8rem !important;
            font-weight: 500 !important;
            padding: 0.5rem 0 !important;
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        h2 {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        h3 {
            font-size: 1.2rem !important;
            font-weight: 500 !important;
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        /* Set default font */
        .stMarkdown, .stText {
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        /* Style selectbox */
        .stSelectbox [data-baseweb="select"] {
            border-radius: 4px;
        }
        /* Style file uploader */
        .stFileUploader {
            border: 2px dashed #0066cc;
            border-radius: 4px;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Add logo container
st.markdown("""
    <div class="logo-container">
        <img src="https://your-logo-url.com/logo.png" alt="Logo">
    </div>
    """, unsafe_allow_html=True)

# Password protection
def check_password():
    """Returns `True` if the user has entered the correct password"""
    
    # Add password to session state if not already there
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
        
    if not st.session_state.password_correct:
        # First run, show inputs for username + password
        st.markdown("## Welcome to Campaign Benchmark Tool")
        st.markdown("Please enter your credentials to access the tool.")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            # Replace these with your desired username/password
            if username == "benchmark" and password == "cnbc2024":
                st.session_state.password_correct = True
                st.rerun()
            else:
                st.error("😕 Incorrect username or password")
                return False
        return False
    
    return True

if not check_password():
    st.stop()

def read_pdf(pdf_file):
    """Extract text and tables from PDF file"""
    text_content = []
    tables = []
    
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text_content.append(page.extract_text())
            if page.extract_tables():
                tables.extend(page.extract_tables())
    
    # Convert tables to dataframes if they exist
    dfs = []
    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        dfs.append(df)
    
    return text_content, dfs

def analyze_data(df):
    """Generate basic statistical insights from dataframe"""
    insights = {
        'summary_stats': df.describe(),
        'missing_values': df.isnull().sum(),
        'column_types': df.dtypes
    }
    
    # Identify numeric columns for additional analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        insights['correlations'] = df[numeric_cols].corr()
    
    return insights

def get_download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for the given object"""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=True)
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()
        return f'<a href="data:text/plain;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def format_percentage(value):
    """Format float as percentage"""
    if pd.isna(value):
        return "N/A"
    try:
        # Convert to float if it's a string
        if isinstance(value, str):
            value = float(value.strip('%')) / 100
        return f"{float(value):.2%}"
    except (ValueError, TypeError):
        return str(value)

def calculate_benchmarks(df, group_by_column):
    """Calculate benchmark metrics for a given grouping"""
    try:
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Convert percentage strings to floats first
        for col in df.columns:
            if df[col].dtype == 'object':  # Only check string columns
                # Check if column contains percentage values
                if df[col].astype(str).str.contains('%').any():
                    df[col] = df[col].apply(lambda x: float(str(x).rstrip('%'))/100 if isinstance(x, str) and '%' in str(x) else x)
        
        # Get numeric columns after conversion
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Initialize metrics dictionary
        metrics = {}
        
        # Add metrics only for numeric columns
        for col in numeric_cols:
            col_lower = col.lower()
            if any(term in col_lower for term in ['impression', 'delivered', 'view']):
                metrics[col] = ['mean', 'sum']
            elif any(term in col_lower for term in ['rate', 'ratio', 'percentage', 'engagement', 'click']):
                metrics[col] = ['mean']
            else:
                metrics[col] = ['mean']
        
        if not metrics:
            st.warning("No numeric metrics found for analysis")
            return pd.DataFrame(), []
        
        # Calculate benchmarks
        benchmarks = df.groupby(group_by_column)[list(metrics.keys())].agg(metrics).round(4)
        
        # Flatten column names if we have multi-level columns
        if isinstance(benchmarks.columns, pd.MultiIndex):
            benchmarks.columns = [f"{col[0]}_{col[1]}" for col in benchmarks.columns]
        
        # Format percentage columns
        percentage_cols = [col for col in benchmarks.columns if any(term in col.lower() for term in ['rate', 'ratio', 'percentage'])]
        for col in percentage_cols:
            benchmarks[col] = benchmarks[col].apply(format_percentage)
        
        return benchmarks, []
        
    except Exception as e:
        st.error(f"Error calculating benchmarks: {str(e)}")
        return pd.DataFrame(), [str(e)]

def set_chart_style(fig):
    """Apply consistent styling to all charts"""
    fig.update_layout(
        font=dict(
            family="Helvetica Neue Light, Helvetica, Arial, sans-serif",
            size=16  # Increased base font size
        ),
        width=1200,  # Wider charts
        height=600,
        margin=dict(t=80, b=50, l=50, r=50),  # Increased top margin for title
        title=dict(
            x=0,  # Left-align title
            y=0.98,  # Position at top
            xanchor='left',
            font=dict(
                size=24,  # Larger title font
                family="Helvetica Neue Light, Helvetica, Arial, sans-serif"
            )
        ),
        xaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=18)
        ),
        yaxis=dict(
            tickfont=dict(size=16),
            title_font=dict(size=18)
        ),
        legend=dict(
            font=dict(size=16),
            title_font=dict(size=18)
        )
    )
    
    # Increase size of data labels if they exist
    if hasattr(fig.data[0], 'text'):
        for trace in fig.data:
            trace.textfont = dict(
                size=16,
                family="Helvetica Neue Light, Helvetica, Arial, sans-serif"
            )
    
    return fig

def create_benchmark_visualization(df, metric, group_by_column, title):
    """Create horizontal bar chart for benchmark metrics"""
    try:
        # Check if metric exists in dataframe
        if metric not in df.columns:
            print(f"Warning: {metric} not found in data")
            return None
            
        # Create the visualization
        fig = px.bar(df, 
                    y=df.index,
                    x=metric,
                    title=f'{title} by {group_by_column}',
                    labels={metric: metric.replace('_', ' ')},
                    text=df[metric],
                    orientation='h')
        
        fig.update_traces(
            texttemplate='%{text}', 
            textposition='outside',
            textfont=dict(size=16)
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title=metric.replace('_', ' '),
            yaxis_title=group_by_column,
            font=dict(
                family="Helvetica Neue Light, Helvetica, Arial, sans-serif",
                size=16
            ),
            width=1200,
            height=600
        )
        
        return fig
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None

def analyze_video_dropoff(df, group_by_column):
    """Analyze video completion rates at different stages"""
    video_metrics = ['Video_User_25_Rate', 'Video_User_50_Rate', 
                    'Video_User_75_Rate', 'Video_User_Completion_Rate']
    
    dropoff = df.groupby(group_by_column)[video_metrics].mean()
    
    fig = go.Figure()
    for metric in video_metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('Video_User_', '').replace('_Rate', ''),
            y=dropoff.index,  # Swap x and y for horizontal bars
            x=dropoff[metric],
            text=dropoff[metric].round(4),
            textfont=dict(size=14),
            orientation='h'  # Set horizontal orientation
        ))
    
    fig.update_layout(
        title=f'Video Completion Rates by {group_by_column}',
        barmode='group',
        yaxis={'categoryorder': 'total ascending'},  # Sort bars
        xaxis_title='Completion Rate',
        yaxis_title=group_by_column
    )
    
    return set_chart_style(fig)

def analyze_device_performance(df, group_by_column):
    """Analyze performance across devices"""
    # Basic metrics by device
    device_metrics = {
        'Desktop': ['Desktop_Delivered_Impressions', 'Desktop_Engagement_Rate', 'Desktop_Click_Rate'],
        'Mobile': ['Mobile_Delivered_Impressions', 'Mobile_Engagement_Rate', 'Mobile_Click_Rate'],
        'Tablet': ['Tablet_Delivered_Impressions', 'Tablet_Engagement_Rate', 'Tablet_Click_Rate']
    }
    
    # Calculate device performance
    device_performance = df.groupby(group_by_column).agg({
        metric: 'mean' for device_list in device_metrics.values() for metric in device_list
    })
    
    # Format percentage columns
    for device in device_metrics:
        rate_cols = [col for col in device_performance.columns if 'Rate' in col]
        for col in rate_cols:
            device_performance[col] = device_performance[col].apply(format_percentage)
    
    return device_performance

def create_device_visualization(df, group_by_column):
    """Create visualization for device performance"""
    try:
        # Check which device columns exist
        device_cols = {
            'Desktop': 'Desktop_Delivered_Impressions',
            'Mobile': 'Mobile_Delivered_Impressions',
            'Tablet': 'Tablet_Delivered_Impressions'
        }
        
        available_devices = {k: v for k, v in device_cols.items() if v in df.columns}
        
        if not available_devices:
            print("No device impression columns found")
            return None, None
            
        # Create device data only for available columns
        device_data = {
            device: df[col].sum() 
            for device, col in available_devices.items()
        }
        
        # Create pie chart
        fig_pie = px.pie(
            values=list(device_data.values()),
            names=list(device_data.keys()),
            title='Impression Distribution by Device'
        )
        fig_pie = set_chart_style(fig_pie)
        
        # Get available rate columns
        device_rates = []
        for device in available_devices.keys():
            eng_rate = f'{device}_Engagement_Rate'
            click_rate = f'{device}_Click_Rate'
            
            if eng_rate in df.columns or click_rate in df.columns:
                rates = {'Device': device}
                if eng_rate in df.columns:
                    rates['Engagement Rate'] = df[eng_rate].mean()
                if click_rate in df.columns:
                    rates['Click Rate'] = df[click_rate].mean()
                device_rates.append(rates)
        
        if not device_rates:
            print("No rate columns found")
            return fig_pie, None
            
        # Create rates dataframe
        rates_df = pd.DataFrame(device_rates)
        
        # Create rates chart
        fig_rates = go.Figure()
        
        if 'Engagement Rate' in rates_df.columns:
            fig_rates.add_trace(go.Bar(
                name='Engagement Rate',
                y=rates_df['Device'],
                x=rates_df['Engagement Rate'],
                orientation='h'
            ))
            
        if 'Click Rate' in rates_df.columns:
            fig_rates.add_trace(go.Bar(
                name='Click Rate',
                y=rates_df['Device'],
                x=rates_df['Click Rate'],
                orientation='h'
            ))
        
        fig_rates.update_layout(
            title='Performance Metrics by Device',
            barmode='group',
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Rate',
            yaxis_title='Device'
        )
        fig_rates = set_chart_style(fig_rates)
        
        return fig_pie, fig_rates
        
    except Exception as e:
        print(f"Error in device visualization: {str(e)}")
        return None, None

def clean_and_validate_data(df):
    """Clean and validate the data, ensuring proper formatting for visualization"""
    try:
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Only convert percentage columns that contain % symbol
        for col in df.columns:
            if df[col].dtype == 'object':  # Only check string columns
                sample_values = df[col].dropna().head()
                if any('%' in str(x) for x in sample_values):
                    df[col] = df[col].apply(lambda x: float(str(x).rstrip('%'))/100 if isinstance(x, str) and '%' in str(x) else x)
        
        return df, None
        
    except Exception as e:
        # If anything fails, return original dataframe
        print(f"Warning in data cleaning: {str(e)}")
        return df, None

def display_data_quality_report(df):
    """Display a comprehensive data quality report"""
    st.subheader("Data Quality Report")
    
    # Basic statistics
    total_rows = len(df)
    total_columns = len(df.columns)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", total_rows)
    with col2:
        st.metric("Total Columns", total_columns)
    with col3:
        st.metric("Complete Rows", df.dropna().shape[0])
    
    # Column-wise analysis
    st.write("### Column Analysis")
    
    column_stats = []
    for column in df.columns:
        stats = {
            'Column': column,
            'Type': str(df[column].dtype),
            'Non-Null Count': df[column].count(),
            'Null %': f"{(df[column].isna().mean()*100):.1f}%",
            'Unique Values': df[column].nunique(),
            'Sample Values': ', '.join(map(str, df[column].dropna().unique()[:3]))
        }
        column_stats.append(stats)
    
    st.dataframe(pd.DataFrame(column_stats))
    
    # Data preview with highlighting
    st.write("### Data Preview (with Issues Highlighted)")
    
    def highlight_nulls(val):
        if pd.isna(val):
            return 'background-color: yellow'
        return ''
    
    styled_df = df.head().style.applymap(highlight_nulls)
    st.dataframe(styled_df)

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return its metadata"""
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join('uploads', filename)
    
    # Save the file
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getvalue())
    
    # Create metadata
    metadata = {
        'original_name': uploaded_file.name,
        'saved_name': filename,
        'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'file_path': file_path
    }
    
    # Update metadata file
    metadata_path = 'uploads/metadata.json'
    all_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
    
    all_metadata[filename] = metadata
    
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    return metadata

def load_saved_files():
    """Load metadata of all saved files"""
    metadata_path = 'uploads/metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def load_file_data(file_path):
    """Load data from saved file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading saved file: {str(e)}")
        return None

def validate_creative_url(url):
    """Validate if the URL points to a supported creative format"""
    try:
        if not url:
            return False
            
        parsed = urlparse(url)
        path = parsed.path.lower()
        
        # Check if URL ends with supported format
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov']
        return any(path.endswith(fmt) for fmt in supported_formats)
    except:
        return False

def display_creative_analysis(creative_url, performance_data=None):
    """Display and analyze creative content"""
    try:
        st.write("### Creative Preview")
        
        # Get file extension
        ext = os.path.splitext(urlparse(creative_url).path)[1].lower()
        
        # Display based on file type
        if ext in ['.jpg', '.jpeg', '.png', '.gif']:
            st.image(creative_url, use_column_width=True)
        elif ext in ['.mp4', '.mov']:
            st.video(creative_url)
            
        # Display performance metrics if available
        if performance_data:
            st.write("### Performance Metrics")
            metrics_df = pd.DataFrame([performance_data])
            st.dataframe(metrics_df)
            
    except Exception as e:
        st.error(f"Error displaying creative: {str(e)}")

def main():
    st.title("Campaign Performance Benchmark Analysis")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    tab1, tab2 = st.tabs(["Performance Metrics", "Creative Analysis"])
    
    with tab1:
        st.write("### Campaign Data")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                # Verify required columns exist
                required_columns = ['Format', 'Size', 'Placement_Name', 'Vertical']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                    st.stop()
                
                st.markdown("---")
                st.write("### Analysis Options")
                
                benchmark_categories = ['Format', 'Size', 'Placement_Name', 'Vertical']
                selected_category = st.selectbox("Select benchmark category:", benchmark_categories)
                
                if selected_category in df.columns:
                    benchmarks, _ = calculate_benchmarks(df, selected_category)
                    
                    if not benchmarks.empty:
                        st.subheader(f"Benchmark Metrics by {selected_category}")
                        st.dataframe(benchmarks)
                        
                        if 'Engagement_Rate' in benchmarks.columns:
                            st.subheader("Engagement Rate Analysis")
                            eng_fig = create_benchmark_visualization(
                                benchmarks, 'Engagement_Rate', 
                                selected_category, 'Average Engagement Rate'
                            )
                            if eng_fig:
                                st.plotly_chart(eng_fig, use_container_width=True)
                        
                        if 'Click_Rate' in benchmarks.columns:
                            st.subheader("Click Rate Analysis")
                            click_fig = create_benchmark_visualization(
                                benchmarks, 'Click_Rate',
                                selected_category, 'Average Click Rate'
                            )
                            if click_fig:
                                st.plotly_chart(click_fig, use_container_width=True)
                        
                        st.subheader("Device Performance Analysis")
                        device_pie, device_rates = create_device_visualization(df, selected_category)
                        if device_pie:
                            st.plotly_chart(device_pie, use_container_width=True)
                        if device_rates:
                            st.plotly_chart(device_rates, use_container_width=True)
                        
                        device_perf = analyze_device_performance(df, selected_category)
                        if not device_perf.empty:
                            st.write("Detailed Device Performance:")
                            st.dataframe(device_perf)
                    else:
                        st.warning("No data available for the selected category.")
                else:
                    st.warning(f"Column {selected_category} not found in the data")
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.stop()

    with tab2:
        st.subheader("Creative Analysis")
        
        if st.session_state.df is not None:
            if 'Creative_URL' in st.session_state.df.columns:
                st.write("Select a creative from your campaign data:")
                valid_urls = st.session_state.df[
                    st.session_state.df['Creative_URL'].apply(validate_creative_url)
                ]['Creative_URL'].unique()
                
                if len(valid_urls) > 0:
                    selected_creative = st.selectbox(
                        "Choose a creative to analyze:",
                        options=[''] + list(valid_urls),
                        format_func=lambda x: f"Creative {list(valid_urls).index(x) + 1}" if x else "Select a creative..."
                    )
                    
                    if selected_creative:
                        try:
                            creative_data = st.session_state.df[
                                st.session_state.df['Creative_URL'] == selected_creative
                            ].iloc[0].to_dict()
                            display_creative_analysis(selected_creative, creative_data)
                        except Exception as e:
                            st.error(f"Error analyzing creative: {str(e)}")
                else:
                    st.warning("No valid creative URLs found in the uploaded data.")
            else:
                st.info("Upload a CSV file with a 'Creative_URL' column to analyze creatives.")
        
        st.write("Or analyze a single creative:")
        creative_url = st.text_input("Enter creative URL (image, video, or interactive)")
        if creative_url:
            if validate_creative_url(creative_url):
                try:
                    display_creative_analysis(creative_url)
                except Exception as e:
                    st.error(f"Error displaying creative: {str(e)}")
            else:
                st.error("Please enter a valid creative URL (supported formats: jpg, jpeg, png, gif, mp4, mov)")

if __name__ == "__main__":
    main()