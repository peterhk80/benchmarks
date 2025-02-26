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
        [data-testid="stSidebarNav"] {
            background-image: url("https://your-logo-url.com/logo.png");
            background-repeat: no-repeat;
            background-position: 20px 20px;
            background-size: 150px auto;
            padding-top: 100px;
        }
        .logo-container {
            position: fixed;
            top: 0;
            left: 0;
            padding: 20px;
            z-index: 999;
        }
        .logo-container img {
            max-width: 150px;
            height: auto;
        }
        /* Custom header styling */
        h1 {
            font-size: 1.8rem !important;
            font-weight: 500 !important;
            padding: 0.5rem 0 !important;
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        h2 {
            font-size: 1.4rem !important;
            font-weight: 700 !important;  /* Made section headers bold */
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        h3 {
            font-size: 1.2rem !important;
            font-weight: 500 !important;
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
        }
        /* Set default font for all text */
        .stMarkdown, .stText {
            font-family: "Helvetica Neue Light", Helvetica, Arial, sans-serif !important;
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
    """Calculate benchmark metrics for a given grouping with flexible column matching"""
    
    def find_columns(keywords):
        """Helper to find columns matching any of the given keywords"""
        return [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords)]
    
    # Find relevant metric columns dynamically
    impression_cols = find_columns(['impression', 'delivered', 'views'])
    engagement_cols = find_columns(['engagement', 'interact'])
    click_cols = find_columns(['click', 'ctr'])
    video_cols = find_columns(['video', 'completion', 'duration'])
    
    # Initialize metrics dictionary with found columns
    metrics = {}
    
    # Add available metrics with appropriate aggregations
    for col in impression_cols:
        metrics[col] = ['mean', 'sum', 'count']
    
    for col in engagement_cols + click_cols:
        metrics[col] = ['mean']
    
    for col in video_cols:
        if any(x in col.lower() for x in ['rate', 'completion']):
            metrics[col] = ['mean']
        elif 'duration' in col.lower():
            metrics[col] = ['mean', 'median']
    
    # If we have no metrics at all, try to use numeric columns as a fallback
    if not metrics:
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            metrics[col] = ['mean']
        
        if not metrics:
            return pd.DataFrame(), ["No numeric columns found for analysis"]
    
    try:
        # Calculate benchmarks with available metrics
        benchmarks = df.groupby(group_by_column).agg(metrics).round(4)
        
        # Flatten column names if we have multi-level columns
        if isinstance(benchmarks.columns, pd.MultiIndex):
            benchmarks.columns = [f"{col[0]}_{col[1]}" for col in benchmarks.columns]
        
        # Format percentage columns
        percentage_cols = [col for col in benchmarks.columns if any(x in col.lower() for x in ['rate', 'percentage', 'ratio'])]
        for col in percentage_cols:
            benchmarks[col] = benchmarks[col].apply(lambda x: format_percentage(x) if pd.notnull(x) else x)
        
        # Get list of metrics we couldn't find
        standard_metrics = {
            'Delivered_Impressions': impression_cols,
            'Engagement_Rate': engagement_cols,
            'Click_Rate': click_cols,
            'Video_Completion_Rate': [col for col in video_cols if 'completion' in col.lower()]
        }
        
        missing_metrics = [metric for metric, cols in standard_metrics.items() if not cols]
        
        return benchmarks, missing_metrics
        
    except Exception as e:
        print(f"Error in calculate_benchmarks: {str(e)}")
        return pd.DataFrame(), [f"Error calculating benchmarks: {str(e)}"]

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
    # Handle aggregated column names
    metric_col = f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
    
    fig = px.bar(df, 
                 y=df.index,  # Swap x and y for horizontal bars
                 x=metric_col,
                 title=f'{title} by {group_by_column}',
                 labels={metric_col: metric.replace('_', ' ')},
                 text=df[metric_col],
                 orientation='h')  # Set horizontal orientation
    
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='outside',
        textfont=dict(
            size=16,
            family="Helvetica Neue Light, Helvetica, Arial, sans-serif"
        )
    )
    
    # Update layout for better label readability
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},  # Sort bars
        xaxis_title=metric.replace('_', ' '),
        yaxis_title=group_by_column
    )
    
    return set_chart_style(fig)

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
    # Prepare data for visualization
    device_data = {
        'Desktop': df['Desktop_Delivered_Impressions'].sum(),
        'Mobile': df['Mobile_Delivered_Impressions'].sum(),
        'Tablet': df['Tablet_Delivered_Impressions'].sum()
    }
    
    # Create pie chart for impression distribution
    fig_pie = px.pie(
        values=list(device_data.values()),
        names=list(device_data.keys()),
        title='Impression Distribution by Device'
    )
    fig_pie = set_chart_style(fig_pie)
    fig_pie.update_traces(textfont=dict(size=14))
    
    # Create horizontal bar chart for engagement rates
    device_rates = pd.DataFrame({
        'Device': ['Desktop', 'Mobile', 'Tablet'],
        'Engagement Rate': [
            df['Desktop_Engagement_Rate'].mean(),
            df['Mobile_Engagement_Rate'].mean(),
            df['Tablet_Engagement_Rate'].mean()
        ],
        'Click Rate': [
            df['Desktop_Click_Rate'].mean(),
            df['Mobile_Click_Rate'].mean(),
            df['Tablet_Click_Rate'].mean()
        ]
    })
    
    fig_rates = go.Figure()
    fig_rates.add_trace(go.Bar(
        name='Engagement Rate',
        y=device_rates['Device'],  # Swap x and y for horizontal bars
        x=device_rates['Engagement Rate'],
        text=[format_percentage(v) for v in device_rates['Engagement Rate']],
        textposition='outside',
        textfont=dict(size=14),
        orientation='h'  # Set horizontal orientation
    ))
    fig_rates.add_trace(go.Bar(
        name='Click Rate',
        y=device_rates['Device'],  # Swap x and y for horizontal bars
        x=device_rates['Click Rate'],
        text=[format_percentage(v) for v in device_rates['Click Rate']],
        textposition='outside',
        textfont=dict(size=14),
        orientation='h'  # Set horizontal orientation
    ))
    
    fig_rates.update_layout(
        title='Performance Metrics by Device',
        barmode='group',
        yaxis={'categoryorder': 'total ascending'},  # Sort bars
        xaxis_title='Rate',
        yaxis_title='Device'
    )
    fig_rates = set_chart_style(fig_rates)
    
    return fig_pie, fig_rates

def clean_percentage_columns(df):
    """Convert percentage strings to float numbers"""
    percentage_columns = [
        'Engagement_Rate', 'Click_Rate', 'Video_User_Completion_Rate',
        'Video_User_25_Rate', 'Video_User_50_Rate', 'Video_User_75_Rate',
        'Desktop_Engagement_Rate', 'Desktop_Click_Rate',
        'Mobile_Engagement_Rate', 'Mobile_Click_Rate',
        'Tablet_Engagement_Rate', 'Tablet_Click_Rate'
    ]
    
    for col in percentage_columns:
        if col in df.columns:
            # Replace '#DIV/0!' with NaN
            df[col] = df[col].replace('#DIV/0!', np.nan)
            
            if df[col].dtype == 'object':  # If column contains strings
                # Remove % and convert to float, handling NaN values
                df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0
            else:  # If column is already numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].max() > 1:  # If percentages are in 0-100 range
                    df[col] = df[col] / 100.0
    
    return df

def validate_creative_url(url):
    """Validate if the URL is a supported creative format"""
    if not url or pd.isna(url):
        return False
    
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc and 
                   any(url.lower().endswith(ext) for ext in 
                       ('.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov')))
    except:
        return False

def analyze_trends(df):
    """Analyze trends and patterns in the data"""
    trends = {}
    
    # Performance by device type
    device_split = {
        'Desktop': df['Desktop_Delivered_Impressions'].sum(),
        'Mobile': df['Mobile_Delivered_Impressions'].sum(),
        'Tablet': df['Tablet_Delivered_Impressions'].sum()
    }
    total_impressions = sum(device_split.values())
    device_split = {k: v/total_impressions for k, v in device_split.items()}
    
    # Video engagement analysis
    video_campaigns = df[df['Video_Starts'].notna()]
    avg_completion_rate = video_campaigns['Video_User_Completion_Rate'].mean()
    avg_video_duration = video_campaigns['Avg_Host_Video_Duration_secs'].mean()
    
    trends['device_distribution'] = device_split
    trends['video_metrics'] = {
        'avg_completion_rate': avg_completion_rate,
        'avg_duration': avg_video_duration
    }
    
    return trends

def display_creative_analysis(creative_url, performance_data):
    """Display creative analysis in Streamlit"""
    st.subheader("Creative Analysis")
    
    # Creative preview
    st.write("Creative Preview:")
    if creative_url.endswith(('.jpg', '.png', '.jpeg')):
        try:
            response = requests.get(creative_url)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                st.image(image, use_column_width=True)
                
                # Basic image analysis
                width, height = image.size
                st.write("### Image Details")
                st.write(f"- Dimensions: {width}x{height} pixels")
                st.write(f"- Aspect ratio: {width/height:.2f}")
                
                # Color mode information
                st.write(f"- Color mode: {image.mode}")
                
                if performance_data:
                    st.write("### Performance Metrics")
                    metrics = ['Delivered_Impressions', 'Engagement_Rate', 'Click_Rate']
                    cols = st.columns(len(metrics))
                    for i, metric in enumerate(metrics):
                        if metric in performance_data:
                            value = performance_data[metric]
                            if metric.endswith('_Rate'):
                                value = format_percentage(value)
                            cols[i].metric(metric.replace('_', ' '), value)
        
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            
    elif creative_url.endswith(('.mp4', '.mov')):
        st.video(creative_url)
        
        if performance_data:
            # Video metrics
            video_metrics = ['Video_User_25_Rate', 'Video_User_50_Rate', 
                           'Video_User_75_Rate', 'Video_User_Completion_Rate']
            
            if any(metric in performance_data for metric in video_metrics):
                st.write("### Video Performance")
                video_data = {
                    metric.replace('Video_User_', '').replace('_Rate', ''): 
                    performance_data.get(metric, 0)
                    for metric in video_metrics
                }
                
                fig_video = px.line(
                    x=list(video_data.keys()),
                    y=list(video_data.values()),
                    title="Video Completion Rates",
                    labels={'x': 'Completion Point', 'y': 'Completion Rate'}
                )
                fig_video = set_chart_style(fig_video)
                st.plotly_chart(fig_video, use_container_width=True)
    else:
        st.write(f"[View Creative]({creative_url})")
    
    # Generate recommendations
    st.subheader("Creative Recommendations")
    recommendations = []
    
    # Basic recommendations based on format
    if creative_url.endswith(('.jpg', '.png', '.jpeg')):
        recommendations.extend([
            "Ensure key message is clearly visible",
            "Check contrast for text readability",
            "Verify branding elements are prominent",
            "Test different creative dimensions for various placements",
            "Optimize image file size for faster loading"
        ])
    elif creative_url.endswith(('.mp4', '.mov')):
        recommendations.extend([
            "Place key messages in first 5 seconds",
            "Include clear call-to-action",
            "Optimize for sound-off viewing",
            "Keep video length under 30 seconds for better completion rates",
            "Add captions or text overlays for accessibility"
        ])
    
    # Performance-based recommendations
    if performance_data:
        eng_rate = performance_data.get('Engagement_Rate', 0)
        if eng_rate < 0.02:  # 2% benchmark
            recommendations.append("Consider testing different creative elements to improve engagement")
        
        click_rate = performance_data.get('Click_Rate', 0)
        if click_rate < 0.001:  # 0.1% benchmark
            recommendations.append("Review call-to-action placement and messaging")
        
        # Video-specific recommendations
        if creative_url.endswith(('.mp4', '.mov')):
            completion_rate = performance_data.get('Video_User_Completion_Rate', 0)
            if completion_rate < 0.5:  # 50% benchmark
                recommendations.append("Consider shortening video length to improve completion rate")
            
            if performance_data.get('Video_User_25_Rate', 0) > completion_rate * 2:
                recommendations.append("High drop-off after first quarter - consider restructuring video content")
    
    for rec in recommendations:
        st.write(f"- {rec}")

def validate_csv_structure(df):
    """Validate CSV structure with more flexible requirements"""
    errors = []
    warnings = []
    
    # Check if we have any data
    if df.empty:
        errors.append("The file contains no data")
        return errors, warnings
    
    # Look for common metric types
    has_impressions = any('impression' in col.lower() for col in df.columns)
    has_engagement = any('engage' in col.lower() for col in df.columns)
    has_clicks = any('click' in col.lower() for col in df.columns)
    
    if not (has_impressions or has_engagement or has_clicks):
        warnings.append("No standard metric columns (impressions, engagement, clicks) found. Will attempt to use available numeric columns.")
    
    # Check for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        warnings.append("No categorical columns found for grouping. This may limit analysis options.")
    
    return errors, warnings

def clean_and_validate_data(df):
    """Clean and validate data with more flexible handling"""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # First, identify likely numeric columns (those with mostly numbers)
        def is_numeric_column(series):
            # Check if more than 50% of non-null values can be converted to numbers
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            numeric_count = sum(pd.to_numeric(non_null, errors='coerce').notna())
            return numeric_count / len(non_null) > 0.5

        # Process each column
        for col in df.columns:
            if df[col].dtype == 'object':  # Only process string columns
                # Try to detect if it's a percentage column
                has_percent = df[col].astype(str).str.contains('%').any()
                
                # If it's a percentage column
                if has_percent:
                    df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100
                # If it looks like a numeric column
                elif is_numeric_column(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                # Otherwise leave it as is (it's probably text data)
        
        # Remove rows where all numeric columns are NaN
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            df = df.dropna(subset=numeric_cols, how='all')
        
        if df.empty:
            return None, "No valid data remains after cleaning"
        
        return df, None
        
    except Exception as e:
        return None, f"Error cleaning data: {str(e)}"

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

def main():
    st.title("Campaign Performance Benchmark Analysis")
    
    # Add tabs for different analyses
    tab1, tab2 = st.tabs(["Performance Metrics", "Creative Analysis"])
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    with tab1:
        st.write("### Campaign Data")
        
        # File selection/upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Load saved files
            saved_files = load_saved_files()
            if saved_files:
                saved_options = ['Select a saved file...'] + [
                    f"{meta['original_name']} (uploaded {meta['upload_date']})"
                    for meta in saved_files.values()
                ]
                selected_saved = st.selectbox("Select from saved files:", saved_options)
                
                if selected_saved and selected_saved != 'Select a saved file...':
                    # Find the corresponding metadata
                    selected_meta = next(
                        meta for meta in saved_files.values()
                        if f"{meta['original_name']} (uploaded {meta['upload_date']})" == selected_saved
                    )
                    
                    if selected_meta['file_path'] != st.session_state.current_file:
                        df = load_file_data(selected_meta['file_path'])
                        if df is not None:
                            st.session_state.df = df
                            st.session_state.current_file = selected_meta['file_path']
                            st.success(f"✅ Loaded saved file: {selected_meta['original_name']}")
        
        with col2:
            st.write("Drag and drop file or select the file:")
            col2_1, col2_2 = st.columns([4, 1])
            with col2_1:
                uploaded_file = st.file_uploader("", type=['csv'])
            with col2_2:
                st.markdown("""
                    <style>
                    .upload-icon {
                        font-size: 24px;
                        color: #0096FF;
                        margin-top: 22px;
                    }
                    </style>
                    <i class="fas fa-upload upload-icon"></i>
                    """, unsafe_allow_html=True)
            
            if uploaded_file is not None:
                try:
                    # Read and validate the new file
                    df = pd.read_csv(uploaded_file)
                    errors, warnings = validate_csv_structure(df)
                    
                    if errors:
                        st.error("Please fix the following errors:")
                        for error in errors:
                            st.error(f"- {error}")
                        st.stop()
                    
                    # Clean and validate data
                    cleaned_df, error = clean_and_validate_data(df)
                    if error:
                        st.error(error)
                        st.stop()
                    
                    # Save the file
                    metadata = save_uploaded_file(uploaded_file)
                    
                    # Update session state
                    st.session_state.df = cleaned_df
                    st.session_state.current_file = metadata['file_path']
                    
                    st.success("✅ File uploaded and saved successfully!")
                    
                    # Show non-critical issues
                    if warnings:
                        with st.expander("ℹ️ View Data Quality Notes"):
                            for warning in warnings:
                                st.info(warning)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.exception(e)
        
        # Only show analysis options if we have data
        if st.session_state.df is not None:
            st.markdown("---")
            st.write("### Analysis Options")
            
            # Rest of your existing analysis code...
            benchmark_categories = ['Format', 'Size', 'Placement_Name', 'Vertical']
            selected_category = st.selectbox("Select benchmark category:", benchmark_categories)
            
            if selected_category in st.session_state.df.columns:
                try:
                    # Calculate and display benchmarks
                    benchmarks, missing_metrics = calculate_benchmarks(st.session_state.df, selected_category)
                    
                    # Show missing metrics in expandable section if there are any
                    if missing_metrics:
                        with st.expander("ℹ️ View Missing Metrics"):
                            st.info("Some metrics are not available in your data:")
                            for metric in missing_metrics:
                                st.write(f"- {metric}")
                    
                    st.subheader(f"Benchmark Metrics by {selected_category}")
                    st.dataframe(benchmarks)
                    
                    # Download button for benchmarks
                    st.markdown(get_download_link(benchmarks, 
                                               f'benchmarks_{selected_category}.csv',
                                               'Download Benchmark Data'), 
                              unsafe_allow_html=True)
                    
                    # Engagement Rate visualization
                    st.subheader("Engagement Rate Analysis")
                    eng_fig = create_benchmark_visualization(
                        benchmarks, 'Engagement_Rate', 
                        selected_category, 'Average Engagement Rate'
                    )
                    st.plotly_chart(eng_fig)
                    st.markdown(get_download_link(eng_fig.to_html(), 
                                               f'engagement_rate_{selected_category}.html',
                                               'Download Chart'), 
                              unsafe_allow_html=True)
                    
                    # Click Rate visualization
                    st.subheader("Click Rate Analysis")
                    click_fig = create_benchmark_visualization(
                        benchmarks, 'Click_Rate',
                        selected_category, 'Average Click Rate'
                    )
                    st.plotly_chart(click_fig)
                    st.markdown(get_download_link(click_fig.to_html(),
                                               f'click_rate_{selected_category}.html',
                                               'Download Chart'),
                              unsafe_allow_html=True)
                    
                    # Video completion analysis
                    if 'Video_User_Completion_Rate' in st.session_state.df.columns:
                        st.subheader("Video Performance Analysis")
                        video_fig = analyze_video_dropoff(st.session_state.df, selected_category)
                        st.plotly_chart(video_fig)
                        st.markdown(get_download_link(video_fig.to_html(),
                                                   f'video_completion_{selected_category}.html',
                                                   'Download Chart'),
                                  unsafe_allow_html=True)
                    
                    # Enhanced Device Performance Analysis
                    st.subheader("Device Performance Analysis")
                    
                    # Overall device metrics
                    device_pie, device_rates = create_device_visualization(st.session_state.df, selected_category)
                    
                    # Display device distribution
                    st.plotly_chart(device_pie)
                    st.markdown(get_download_link(device_pie.to_html(),
                                               'device_distribution.html',
                                               'Download Distribution Chart'),
                              unsafe_allow_html=True)
                    
                    # Display device performance metrics
                    st.plotly_chart(device_rates)
                    st.markdown(get_download_link(device_rates.to_html(),
                                               'device_performance.html',
                                               'Download Performance Chart'),
                              unsafe_allow_html=True)
                    
                    # Detailed device performance by category
                    st.write("Detailed Device Performance by Category:")
                    device_perf = analyze_device_performance(st.session_state.df, selected_category)
                    st.dataframe(device_perf)
                    st.markdown(get_download_link(device_perf,
                                               f'device_performance_{selected_category}.csv',
                                               'Download Detailed Device Data'),
                              unsafe_allow_html=True)
                    
                    # Additional Insights
                    st.subheader("Additional Insights")
                    trends = analyze_trends(st.session_state.df)
                    
                    # Device Distribution
                    st.write("Device Distribution:")
                    device_dist = pd.DataFrame(trends['device_distribution'].items(), 
                                             columns=['Device', 'Percentage'])
                    device_dist['Percentage'] = device_dist['Percentage'].apply(format_percentage)
                    st.dataframe(device_dist)
                    
                    # Video Performance
                    if trends['video_metrics']['avg_completion_rate'] > 0:
                        st.write("Video Performance Metrics:")
                        st.write(f"- Average Video Completion Rate: {format_percentage(trends['video_metrics']['avg_completion_rate'])}")
                        st.write(f"- Average Video Duration: {trends['video_metrics']['avg_duration']:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Error processing benchmarks: {str(e)}")
                    st.exception(e)
            else:
                st.warning(f"Column {selected_category} not found in the data")

    with tab2:
        st.subheader("Creative Analysis")
        
        # Option to analyze creatives from CSV
        if st.session_state.df is not None and 'Creative_URL' in st.session_state.df.columns:
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
                    # Get performance data for this creative
                    creative_data = st.session_state.df[
                        st.session_state.df['Creative_URL'] == selected_creative
                    ].iloc[0].to_dict()
                    display_creative_analysis(selected_creative, creative_data)
            else:
                st.warning("No valid creative URLs found in the uploaded data.")
        
        # Option to analyze a single creative
        st.write("Or analyze a single creative:")
        creative_url = st.text_input("Enter creative URL (image, video, or interactive)")
        if creative_url:
            if validate_creative_url(creative_url):
                display_creative_analysis(
                    creative_url, 
                    st.session_state.df.iloc[0].to_dict() if st.session_state.df is not None else None
                )
            else:
                st.error("Please enter a valid creative URL (supported formats: jpg, jpeg, png, gif, mp4, mov)")

if __name__ == "__main__":
    main()