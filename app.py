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

# Configure Streamlit page
st.set_page_config(
    page_title="Campaign Performance Benchmark Analysis",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for logo positioning
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
    benchmarks = df.groupby(group_by_column).agg({
        'Delivered_Impressions': ['mean', 'sum', 'count'],
        'Engagement_Rate': 'mean',
        'Click_Rate': 'mean',
        'Video_User_Completion_Rate': 'mean',
        'Avg_Host_Video_Duration_secs': 'mean'
    }).round(4)
    
    benchmarks.columns = ['Avg_Impressions', 'Total_Impressions', 'Campaign_Count', 
                         'Avg_Engagement_Rate', 'Avg_Click_Rate', 
                         'Avg_Video_Completion_Rate', 'Avg_Video_Duration']
    
    # Format percentage columns
    percentage_cols = ['Avg_Engagement_Rate', 'Avg_Click_Rate', 'Avg_Video_Completion_Rate']
    for col in percentage_cols:
        benchmarks[col] = benchmarks[col].apply(format_percentage)
    
    return benchmarks

def set_chart_style(fig):
    """Apply consistent styling to all charts"""
    fig.update_layout(
        font=dict(
            family="Helvetica Neue Light, Helvetica, Arial, sans-serif",
            size=12  # Base font size
        ),
        width=1200,  # Wider charts
        height=600,
        margin=dict(t=50, b=50, l=50, r=50),
        title_x=0.5,  # Center title
        title_y=0.95,
        xaxis=dict(
            tickfont=dict(size=14),  # Larger tick labels
            title_font=dict(size=14)  # Larger axis titles
        ),
        yaxis=dict(
            tickfont=dict(size=14),  # Larger tick labels
            title_font=dict(size=14)  # Larger axis titles
        ),
        legend=dict(
            font=dict(size=14)  # Larger legend text
        )
    )
    
    # Increase size of data labels if they exist
    if 'text' in fig.data[0]:
        for trace in fig.data:
            trace.textfont = dict(size=14)
    
    return fig

def create_benchmark_visualization(df, metric, group_by_column, title):
    """Create horizontal bar chart for benchmark metrics"""
    fig = px.bar(df, 
                 y=df.index,  # Swap x and y for horizontal bars
                 x=metric,
                 title=f'{title} by {group_by_column}',
                 labels={metric: metric.replace('_', ' ')},
                 text=df[metric],
                 orientation='h')  # Set horizontal orientation
    
    fig.update_traces(
        texttemplate='%{text}', 
        textposition='outside',
        textfont=dict(size=14)
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
    """Validate CSV file structure and required columns"""
    required_columns = {
        'Delivered_Impressions': {
            'type': 'number',
            'min_value': 0,
            'allow_null': False
        },
        'Engagement_Rate': {
            'type': 'percentage',
            'min_value': 0,
            'max_value': 100,
            'allow_null': True
        },
        'Click_Rate': {
            'type': 'percentage',
            'min_value': 0,
            'max_value': 100,
            'allow_null': True
        },
        'Format': {
            'type': 'string',
            'allow_null': False
        },
        'Size': {
            'type': 'string',
            'allow_null': False
        },
        'Placement_Name': {
            'type': 'string',
            'min_length': 1,
            'allow_null': False
        },
        'Vertical': {
            'type': 'string',
            'allow_null': False
        }
    }
    
    errors = []
    warnings = []
    
    # Check for empty dataframe
    if df.empty:
        errors.append("The file is empty. Please upload a file with data.")
        return errors, warnings
    
    # Check for required columns
    missing_columns = [col for col in required_columns.keys() if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return errors, warnings
    
    # Check data types and validation rules
    for col, rules in required_columns.items():
        if col in df.columns:
            # Check for empty values
            null_count = df[col].isna().sum()
            if not rules['allow_null'] and null_count > 0:
                errors.append(f"Column '{col}' contains {null_count} empty values which are not allowed")
            elif null_count/len(df) > 0.5:
                warnings.append(f"Column '{col}' is more than 50% empty ({null_count} empty values)")
            
            # Type-specific validations
            non_null_values = df[col].dropna()
            
            if rules['type'] == 'number':
                try:
                    numeric_values = pd.to_numeric(non_null_values, errors='raise')
                    if 'min_value' in rules and (numeric_values < rules['min_value']).any():
                        errors.append(f"Column '{col}' contains values below minimum ({rules['min_value']})")
                    if 'max_value' in rules and (numeric_values > rules['max_value']).any():
                        errors.append(f"Column '{col}' contains values above maximum ({rules['max_value']})")
                except:
                    errors.append(f"Column '{col}' contains non-numeric values")
            
            elif rules['type'] == 'percentage':
                try:
                    # Convert percentage strings to float values
                    cleaned_values = non_null_values.apply(lambda x: 
                        float(str(x).strip('%'))/100 if isinstance(x, str) else float(x)
                    )
                    if (cleaned_values < 0).any() or (cleaned_values > 1).any():
                        errors.append(f"Column '{col}' contains invalid percentage values (should be between 0% and 100%)")
                except:
                    errors.append(f"Column '{col}' contains invalid percentage values")
            
            elif rules['type'] == 'string':
                if 'min_length' in rules:
                    short_values = non_null_values[non_null_values.str.len() < rules['min_length']]
                    if not short_values.empty:
                        errors.append(f"Column '{col}' contains values shorter than {rules['min_length']} characters")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate rows in the data")
    
    # Check for suspicious values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # Check for outliers (values more than 3 standard deviations from mean)
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[abs(df[col] - mean) > 3*std]
        if len(outliers) > 0:
            warnings.append(f"Found {len(outliers)} potential outliers in column '{col}'")
    
    return errors, warnings

def clean_and_validate_data(df):
    """Clean and validate the uploaded data"""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Basic cleaning
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Remove leading/trailing whitespace from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].str.strip()
        
        # Handle special characters in numeric columns
        numeric_columns = ['Delivered_Impressions']
        for col in numeric_columns:
            if col in df.columns:
                # Remove commas and other formatting
                df[col] = df[col].replace({',': '', '$': '', '%': ''}, regex=True)
                # Convert to numeric, replacing errors with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize percentage columns
        percentage_columns = ['Engagement_Rate', 'Click_Rate', 'Video_User_Completion_Rate']
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 
                    float(str(x).strip('%'))/100 if isinstance(x, str) and '%' in str(x)
                    else float(x)/100 if isinstance(x, (int, float)) and x > 1
                    else x
                )
        
        # Standardize text columns
        text_columns = ['Format', 'Vertical']
        for col in text_columns:
            if col in df.columns:
                # Capitalize first letter of each word
                df[col] = df[col].str.title()
                # Remove special characters
                df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        # Validate Creative_URL format if present
        if 'Creative_URL' in df.columns:
            df['Creative_URL'] = df['Creative_URL'].apply(lambda x: 
                x if validate_creative_url(x) else None
            )
        
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

def main():
    st.title("Campaign Performance Benchmark Analysis")
    
    # Add tabs for different analyses
    tab1, tab2 = st.tabs(["Performance Metrics", "Creative Analysis"])
    
    # Store the dataframe in session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    with tab1:
        # File upload section with help text
        st.write("### Upload Campaign Data")
        st.write("""
        Please upload a CSV file with campaign performance data.
        
        **Required columns:**
        - Delivered_Impressions (numeric)
        - Engagement_Rate (percentage)
        - Click_Rate (percentage)
        - Format (text)
        - Size (text)
        - Placement_Name (text)
        - Vertical (text)
        
        **Optional columns:**
        - Creative_URL (for creative analysis)
        - Video metrics (for video performance analysis)
        - Device metrics (for device performance analysis)
        """)
        
        uploaded_file = st.file_uploader("Upload your campaign data CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                try:
                    df = pd.read_csv(uploaded_file)
                except pd.errors.EmptyDataError:
                    st.error("The uploaded file is empty. Please check the file contents.")
                    st.stop()
                except pd.errors.ParserError:
                    st.error("Unable to parse the CSV file. Please ensure it's a valid CSV format.")
                    st.stop()
                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}")
                    st.stop()
                
                # Validate CSV structure
                errors, warnings = validate_csv_structure(df)
                
                if errors:
                    st.error("Please fix the following errors in your CSV file:")
                    for error in errors:
                        st.error(f"- {error}")
                    st.stop()
                
                if warnings:
                    st.warning("Potential issues found in your data:")
                    for warning in warnings:
                        st.warning(f"- {warning}")
                
                # Clean and validate data
                cleaned_df, error = clean_and_validate_data(df)
                if error:
                    st.error(error)
                    st.stop()
                
                df = cleaned_df
                df = clean_percentage_columns(df)
                st.session_state.df = df  # Store in session state
                
                # Display data summary
                st.success("Data successfully loaded and cleaned!")
                st.write("### Data Summary")
                st.write(f"- Total rows: {len(df)}")
                st.write(f"- Total columns: {len(df.columns)}")
                
                # Display sample of the data
                with st.expander("Preview Data"):
                    st.dataframe(df.head())
                
                # Check if Creative_URL column exists
                has_creative_urls = 'Creative_URL' in df.columns
                if has_creative_urls:
                    valid_urls = df['Creative_URL'].apply(validate_creative_url).sum()
                    st.info(f"Found {valid_urls} valid creative URLs! You can analyze them in the Creative Analysis tab.")
                
                # Main benchmark categories
                benchmark_categories = ['Format', 'Size', 'Placement_Name', 'Vertical']
                selected_category = st.selectbox("Select benchmark category:", benchmark_categories)
                
                if selected_category in df.columns:
                    # Calculate and display benchmarks
                    benchmarks = calculate_benchmarks(df, selected_category)
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
                        benchmarks, 'Avg_Engagement_Rate', 
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
                        benchmarks, 'Avg_Click_Rate',
                        selected_category, 'Average Click Rate'
                    )
                    st.plotly_chart(click_fig)
                    st.markdown(get_download_link(click_fig.to_html(),
                                               f'click_rate_{selected_category}.html',
                                               'Download Chart'),
                              unsafe_allow_html=True)
                    
                    # Video completion analysis
                    if 'Video_User_Completion_Rate' in df.columns:
                        st.subheader("Video Performance Analysis")
                        video_fig = analyze_video_dropoff(df, selected_category)
                        st.plotly_chart(video_fig)
                        st.markdown(get_download_link(video_fig.to_html(),
                                                   f'video_completion_{selected_category}.html',
                                                   'Download Chart'),
                                  unsafe_allow_html=True)
                    
                    # Enhanced Device Performance Analysis
                    st.subheader("Device Performance Analysis")
                    
                    # Overall device metrics
                    device_pie, device_rates = create_device_visualization(df, selected_category)
                    
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
                    device_perf = analyze_device_performance(df, selected_category)
                    st.dataframe(device_perf)
                    st.markdown(get_download_link(device_perf,
                                               f'device_performance_{selected_category}.csv',
                                               'Download Detailed Device Data'),
                              unsafe_allow_html=True)
                    
                    # Additional Insights
                    st.subheader("Additional Insights")
                    trends = analyze_trends(df)
                    
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
                    
                else:
                    st.warning(f"Column {selected_category} not found in the data")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Full error details:")
                st.exception(e)

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