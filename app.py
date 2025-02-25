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

# Configure Streamlit page
st.set_page_config(
    page_title="Campaign Performance Benchmark Analysis",
    page_icon="📊",
    layout="wide"
)

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
                st.experimental_rerun()
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

def create_benchmark_visualization(df, metric, group_by_column, title):
    """Create bar chart for benchmark metrics"""
    fig = px.bar(df, x=df.index, y=metric,
                 title=f'{title} by {group_by_column}',
                 labels={metric: metric.replace('_', ' ')},
                 text=df[metric])
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    return fig

def analyze_video_dropoff(df, group_by_column):
    """Analyze video completion rates at different stages"""
    video_metrics = ['Video_User_25_Rate', 'Video_User_50_Rate', 
                    'Video_User_75_Rate', 'Video_User_Completion_Rate']
    
    dropoff = df.groupby(group_by_column)[video_metrics].mean()
    
    fig = go.Figure()
    for metric in video_metrics:
        fig.add_trace(go.Bar(
            name=metric.replace('Video_User_', '').replace('_Rate', ''),
            x=dropoff.index,
            y=dropoff[metric],
            text=dropoff[metric].round(4)
        ))
    
    fig.update_layout(
        title=f'Video Completion Rates by {group_by_column}',
        barmode='group'
    )
    return fig

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
    
    # Create bar chart for engagement rates
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
        x=device_rates['Device'],
        y=device_rates['Engagement Rate'],
        text=[format_percentage(v) for v in device_rates['Engagement Rate']],
        textposition='outside'
    ))
    fig_rates.add_trace(go.Bar(
        name='Click Rate',
        x=device_rates['Device'],
        y=device_rates['Click Rate'],
        text=[format_percentage(v) for v in device_rates['Click Rate']],
        textposition='outside'
    ))
    
    fig_rates.update_layout(
        title='Performance Metrics by Device',
        barmode='group'
    )
    
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

def main():
    st.title("Campaign Performance Benchmark Analysis")
    
    uploaded_file = st.file_uploader("Upload your campaign data CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read and clean the data
            df = pd.read_csv(uploaded_file)
            df = clean_percentage_columns(df)
            st.success("Data successfully loaded and cleaned!")
            
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

if __name__ == "__main__":
    main()