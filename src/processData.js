const processCSVData = (csvText) => {
    try {
      const lines = csvText.split('\n');
      
      // Get headers and normalize them
      const headers = lines[0].split(',').map(header => 
        header.trim()
          .toLowerCase()
          .replace(/[\s-]/g, '_')  // Replace spaces and hyphens with underscores
          .replace(/[^a-z0-9_]/g, '')  // Remove special characters
      );

      const processedData = lines
        .slice(1)  // Skip header row
        .filter(line => line.trim() !== '')
        .map(line => {
          const values = line.split(',').map(val => val.trim());
          const row = {};

          // Map values to normalized headers
          headers.forEach((header, index) => {
            let value = values[index];

            // Skip empty values
            if (!value) return;

            // Try to convert numbers and percentages
            if (value.includes('%')) {
              value = parseFloat(value.replace('%', '')) / 100;
            } else if (!isNaN(value)) {
              value = parseFloat(value);
            }

            row[header] = value;
          });

          // Only include rows that have at least some data
          if (Object.keys(row).length > 0) {
            return row;
          }
          return null;
        })
        .filter(item => item !== null);

      if (processedData.length === 0) {
        throw new Error('No valid data found in file');
      }

      // Add date sorting if a date column exists
      const dateColumn = headers.find(h => h.includes('date'));
      if (dateColumn) {
        processedData.sort((a, b) => new Date(a[dateColumn]) - new Date(b[dateColumn]));
      }
      
      return processedData;
    } catch (err) {
      console.error('Error processing CSV:', err);
      throw new Error('Failed to process CSV data. Please check the file format.');
    }
  };

export const processBenchmarkData = (data) => {
  // Helper function to find matching column names
  const findColumn = (keywords) => {
    const columns = Object.keys(data[0] || {});
    return columns.find(col => 
      keywords.some(keyword => 
        col.toLowerCase().includes(keyword.toLowerCase())
      )
    );
  };

  // Process engagement metrics with flexible column matching
  const processEngagement = (rawData) => {
    const impressionsCol = findColumn(['impression', 'delivered', 'views']);
    const engagementCol = findColumn(['engagement', 'interact']);
    const clickCol = findColumn(['click', 'ctr']);
    const videoCompletionCol = findColumn(['completion', 'complete']);
    const durationCol = findColumn(['duration', 'length']);

    return {
      engagementRate: calculateMetric(rawData, engagementCol),
      clickRates: calculateMetric(rawData, clickCol),
      impressions: calculateMetric(rawData, impressionsCol),
      videoMetrics: {
        completionRate: calculateMetric(rawData, videoCompletionCol),
        averageDuration: calculateMetric(rawData, durationCol)
      }
    };
  };

  // Process external metrics with flexible column matching
  const processExternalMetrics = (rawData) => {
    const findMetric = (keywords) => {
      const col = findColumn(keywords);
      return col ? rawData[col] : null;
    };

    return {
      marketIndicators: {
        spxChange: findMetric(['spx', 'sp500']),
        vixChange: findMetric(['vix', 'volatility']),
        fearGreedIndex: findMetric(['fear', 'greed'])
      },
      economicData: {
        fomcRates: findMetric(['fomc', 'fed', 'rate']),
        cpi: findMetric(['cpi', 'consumer', 'price']),
        ppi: findMetric(['ppi', 'producer', 'price']),
        gdp: findMetric(['gdp', 'gross']),
        employment: findMetric(['employment', 'jobs'])
      }
    };
  };

  return {
    benchmarks: processEngagement(data),
    externalMetrics: processExternalMetrics(data)
  };
};

// Helper function to calculate metrics with error handling
const calculateMetric = (data, column) => {
  if (!column || !data || data.length === 0) return null;
  
  try {
    const values = data.map(row => row[column]).filter(val => val != null);
    if (values.length === 0) return null;
    
    return values.reduce((sum, val) => sum + (parseFloat(val) || 0), 0) / values.length;
  } catch (err) {
    console.warn(`Error calculating metric for column ${column}:`, err);
    return null;
  }
};

// Helper functions for calculations
const calculateEngagementRate = (data) => {
  // Implementation needed based on your data structure
};

const calculateClickRates = (data) => {
  // Implementation needed based on your data structure
};

const calculateImpressions = (data) => {
  // Implementation needed based on your data structure
};

const calculateVideoCompletionRate = (data) => {
  // Implementation needed based on your data structure
};

const calculateAverageDuration = (data) => {
  // Implementation needed based on your data structure
};