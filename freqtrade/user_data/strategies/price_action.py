# price_action_analysis.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def identify_pivots(df, window_size=5):
    """
    Identify local maxima (pivot highs) and minima (pivot lows) in price data.

    Parameters:
    - df: DataFrame containing 'high' and 'low' price columns.
    - window_size: Number of periods to consider for local peaks/troughs.

    Returns:
    - df: DataFrame with added 'pivot_high' and 'pivot_low' columns.
    """
    df = df.copy()
    df['pivot_high'] = df['high'][(df['high'] == df['high'].rolling(window=window_size, center=True).max())]
    df['pivot_low'] = df['low'][(df['low'] == df['low'].rolling(window=window_size, center=True).min())]
    return df

def cluster_price_levels(df, n_clusters=5):
    """
    Cluster pivot points to identify significant support/resistance zones.

    Parameters:
    - df: DataFrame with 'pivot_high' and 'pivot_low' columns.
    - n_clusters: Number of clusters to form.

    Returns:
    - cluster_centers: DataFrame with cluster centers representing price levels.
    """
    # Extract pivot levels
    pivot_levels = pd.concat([
        df['pivot_high'].dropna(),
        df['pivot_low'].dropna()
    ]).values.reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pivot_levels)
    centers = kmeans.cluster_centers_.flatten()
    cluster_centers = pd.DataFrame({'level': centers})
    return cluster_centers

def get_support_resistance_zones(df, cluster_centers, tolerance=0.005):
    """
    Define support and resistance zones based on clustered pivot levels.

    Parameters:
    - df: DataFrame with 'close' price column.
    - cluster_centers: DataFrame with 'level' column from cluster_price_levels.
    - tolerance: Percentage tolerance around cluster centers to define zones.

    Returns:
    - zones: List of dictionaries with 'type' ('support' or 'resistance') and 'level'.
    """
    median_price = df['close'].median()
    zones = []

    for center in cluster_centers['level']:
        # Determine if the level is support or resistance
        zone_type = 'support' if center < median_price else 'resistance'
        zones.append({'type': zone_type, 'level': center})

    return zones

def recentness_weighted_levels(df, decay_factor=0.99):
    """
    Weight pivot levels based on their recency using exponential decay.

    Parameters:
    - df: DataFrame with 'pivot_high' and 'pivot_low' columns.
    - decay_factor: Exponential decay factor between 0 and 1.

    Returns:
    - weighted_levels: DataFrame with 'level' and 'weight' columns.
    """
    df = df.copy()
    df['time_index'] = np.arange(len(df))

    # Extract pivot points
    pivot_highs = df[['time_index', 'pivot_high']].dropna()
    pivot_lows = df[['time_index', 'pivot_low']].dropna()

    # Combine and calculate weights
    pivot_highs['weight'] = decay_factor ** (df['time_index'].max() - pivot_highs['time_index'])
    pivot_lows['weight'] = decay_factor ** (df['time_index'].max() - pivot_lows['time_index'])

    # Combine highs and lows
    weighted_levels = pd.concat([
        pivot_highs[['pivot_high', 'weight']].rename(columns={'pivot_high': 'level'}),
        pivot_lows[['pivot_low', 'weight']].rename(columns={'pivot_low': 'level'})
    ])

    return weighted_levels

def get_price_density(df, bin_size=10):
    """
    Create a density map of price zones based on interaction frequency.

    Parameters:
    - df: DataFrame with 'close' price column.
    - bin_size: Number of bins to divide the price range.

    Returns:
    - density: DataFrame with 'price_bin' and 'count' columns.
    """
    price_min = df['close'].min()
    price_max = df['close'].max()
    bins = np.linspace(price_min, price_max, bin_size)
    df['price_bin'] = pd.cut(df['close'], bins=bins)
    density = df['price_bin'].value_counts().reset_index()
    density.columns = ['price_bin', 'count']
    density.sort_values('price_bin', inplace=True)
    return density

def detect_trendlines(df, sensitivity=5):
    """
    Identify trendlines by fitting lines to sequences of pivot points.

    Parameters:
    - df: DataFrame with 'time_index', 'pivot_high', and 'pivot_low' columns.
    - sensitivity: Number of pivot points to consider for each trendline.

    Returns:
    - trendlines: List of dictionaries with 'type', 'slope', and 'intercept'.
    """
    trendlines = []

    # Trendlines from pivot highs (resistance)
    pivot_highs = df[['time_index', 'pivot_high']].dropna()
    for i in range(len(pivot_highs) - sensitivity + 1):
        x = pivot_highs['time_index'].iloc[i:i+sensitivity]
        y = pivot_highs['pivot_high'].iloc[i:i+sensitivity]
        slope, intercept = np.polyfit(x, y, 1)
        trendlines.append({'type': 'resistance', 'slope': slope, 'intercept': intercept})

    # Trendlines from pivot lows (support)
    pivot_lows = df[['time_index', 'pivot_low']].dropna()
    for i in range(len(pivot_lows) - sensitivity + 1):
        x = pivot_lows['time_index'].iloc[i:i+sensitivity]
        y = pivot_lows['pivot_low'].iloc[i:i+sensitivity]
        slope, intercept = np.polyfit(x, y, 1)
        trendlines.append({'type': 'support', 'slope': slope, 'intercept': intercept})

    return trendlines

def measure_price_acceleration(df):
    """
    Calculate the velocity and acceleration of the closing price.

    Parameters:
    - df: DataFrame with 'close' price column.

    Returns:
    - df: DataFrame with added 'velocity' and 'acceleration' columns.
    """
    df = df.copy()
    df['velocity'] = df['close'].diff()
    df['acceleration'] = df['velocity'].diff()
    return df

def detect_reversal_points(df, threshold=1.5):
    """
    Detect sharp reversals in price based on acceleration.

    Parameters:
    - df: DataFrame with 'acceleration' column.
    - threshold: Multiplier for standard deviation to define significant reversal.

    Returns:
    - df: DataFrame with added 'reversal' column (1 if reversal detected, 0 otherwise).
    """
    df = df.copy()
    acc_std = df['acceleration'].std()
    df['reversal'] = np.where(abs(df['acceleration']) > threshold * acc_std, 1, 0)
    return df

def analyze_price_action(df, window_size=5, n_clusters=5, tolerance=0.005, decay_factor=0.99, bin_size=10, sensitivity=5, threshold=1.5):
    """
    Main function to analyze price action by calling all individual methods.

    Parameters:
    - df: DataFrame containing OHLCV data with columns 'open', 'high', 'low', 'close', 'volume'.
    - window_size: Window size for identifying pivots.
    - n_clusters: Number of clusters for price levels.
    - tolerance: Tolerance for defining support/resistance zones.
    - decay_factor: Decay factor for weighting recent levels.
    - bin_size: Bin size for price density calculation.
    - sensitivity: Sensitivity for detecting trendlines.
    - threshold: Threshold for detecting reversal points.

    Returns:
    - df: DataFrame with additional columns populated.
    - results: Dictionary containing support/resistance zones, weighted levels, price density, trendlines.
    """
    df = df.copy()

    # Ensure 'time_index' column exists
    df.reset_index(drop=True, inplace=True)
    df['time_index'] = df.index

    # Step 1: Identify pivot points
    df = identify_pivots(df, window_size=window_size)

    # Step 2: Cluster pivot levels
    cluster_centers = cluster_price_levels(df, n_clusters=n_clusters)

    # Step 3: Define support and resistance zones
    zones = get_support_resistance_zones(df, cluster_centers, tolerance=tolerance)

    # Step 4: Weight levels based on recency
    weighted_levels = recentness_weighted_levels(df, decay_factor=decay_factor)

    # Step 5: Create price density heatmap data
    density = get_price_density(df, bin_size=bin_size)

    # Step 6: Detect trendlines
    trendlines = detect_trendlines(df, sensitivity=sensitivity)

    # Step 7: Measure price acceleration
    df = measure_price_acceleration(df)

    # Step 8: Detect reversal points
    df = detect_reversal_points(df, threshold=threshold)

    # Step 9: Consensus Columns
    # For example, mark points where reversal detected at support/resistance zones
    df['reversal_at_zone'] = 0
    for zone in zones:
        level = zone['level']
        zone_type = zone['type']
        # Define the zone boundaries
        lower_bound = level * (1 - tolerance)
        upper_bound = level * (1 + tolerance)
        # Check for reversals within the zone
        df['reversal_at_zone'] = np.where(
            (df['reversal'] == 1) &
            (df['close'] >= lower_bound) &
            (df['close'] <= upper_bound),
            1, df['reversal_at_zone']
        )

    # Collect all results in a dictionary
    results = {
        'support_resistance_zones': zones,
        'weighted_levels': weighted_levels,
        'price_density': density,
        'trendlines': trendlines
    }

    return df, results

# Example usage:
if __name__ == "__main__":
    # Load your OHLCV data into a DataFrame 'df'
    # df = pd.read_csv('your_data.csv')

    # For demonstration, let's create a sample DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100)
    prices = np.sin(np.linspace(0, 20, 100)) * 10 + 100 + np.random.normal(0, 1, 100)
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.normal(0, 1, 100),
        'high': prices + np.random.normal(0, 1, 100),
        'low': prices - np.random.normal(0, 1, 100),
        'close': prices,
        'volume': np.random.randint(100, 1000, size=100)
    })

    # Analyze price action
    df, results = analyze_price_action(df)

    # Now 'df' has additional columns, and 'results' contains analysis outputs
    print(df.head())
    print(results['support_resistance_zones'])
