def handle_missing_values(self, df):
    """
    Handle missing values in the dataset.
    
    Strategy:
    - 'Arrival Delay in Minutes': Median imputation (robust to right-skewed distribution)
    - Other columns: Report and optionally handle if needed
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with imputed missing values
    """
    print("\nHandling missing values...")
    
    # Calculate missing values per column
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if missing_cols.empty:
        print("✓ No missing values found.")
        return df
    
    print("Missing values detected:")
    for col, count in missing_cols.items():
        print(f"  - {col}: {count} missing values")
    
    # Median imputation for Arrival Delay
    if 'Arrival Delay in Minutes' in missing_cols.index:
        median_value = df['Arrival Delay in Minutes'].median()
        df['Arrival Delay in Minutes'].fillna(median_value, inplace=True)
        print(f"✓ Imputed 'Arrival Delay in Minutes' with median = {median_value:.2f}")
    
    # Warn about any remaining missing values
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"⚠ Warning: {remaining_missing} missing values remain in other columns.")
        print("  Consider adding imputation strategies for these features.")
    else:
        print("✓ All missing values handled.")
    
    return df
