import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read_data(filename):
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # rename remaining columns
    df = df.rename(columns={'Country Name': 'Country'})

    # melt the dataframe to convert years to a single column
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # convert year column to integer and value column to float
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # separate dataframes with years and countries as columns
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # clean the data
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def calculate_summary_stats(df_years, countries, indicators):
    # create a dictionary to store the summary statistics
    summary_stats = {}

    # calculate summary statistics for each indicator and country
    for indicator in indicators:
        for country in countries:
            # summary statistics for individual countries
            stats = df_years.loc[(country, indicator)].describe()
            summary_stats[f'{country} - {indicator}'] = stats

        # summary statistics for the world
        stats = df_years.loc[('World', indicator)].describe()
        summary_stats[f'World - {indicator}'] = stats

    return summary_stats


def print_summary_stats(summary_stats):
    # print the summary statistics
    for key, value in summary_stats.items():
        print(key)
        print(value)
        print()


# create scatter plots
def create_scatter_plots(df_years, indicators, countries):
    for country in countries:
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                x = df_years.loc[(country, indicators[i])]
                y = df_years.loc[(country, indicators[j])]
                plt.scatter(x, y)
                plt.xlabel(indicators[i])
                plt.ylabel(indicators[j])
                plt.title(country)
                plt.show()


def subset_data(df_years, countries, indicators):
    """
    Subsets the data to include only the selected countries and indicators.
    Returns the subsetted data as a new DataFrame.
    """
    df = df_years.loc[(countries, indicators), :]
    df = df.transpose()
    return df


def calculate_correlations(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr = df.corr()
    return corr


def visualize_correlations(corr):
    """
    Plots the correlation matrix as a heatmap using Seaborn.
    """
    sns.heatmap(corr, cmap='winter', annot=True, square=True)
    plt.title('Correlation Matrix of Indicators')
    plt.show()


def plot_line_Agricultural_land(df_years):
    country_list = ['United States', 'Brazil', 'China', 'Australia']
    indicator = 'Agricultural land (sq. km)'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Agricultural land (sq. km)')
    plt.legend()
    plt.show()


def plot_line_Population_growth(df_years):
    country_list = ['United States', 'Brazil', 'China', 'Australia']
    indicator = 'Population growth (annual %)'
    for country in country_list:
        df_subset = df_years.loc[(country, indicator), :]
        plt.plot(df_subset.index, df_subset.values, label=country)
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.title('Population growth (annual %)')
    plt.legend()
    plt.show()


def plot_Agricultural_land(df_years):
    country_list = ['United States', 'Brazil', 'China', 'Australia']
    
    Agricultural_land_indicator = 'Agricultural land (sq. km)'
    years = [1960, 1970, 1980, 1990, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
       
        Agricultural_land_values = []
        for country in country_list:
            
            Agricultural_land_values.append(
                df_years.loc[(country, Agricultural_land_indicator), year])
       
        rects2 = ax.bar(x + width/2 + i*width/len(years), Agricultural_land_values,
                        width/len(years), label=str(year)+" "+Agricultural_land_indicator)

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(
        'Agricultural land (sq. km)')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()

def plot_Population_growth(df_years):
    country_list = ['United States', 'Brazil', 'China', 'Australia']
    Population_growth_indicator = 'Population growth (annual %)'
   
    years = [1960, 1970, 1980, 1990, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
        Population_growth_values = []
        Population_growth_values = []
        for country in country_list:
           Population_growth_values.append(
                df_years.loc[(country, Population_growth_indicator), year])
           
        rects1 = ax.bar(x - width/2 + i*width/len(years), Population_growth_values,
                        width/len(years), label=str(year)+" "+Population_growth_indicator)
       

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title(
        'Population growth (annual %)')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    df_years, df_countries = read_data(
        r"C:\Users\mouli\Downloads\wbdata.csv")

    # Call plot_line_Agricultural_land to create the  plot
    plot_line_Agricultural_land(df_years)

    # Call plot_line_Population_total to create the  plot
    plot_line_Population_growth(df_years)

    # Call plot_Agricultural_land
    plot_Agricultural_land(df_years)

    # Call plot_population_growth
    plot_Population_growth(df_years)
 
   # select the indicators of interest
indicators = ['Agricultural land (sq. km)',
               'Population growth (annual %)']

# select a few countries for analysis
countries = ['United States', 'Brazil', 'China', 'Australia']

# calculate summary statistics
summary_stats = calculate_summary_stats(df_years, countries, indicators)

# print the summary statistics
print_summary_stats(summary_stats)

# Use the describe method to explore the data for the 'United States'
us_data = df_years.loc[('United States', slice(None)), :]
us_data_describe = us_data.describe()
print("Data for United States")
print(us_data_describe)

# Use the mean method to find the mean Agricultural land (sq. km) for each country
Agricultural_land = df_years.loc[(
    slice(None), 'Agricultural land (sq. km)'), :]
Agricultural_land_mean = Agricultural_land.mean()
print("\nMean Agricultural land for each country")
print(Agricultural_land)

# Use the mean method to find the mean Population growth (annual %) for each year
Population_growth = df_years.loc[(slice(
    None), 'Population growth (annual %)'), :]
Population_growth_mean = Population_growth.mean()
print("\nMean Population total for each country")
print(Population_growth)

df = subset_data(df_years, countries, indicators)
corr = calculate_correlations(df)
visualize_correlations(corr)


# create scatter plots
create_scatter_plots(df_years, indicators, countries)
