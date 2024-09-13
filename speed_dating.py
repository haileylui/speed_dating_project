import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv(
    "/Users/henkalui/Desktop/JDE/Speed_dating_project/Cleaned_Speed_Dating_Data.csv"
)
# Connect to SQLite database
conn = sqlite3.connect(
    "/Users/henkalui/Desktop/JDE/Speed_dating_project/Speed_Dating_database.db",
    timeout=10,
)
# Load the data from the database
df = pd.read_sql_query("SELECT * FROM 'Speed dating'", conn)
# Analysis 1: Radar chart - preference evaluation of 6 attributes that participants perceived
perceived_attributes = [
    "pf_o_att",
    "pf_o_sin",
    "pf_o_int",
    "pf_o_fun",
    "pf_o_amb",
    "pf_o_sha",
]
# Calculating the mean of each gender in each attribute
perceived_mean_men = df[df["gender"] == 0][perceived_attributes].mean().values
perceived_mean_women = df[df["gender"] == 1][perceived_attributes].mean().values
# Print the mean values
print(perceived_mean_men, perceived_mean_women)
# Custom labels
custom_labels = [
    "Attractiveness",
    "Sincerity",
    "Intelligence",
    "Fun",
    "Ambition",
    "Shared Interests",
]
# Create a radar chart using Plotly
fig1 = go.Figure()
# Add trace for men
fig1.add_trace(
    go.Scatterpolar(
        r=perceived_mean_men,  # Close the loop
        theta=custom_labels + [custom_labels[0]],  # Close the loop
        fill="toself",
        name="Men",
        fillcolor="rgba(100, 149, 237, 0.5)",  # Fill color for men
        line=dict(color="cadetblue", width=3),  # Line properties for men
    )
)
# Add trace for women
fig1.add_trace(
    go.Scatterpolar(
        r=perceived_mean_women,  # Close the loop
        theta=custom_labels + [custom_labels[0]],  # Close the loop
        fill="toself",
        name="Women",
        fillcolor="rgba(255, 105, 180, 0.5)",  # Fill color for women
        line=dict(color="hotpink", width=3),  # Line properties for women
    )
)
# Update layout to include legend next to the chart
fig1.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
    showlegend=True,  # Show the legend
    legend=dict(
        title="Gender",
        orientation="v",  # Vertical legend
        yanchor="middle",
        y=0.5,  # Centered vertically
        xanchor="left",
        x=0.8,  # Positioned to the right of the chart
    ),
)
# Show figure
# fig1.show()

# Analysis 2: Correlation - [Preferences vs positive response rate]
# create dataframe from SQLite
# Group by 'iid' and calculate the desired metrics
grouped_df = (
    df.groupby("iid")
    .agg(
        participant=("iid", "first"),  # Use 'iid' as participant identifier
        gender=("gender", "first"),  # Assuming gender is constant for each iid
        avg_attr=("attr_o", "mean"),  # Mean score of attractiveness
        avg_sinc=("sinc_o", "mean"),  # Mean score of sincerity
        avg_intel=("intel_o", "mean"),  # Mean score of intelligence
        avg_fun=("fun_o", "mean"),  # Mean score of fun
        avg_amb=("amb_o", "mean"),  # Mean score of ambition
        avg_shar=("shar_o", "mean"),  # Mean score of shared interests
        positive_response_rate=(
            "dec_o",
            lambda x: (x == 1).mean(),
        ),  # Percentage of getting [1] in dec_o
    )
    .reset_index(drop=True)  # Reset index for a clean DataFrame
)

# Rename columns to match specified names
grouped_df.rename(
    columns={
        "participant": "Participant",
        "gender": "Gender",
        "avg_attr": "Mean Score of Attractiveness",
        "avg_sinc": "Mean Score of Sincerity",
        "avg_intel": "Mean Score of Intelligence",
        "avg_fun": "Mean Score of Fun",
        "avg_amb": "Mean Score of Ambition",
        "avg_shar": "Mean Score of Shared Interests",
        "positive_response_rate": "Positive Response Rate",
    },
    inplace=True,
)

# Display the resulting DataFrame
# print(grouped_df)

# Actual preference for men
men_df = grouped_df[grouped_df["Gender"] == 0]

# Set up the scatter plots
attributes = [
    "Mean Score of Attractiveness",
    "Mean Score of Sincerity",
    "Mean Score of Intelligence",
    "Mean Score of Fun",
    "Mean Score of Ambition",
    "Mean Score of Shared Interests",
]

# Create a subplot layout (2 rows, 3 columns) with increased vertical spacing
fig2 = make_subplots(rows=2, cols=3, vertical_spacing=0.25, horizontal_spacing=0.1)

# Add scatter plots to the subplots
for i, attr in enumerate(attributes):
    row = i // 3 + 1
    col = i % 3 + 1

    fig2.add_trace(
        go.Scatter(
            x=men_df["Positive Response Rate"],
            y=men_df[attr],
            mode="markers",
            name=attr,
        ),
        row=row,
        col=col,
    )

    # Calculate regression line and R
    trend = np.polyfit(men_df["Positive Response Rate"], men_df[attr], 1)
    trend_line = np.polyval(trend, men_df["Positive Response Rate"])
    r_value = np.corrcoef(men_df["Positive Response Rate"], men_df[attr])[0, 1]

    # Add trend line
    fig2.add_trace(
        go.Scatter(
            x=men_df["Positive Response Rate"],
            y=trend_line,
            mode="lines",
            name="Trend Line",
            line=dict(color="red"),
        ),
        row=row,
        col=col,
    )

    # Update axis titles and ticks
    fig2.update_xaxes(
        title_text="Positive Response Rate",
        row=row,
        col=col,
        tickvals=[0.5, 1.0],
        ticktext=["50%", "100%"],
    )

    # Set y-axis range to start from 2
    fig2.update_yaxes(
        title_text=attr, row=row, col=col, range=[2, 9]
    )  # Adjust the upper limit as needed

    # Add R annotation
    fig2.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=2.3,
        text=f"R = {r_value:.2f}",
        showarrow=False,
        font=dict(size=12),
        row=row,
        col=col,
    )

# Update layout
fig2.update_layout(
    title_text="Scatter Correlation Graphs for Female Participants",
    height=600,
    width=1000,
)

# Show the figure
# fig2.show()

# Actual preference of women
# Filter for gender = 1
women_df = grouped_df[grouped_df["Gender"] == 1]

# Set up the scatter plots
attributes = [
    "Mean Score of Attractiveness",
    "Mean Score of Sincerity",
    "Mean Score of Intelligence",
    "Mean Score of Fun",
    "Mean Score of Ambition",
    "Mean Score of Shared Interests",
]

# Create a subplot layout (2 rows, 3 columns) with increased vertical spacing
fig3 = make_subplots(rows=2, cols=3, vertical_spacing=0.25, horizontal_spacing=0.1)

# Add scatter plots to the subplots
for i, attr in enumerate(attributes):
    row = i // 3 + 1
    col = i % 3 + 1

    fig3.add_trace(
        go.Scatter(
            x=women_df["Positive Response Rate"],
            y=women_df[attr],
            mode="markers",
            name=attr,
        ),
        row=row,
        col=col,
    )

    # Calculate regression line and R
    trend = np.polyfit(women_df["Positive Response Rate"], women_df[attr], 1)
    trend_line = np.polyval(trend, women_df["Positive Response Rate"])
    r_value = np.corrcoef(women_df["Positive Response Rate"], women_df[attr])[0, 1]

    # Add trend line
    fig3.add_trace(
        go.Scatter(
            x=women_df["Positive Response Rate"],
            y=trend_line,
            mode="lines",
            name="Trend Line",
            line=dict(color="red"),
        ),
        row=row,
        col=col,
    )

    # Update axis titles and ticks
    fig3.update_xaxes(
        title_text="Positive Response Rate",
        row=row,
        col=col,
        tickvals=[0.5, 1.0],
        ticktext=["50%", "100%"],
    )

    # Set y-axis range to start from 2
    fig3.update_yaxes(
        title_text=attr, row=row, col=col, range=[2, 9]
    )  # Adjust the upper limit as needed

    # Add R annotation
    fig3.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=2.3,
        text=f"R = {r_value:.2f}",
        showarrow=False,
        font=dict(size=12),
        row=row,
        col=col,
    )

# Update layout
fig3.update_layout(
    title_text="Scatter Correlation Graphs for Male Participants",
    height=600,
    width=1000,
)

# Show the figure
# fig3.show()

# Difference between men and women
# Calculate correlation (R) values for each attribute for males and females
correlation_values = {
    "Attractiveness": [
        np.corrcoef(
            men_df["Positive Response Rate"], men_df["Mean Score of Attractiveness"]
        )[0, 1],
        np.corrcoef(
            women_df["Positive Response Rate"], women_df["Mean Score of Attractiveness"]
        )[0, 1],
    ],
    "Sincerity": [
        np.corrcoef(
            men_df["Positive Response Rate"], men_df["Mean Score of Sincerity"]
        )[0, 1],
        np.corrcoef(
            women_df["Positive Response Rate"], women_df["Mean Score of Sincerity"]
        )[0, 1],
    ],
    "Intelligence": [
        np.corrcoef(
            men_df["Positive Response Rate"], men_df["Mean Score of Intelligence"]
        )[0, 1],
        np.corrcoef(
            women_df["Positive Response Rate"], women_df["Mean Score of Intelligence"]
        )[0, 1],
    ],
    "Fun": [
        np.corrcoef(men_df["Positive Response Rate"], men_df["Mean Score of Fun"])[
            0, 1
        ],
        np.corrcoef(women_df["Positive Response Rate"], women_df["Mean Score of Fun"])[
            0, 1
        ],
    ],
    "Ambition": [
        np.corrcoef(men_df["Positive Response Rate"], men_df["Mean Score of Ambition"])[
            0, 1
        ],
        np.corrcoef(
            women_df["Positive Response Rate"], women_df["Mean Score of Ambition"]
        )[0, 1],
    ],
    "Shared Interests": [
        np.corrcoef(
            men_df["Positive Response Rate"], men_df["Mean Score of Shared Interests"]
        )[0, 1],
        np.corrcoef(
            women_df["Positive Response Rate"],
            women_df["Mean Score of Shared Interests"],
        )[0, 1],
    ],
}

# Convert the dictionary into a DataFrame for easier plotting
correlation_df = pd.DataFrame(correlation_values, index=["Women", "Men"])

# Create a bar chart for correlation values
fig_corr = go.Figure()

# Add bars for men and women
fig_corr.add_trace(
    go.Bar(
        x=correlation_df.columns,
        y=correlation_df.loc["Women"],
        name="Women",
        marker_color="hotpink",
    )
)
fig_corr.add_trace(
    go.Bar(
        x=correlation_df.columns,
        y=correlation_df.loc["Men"],
        name="Men",
        marker_color="cadetblue",
    )
)

# Update layout for the correlation values bar chart
fig_corr.update_layout(
    title="Correlation (R) of Attributes with Positive Response Rate",
    xaxis_title="Attributes",
    yaxis_title="Correlation (R)",
    barmode="group",
    legend=dict(title="Gender"),
)

# Show the correlation values figure
fig_corr.show()
# Ensure that the connection is closed
conn.close()
