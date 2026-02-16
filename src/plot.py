import pandas as pd
import plotly.express as px

# 1. Load the data
# Adjust the filename if necessary
df = pd.read_csv('data/cost_centre_hierarchy_20250408_edit.csv')

# 2. Clean up column names (handling the colon prefix in your snippet)
df.columns = [col.replace(':', '') for col in df.columns]

# 3. Create the Interactive Sunburst Chart
fig = px.sunburst(
    df,
    names='child',
    parents='parent',
    title="Cost Centre Hierarchy (Click sectors to zoom in/out)",
    height=800
)

# 4. Enhance the layout
fig.update_layout(
    margin=dict(t=50, l=10, r=10, b=10),
    extendsunburstcolors=True
)

# 5. Show the plot
# This will open in your default web browser
fig.show()