import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components
import math
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# Ensure MAX_DISTANCE is stored in the session state
if 'max_distance' not in st.session_state:
    st.session_state.max_distance = 0

# Full-page layout
st.set_page_config(layout="wide")

# Title of the app centered
st.markdown("<h1 style='text-align: center;'>Taxi Grouping Optimization</h1>", unsafe_allow_html=True)

# Create sidebar for file uploads and processing actions
with st.sidebar:
    st.header("Upload Master File")
    st.write("(PostalCode, Latitude, Longitude)")
    master_file = st.file_uploader("Choose the master file", type=["xlsx"], key="master")

    st.write("###")

    st.header("Upload File with PickUpPostal and DropOffPostal")
    upload_file = st.file_uploader("Choose the file to process", type=["xlsx"], key="upload")

    # Spacer for better layout
    st.write("###")

    # Add a button to run the processing
    run_button = st.button('Run Grouping and Clustering')

    # Placeholder for the processing status
    status_placeholder = st.empty()

    # Spacer for better layout
    st.write("###")

    # Placeholder for the download button
    download_placeholder = st.empty()

# Map display in the main area
if run_button:
    if master_file and upload_file:
        # Update the status to show "Processing..."
        status_placeholder.text("Processing...")

        # Load the master file
        master_df = pd.read_excel(master_file)

        # Load the uploaded file
        upload_df = pd.read_excel(upload_file)

        # Remove duplicates based on a unique identifier (e.g., StaffID) if it exists
        if 'StaffID' in upload_df.columns:
            upload_df = upload_df.drop_duplicates(subset=['StaffID'], keep='first')

        # Merge to get Latitude and Longitude for Pick-Up and Drop-Off Postals
        pickup_df = upload_df.merge(master_df, left_on='PickUpPostal', right_on='PostalCode', how='left', suffixes=('', '_Pickup'))
        pickup_df.rename(columns={'Latitude': 'PickUp_Latitude', 'Longitude': 'PickUp_Longitude'}, inplace=True)

        dropoff_df = pickup_df.merge(master_df, left_on='DropOffPostal', right_on='PostalCode', how='left', suffixes=('', '_DropOff'))
        dropoff_df.rename(columns={'Latitude': 'DropOff_Latitude', 'Longitude': 'DropOff_Longitude'}, inplace=True)

        # Identify missing postal codes for both PickUp and DropOff
        missing_pickup = dropoff_df[dropoff_df['PickUp_Latitude'].isna() | dropoff_df['PickUp_Longitude'].isna()]
        missing_dropoff = dropoff_df[dropoff_df['DropOff_Latitude'].isna() | dropoff_df['DropOff_Longitude'].isna()]

        if not missing_pickup.empty or not missing_dropoff.empty:
            st.warning("Some postal codes in the uploaded file do not have matching latitude and longitude in the master file.")
            
            if not missing_pickup.empty:
                st.subheader("Missing Pick-Up Postal Codes")
                st.dataframe(missing_pickup[['PickUpPostal']].drop_duplicates())
            
            if not missing_dropoff.empty:
                st.subheader("Missing Drop-Off Postal Codes")
                st.dataframe(missing_dropoff[['DropOffPostal']].drop_duplicates())

            # Update the status to show "Completed!"
            status_placeholder.success("Completed!")
        else:
            # Drop rows with missing latitude or longitude
            dropoff_df.dropna(subset=['PickUp_Latitude', 'PickUp_Longitude', 'DropOff_Latitude', 'DropOff_Longitude'], inplace=True)

            # Function to calculate the distance between two points
            def calculate_distance(point1, point2):
                return geodesic(point1, point2).kilometers

            # Function to calculate the centroid of pickup points
            def calculate_centroid(df):
                avg_lat = df['PickUp_Latitude'].mean()
                avg_long = df['PickUp_Longitude'].mean()
                return avg_lat, avg_long

            # Function to sort postal codes by distance from the centroid
            def sort_by_distance_from_centroid(df, centroid):
                df['DistanceFromCentroid'] = df.apply(
                    lambda row: calculate_distance(centroid, (row['PickUp_Latitude'], row['PickUp_Longitude'])), axis=1
                )
                df = df.sort_values(by='DistanceFromCentroid')
                return df

            # Function to determine the number of clusters needed based on max group size
            def determine_clusters_needed(df, max_group_size):
                num_clusters = max(math.ceil(len(df) / max_group_size), 1)
                return num_clusters

            # Initial clustering using KMeans to create groups based on combined pick-up and drop-off proximity
            def cluster_passengers(df, n_clusters):
                combined_coords = df[['PickUp_Latitude', 'PickUp_Longitude', 'DropOff_Latitude', 'DropOff_Longitude']].copy()
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(combined_coords)
                df['Cluster'] = kmeans.labels_
                return df, kmeans

            # Adjust groups to minimize distance and taxi usage, ensuring max 4 unique postal codes per taxi and max 4 passengers per group
            def adjust_groups(df, max_unique_postals=4, max_group_size=4):
                adjusted_df = pd.DataFrame(columns=df.columns)
                adjusted_df['TaxiGroup'] = ''  # Initialize the 'TaxiGroup' column with empty strings

                taxi_group_counter = 1
                for cluster in df['Cluster'].unique():
                    cluster_df = df[df['Cluster'] == cluster].copy()

                    # Sort the cluster by distance from the pick-up point
                    cluster_df['Distance'] = cluster_df.apply(
                        lambda row: calculate_distance(
                            (row['PickUp_Latitude'], row['PickUp_Longitude']),
                            (row['DropOff_Latitude'], row['DropOff_Longitude'])
                        ), axis=1)
                    cluster_df = cluster_df.sort_values(by='Distance')

                    while len(cluster_df) > 0:
                        group = pd.DataFrame()
                        unique_postals = set()
                        group_size = 0

                        # Add to group until constraints are met (max 4 unique postals and max 4 passengers)
                        for i, row in cluster_df.iterrows():
                            potential_postals = unique_postals.union([row['PickUpPostal'], row['DropOffPostal']])
                            
                            if len(potential_postals) > max_unique_postals or group_size >= max_group_size:
                                break

                            unique_postals = potential_postals
                            group = pd.concat([group, pd.DataFrame([row])])
                            group_size += 1

                        group['TaxiGroup'] = f'Taxi {taxi_group_counter}'
                        adjusted_df = pd.concat([adjusted_df, group], ignore_index=True)

                        # Remove grouped entries from the cluster
                        cluster_df = cluster_df.drop(group.index)

                        taxi_group_counter += 1

                return adjusted_df

            # Calculate the centroid of all pickup points
            centroid = calculate_centroid(dropoff_df)

            # Sort the data by distance from the centroid
            dropoff_df = sort_by_distance_from_centroid(dropoff_df, centroid)

            # Calculate the number of clusters needed
            num_clusters = determine_clusters_needed(dropoff_df, max_group_size=4)

            # Perform initial clustering
            dropoff_df, kmeans = cluster_passengers(dropoff_df, num_clusters)

            # Adjust groups to ensure max 4 unique postal codes per taxi and max 4 passengers per group
            dropoff_df = adjust_groups(dropoff_df, max_unique_postals=4, max_group_size=4)

            # Save the final DataFrame to an Excel file
            output_excel_file_path = 'Taxi_Grouped_Data.xlsx'
            dropoff_df.to_excel(output_excel_file_path, index=False)

            # Update the status to show "Completed!"
            status_placeholder.success("Completed!")

            # Create a download button for the Excel file
            with download_placeholder:
                with open(output_excel_file_path, "rb") as file:
                    st.download_button(
                        label="Download Optimized Excel File",
                        data=file,
                        file_name=output_excel_file_path,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            # Create a map centered around Singapore, zoomed to show the area properly
            map_center = [1.3521, 103.8198]  # Coordinates for Singapore
            m = folium.Map(location=map_center, zoom_start=12)

            # Function to add markers to the map
            def add_markers_to_map(m, df):
                for _, row in df.iterrows():
                    # Add pick-up marker
                    folium.Marker(
                        location=[row['PickUp_Latitude'], row['PickUp_Longitude']],
                        popup=f"Pick-Up: {row['PickUpPostal']} | Group: {row['TaxiGroup']}",
                        icon=folium.Icon(color='red', icon='home')
                    ).add_to(m)

                    # Add drop-off marker
                    folium.Marker(
                        location=[row['DropOff_Latitude'], row['DropOff_Longitude']],
                        popup=f"Drop-Off: {row['DropOffPostal']} | Group: {row['TaxiGroup']}",
                        icon=folium.Icon(color='green', icon='flag')
                    ).add_to(m)

            # Add markers to the map
            add_markers_to_map(m, dropoff_df)

            # Render the updated map HTML
            map_html = m._repr_html_()
            components.html(map_html, height=800, scrolling=True)

    else:
        st.error("Please upload both the master file and the file with PickUpPostal and DropOffPostal.")
