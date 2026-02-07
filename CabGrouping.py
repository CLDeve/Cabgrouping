import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components
import math
from sklearn.cluster import DBSCAN
import numpy as np
from geopy.distance import geodesic

__version__ = "v0.0.0.2"

if 'max_distance' not in st.session_state:
    st.session_state.max_distance = 0

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Taxi Grouping Optimization</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Upload Master File")
    st.write("(PostalCode, Latitude, Longitude)")
    master_file = st.file_uploader("Choose the master file", type=["xlsx"], key="master")

    st.write("###")

    st.header("Upload File with PickUpPostal and DropOffPostal")
    upload_file = st.file_uploader("Choose the file to process", type=["xlsx"], key="upload")

    st.write("###")

    run_button = st.button('Run Grouping and Clustering')

    status_placeholder = st.empty()

    st.write("###")

    download_placeholder = st.empty()

if run_button:
    if master_file and upload_file:
        # Update the status to show "Processing..."
        status_placeholder.text("Processing...")

        master_df = pd.read_excel(master_file, dtype={'PostalCode': str})

        upload_df = pd.read_excel(upload_file, dtype={'PickUpPostal': str, 'DropOffPostal': str})

        if 'StaffID' in upload_df.columns:
            upload_df = upload_df.drop_duplicates(subset=['StaffID'], keep='first')

        pickup_df = upload_df.merge(master_df, left_on='PickUpPostal', right_on='PostalCode', how='left', suffixes=('', '_Pickup'))
        pickup_df.rename(columns={'Latitude': 'PickUp_Latitude', 'Longitude': 'PickUp_Longitude'}, inplace=True)

        dropoff_df = pickup_df.merge(master_df, left_on='DropOffPostal', right_on='PostalCode', how='left', suffixes=('', '_DropOff'))
        dropoff_df.rename(columns={'Latitude': 'DropOff_Latitude', 'Longitude': 'DropOff_Longitude'}, inplace=True)

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

            status_placeholder.success("Completed!")
        else:
            dropoff_df.dropna(subset=['PickUp_Latitude', 'PickUp_Longitude', 'DropOff_Latitude', 'DropOff_Longitude'], inplace=True)

            def calculate_distance(point1, point2):
                return geodesic(point1, point2).kilometers

            def calculate_centroid(df):
                avg_lat = df['PickUp_Latitude'].mean()
                avg_long = df['PickUp_Longitude'].mean()
                return avg_lat, avg_long

            def sort_by_distance_from_centroid(df, centroid):
                df['DistanceFromCentroid'] = df.apply(
                    lambda row: calculate_distance(centroid, (row['PickUp_Latitude'], row['PickUp_Longitude'])), axis=1
                )
                df = df.sort_values(by='DistanceFromCentroid')
                return df

            def determine_clusters_needed(df, max_group_size):
                num_clusters = max(math.ceil(len(df) / max_group_size), 1)
                return num_clusters

            def normalize_postal(code):
                if pd.isna(code):
                    return None
                digits = ''.join(ch for ch in str(code).strip() if ch.isdigit())
                if len(digits) < 2:
                    return None
                if len(digits) <= 6:
                    return digits.zfill(6)
                return digits[:6]

            def build_postal_groups(df):
                groups = []
                for postal, g in df.groupby('PickUpPostalNorm'):
                    groups.append({
                        'postal': postal,
                        'lat': g['PickUp_Latitude'].mean(),
                        'lon': g['PickUp_Longitude'].mean(),
                        'indices': g.index.tolist(),
                        'size': len(g)
                    })
                return groups

            def cluster_postal_groups(postal_groups, eps_km=1.5, min_samples=2):
                if not postal_groups:
                    return {}
                coords = np.radians([[g['lat'], g['lon']] for g in postal_groups])
                if len(coords) == 1:
                    return {0: postal_groups}
                db = DBSCAN(
                    eps=eps_km / 6371.0,
                    min_samples=min_samples,
                    metric='haversine'
                ).fit(coords)
                labels = db.labels_
                clustered = {}
                next_cluster = max(labels) + 1 if labels.size > 0 else 0
                for g, label in zip(postal_groups, labels):
                    if label == -1:
                        label = next_cluster
                        next_cluster += 1
                    clustered.setdefault(label, []).append(g)
                return clustered

            def pack_into_taxis(cluster_groups, max_group_size=4, max_unique_postals=4, start_counter=1):
                taxi_group_counter = start_counter
                assignments = {}

                for _, groups in cluster_groups.items():
                    taxis = []
                    # Prioritize larger postal groups so they stay together when possible.
                    groups_sorted = sorted(groups, key=lambda x: x['size'], reverse=True)

                    for g in groups_sorted:
                        postal = g['postal']
                        indices = g['indices']

                        # Split only if the same postal exceeds capacity.
                        chunks = [indices[i:i + max_group_size] for i in range(0, len(indices), max_group_size)]

                        for chunk in chunks:
                            placed = False
                            for taxi in taxis:
                                capacity_left = max_group_size - len(taxi['indices'])
                                if len(chunk) > capacity_left:
                                    continue
                                if postal not in taxi['postals'] and len(taxi['postals']) >= max_unique_postals:
                                    continue
                                taxi['indices'].extend(chunk)
                                taxi['postals'].add(postal)
                                placed = True
                                break

                            if not placed:
                                taxis.append({
                                    'indices': list(chunk),
                                    'postals': {postal}
                                })

                    for taxi in taxis:
                        taxi_label = f'Taxi {taxi_group_counter}'
                        for idx in taxi['indices']:
                            assignments[idx] = taxi_label
                        taxi_group_counter += 1

                return assignments, taxi_group_counter

            dropoff_df['PickUpPostalNorm'] = dropoff_df['PickUpPostal'].apply(normalize_postal)

            postal_groups = build_postal_groups(dropoff_df)
            clustered_groups = cluster_postal_groups(postal_groups, eps_km=1.5, min_samples=2)

            assignments, _ = pack_into_taxis(
                clustered_groups,
                max_unique_postals=4,
                max_group_size=4,
                start_counter=1
            )

            dropoff_df['TaxiGroup'] = dropoff_df.index.map(assignments.get)

            output_excel_file_path = 'Taxi_Grouped_Data.xlsx'
            dropoff_df.to_excel(output_excel_file_path, index=False)

            status_placeholder.success("Completed!")

            with download_placeholder:
                with open(output_excel_file_path, "rb") as file:
                    st.download_button(
                        label="Download Optimized Excel File",
                        data=file,
                        file_name=output_excel_file_path,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            map_center = [1.3521, 103.8198]  # Coordinates for Singapore
            m = folium.Map(location=map_center, zoom_start=12)

            def add_markers_to_map(m, df):
                for _, row in df.iterrows():
                    # Add pick-up marker
                    folium.Marker(
                        location=[row['PickUp_Latitude'], row['PickUp_Longitude']],
                        popup=f"Pick-Up: {row['PickUpPostal']} | Group: {row['TaxiGroup']}",
                        icon=folium.Icon(color='red', icon='home')
                    ).add_to(m)

                    folium.Marker(
                        location=[row['DropOff_Latitude'], row['DropOff_Longitude']],
                        popup=f"Drop-Off: {row['DropOffPostal']} | Group: {row['TaxiGroup']}",
                        icon=folium.Icon(color='green', icon='flag')
                    ).add_to(m)

            add_markers_to_map(m, dropoff_df)

            map_html = m._repr_html_()
            components.html(map_html, height=800, scrolling=True)

    else:
        st.error("Please upload both the master file and the file with PickUpPostal and DropOffPostal.")
