import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components
import math
from sklearn.cluster import KMeans
from geopy.distance import geodesic

__version__ = "v0.0.1.5"

if 'max_distance' not in st.session_state:
    st.session_state.max_distance = 0

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center;'>Taxi Grouping Optimization</h1>", unsafe_allow_html=True)
st.caption(f"Version {__version__}")

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

        master_df = pd.read_excel(master_file)

        upload_df = pd.read_excel(upload_file)

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

            def cluster_passengers(df, n_clusters):
                combined_coords = df[['PickUp_Latitude', 'PickUp_Longitude', 'DropOff_Latitude', 'DropOff_Longitude']].copy()
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(combined_coords)
                df['Cluster'] = kmeans.labels_
                return df, kmeans

            def get_sector(code):
                if pd.isna(code):
                    return None
                digits = ''.join(ch for ch in str(code).strip() if ch.isdigit())
                if len(digits) < 2:
                    return None
                if len(digits) <= 6:
                    digits = digits.zfill(6)
                return digits[:2]

            def adjust_groups(df, max_unique_postals=4, max_group_size=4):
                adjusted_df = pd.DataFrame(columns=df.columns)
                adjusted_df['TaxiGroup'] = ''  # Initialize the 'TaxiGroup' column with empty strings

                taxi_group_counter = 1
                for cluster in df['Cluster'].unique():
                    cluster_df = df[df['Cluster'] == cluster].copy()

                    cluster_df['Distance'] = cluster_df.apply(
                        lambda row: calculate_distance(
                            (row['PickUp_Latitude'], row['PickUp_Longitude']),
                            (row['DropOff_Latitude'], row['DropOff_Longitude'])
                        ), axis=1)
                    cluster_df = cluster_df.sort_values(by='Distance')

                    while len(cluster_df) > 0:
                        group_rows = []
                        unique_postals = set()
                        group_size = 0
                        group_sectors = set()

                        while group_size < max_group_size and len(cluster_df) > 0:
                            if group_size == 0:
                                candidates = cluster_df.index.tolist()
                            else:
                                same_sector = cluster_df[cluster_df['DropOffSector'].isin(group_sectors)]
                                other_sector = cluster_df[~cluster_df['DropOffSector'].isin(group_sectors)]
                                candidates = list(same_sector.index) + list(other_sector.index)

                            added = False
                            for idx in candidates:
                                row = cluster_df.loc[idx]
                                potential_postals = unique_postals.union([row['PickUpPostal'], row['DropOffPostal']])
                                if len(potential_postals) > max_unique_postals:
                                    continue
                                unique_postals = potential_postals
                                group_rows.append(row)
                                group_size += 1
                                group_sectors.add(row['DropOffSector'])
                                cluster_df = cluster_df.drop(idx)
                                added = True
                                break

                            if not added:
                                break

                        group = pd.DataFrame(group_rows)
                        group['TaxiGroup'] = f'Taxi {taxi_group_counter}'
                        adjusted_df = pd.concat([adjusted_df, group], ignore_index=True)

                        taxi_group_counter += 1

                return adjusted_df

            def soft_merge_by_sector(df, max_group_size=4, max_unique_postals=4, max_distance_km=3):
                def normalize_postal(code):
                    if pd.isna(code):
                        return None
                    digits = ''.join(ch for ch in str(code).strip() if ch.isdigit())
                    if len(digits) < 2:
                        return None
                    if len(digits) <= 6:
                        return digits.zfill(6)
                    return digits[:6]

                def get_sector(code):
                    postal = normalize_postal(code)
                    return postal[:2] if postal else None

                df = df.copy()
                df['_DropOffSector'] = df['DropOffPostal'].apply(get_sector)

                def build_taxis():
                    taxis = {}
                    for taxi, g in df.groupby('TaxiGroup'):
                        taxis[taxi] = {
                            'indices': g.index.tolist(),
                            'sectors': set(g['_DropOffSector']),
                            'postals': set(g['PickUpPostal']).union(set(g['DropOffPostal'])),
                            'centroid': (g['DropOff_Latitude'].mean(), g['DropOff_Longitude'].mean())
                        }
                    return taxis

                taxis = build_taxis()
                donors = sorted(taxis.keys(), key=lambda t: len(taxis[t]['indices']))

                for donor in donors:
                    if donor not in taxis:
                        continue
                    donor_indices = list(taxis[donor]['indices'])

                    temp_taxis = {}
                    for t, info in taxis.items():
                        if t == donor:
                            continue
                        temp_taxis[t] = {
                            'indices': list(info['indices']),
                            'sectors': set(info['sectors']),
                            'postals': set(info['postals']),
                            'centroid': info['centroid']
                        }

                    assign_map = {}
                    success = True

                    for idx in donor_indices:
                        row = df.loc[idx]
                        row_sector = row['_DropOffSector']
                        row_postals = {row['PickUpPostal'], row['DropOffPostal']}
                        row_point = (row['DropOff_Latitude'], row['DropOff_Longitude'])

                        candidates = []
                        for t, info in temp_taxis.items():
                            if len(info['indices']) >= max_group_size:
                                continue
                            if calculate_distance(row_point, info['centroid']) > max_distance_km:
                                continue
                            new_postals = info['postals'].union(row_postals)
                            if len(new_postals) > max_unique_postals:
                                continue
                            same_sector = row_sector in info['sectors']
                            capacity_left = max_group_size - len(info['indices'])
                            score = (0 if same_sector else 1, capacity_left)
                            candidates.append((score, t))

                        if not candidates:
                            success = False
                            break

                        candidates.sort()
                        target = candidates[0][1]
                        info = temp_taxis[target]
                        info['indices'].append(idx)
                        info['sectors'].add(row_sector)
                        info['postals'].update(row_postals)
                        info['centroid'] = (
                            df.loc[info['indices']]['DropOff_Latitude'].mean(),
                            df.loc[info['indices']]['DropOff_Longitude'].mean()
                        )
                        assign_map[idx] = target

                    if success:
                        for idx, target in assign_map.items():
                            df.at[idx, 'TaxiGroup'] = target
                        taxis = build_taxis()

                df.drop(columns=['_DropOffSector'], inplace=True)
                return df

            dropoff_df['DropOffSector'] = dropoff_df['DropOffPostal'].apply(get_sector)
            centroid = calculate_centroid(dropoff_df)

            dropoff_df = sort_by_distance_from_centroid(dropoff_df, centroid)

            num_clusters = determine_clusters_needed(dropoff_df, max_group_size=4)

            dropoff_df, kmeans = cluster_passengers(dropoff_df, num_clusters)

            dropoff_df = adjust_groups(dropoff_df, max_unique_postals=4, max_group_size=4)
            dropoff_df = soft_merge_by_sector(dropoff_df, max_group_size=4, max_unique_postals=4, max_distance_km=5)

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
