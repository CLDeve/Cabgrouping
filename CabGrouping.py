import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components
import time
import math
from sklearn.cluster import KMeans
from geopy.distance import geodesic

__version__ = "v0.0.1.6"

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
    progress_placeholder = st.empty()
    eta_placeholder = st.empty()

    st.write("###")

    download_placeholder = st.empty()

if run_button:
    if master_file and upload_file:
        # Update the status to show "Processing..."
        status_placeholder.text("Processing...")
        start_time = time.perf_counter()
        progress_bar = progress_placeholder.progress(0)
        eta_placeholder.text("ETA: --")

        def update_progress(pct):
            progress_bar.progress(pct)
            if pct <= 0:
                eta_placeholder.text("ETA: --")
                return
            elapsed = time.perf_counter() - start_time
            total_est = elapsed / (pct / 100.0)
            remaining = max(0.0, total_est - elapsed)
            eta_placeholder.text(f"ETA: {remaining:.1f} s")

        master_df = pd.read_excel(master_file)
        update_progress(10)

        upload_df = pd.read_excel(upload_file)
        update_progress(20)

        if 'StaffID' in upload_df.columns:
            upload_df = upload_df.drop_duplicates(subset=['StaffID'], keep='first')

        pickup_df = upload_df.merge(master_df, left_on='PickUpPostal', right_on='PostalCode', how='left', suffixes=('', '_Pickup'))
        pickup_df.rename(columns={'Latitude': 'PickUp_Latitude', 'Longitude': 'PickUp_Longitude'}, inplace=True)
        update_progress(30)

        dropoff_df = pickup_df.merge(master_df, left_on='DropOffPostal', right_on='PostalCode', how='left', suffixes=('', '_DropOff'))
        dropoff_df.rename(columns={'Latitude': 'DropOff_Latitude', 'Longitude': 'DropOff_Longitude'}, inplace=True)
        update_progress(40)

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
            update_progress(100)
        else:
            dropoff_df.dropna(subset=['PickUp_Latitude', 'PickUp_Longitude', 'DropOff_Latitude', 'DropOff_Longitude'], inplace=True)
            update_progress(50)

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

            def adjust_groups(df, max_unique_postals=4, max_group_size=4, start_counter=1):
                adjusted_df = pd.DataFrame(columns=df.columns)
                adjusted_df['TaxiGroup'] = ''  # Initialize the 'TaxiGroup' column with empty strings

                taxi_group_counter = start_counter
                for cluster in df['Cluster'].unique():
                    cluster_df = df[df['Cluster'] == cluster].copy()

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

                        for i, row in cluster_df.iterrows():
                            potential_postals = unique_postals.union([row['PickUpPostal'], row['DropOffPostal']])

                            if len(potential_postals) > max_unique_postals or group_size >= max_group_size:
                                break

                            unique_postals = potential_postals
                            group = pd.concat([group, pd.DataFrame([row])])
                            group_size += 1

                        group['TaxiGroup'] = f'Taxi {taxi_group_counter}'
                        adjusted_df = pd.concat([adjusted_df, group], ignore_index=True)

                        cluster_df = cluster_df.drop(group.index)

                        taxi_group_counter += 1

                return adjusted_df, taxi_group_counter

            def normalize_postal(code):
                if pd.isna(code):
                    return None
                digits = ''.join(ch for ch in str(code).strip() if ch.isdigit())
                if len(digits) < 2:
                    return None
                if len(digits) <= 6:
                    return digits.zfill(6)
                return digits[:6]

            def assign_groups_dropoff_first(df, max_group_size=4):
                # Keep the same drop-off together (split only if > max_group_size).
                centroid = (df['DropOff_Latitude'].mean(), df['DropOff_Longitude'].mean())
                dropoff_groups = []
                for dropoff, g in df.groupby('DropOffPostal'):
                    lat = g['DropOff_Latitude'].mean()
                    lon = g['DropOff_Longitude'].mean()
                    dist = calculate_distance(centroid, (lat, lon))
                    dropoff_groups.append({
                        'dropoff': dropoff,
                        'indices': g.index.tolist(),
                        'dist': dist
                    })

                dropoff_groups = sorted(dropoff_groups, key=lambda x: x['dist'])

                taxis = []
                for group in dropoff_groups:
                    indices = group['indices']
                    chunks = [indices[i:i + max_group_size] for i in range(0, len(indices), max_group_size)]

                    for chunk in chunks:
                        best_taxi = None
                        best_score = None
                        for taxi in taxis:
                            capacity_left = max_group_size - len(taxi['indices'])
                            if len(chunk) > capacity_left:
                                continue
                            score = capacity_left - len(chunk)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_taxi = taxi

                        if best_taxi is None:
                            taxis.append({'indices': list(chunk)})
                        else:
                            best_taxi['indices'].extend(chunk)

                return taxis

            def order_taxis_by_centroid_kmeans(taxis, df):
                if len(taxis) <= 1:
                    return taxis

                centroid_rows = []
                for i, taxi in enumerate(taxis):
                    sub = df.loc[taxi['indices']]
                    centroid_rows.append({
                        'idx': i,
                        'lat': sub['DropOff_Latitude'].mean(),
                        'lon': sub['DropOff_Longitude'].mean()
                    })

                centroid_df = pd.DataFrame(centroid_rows)
                n_clusters = min(len(centroid_df), max(1, math.ceil(len(centroid_df) / 4)))
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(
                        centroid_df[['lat', 'lon']]
                    )
                    centroid_df['Cluster'] = kmeans.labels_
                    cluster_centroids = centroid_df.groupby('Cluster')[['lat', 'lon']].mean()
                    centroid_df = centroid_df.join(cluster_centroids, on='Cluster', rsuffix='_c')
                    centroid_df['DistToCluster'] = centroid_df.apply(
                        lambda r: calculate_distance((r['lat'], r['lon']), (r['lat_c'], r['lon_c'])),
                        axis=1
                    )
                    centroid_df = centroid_df.sort_values(by=['Cluster', 'DistToCluster'])
                else:
                    sector_centroid = (centroid_df['lat'].mean(), centroid_df['lon'].mean())
                    centroid_df['DistToSector'] = centroid_df.apply(
                        lambda r: calculate_distance((r['lat'], r['lon']), sector_centroid),
                        axis=1
                    )
                    centroid_df = centroid_df.sort_values(by='DistToSector')

                ordered = []
                for i in centroid_df['idx'].tolist():
                    ordered.append(taxis[i])
                return ordered

            def mix_nearby_sectors(df, max_group_size=4, speed_kmh=80, max_travel_minutes=45):
                df = df.copy()

                def build_taxi_groups():
                    groups = {}
                    for taxi, g in df.groupby('TaxiGroup'):
                        centroid = (g['DropOff_Latitude'].mean(), g['DropOff_Longitude'].mean())
                        dropoff_groups = []
                        for dropoff, dg in g.groupby('DropOffPostal'):
                            dropoff_groups.append({
                                'dropoff': dropoff,
                                'indices': dg.index.tolist(),
                                'size': len(dg),
                                'lat': dg['DropOff_Latitude'].mean(),
                                'lon': dg['DropOff_Longitude'].mean()
                            })
                        groups[taxi] = {
                            'indices': g.index.tolist(),
                            'centroid': centroid,
                            'dropoff_groups': dropoff_groups
                        }
                    return groups

                changed = True
                while changed:
                    changed = False
                    taxi_groups = build_taxi_groups()
                    donors = sorted(taxi_groups.items(), key=lambda kv: len(kv[1]['indices']))

                    for donor_name, donor in donors:
                        if donor_name not in taxi_groups:
                            continue
                        for group in donor['dropoff_groups']:
                            best_target = None
                            best_score = None

                            for target_name, target in taxi_groups.items():
                                if target_name == donor_name:
                                    continue
                                capacity_left = max_group_size - len(target['indices'])
                                if group['size'] > capacity_left:
                                    continue
                                dist_km = calculate_distance(
                                    (group['lat'], group['lon']),
                                    target['centroid']
                                )
                                travel_minutes = (dist_km / speed_kmh) * 60
                                if travel_minutes > max_travel_minutes:
                                    continue
                                score = capacity_left - group['size']
                                if best_score is None or score < best_score:
                                    best_score = score
                                    best_target = target_name

                            if best_target is not None:
                                df.loc[group['indices'], 'TaxiGroup'] = best_target
                                changed = True
                                break

                        if changed:
                            break

                # Renumber taxis to keep labels tidy
                def taxi_sort_key(name):
                    try:
                        return int(str(name).split()[-1])
                    except Exception:
                        return 10**9

                ordered = sorted(df['TaxiGroup'].unique(), key=taxi_sort_key)
                mapping = {old: f'Taxi {i}' for i, old in enumerate(ordered, start=1)}
                df['TaxiGroup'] = df['TaxiGroup'].map(mapping)
                return df

            dropoff_df['DropOffPostalNorm'] = dropoff_df['DropOffPostal'].apply(normalize_postal)
            dropoff_df['DropOffSector'] = dropoff_df['DropOffPostalNorm'].apply(
                lambda x: x[:2] if x else None
            )

            grouped_results = []
            taxi_counter = 1
            for _, sector_df in dropoff_df.groupby('DropOffSector', dropna=False):
                taxis = assign_groups_dropoff_first(sector_df, max_group_size=4)
                taxis = order_taxis_by_centroid_kmeans(taxis, sector_df)

                for taxi in taxis:
                    for row_idx in taxi['indices']:
                        sector_df.at[row_idx, 'TaxiGroup'] = f'Taxi {taxi_counter}'
                    taxi_counter += 1

                grouped_results.append(sector_df)

            dropoff_df = pd.concat(grouped_results, ignore_index=True)
            update_progress(80)
            dropoff_df = mix_nearby_sectors(
                dropoff_df,
                max_group_size=4,
                speed_kmh=80,
                max_travel_minutes=45
            )
            update_progress(90)

            output_excel_file_path = 'Taxi_Grouped_Data.xlsx'
            dropoff_df.to_excel(output_excel_file_path, index=False)
            update_progress(95)

            status_placeholder.success("Completed!")
            update_progress(100)

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
