import folium
from geopy.geocoders import GoogleV3, Nominatim, ArcGIS
from louvain_method_copy import *


# Tạo bản đồ
m = folium.Map(location=[51.5074, 0.1278], zoom_start=5)
# Đọc file
labels = []

with open('dataset/label.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        label = line.strip()
        labels.append(label)

# Tạo đối tượng geolocator
geolocator = Nominatim(user_agent="dsafssfcxvsdfsdvsd")
geolocator_google = GoogleV3(api_key='AIzaSyDcr7N1MJhtrFJzLncjIHSXXF9WOs9YtOE')

true_partition1, frame = louvain_algorithm(adj_matrix)
true_partition = list(filter(lambda x: len(x)>3, true_partition1))
print(true_partition)

colors = ['blue', 'darkpurple', 'black', 'cadetblue', 'lightblue', 'darkgreen', 'gray', 'green', 'orange', 'purple', 'lightgreen', 'darkblue', 'white', 'red', 'darkred', 'pink', 'lightgray', 'lightred', 'beige', 'crimson', 'chocolate', 'peru', 'darkslategray', 'steelblue', 'darkolivegreen', 'saddlebrown', 'salmon', 'darkturquoise', 'slategray', 'darkkhaki', 'olive', 'teal', 'indigo', 'mediumvioletred', 'mediumblue', 'darkorchid', 'mediumseagreen', 'mediumslateblue', 'sienna', 'mediumturquoise', 'lightseagreen', 'goldenrod', 'tomato', 'seagreen', 'dodgerblue', 'lightslategray', 'darkseagreen', 'deeppink', 'darkgray', 'darkmagenta']
print(len(true_partition),len(colors))
# Tìm kiếm tọa độ của các điểm và đánh dấu trên bản đồ
for i,cluster in enumerate(true_partition):
    if i>=len(colors):
        break
    for point in cluster:       
        location = geolocator.geocode(labels[point]+ ", Europe")
        if location is not None:
            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=labels[point],
                icon=folium.Icon(color=colors[i])
            ).add_to(m)

# Hiển thị bản đồ

# Lưu bản đồ vào một file HTML
m.save('map.html')


import csv

data = true_partition1

# Define the output file name
filename = "result_louvain.csv"

# Open the output file in write mode
with open(filename, mode='w', newline='') as file:
    # Create a writer object
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['id', 'group'])

    # Write the data rows
    for group_index, group in enumerate(data):
        for id in group:
            writer.writerow([id, group_index])